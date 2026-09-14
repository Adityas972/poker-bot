import os

import pytest

import poker_bot.web.app as web_app
from poker_bot.deep_cfr.networks import InfosetMLP
from poker_bot.deep_cfr.trainer import save_policy_net


@pytest.fixture
def client(monkeypatch):
    # Use an untrained (random-weight) network so tests don't depend on a
    # real trained model file existing on disk, and bypass the on-disk
    # reload check entirely (that's exercised separately below).
    fake_net = InfosetMLP(hidden_dim=8)
    monkeypatch.setattr(web_app, "_get_policy_net", lambda: fake_net)
    web_app.app.config["TESTING"] = True
    with web_app.app.test_client() as c:
        # Each test client gets its own cookie jar, so the first request
        # below allocates a fresh, isolated SessionState automatically.
        yield c


def test_state_before_any_hand(client):
    resp = client.get("/api/state")
    assert resp.status_code == 200
    assert resp.json["in_hand"] is False


def test_new_hand_deals_and_returns_legal_actions_for_whoever_acts_first(client):
    resp = client.post("/api/new_hand")
    assert resp.status_code == 200
    data = resp.json
    assert data["in_hand"] is True
    assert data["street"] == "Preflop"
    assert len(data["your_hole"]) == 2
    # Player 0 (human) always acts first preflop in this engine.
    assert data["to_act"] == "you"
    assert len(data["legal_actions"]) > 0


def test_action_endpoint_rejects_illegal_action(client):
    client.post("/api/new_hand")
    resp = client.post("/api/action", json={"action": "NOT_A_REAL_ACTION"})
    assert resp.status_code == 400
    assert "error" in resp.json


def test_action_endpoint_rejects_when_no_hand_in_progress(client):
    resp = client.post("/api/action", json={"action": "CALL"})
    assert resp.status_code == 400


def test_folding_ends_hand_and_updates_session_score(client):
    client.post("/api/new_hand")
    resp = client.post("/api/action", json={"action": "FOLD"})
    data = resp.json
    assert data["terminal"] is True
    assert data["outcome"] == "you_folded"
    assert data["net"]["you"] == -50  # lost the small blind
    assert data["session_net"]["you"] == -50


def test_get_policy_net_reloads_when_file_changes_on_disk(tmp_path, monkeypatch):
    """Regression test: a long-running server used to cache the policy
    network forever, silently ignoring a retrain that finished later --
    _get_policy_net must notice the file's mtime changed and reload."""
    model_path = tmp_path / "policy_net.pt"
    monkeypatch.setattr(web_app, "MODEL_PATH", model_path)
    monkeypatch.setattr(web_app, "_policy_net", None)
    monkeypatch.setattr(web_app, "_policy_net_mtime", None)

    save_policy_net(InfosetMLP(hidden_dim=8), 8, str(model_path))
    os.utime(model_path, (1_000_000_000, 1_000_000_000))
    first = web_app._get_policy_net()
    again = web_app._get_policy_net()
    assert again is first  # unchanged file -> served from cache

    save_policy_net(InfosetMLP(hidden_dim=8), 8, str(model_path))
    os.utime(model_path, (2_000_000_000, 2_000_000_000))  # simulates a later retrain
    after_retrain = web_app._get_policy_net()
    assert after_retrain is not first  # changed file -> reloaded


def test_playing_to_showdown_reveals_both_hands(client):
    client.post("/api/new_hand")
    data = None
    for _ in range(20):
        state = client.get("/api/state").json
        if state["terminal"]:
            break
        legal = [a["action"] for a in state["legal_actions"]]
        action = "CALL" if "CALL" in legal else legal[0]
        data = client.post("/api/action", json={"action": action}).json
        if data["terminal"]:
            break

    assert data is not None and data["terminal"] is True
    if data["outcome"] == "showdown":
        assert "your_hand_desc" in data and "bot_hand_desc" in data
        assert len(data["full_board"]) == 5
        assert sum(data["net"].values()) == 0
