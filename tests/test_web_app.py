import pytest

import poker_bot.web.app as web_app
from poker_bot.deep_cfr.networks import InfosetMLP


@pytest.fixture
def client(monkeypatch):
    # Use an untrained (random-weight) network so tests don't depend on a
    # real trained model file existing on disk.
    monkeypatch.setattr(web_app, "_policy_net", InfosetMLP(hidden_dim=8))
    monkeypatch.setattr(web_app, "_state", None)
    monkeypatch.setattr(web_app, "_session_net", [0, 0])
    web_app.app.config["TESTING"] = True
    with web_app.app.test_client() as c:
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
