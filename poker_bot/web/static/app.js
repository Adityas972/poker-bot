const SUIT_SYMBOL = { s: "♠", h: "♥", d: "♦", c: "♣" };
const RED_SUITS = new Set(["h", "d"]);

function cardEl(cardStr) {
  const rank = cardStr.slice(0, -1);
  const suit = cardStr.slice(-1);
  const el = document.createElement("div");
  el.className = "card" + (RED_SUITS.has(suit) ? " red" : "");
  el.textContent = rank + SUIT_SYMBOL[suit];
  return el;
}

function backCardEl() {
  const el = document.createElement("div");
  el.className = "card back";
  return el;
}

function renderCards(container, cardStrs) {
  container.innerHTML = "";
  cardStrs.forEach((c) => container.appendChild(cardEl(c)));
}

function renderHiddenCards(container, count) {
  container.innerHTML = "";
  for (let i = 0; i < count; i++) container.appendChild(backCardEl());
}

const els = {
  error: document.getElementById("error-banner"),
  session: document.getElementById("session-score"),
  botStack: document.getElementById("bot-stack"),
  yourStack: document.getElementById("your-stack"),
  botCards: document.getElementById("bot-cards"),
  boardCards: document.getElementById("board-cards"),
  yourCards: document.getElementById("your-cards"),
  pot: document.getElementById("pot"),
  streetLabel: document.getElementById("street-label"),
  resultPanel: document.getElementById("result-panel"),
  resultText: document.getElementById("result-text"),
  newHandBtn: document.getElementById("new-hand-btn"),
  actionPanel: document.getElementById("action-panel"),
  actionButtons: document.getElementById("action-buttons"),
  startPanel: document.getElementById("start-panel"),
  startBtn: document.getElementById("start-btn"),
  botLog: document.getElementById("bot-log"),
};

function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = false;
}

function clearError() {
  els.error.hidden = true;
}

function appendBotLog(entries) {
  for (const entry of entries) {
    const div = document.createElement("div");
    div.className = "bot-log-entry";
    const strategyStr = Object.entries(entry.strategy)
      .sort((a, b) => b[1] - a[1])
      .map(([a, p]) => `${a}=${p.toFixed(2)}`)
      .join(", ");
    div.innerHTML =
      `<span class="street">${entry.street}</span>` +
      `<span class="chosen">bot chooses ${entry.chosen_label}</span>` +
      `<span class="strategy">considered: ${strategyStr}</span>`;
    els.botLog.prepend(div);
  }
}

function render(data) {
  if (!data.in_hand) {
    els.startPanel.hidden = false;
    els.actionPanel.hidden = true;
    els.resultPanel.hidden = true;
    renderHiddenCards(els.botCards, 0);
    renderCards(els.yourCards, []);
    renderCards(els.boardCards, []);
    els.pot.textContent = "0";
    els.streetLabel.textContent = "";
    updateSession(data.session_net);
    return;
  }

  els.startPanel.hidden = true;
  els.pot.textContent = data.pot;
  els.streetLabel.textContent = data.street;
  els.yourStack.textContent = data.your_stack;
  els.botStack.textContent = data.bot_stack;
  renderCards(els.boardCards, data.board);
  renderCards(els.yourCards, data.your_hole);
  updateSession(data.session_net);

  if (data.terminal) {
    renderCards(els.botCards, data.bot_hole);
    renderCards(els.boardCards, data.full_board);
    els.actionPanel.hidden = true;
    els.resultPanel.hidden = false;
    els.resultText.textContent = resultText(data);
  } else {
    renderHiddenCards(els.botCards, 2);
    els.resultPanel.hidden = true;
    if (data.to_act === "you") {
      els.actionPanel.hidden = false;
      renderActionButtons(data.legal_actions);
    } else {
      els.actionPanel.hidden = true;
    }
  }
}

function resultText(data) {
  if (data.outcome === "you_folded") return "You folded. Bot wins the pot.";
  if (data.outcome === "bot_folded") return "Bot folded. You win the pot.";
  const net = data.net.you;
  const outcome = net > 0 ? `You win ${net} chips` : net < 0 ? `You lose ${-net} chips` : "Chop";
  return `Showdown — you: ${data.your_hand_desc}, bot: ${data.bot_hand_desc}. ${outcome}.`;
}

function updateSession(session) {
  const fmt = (n) => (n >= 0 ? `+${n}` : `${n}`);
  els.session.textContent = `Session: you ${fmt(session.you)} · bot ${fmt(session.bot)}`;
}

function renderActionButtons(actions) {
  els.actionButtons.innerHTML = "";
  for (const a of actions) {
    const btn = document.createElement("button");
    btn.textContent = a.label;
    if (a.action === "FOLD") btn.className = "fold";
    btn.addEventListener("click", () => sendAction(a.action));
    els.actionButtons.appendChild(btn);
  }
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "request failed");
  return data;
}

async function startHand() {
  clearError();
  try {
    const data = await postJSON("/api/new_hand");
    appendBotLog(data.bot_log);
    render(data);
  } catch (e) {
    showError(e.message);
  }
}

async function sendAction(action) {
  clearError();
  try {
    const data = await postJSON("/api/action", { action });
    appendBotLog(data.bot_log);
    render(data);
  } catch (e) {
    showError(e.message);
  }
}

els.startBtn.addEventListener("click", startHand);
els.newHandBtn.addEventListener("click", startHand);

fetch("/api/state").then((r) => r.json()).then(render).catch(() => {});
