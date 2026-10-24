/* BridgeSign – Sentence mode
 * Signed words (from the live recognizer, taps, or typing) go into a tray;
 * the local sentence brain suggests English sentences; the signer picks one,
 * which is shown and spoken to the other person. The other person replies by
 * voice, typing, or quick replies. Depends on app.js (currentMode, speak,
 * showToast, refreshIcons).
 */
(function () {
  "use strict";

  const QUICK_REPLIES = [
    "Yes.", "No.", "Can you repeat that?", "Please wait a moment.",
    "I understand.", "I don't understand.", "Can you show me?", "Thank you.",
  ];

  const state = {
    tray: [],               // [{gloss, source}]
    consumed: 0,            // server signed_glosses already moved into the tray
    skipNextServer: false,  // after Clear, ignore glosses an in-flight response still carries
    spelling: "",
    question: false,
    vocab: null,            // {groups, glosses:Set}
    vocabPromise: null,
    activeGroup: "",
    suggestTimer: null,
    suggestSeq: 0,
    log: [],
    recognizer: null,
  };

  const $ = id => document.getElementById(id);

  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    Object.entries(props).forEach(([key, value]) => {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value);
    });
    children.forEach(child => node.appendChild(child));
    return node;
  }

  function displayGloss(gloss) {
    if (gloss.startsWith("fs-")) return gloss.slice(3) + " ✋";
    return gloss.replace(/-/g, " ");
  }

  function normalizeTyped(word) {
    const upper = String(word || "").trim().toUpperCase().replace(/\s+/g, "-");
    if (!upper) return "";
    if (state.vocab && state.vocab.glosses.has(upper)) return upper;
    return "fs-" + upper.replace(/[^A-Z0-9'-]/g, "");
  }

  // ── Vocabulary / picker ──────────────────────────────────────────────
  function loadVocab() {
    // One shared request: opening the mode and the picker both ask for it.
    if (!state.vocabPromise) state.vocabPromise = fetchVocab();
    return state.vocabPromise;
  }

  async function fetchVocab() {
    try {
      const res = await fetch("/api/sentence/vocab", { credentials: "same-origin" });
      const data = await res.json();
      const groups = {};
      (data.groups || []).forEach(group => { groups[group.name] = group.glosses; });
      const glosses = new Set(Object.values(groups).flat());
      state.vocab = { groups, glosses };
      state.activeGroup = Object.keys(state.vocab.groups)[0] || "";
      setBrainStatus(data.brain_available, null, data.brain_error);
      renderPicker();
    } catch (error) {
      state.vocabPromise = null;  // allow a retry
      setBrainStatus(false, "Could not load the sign list.");
    }
  }

  function setBrainStatus(available, message, problem) {
    const status = $("sentBrainStatus");
    if (!status) return;
    status.textContent = message || (available
      ? "Suggestions come from the local sentence model on this server. Check before sending."
      : "Sentence model not loaded – showing a literal reading of the signs." + (problem ? " " + problem : ""));
  }

  function renderPicker() {
    const groups = $("sentGroups");
    const chips = $("sentChips");
    if (!groups || !chips || !state.vocab) return;
    groups.replaceChildren();
    Object.keys(state.vocab.groups).forEach(name => {
      groups.appendChild(el("button", {
        type: "button",
        class: "sent-group" + (name === state.activeGroup ? " active" : ""),
        role: "tab",
        "aria-selected": String(name === state.activeGroup),
        text: name,
        onclick: () => { state.activeGroup = name; renderPicker(); },
      }));
    });
    chips.replaceChildren();
    (state.vocab.groups[state.activeGroup] || []).filter(g => g !== "Q").forEach(gloss => {
      chips.appendChild(el("button", {
        type: "button", class: "sent-chip", text: displayGloss(gloss),
        onclick: () => addSign(gloss, "tap"),
      }));
    });
  }

  window.toggleSignPicker = function () {
    const picker = $("sentPicker");
    const button = $("sentPickerBtn");
    if (!picker) return;
    picker.hidden = !picker.hidden;
    if (button) button.setAttribute("aria-expanded", String(!picker.hidden));
    if (!picker.hidden) loadVocab();
  };

  window.addTypedSign = function (event) {
    event.preventDefault();
    const input = $("sentTypeInput");
    const gloss = normalizeTyped(input && input.value);
    if (gloss) addSign(gloss, "typed");
    if (input) input.value = "";
  };

  // ── Tray ─────────────────────────────────────────────────────────────
  function addSign(gloss, source) {
    if (!gloss) return;
    state.tray.push({ gloss, source });
    renderTray();
    scheduleSuggest();
  }

  function removeSign(index) {
    state.tray.splice(index, 1);
    renderTray();
    scheduleSuggest();
  }

  function renderTray() {
    const tray = $("sentTray");
    if (!tray) return;
    tray.replaceChildren();
    state.tray.forEach((item, index) => {
      const chip = el("span", { class: "sent-token" + (item.source === "sign" ? " signed" : "") }, [
        el("span", { text: displayGloss(item.gloss) }),
        el("button", {
          type: "button", class: "sent-token-x", "aria-label": `Remove ${displayGloss(item.gloss)}`, text: "×",
          onclick: () => removeSign(index),
        }),
      ]);
      tray.appendChild(chip);
    });
    if (state.question) tray.appendChild(el("span", { class: "sent-token question", text: "?" }));
    if (state.spelling) tray.appendChild(el("span", { class: "sent-token spelling", text: `spelling ${state.spelling}…` }));
    if (!tray.childNodes.length) {
      tray.appendChild(el("span", { class: "sent-empty", text: "Sign words, fingerspell, or tap “Add sign”." }));
    }
  }

  window.toggleSentenceQuestion = function () {
    state.question = !state.question;
    const button = $("sentQuestionBtn");
    if (button) {
      button.setAttribute("aria-pressed", String(state.question));
      button.classList.toggle("active", state.question);
    }
    renderTray();
    scheduleSuggest();
  };

  window.clearSentence = function () {
    state.tray = [];
    state.question = false;
    const button = $("sentQuestionBtn");
    if (button) { button.setAttribute("aria-pressed", "false"); button.classList.remove("active"); }
    state.skipNextServer = true;
    fetch("/api/sentence/clear", { method: "POST", credentials: "same-origin" }).catch(() => {});
    renderTray();
    renderSuggestions([]);
  };

  // ── Live recognizer feed ─────────────────────────────────────────────
  window.onSentenceInfer = function (d) {
    const glosses = Array.isArray(d.signed_glosses) ? d.signed_glosses : [];
    if (state.skipNextServer) {
      state.consumed = glosses.length;
      state.skipNextServer = false;
    } else if (glosses.length < state.consumed) {
      state.consumed = 0;   // server session was reset (camera restart / mode switch)
    }
    const fresh = glosses.slice(state.consumed);
    state.consumed = glosses.length;
    const spelling = d.word_buffer || "";
    const spellingChanged = spelling !== state.spelling;
    state.spelling = spelling;
    if (fresh.length) {
      fresh.forEach(gloss => state.tray.push({ gloss, source: "sign" }));
      renderTray();
      scheduleSuggest();
    } else if (spellingChanged) {
      renderTray();
    }
  };

  window.onSentenceModeOpened = function () {
    state.consumed = 0;
    state.skipNextServer = true;
    loadVocab();
    renderTray();
    renderQuickReplies();
    renderLog();
  };

  // ── Suggestions ──────────────────────────────────────────────────────
  function scheduleSuggest() {
    clearTimeout(state.suggestTimer);
    state.suggestTimer = setTimeout(fetchSuggestions, 350);
  }

  async function fetchSuggestions() {
    const signs = state.tray.map(item => item.gloss);
    if (!signs.length) { renderSuggestions([]); return; }
    if (state.question) signs.push("Q");
    const seq = ++state.suggestSeq;
    const box = $("sentSuggestions");
    if (box) box.setAttribute("aria-busy", "true");
    try {
      const res = await fetch("/api/sentence/suggest", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ signs }),
      });
      const data = await res.json();
      if (seq !== state.suggestSeq) return;  // a newer request superseded this one
      if (!res.ok) throw new Error(data.error || "Suggestion request failed");
      setBrainStatus(data.brain_available, null, data.brain_error);
      renderSuggestions(data.suggestions || []);
    } catch (error) {
      if (seq === state.suggestSeq) {
        renderSuggestions([]);
        setBrainStatus(false, "Could not get suggestions. You can still type or use quick phrases.");
      }
    } finally {
      if (box && seq === state.suggestSeq) box.removeAttribute("aria-busy");
    }
  }

  function renderSuggestions(list) {
    const box = $("sentSuggestions");
    if (!box) return;
    box.replaceChildren();
    if (!list.length) {
      box.appendChild(el("span", { class: "sent-empty", text: "Suggestions appear here." }));
      return;
    }
    list.forEach((item, index) => {
      const percent = Math.round((item.score || 0) * 100);
      box.appendChild(el("button", {
        type: "button",
        class: "sent-suggestion" + (index === 0 ? " best" : ""),
        title: item.source === "literal" ? "Literal reading of the signs" : `Model preference ${percent}%`,
        onclick: () => chooseSentence(item.text),
      }, [
        el("span", { class: "sent-suggestion-text", text: item.text }),
        el("span", { class: "sent-suggestion-meta", text: item.source === "literal" ? "literal" : `${percent}%` }),
      ]));
    });
  }

  function chooseSentence(text) {
    addTurn("signer", text);
    if (typeof speak === "function") speak(text);
    window.clearSentence();
  }

  // ── Conversation ─────────────────────────────────────────────────────
  function addTurn(who, text) {
    const clean = String(text || "").trim();
    if (!clean) return;
    state.log.push({ who, text: clean, at: new Date() });
    if (state.log.length > 50) state.log.shift();
    renderLog();
  }

  function renderLog() {
    const log = $("convLog");
    if (!log) return;
    log.replaceChildren();
    state.log.forEach((turn, index) => {
      const time = turn.at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      log.appendChild(el("li", {
        class: `conv-turn ${turn.who}` + (index === state.log.length - 1 ? " latest" : ""),
      }, [
        el("span", { class: "conv-who", text: `${turn.who === "signer" ? "Signer" : "Other person"} · ${time}` }),
        el("span", { class: "conv-text", text: turn.text }),
      ]));
    });
    if (!state.log.length) {
      log.appendChild(el("li", { class: "sent-empty", text: "Confirmed sentences and replies appear here in large text." }));
    }
    log.scrollTop = log.scrollHeight;
  }

  window.clearConversation = function () {
    state.log = [];
    renderLog();
  };

  function renderQuickReplies() {
    const box = $("convQuick");
    if (!box || box.childNodes.length) return;
    QUICK_REPLIES.forEach(text => {
      box.appendChild(el("button", {
        type: "button", class: "sent-chip", text,
        onclick: () => addTurn("other", text),
      }));
    });
  }

  window.sendTypedReply = function (event) {
    event.preventDefault();
    const input = $("convReplyInput");
    addTurn("other", input && input.value);
    if (input) input.value = "";
  };

  window.toggleReplyMic = function () {
    const button = $("convMicBtn");
    if (state.recognizer) {
      state.recognizer.stop();
      return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      if (typeof showToast === "function") showToast("Speech input is not available in this browser – type the reply instead.");
      return;
    }
    const recognizer = new SR();
    recognizer.lang = "en-US";
    recognizer.interimResults = true;
    recognizer.continuous = false;
    const input = $("convReplyInput");
    recognizer.onresult = event => {
      const text = Array.from(event.results).map(r => r[0].transcript).join(" ");
      if (input) input.value = text;
      if (event.results[event.results.length - 1].isFinal) {
        addTurn("other", text);
        if (input) input.value = "";
      }
    };
    recognizer.onerror = event => {
      if (typeof showToast === "function") showToast(`Speech input stopped (${event.error}). You can type instead.`);
    };
    recognizer.onend = () => {
      state.recognizer = null;
      if (button) { button.setAttribute("aria-pressed", "false"); button.classList.remove("listening"); }
    };
    state.recognizer = recognizer;
    if (button) { button.setAttribute("aria-pressed", "true"); button.classList.add("listening"); }
    recognizer.start();
  };
})();
