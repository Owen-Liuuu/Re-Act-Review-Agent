(function () {
  var status = document.body.getAttribute("data-status") || "";
  setupAccount();
  if (status === "idle") {
    setupHome();
    return;
  }
  var runId = document.body.getAttribute("data-run-id") || "";
  var liveStage = document.body.getAttribute("data-live-stage") || "";
  window.deskEstimateS = parseFloat(document.body.getAttribute("data-estimate-s") || "") || 0;
  if (!runId) return;
  if (status !== "running" && status !== "queued") return;

  var stepCount = -1;

  function poll() {
    fetch("/runs/" + encodeURIComponent(runId), {
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data || typeof data !== "object") return;
        if (data.status !== status) {
          location.reload();
          return;
        }
        var n = Array.isArray(data.steps) ? data.steps.length : 0;
        if (stepCount >= 0 && n !== stepCount) {
          location.reload();
          return;
        }
        stepCount = n;
        var live = data.live;
        if (live && live.stage && liveStage && live.stage !== liveStage) {
          location.reload();
          return;
        }
        if (live && live.stage) liveStage = live.stage;
        patchLive(live);
      })
      .catch(function () {});
  }

  function patchLive(live) {
    if (!live) return;
    var title = document.getElementById("desk-live-title");
    if (title && live.title) title.textContent = live.title;
    var cap = document.getElementById("desk-live-caption");
    if (cap) {
      cap.textContent = live.caption || live.label || live.title || cap.textContent;
    }
    var box = document.getElementById("desk-live");
    if (box && live.started_unix) {
      box.setAttribute("data-started-unix", String(live.started_unix));
    }
    if (live.estimate_label || live.estimate_s) {
      window.deskEstimateS = live.estimate_max_s || live.estimate_s;
      var est = document.getElementById("desk-estimate");
      if (est) {
        est.textContent = live.estimate_label || fmtRange(
          live.estimate_min_s, live.estimate_max_s || live.estimate_s);
      }
    }
    tickElapsed();
  }

  function fmtDur(seconds, approx) {
    var sec = Math.max(0, Math.round(Number(seconds) || 0));
    var body = sec < 90 ? sec + "s" : Math.max(1, Math.round(sec / 60)) + " min";
    return approx ? "~" + body : body;
  }

  function fmtRange(lo, hi) {
    var a = Math.max(0, Math.round(Number(lo) || 0));
    var b = Math.max(a, Math.round(Number(hi) || 0));
    if (!b) return "";
    if (a === b) return fmtDur(a, false);
    if (b <= 90) return a + "–" + b + "s";
    var hiM = Math.max(1, Math.ceil(b / 60));
    if (a < 60) return a + "s–" + hiM + " min";
    var loM = Math.max(1, Math.floor(a / 60));
    return loM === hiM ? hiM + " min" : loM + "–" + hiM + " min";
  }

  function tickElapsed() {
    var live = document.getElementById("desk-live");
    var el = document.getElementById("desk-elapsed");
    if (!live || !el) return;
    var started = parseFloat(live.getAttribute("data-started-unix") || "");
    if (!started) return;
    var sec = Math.max(0, Math.round(Date.now() / 1000 - started));
    el.textContent = fmtDur(sec, false);
    var bar = document.getElementById("desk-eta-bar");
    if (bar && window.deskEstimateS) {
      var pct = Math.min(100, Math.round(100 * sec / window.deskEstimateS));
      var inner = bar.querySelector("i");
      if (inner) inner.style.width = pct + "%";
    }
  }

  setInterval(poll, 2000);
  poll();
  setInterval(tickElapsed, 1000);

  function catalog() {
    var el = document.getElementById("desk-catalog");
    if (!el) return null;
    try { return JSON.parse(el.textContent || "{}"); } catch (e) { return null; }
  }

  function fillModels(sel, vendor, vision, data) {
    var list = ((vision ? data.vision : data.text)[vendor] || []);
    sel.innerHTML = list.map(function (m) {
      return "<option value=\"" + m + "\">" + m + "</option>";
    }).join("");
  }

  function gearVendor(letter) {
    var id = letter === "c" ? "c-vendor" : letter === "s" ? "s-vendor" : "v-vendor";
    var el = document.getElementById(id);
    return el ? el.value : "";
  }

  function syncKeyLabels(data) {
    if (!data || !data.hints) return;
    [["c", "Complex"], ["s", "Simple"], ["v", "Visual"]].forEach(function (pair) {
      var letter = pair[0];
      var title = pair[1];
      var vendor = gearVendor(letter) || "DeepSeek";
      var hint = data.hints[vendor] || { placeholder: "…", where: "" };
      var lab = document.getElementById("lab-" + letter);
      var input = document.getElementById("key-" + letter);
      var hintEl = document.getElementById("hint-" + letter);
      if (lab) lab.textContent = title + " · " + vendor;
      if (input) input.placeholder = hint.placeholder;
      if (hintEl) hintEl.textContent = "Native " + vendor + " key · " + hint.where;
    });
  }

  function openSheet() {
    var veil = document.getElementById("veil");
    var sheet = document.getElementById("sheet");
    if (!veil || !sheet) return;
    veil.hidden = false;
    sheet.hidden = false;
    syncKeyLabels(catalog());
    var who = document.getElementById("who");
    if (who) who.focus();
  }

  function closeSheet() {
    var veil = document.getElementById("veil");
    var sheet = document.getElementById("sheet");
    if (veil) veil.hidden = true;
    if (sheet) sheet.hidden = true;
    ["wrap-c", "wrap-s", "wrap-v"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.classList.remove("need");
    });
  }

  function accountLabel() {
    var nameEl = document.getElementById("who");
    var name = nameEl ? nameEl.value.trim() : "";
    var n = ["key-c", "key-s", "key-v"].filter(function (id) {
      var el = document.getElementById(id);
      return el && el.value.trim();
    }).length;
    var btn = document.getElementById("account-btn");
    if (!btn) return;
    if (!name && n === 0) { btn.textContent = "Account"; return; }
    btn.textContent = (name || "Account") + " · " + n + " key" + (n === 1 ? "" : "s");
  }

  function setupAccount() {
    var btn = document.getElementById("account-btn");
    var veil = document.getElementById("veil");
    var close = document.getElementById("sheet-close");
    var form = document.getElementById("account-form");
    if (btn) btn.onclick = openSheet;
    if (close) close.onclick = closeSheet;
    if (veil) veil.onclick = closeSheet;
    if (form) {
      form.onsubmit = function (e) {
        e.preventDefault();
        accountLabel();
        syncKeyHint();
        closeSheet();
      };
    }
    syncKeyLabels(catalog());
    accountLabel();
    syncKeyHint();
  }

  function syncKeyHint() {
    var hint = document.getElementById("key-hint");
    if (!hint) return;
    var n = ["key-c", "key-s", "key-v"].filter(function (id) {
      var el = document.getElementById(id);
      return el && el.value.trim();
    }).length;
    hint.hidden = n > 0;
  }

  function names(input) {
    if (!input.files || !input.files.length) return "";
    return Array.prototype.map.call(input.files, function (f) { return f.name; }).join(", ");
  }

  function showAlert(html) {
    var box = document.getElementById("missing");
    if (!box) return;
    box.innerHTML = html;
    box.hidden = false;
  }

  function setupHome() {
    var data = catalog();
    if (data) {
      function bind(vId, mId, vision) {
        var v = document.getElementById(vId);
        var m = document.getElementById(mId);
        if (!v || !m) return;
        v.onchange = function () {
          fillModels(m, v.value, vision, data);
          syncKeyLabels(data);
        };
      }
      bind("c-vendor", "c-model", false);
      bind("s-vendor", "s-model", false);
      bind("v-vendor", "v-model", true);
    }
    var review = document.getElementById("review");
    var sources = document.getElementById("sources");
    var start = document.getElementById("start");
    if (review) {
      review.onchange = function () {
        var n = names(this);
        var chip = document.getElementById("review-chip");
        var hint = document.getElementById("review-hint");
        var row = document.getElementById("review-row");
        if (chip) chip.textContent = n ? "Ready" : "Choose";
        if (hint) hint.textContent = n || "Required · published systematic review";
        if (row) {
          row.classList.toggle("ready", !!n);
          row.classList.remove("need");
        }
        var box = document.getElementById("missing");
        if (box) box.hidden = true;
      };
    }
    if (sources) {
      sources.onchange = function () {
        var n = names(this);
        var chip = document.getElementById("src-chip");
        var hint = document.getElementById("src-hint");
        var row = document.getElementById("src-row");
        if (chip) chip.textContent = n ? "Ready" : "Choose";
        if (hint) hint.textContent = n || "Optional · local papers skip retrieval";
        if (row) row.classList.toggle("ready", !!n);
      };
    }
    ["key-c", "key-s", "key-v"].forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      el.oninput = function () {
        var wrap = this.closest(".key-field");
        if (wrap) wrap.classList.remove("need");
        syncKeyHint();
      };
    });
    syncKeyHint();
    if (!start) return;
    start.onsubmit = function (e) {
      var noPdf = !review || !review.files.length;
      var missing = ["c", "s", "v"].filter(function (letter) {
        var el = document.getElementById("key-" + letter);
        return !el || !el.value.trim();
      });
      var requireKeys = document.body.getAttribute("data-require-keys") === "1";
      if (noPdf) {
        e.preventDefault();
        var row = document.getElementById("review-row");
        if (row) row.classList.add("need");
        showAlert("<strong>No review PDF detected.</strong> Use Choose on Review, pick a <span class=\"mono\">.pdf</span>, then Start run.");
        if (review) review.focus();
        return;
      }
      if (missing.length && missing.length < 3) {
        e.preventDefault();
        missing.forEach(function (letter) {
          var wrap = document.getElementById("wrap-" + letter);
          if (wrap) wrap.classList.add("need");
        });
        showAlert("<strong>API keys missing.</strong> Open Account (top right) and paste the native vendor key for each gear, then Start. Not an OpenRouter key.");
        openSheet();
        return;
      }
      if (requireKeys && missing.length) {
        e.preventDefault();
        missing.forEach(function (letter) {
          var wrap = document.getElementById("wrap-" + letter);
          if (wrap) wrap.classList.add("need");
        });
        showAlert("<strong>API keys missing.</strong> Open Account (top right) and paste the native vendor key for each gear, then Start. Not an OpenRouter key.");
        openSheet();
      }
    };
  }
})();
