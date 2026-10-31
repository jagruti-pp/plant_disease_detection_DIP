(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const fileInput = $("file"), drop = $("drop"), goBtn = $("go"), pickBtn = $("pick");
  let selected = null, data = null, activeStage = 0;

  /* ---------------- model status banner ---------------- */
  function loadModelStatus() {
    return fetch("/api/status").then((r) => r.json()).then((s) => {
      const b = $("modelBanner");
      b.classList.remove("hidden", "ok");
      if (s.model_ready) { b.classList.add("ok"); b.textContent = "Trained model loaded: " + s.classes.length + " classes ready."; }
      else b.textContent = (s.message || "Disease classification model not available. Please train/load the model.") + " Use the Train Model section below.";
    }).catch(() => {});
  }
  loadModelStatus();

  /* ---------------- upload handling ---------------- */
  function showError(msg) { const e = $("errorBox"); e.textContent = msg; e.classList.toggle("hidden", !msg); }

  function setFile(f) {
    showError("");
    if (!f) return;
    if (!/\.(jpe?g|png)$/i.test(f.name)) { showError("Unsupported file format. Please upload a JPG, JPEG or PNG image."); return; }
    if (f.size > 10 * 1024 * 1024) { showError("Image is too large. Maximum size is 10 MB."); return; }
    selected = f;
    $("previewImg").src = URL.createObjectURL(f);
    $("previewName").textContent = f.name;
    $("preview").classList.remove("hidden");
    goBtn.classList.remove("hidden");
  }

  pickBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  goBtn.addEventListener("click", async () => {
    showError("");
    if (!selected) { showError("No image uploaded. Please choose a leaf image."); return; }
    const fd = new FormData(); fd.append("image", selected);
    $("loading").classList.remove("hidden"); $("results").classList.add("hidden"); goBtn.disabled = true;
    try {
      const res = await fetch("/api/analyze", { method: "POST", body: fd });
      let json = null;
      try { json = await res.json(); } catch (_) {}
      if (!res.ok || !json || json.error) throw new Error((json && json.error) || "Server error. Please try again.");
      data = json; render();
    } catch (err) {
      showError(err.message || "Could not analyse the image.");
    } finally {
      $("loading").classList.add("hidden"); goBtn.disabled = false;
    }
  });


  /* ---------------- training on the website ---------------- */
  let training = false, pollTimer = null;
  const trainErr = (m) => { const e = $("trainError"); e.textContent = m || ""; e.classList.toggle("hidden", !m); };

  function renderDataset(ds) {
    const box = $("datasetList");
    $("classHints").innerHTML = ds.classes.map((c) => `<option value="${esc(c.name)}">`).join("");
    if (!ds.classes.length) { box.innerHTML = '<p class="muted small">No images uploaded yet.</p>'; return; }
    box.innerHTML = ds.classes.map((c) => `<div class="ds-row"><span>${esc(c.label)}<span class="n">${c.count} images</span>${c.count < 10 ? '<div class="ds-warn">Few images – accuracy improves with 20+ per class.</div>' : ""}</span>
      <button title="Remove class" data-c="${esc(c.name)}" ${training ? "disabled" : ""}>✕</button></div>`).join("") +
      `<div class="small muted" style="padding:8px 0">${ds.classes.length} classes · ${ds.total} images</div>`;
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm("Remove this class and its images?")) return;
      const r = await fetch("/api/train/class/" + encodeURIComponent(b.dataset.c), { method: "DELETE" });
      const j = await r.json(); if (j.error) trainErr(j.error); else renderDataset(j);
    }));
  }
  const refreshDataset = () => fetch("/api/train/dataset").then((r) => r.json()).then(renderDataset).catch(() => {});
  refreshDataset();

  async function uploadGroups(groups) {            // groups: {className: [File,...]}
    trainErr("");
    const total = Object.values(groups).reduce((n, a) => n + a.length, 0);
    if (!total) { trainErr("No JPG/PNG images were found in your selection."); return; }
    let done = 0, skipped = 0;
    $("trainBtn").disabled = true;
    try {
      for (const [cls, files] of Object.entries(groups)) {
        for (let i = 0; i < files.length; i += 10) {
          const fd = new FormData();
          fd.append("class_name", cls);
          files.slice(i, i + 10).forEach((f) => fd.append("images", f));
          const res = await fetch("/api/train/upload", { method: "POST", body: fd });
          let j = null; try { j = await res.json(); } catch (_) {}
          if (!res.ok || !j || j.error) throw new Error((j && j.error) || "Upload failed.");
          done += j.saved; skipped += j.skipped; renderDataset(j);
          $("uploadStatus").textContent = `Uploading… ${done + skipped} / ${total}`;
        }
      }
      $("uploadStatus").textContent = `Added ${done} image(s)` + (skipped ? `, skipped ${skipped} invalid/too-large file(s).` : ".");
    } catch (e) { trainErr(e.message); }
    finally { $("trainBtn").disabled = training; }
  }

  $("folderBtn").addEventListener("click", () => $("folderInput").click());
  $("folderInput").addEventListener("change", () => {
    const groups = {};
    Array.from($("folderInput").files).forEach((f) => {
      if (!/\.(jpe?g|png)$/i.test(f.name)) return;
      const parts = (f.webkitRelativePath || "").split("/");
      if (parts.length < 2) return;
      const cls = parts[parts.length - 2];
      (groups[cls] = groups[cls] || []).push(f);
    });
    $("folderInput").value = "";
    uploadGroups(groups);
  });
  $("classBtn").addEventListener("click", () => {
    if (!$("className").value.trim()) { trainErr("Please type the class name first (e.g. Healthy)."); $("className").focus(); return; }
    trainErr(""); $("classFiles").click();
  });
  $("classFiles").addEventListener("change", () => {
    const files = Array.from($("classFiles").files);
    $("classFiles").value = "";
    uploadGroups({ [$("className").value.trim()]: files });
  });

  function showTrain(st) {
    const prog = $("trainProgress"), msg = $("trainMsg");
    training = st.state === "running";
    $("trainBtn").disabled = training; $("trainBtn").textContent = training ? "Training…" : "Train Model";
    $("folderBtn").disabled = $("classBtn").disabled = training;
    if (training || st.state === "done") {
      prog.classList.remove("hidden");
      const pct = st.state === "done" ? 100 : Math.min(99, Math.round(100 * (st.epoch || 0) / Math.max(1, st.total_epochs)));
      $("trainBar").style.width = pct + "%";
      $("trainText").textContent = training
        ? `${st.phase || "Training"} – epoch ${st.epoch}/${st.total_epochs}` + (st.val_accuracy != null ? ` · train acc ${st.accuracy}% · val acc ${st.val_accuracy}%` : "")
        : "Finished";
    } else prog.classList.add("hidden");
    msg.classList.toggle("hidden", !(st.message || st.warning) );
    msg.classList.remove("ok", "error");
    if (st.state === "done") msg.classList.add("ok"); else if (st.state === "error") msg.classList.add("error");
    msg.textContent = [st.message, st.warning].filter(Boolean).join(" ");
  }

  function poll() {
    fetch("/api/train/status").then((r) => r.json()).then((st) => {
      showTrain(st);
      if (st.state === "running") pollTimer = setTimeout(poll, 2000);
      else { pollTimer = null; refreshDataset(); if (st.state === "done") loadModelStatus(); }
    }).catch(() => { pollTimer = setTimeout(poll, 4000); });
  }
  poll();

  $("trainBtn").addEventListener("click", async () => {
    trainErr("");
    const res = await fetch("/api/train/start", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: +$("epochs").value || 8 }) });
    let j = null; try { j = await res.json(); } catch (_) {}
    if (!res.ok || !j || j.error) { trainErr((j && j.error) || "Could not start training."); return; }
    training = true; poll();
  });

  /* ---------------- stage definitions ---------------- */
  const fig = (src, cap) => `<figure class="fig"><img src="${src}" alt="${esc(cap)}" loading="lazy"><figcaption>${esc(cap)}</figcaption></figure>`;

  function stages() {
    const im = data.images, a = data.acquisition, d = data.disease;
    const dState = d.status === "unavailable" ? "warn" : "";
    return [
      { name: "Image Acquisition", icon: "✓", cls: "", title: "Image Acquisition",
        desc: "The leaf image is read and its basic properties are measured.",
        stats: [["Width", a.width + " px"], ["Height", a.height + " px"], ["Channels", a.channels], ["Min intensity", a.min], ["Max intensity", a.max], ["Mean intensity", a.mean]],
        figs: [[im.original, "Original image"], [im.gray, "Grayscale"], [im.hsv, "HSV colour space"]] },
      { name: "Enhancement", icon: "✓", cls: "", title: "Image Enhancement", decision: 0,
        desc: "The image is measured first, then the enhancement that suits it is applied." },
      { name: "Filtering", icon: "✓", cls: "", title: "Filtering / Sharpening", decision: 1,
        desc: "Noise and sharpness are measured on the enhanced image, then the matching filter is applied." },
      { name: "Edge Detection", icon: "✓", cls: "", title: "Edge Detection", decision: 2,
        desc: "All four edge operators run on the filtered image and are scored; the best one is selected." },
      { name: "Thresholding", icon: "✓", cls: "", title: "Thresholding", decision: 3,
        desc: "The histogram and lighting of the filtered image decide which thresholding method is used.",
        stats: [["Otsu threshold", data.otsu], ["Global threshold", 120]] },
      { name: "Morphology", icon: "✓", cls: "", title: "Morphological Processing", decision: 4,
        desc: "The binary mask is inspected for specks and holes, then the matching operation is applied (3×3 ellipse)." },
      { name: "Disease Detection", icon: d.status === "unavailable" ? "⚠" : "🔬", cls: "disease " + dState, title: "🔬 Disease Detection",
        desc: "The enhanced + filtered image is passed to a trained classifier. Segmentation only locates suspicious regions; the disease name comes from the model.",
        custom: () => diseaseStage(),
        figs: [[im.classifier_input, "Processed image given to classifier"], [im.leaf_mask, "Leaf mask (segmentation)"], [im.lesion_mask, "Suspected lesion mask (morphology)"]] },
      { name: "Final Result", icon: "⭐", cls: "final", title: "Final Result",
        desc: "Annotated leaf with the model's prediction.",
        figs: [[im.final, "Final annotated image"]], scroll: true },
    ];
  }

  function decisionPanel(s, i) {
    const d = data.decisions[s.decision], im = data.images;
    const out = im[d.output_key];
    const next = stages()[i + 1];
    const cands = d.candidates.map((c) => {
      let name = c.name;
      if (c.key === "gamma") name = "Gamma Transformation (γ = " + data.gamma + ")";
      return `<figure class="fig cand ${c.selected ? "picked" : "skipped"}">
        <span class="pick-badge">${c.selected ? "✔ Selected" : "Not selected"}</span>
        <img src="${im[c.key]}" alt="${esc(name)}" loading="lazy">
        <figcaption>${esc(name)}</figcaption><p class="cand-reason">${esc(c.reason)}</p></figure>`;
    }).join("");
    return `
      <div class="choice">
        <div class="choice-flow"><span>Input</span><b>${esc(d.input)}</b></div>
        <div class="choice-main"><span class="choice-tag">✔ Filter chosen</span><h4>${esc(d.chosen)}</h4></div>
        <div class="why"><b>Why?</b> ${esc(d.why)}</div>
        <div class="stats">${d.metrics.map((m) => `<div class="stat"><small>${esc(m[0])}</small><b>${esc(m[1])}</b></div>`).join("")}</div>
      </div>
      <div class="gallery cand-gallery">${cands}</div>
      <div class="handoff"><img src="${out}" alt="Output of this step">
        <div><small>Output passed to the next step</small><b>${esc(d.chosen)}</b>
        ${next ? `<button type="button" class="btn accent next-btn" id="nextStep">Next step: ${esc(next.name)} →</button>` : ""}</div></div>`;
  }

  function diseaseStage() {
    const d = data.disease;
    if (d.status === "unavailable")
      return `<div class="note">${esc(d.message)}<br>No disease is reported because no classifier is available.</div>`;
    return `<div class="stats"><div class="stat"><small>Status</small><b>${d.status === "healthy" ? "Healthy" : "Disease Detected"}</b></div>
      <div class="stat"><small>Prediction</small><b>${esc(d.name)}</b></div>
      <div class="stat"><small>Confidence</small><b>${d.confidence.toFixed(1)}%</b></div></div>`;
  }

  /* ---------------- rendering ---------------- */
  function render() {
    $("results").classList.remove("hidden");
    stopPlay(); renderTimeline(); selectStage(0, false);
    renderResult(); renderInfo(); renderCompare();
    $("dashboard").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ---------------- step-by-step walkthrough ---------------- */
  let playTimer = null;
  function stopPlay() { if (playTimer) { clearInterval(playTimer); playTimer = null; } const b = $("playBtn"); if (b) b.textContent = "▶ Walk through step by step"; }
  function startPlay() {
    selectStage(0, false);
    $("playBtn").textContent = "⏸ Stop walkthrough";
    playTimer = setInterval(() => {
      if (activeStage >= stages().length - 1) { stopPlay(); return; }
      selectStage(activeStage + 1, false);
    }, 6000);
  }
  $("playBtn").addEventListener("click", () => (playTimer ? stopPlay() : startPlay()));

  function renderTimeline() {
    $("timeline").innerHTML = stages().map((s, i) =>
      `<button class="stage ${s.cls}" data-i="${i}" role="tab"><span class="num">${i + 1}</span><span class="name">${esc(s.name)}</span><span class="tick">${s.icon}</span></button>`).join("");
    $("timeline").querySelectorAll(".stage").forEach((b) => b.addEventListener("click", () => selectStage(+b.dataset.i, true)));
  }

  function selectStage(i, scroll) {
    activeStage = i;
    const s = stages()[i];
    document.querySelectorAll(".stage").forEach((b, k) => b.classList.toggle("active", k === i));
    let html = `<h3>${i + 1}. ${esc(s.title)}</h3><p class="desc">${esc(s.desc)}</p>`;
    if (s.stats) html += `<div class="stats">${s.stats.map((x) => `<div class="stat"><small>${esc(x[0])}</small><b>${esc(x[1])}</b></div>`).join("")}</div>`;
    if (s.decision != null) html += decisionPanel(s, i);
    if (s.custom) html += s.custom();
    if (s.figs) html += `<div class="gallery">${s.figs.map((f) => fig(f[0], f[1])).join("")}</div>`;
    if (s.decision == null && i < stages().length - 1)
      html += `<div class="handoff"><div><button type="button" class="btn accent next-btn" id="nextStep">Next step: ${esc(stages()[i + 1].name)} →</button></div></div>`;
    $("stageView").innerHTML = html;
    bindZoom($("stageView"));
    const nb = $("nextStep"); if (nb) nb.addEventListener("click", () => { stopPlay(); selectStage(i + 1, true); $("stageView").scrollIntoView({ behavior: "smooth", block: "start" }); });
    if (s.scroll && scroll) $("resultSection").scrollIntoView({ behavior: "smooth" });
  }

  function renderResult() {
    const d = data.disease, card = $("resultCard");
    if (d.status === "unavailable") {
      card.className = "result-card unavailable";
      card.innerHTML = `<div class="result-banner"><div class="status">Final Disease Detection</div><h3>Model Not Available</h3></div>
        <div class="unavail-msg"><p><b>${esc(d.message)}</b></p>
        <p class="muted">The DIP stages above still ran normally, but segmentation alone is not disease classification, so no disease name or confidence is shown.</p>
        <p class="muted">To enable detection: scroll to <b>Train the Disease Classifier</b> at the top of this page, upload your labelled leaf images and press <b>Train Model</b>.</p></div>`;
      return;
    }
    const healthy = d.status === "healthy";
    card.className = "result-card " + d.status;
    const top = (d.top || []).map((t) => `<li><span>${esc(t.name)}</span><b>${t.confidence.toFixed(1)}%</b></li>`).join("");
    card.innerHTML = `
      <div class="result-banner">
        <div class="status">${healthy ? "Healthy" : "Disease Detected"}</div>
        <h3>${healthy ? "Healthy Leaf" : esc(d.name)}</h3>
        <div class="conf">Confidence: ${d.confidence.toFixed(1)}%</div>
      </div>
      <div class="result-body">
        <div>
          <div class="final-img"><img id="finalImg" src="${d.final_image}" alt="Final annotated leaf" style="cursor:zoom-in"></div>
          <p class="legend-note">${healthy ? "Leaf shown without any disease marking." : "Red overlay/outline = suspected affected region."}</p>
        </div>
        <div class="details">
          <div class="detail-block"><h4>Status</h4><div>${healthy ? "Healthy" : "Disease Detected"}</div></div>
          <div class="detail-block"><h4>${healthy ? "Result" : "Disease"}</h4><div>${healthy ? "Healthy Leaf" : esc(d.name)}</div></div>
          <div class="detail-block"><h4>Confidence</h4><div class="meter"><span style="width:${Math.min(100, d.confidence)}%"></span></div>
            <div>${d.confidence.toFixed(1)}%</div>
            ${d.low_confidence ? '<div class="low">⚠ Low confidence – the prediction may be unreliable. Try a clearer, well-lit photo.</div>' : ""}</div>
          <div class="detail-block"><h4>Affected Region</h4><div>${esc(d.region.text)}</div></div>
          <div class="detail-block"><h4>About this result</h4><div>${esc(d.info.symptoms)}</div></div>
          <div class="detail-block"><h4>Top predictions</h4><ul class="top-list">${top}</ul></div>
          <a class="btn download" href="${d.download_url}" download="leafvision_final_result.png">⬇ Download Final Result</a>
        </div>
      </div>`;
    $("finalImg").addEventListener("click", () => zoom(d.final_image));
  }

  function renderInfo() {
    const d = data.disease, box = $("infoCard");
    if (d.status === "unavailable") {
      box.innerHTML = `<p class="muted">Disease information appears here after a trained model makes a prediction.</p>
        <div class="disclaimer">${esc(data.disclaimer)}</div>`;
      return;
    }
    const i = d.info;
    box.innerHTML = `
      <div class="info-grid">
        <div class="info-box"><h4>Disease name</h4><p>${esc(i.name)}</p></div>
        <div class="info-box"><h4>Common symptoms</h4><p>${esc(i.symptoms)}</p></div>
        <div class="info-box"><h4>Cause / pathogen</h4><p>${esc(i.cause)}</p></div>
        <div class="info-box"><h4>General management / prevention</h4><ul>${i.management.map((m) => `<li>${esc(m)}</li>`).join("")}</ul></div>
      </div>
      ${i.known ? "" : '<p class="muted">Detailed information for this class is not stored in the project, so general advice is shown.</p>'}
      <div class="disclaimer">${esc(i.disclaimer)}</div>`;
  }

  function renderCompare() {
    const im = data.images;
    const items = [["Original", im.original], ["Enhanced", im.contrast], ["Filtered", im.gaussian],
      ["Segmented", im.otsu], ["Morphology", im.morph_clean], ["Final Result", im.final]];
    $("compareCards").innerHTML = items.map((x) => fig(x[1], x[0])).join("");
    bindZoom($("compareCards"));
    const sel = $("compareSelect");
    sel.innerHTML = items.slice(1).map((x, k) => `<option value="${k + 1}">${x[0]}</option>`).join("");
    sel.onchange = setSlider;
    $("baBeforeImg").src = im.original;
    $("baRange").value = 50; setSlider(); moveSlider();
    function setSlider() { $("baAfter").src = items[+sel.value][1]; }
  }

  function moveSlider() {
    const v = $("baRange").value;
    $("baBefore").style.clipPath = `inset(0 ${100 - v}% 0 0)`;
    $("baLine").style.left = v + "%";
  }
  $("baRange").addEventListener("input", moveSlider);

  /* ---------------- image zoom ---------------- */
  function zoom(src) {
    const l = document.createElement("div"); l.className = "lightbox";
    l.innerHTML = `<img src="${src}" alt="Enlarged result">`;
    l.addEventListener("click", () => l.remove());
    document.body.appendChild(l);
  }
  function bindZoom(root) { root.querySelectorAll(".fig img").forEach((im) => im.addEventListener("click", () => zoom(im.src))); }
})();
