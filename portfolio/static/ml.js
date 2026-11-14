"use strict";
const $ = (s) => document.querySelector(s);
const esc = (v) =>
  String(v).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const pct = (n) => (n === null ? "N/A" : `${(n * 100).toFixed(1)}%`);
const dec = (n) => (n === null ? "N/A" : Number(n.toPrecision(5)));
let mode = "train",
  result = null,
  busy = false,
  fileRevision = 0;
const config = {
  train: [
    "Training data",
    "Model report",
    "Train classifier →",
    "CSV: 8–1,000 rows, 1–12 numeric features, binary target. No missing values or duplicate feature rows.",
  ],
  search: [
    "Document collection",
    "Retrieval results",
    "Search documents →",
    "Separate 2–100 documents with a line containing ---. Up to 200 KB total.",
  ],
  evaluate: [
    "Labeled predictions",
    "Evaluation report",
    "Evaluate predictions →",
    "CSV with actual (0/1) and score (0–1) columns. Up to 5,000 rows.",
  ],
};
const trainSample =
  "signal,noise,label\n" +
  Array.from({ length: 64 }, (_, i) => {
    const signal = (((i * 17) % 64) - 32) / 8,
      noise = (((i * 11) % 23) - 11) / 5;
    const label = signal + 0.35 * noise + (i % 7 === 0 ? 2 : 0) > 0 ? 1 : 0;
    return `${signal},${noise},${label}`;
  }).join("\n");
const samples = {
  train: trainSample,
  search:
    "Database transactions\nA database transaction groups writes into an atomic unit. Rollback reverses uncommitted changes after an error. Isolation helps protect concurrent operations.\n---\nModel evaluation\nPrecision and recall reveal different classification errors. A confusion matrix counts true and false predictions.\n---\nDatabase backups\nBackups preserve durable copies of database records. Recovery procedures should be tested regularly.\n---\nWeb security\nA content security policy restricts browser resources. Cookies may need Secure, HttpOnly, and SameSite attributes.\n---\nTransaction consistency\nAn inventory transaction can check stock and write a movement together. If stock is insufficient, rollback prevents partial updates.",
  evaluate:
    "actual,score\n1,0.95\n0,0.80\n1,0.75\n1,0.65\n0,0.60\n1,0.45\n0,0.35\n0,0.20\n1,0.15\n0,0.05",
};
function message(text, error = false) {
  $("#notice").textContent = text;
  $("#notice").className = error ? "error" : "";
}
function invalidate() {
  result = null;
  $("#download").disabled = true;
  $("#result").innerHTML = window.analysisEmpty(mode);
}
function switchMode(next) {
  if (busy) return;
  fileRevision++;
  mode = "train";
  window.designIdentity(mode);
  document
    .querySelectorAll("[data-mode]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.mode === mode)),
    );
  for (const key of Object.keys(config))
    $(`#${key}-options`).hidden = key !== mode;
  $("#input-title").textContent = config[mode][0];
  $("#result-title").textContent = config[mode][1];
  $("#run").textContent = config[mode][2];
  $("#input-help").textContent =
    config[mode][3] +
    (mode === "search" ? " UTF-8 text." : " UTF-8 CSV, up to 1 MB.");
  $("#file-label").textContent =
    mode === "search" ? "Choose a document collection" : "Choose a CSV file";
  $("#file").accept = mode === "search" ? ".txt,text/plain" : ".csv,text/csv";
  $("#source").value = "";
  invalidate();
  message("");
}
const stats = (items) =>
  `<div class="stats">${items.map(([k, v]) => `<div class="stat"><small>${esc(k)}</small><strong>${esc(v)}</strong></div>`).join("")}</div>`;
const table = (headers, rows) =>
  `<div class="table-scroll"><table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
function matrix(m) {
  const c = m.confusion;
  return `<h3>Confusion matrix</h3><div class="matrix"><div>True negatives<strong>${c.tn}</strong></div><div class="error-cell">False positives<strong>${c.fp}</strong></div><div class="error-cell">False negatives<strong>${c.fn}</strong></div><div>True positives<strong>${c.tp}</strong></div></div>`;
}
function plot(history) {
  const high = Math.max(...history.map((p) => p.log_loss), 0.01);
  const points = history
    .map(
      (p) => `${50 + (p.epoch / 200) * 650},${240 - (p.log_loss / high) * 190}`,
    )
    .join(" ");
  return `<div class="plot"><div class="plot-heading"><h3>Training log loss</h3><span>${dec(history.at(-1).log_loss)} <small>final loss</small></span></div><svg viewBox="0 0 740 280" role="img" aria-label="Training log loss over 200 epochs, final loss ${dec(history.at(-1).log_loss)}"><defs><linearGradient id="loss-fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#bedf85" stop-opacity=".25"/><stop offset="1" stop-color="#bedf85" stop-opacity="0"/></linearGradient></defs>${[0, 0.25, 0.5, 0.75, 1].map((v) => `<line x1="50" y1="${240 - v * 190}" x2="700" y2="${240 - v * 190}"/><text x="3" y="${244 - v * 190}">${(v * high).toFixed(2)}</text>`).join("")}<polygon points="50,240 ${points} 700,240" fill="url(#loss-fill)"/><polyline pathLength="1" points="${points}"/><text x="50" y="269">Epoch 0</text><text x="640" y="269">Epoch 200</text></svg><p class="hint">Training loss is not held-out performance.</p></div>`;
}
function render(r) {
  if (mode === "train") {
    $("#result").innerHTML =
      stats([
        ["Test accuracy", pct(r.metrics.accuracy)],
        ["Baseline accuracy", pct(r.baseline_accuracy)],
        ["Test ROC AUC", dec(r.metrics.roc_auc)],
      ]) +
      `<p class="summary">${r.train_rows} training rows · ${r.test_rows} held-out rows · seed ${r.seed}. Scaling is fitted on training rows only. Baseline always predicts the training majority class.</p>` +
      plot(r.loss_history) +
      matrix(r.metrics) +
      "<h3>Standardized feature coefficients</h3>" +
      table(
        ["Feature", "Coefficient", "Train mean", "Train scale"],
        r.features.map((f) => [
          f.name,
          dec(f.weight),
          dec(f.mean),
          dec(f.scale),
        ]),
      ) +
      '<p class="hint">Coefficients describe association in this fitted linear model, not causation. Download includes scaler, coefficients, split records, and held-out scores. This small baseline is not a production-ready model.</p>';
  } else if (mode === "search") {
    $("#result").innerHTML =
      stats([
        ["Documents", r.documents],
        ["Vocabulary", r.vocabulary],
        ["Matches shown", r.results.length],
      ]) +
      `<p class="summary">${esc(r.method)}. Scores measure term similarity, not correctness or probability. Term badges show ${r.algorithm === "bm25" ? "BM25" : "TF-IDF"} score contributions.</p>` +
      r.results
        .map(
          (d) =>
            `<article class="result-card"><h3>Document ${d.document}<small>${r.algorithm === "bm25" ? "BM25" : "Cosine"} ${d.score.toFixed(3)}</small></h3><p>${esc(d.excerpt)}</p><div>${d.matched_terms.map((t) => `<span class="tag">${esc(t.term)} · ${t.contribution.toFixed(3)}</span>`).join("")}</div></article>`,
        )
        .join("") +
      (r.results.length
        ? ""
        : "<p>No overlapping searchable terms. Try vocabulary present in the documents.</p>") +
      `<p class="hint">Unseen query terms: ${esc(r.unseen_terms.join(", ") || "none")}. Top ${r.top_k} results; excerpts limited to 600 characters.</p>`;
  } else {
    const m = r.metrics;
    $("#result").innerHTML =
      stats([
        ["Precision", pct(m.precision)],
        ["Recall", pct(m.recall)],
        ["ROC AUC", dec(m.roc_auc)],
      ]) +
      `<p class="summary">${r.rows} labeled predictions · threshold ${r.threshold} · positive prevalence ${pct(r.positive_rate)}. Undefined metrics display N/A.</p>` +
      matrix(m) +
      table(
        ["Accuracy", "F1", "Brier loss", "Log loss"],
        [[pct(m.accuracy), dec(m.f1), dec(m.brier), dec(m.log_loss)]],
      ) +
      "<h3>Threshold tradeoffs</h3>" +
      table(
        [
          "Threshold",
          "Precision",
          "Recall",
          "False positives",
          "False negatives",
        ],
        r.thresholds.map((t) => [
          t.threshold,
          pct(t.precision),
          pct(t.recall),
          t.confusion.fp,
          t.confusion.fn,
        ]),
      ) +
      "<h3>Calibration by score bin</h3>" +
      table(
        ["Bin", "Count", "Mean score", "Observed positives"],
        r.calibration.map((b) => [
          b.range,
          b.count,
          pct(b.mean_score),
          pct(b.observed_rate),
        ]),
      ) +
      '<p class="hint">Bins are left-inclusive and right-exclusive, except the final bin includes 1.0. Small bins are noisy. Repeated tuning on test labels leaks information into your decision.</p>';
  }
}
document.querySelectorAll("[data-mode]").forEach((b) =>
  b.addEventListener("click", () => {
    location.hash = b.dataset.mode;
  }),
);
$("#sample").addEventListener("click", () => {
  fileRevision++;
  $("#source").value = samples[mode];
  invalidate();
  message("Synthetic sample loaded.");
});
document.querySelectorAll("textarea,input:not([type=file])").forEach((input) =>
  input.addEventListener("input", () => {
    fileRevision++;
    invalidate();
    message("Input changed. Run the analysis to update the report.");
  }),
);
$("#file").addEventListener("change", async () => {
  const file = $("#file").files?.[0];
  $("#file").value = "";
  if (!file) return;
  const revision = ++fileRevision,
    limit = mode === "search" ? 200000 : 1000000;
  try {
    message(`Reading ${file.name}…`);
    const text = await window.localTextFile.read(
      file,
      limit,
      mode === "search" ? "200 KB" : "1 MB",
    );
    if (revision !== fileRevision || busy) return;
    $("#source").value = text;
    invalidate();
    message(`Loaded ${file.name}. Review the input, then run the analysis.`);
    document.dispatchEvent(new Event("source-loaded"));
  } catch (error) {
    if (revision === fileRevision) message(error.message, true);
  }
});
$("#run").addEventListener("click", async () => {
  if (busy) return;
  fileRevision++;
  busy = true;
  window.analysisBusy(true);
  result = null;
  const controls = [
    ...document.querySelectorAll("input,select,textarea,button"),
  ];
  const disabled = controls.map((c) => c.disabled);
  controls.forEach((c) => (c.disabled = true));
  message("Computing locally…");
  try {
    const text = $("#source").value;
    const body =
      mode === "search"
        ? { documents: text, query: $("#query").value }
        : {
            csv: text,
            target: $("#target").value,
            seed: Number($("#seed").value),
            threshold: Number($("#threshold").value),
          };
    const data = await window.analysisRequest(`/api/ml/${mode}`, {
      ...body,
      ...window.capabilityInputs(),
    });
    result = data;
    render(data);
    window.renderCapabilities(data);
    if (
      !matchMedia("(prefers-reduced-motion: reduce)").matches &&
      !document.body.classList.contains("still-life")
    )
      document.querySelector(".plot polyline")?.animate(
        [
          { strokeDasharray: "1", strokeDashoffset: "1" },
          { strokeDasharray: "1", strokeDashoffset: "0" },
        ],
        { duration: 850, easing: "ease-out" },
      );
    message("Complete. Download the report to keep these results.");
  } catch (e) {
    invalidate();
    message(e.message, true);
  } finally {
    busy = false;
    controls.forEach((c, index) => (c.disabled = disabled[index]));
    $("#download").disabled = !result;
    window.analysisBusy(false, !!result);
  }
});
$("#download").addEventListener("click", () => {
  if (!result) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `ml-${mode}-report.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
switchMode("train");
