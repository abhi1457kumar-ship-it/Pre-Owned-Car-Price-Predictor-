const palette = ["#3ee0ff", "#7cffc4", "#8cb4ff", "#ffb45c", "#f08fff", "#9ad7c2"];

function inr(value) {
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(value);
}

function makeChart(ctx, type, labels, values, extra = {}) {
  return new Chart(ctx, {
    type,
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: palette,
        borderColor: "#3ee0ff",
        borderWidth: type === "line" ? 2 : 0,
        tension: 0.3,
      }],
    },
    options: {
      plugins: { legend: { display: type === "doughnut" || type === "pie", labels: { color: "#e8f4f8" } } },
      scales: type === "doughnut" || type === "pie" ? {} : {
        x: { ticks: { color: "#93a8b5" }, grid: { color: "rgba(118,214,255,0.08)" } },
        y: { ticks: { color: "#93a8b5" }, grid: { color: "rgba(118,214,255,0.08)" } },
      },
      ...extra,
    },
  });
}

async function loadInsights() {
  const errorBox = document.getElementById("insights-error");
  try {
    const response = await fetch("/api/insights");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load insights.");
    renderStats(data);
    renderMetrics(data);
    renderCharts(data);
  } catch (error) {
    errorBox.hidden = false;
    errorBox.textContent = error.message;
  }
}

function renderStats(data) {
  const items = [
    ["Usable records", data.catalog_summary.record_count.toLocaleString("en-IN")],
    ["Brands", data.catalog_summary.brand_count],
    ["Models", data.catalog_summary.model_count],
    ["Cities", data.catalog_summary.city_count],
  ];
  const grid = document.getElementById("stat-grid");
  grid.innerHTML = items.map(([label, value]) => `
    <article class="glass-card stat-card">
      <p class="eyebrow">${label}</p>
      <h3>${value}</h3>
    </article>
  `).join("");
}

function renderMetrics(data) {
  const selected = data.selected_model;
  const rows = Object.entries(data.metrics).map(([name, info]) => {
    const test = info.test || {};
    const marker = name === selected ? " (selected)" : "";
    return `<tr><td>${name}${marker}</td><td>${inr(test.mae || 0)}</td><td>${inr(test.rmse || 0)}</td><td>${test.r2}</td></tr>`;
  }).join("");
  document.getElementById("metrics-panel").innerHTML = `
    <h2>Held-out test metrics</h2>
    <p class="muted">Selected algorithm: ${selected}. Metrics are calculated on the 20% test split after preprocessing was fit on training data only.</p>
    <div class="table-responsive">
      <table class="table table-dark table-sm">
        <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderCharts(data) {
  const dist = data.distributions;
  makeChart(document.getElementById("fuel-chart"), "doughnut", Object.keys(dist.fuel_type), Object.values(dist.fuel_type));
  makeChart(document.getElementById("transmission-chart"), "doughnut", Object.keys(dist.transmission), Object.values(dist.transmission));
  makeChart(document.getElementById("year-chart"), "bar", Object.keys(dist.manufacturing_year), Object.values(dist.manufacturing_year));
  makeChart(document.getElementById("price-chart"), "bar", Object.keys(dist.selling_price_bins), Object.values(dist.selling_price_bins));

  const importance = data.feature_importance || [];
  const note = document.getElementById("importance-note");
  if (!importance.length) {
    note.textContent = "Feature importance is unavailable for the selected algorithm.";
    return;
  }
  note.textContent = "Permutation importance on original input columns using a sample from the held-out test set. This is not a causal ranking.";
  makeChart(
    document.getElementById("importance-chart"),
    "bar",
    importance.map((item) => item.feature),
    importance.map((item) => item.importance),
    { indexAxis: "y" }
  );
}

if (document.getElementById("stat-grid")) {
  loadInsights();
}
