document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    window.lucide.createIcons();
  }
  const form = document.getElementById("predict-form");
  if (form) {
    initPredictForm(form);
  }
});

function fillSelect(select, values, placeholder) {
  select.innerHTML = "";
  const option = document.createElement("option");
  option.value = "";
  option.textContent = placeholder;
  select.appendChild(option);
  values.forEach((value) => {
    const item = document.createElement("option");
    item.value = value;
    item.textContent = value;
    select.appendChild(item);
  });
}

async function initPredictForm(form) {
  const errorBox = document.getElementById("form-error");
  const submitBtn = document.getElementById("submit-btn");
  const resultPanel = document.getElementById("result-panel");
  const brand = document.getElementById("brand");
  const model = document.getElementById("model");
  const variant = document.getElementById("variant");

  let options;
  try {
    const response = await fetch("/api/form-options");
    options = await response.json();
    if (!response.ok) {
      throw new Error(options.error || "Could not load form options.");
    }
  } catch (error) {
    errorBox.hidden = false;
    errorBox.textContent = error.message;
    submitBtn.disabled = true;
    return;
  }

  fillSelect(brand, options.brands, "Select brand");
  fillSelect(document.getElementById("fuel_type"), options.fuel_types, "Select fuel type");
  fillSelect(document.getElementById("transmission"), options.transmissions, "Select transmission");
  fillSelect(document.getElementById("condition"), options.conditions, "Select condition");
  fillSelect(document.getElementById("city"), options.cities, "Select city");

  const bounds = options.bounds;
  applyBounds("manufacturing_year", bounds.manufacturing_year);
  applyBounds("km_driven", bounds.km_driven);
  applyBounds("previous_owners", bounds.previous_owners);
  applyBounds("engine_cc", bounds.engine_cc);
  applyBounds("mileage_kmpl", bounds.mileage_kmpl);

  fillSelect(model, [], "Select a brand first");
  fillSelect(variant, [], "Select a model first");

  brand.addEventListener("change", () => {
    const models = options.brand_models[brand.value] || [];
    fillSelect(model, models, models.length ? "Select model" : "No models available");
    fillSelect(variant, [], "Select a model first");
  });

  model.addEventListener("change", () => {
    const key = `${brand.value}||${model.value}`;
    const variants = options.model_variants[key] || [];
    fillSelect(variant, variants, variants.length ? "Select variant" : "No variants available");
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    if (!form.reportValidity()) {
      return;
    }
    const payload = Object.fromEntries(new FormData(form).entries());
    ["manufacturing_year", "km_driven", "previous_owners", "engine_cc", "mileage_kmpl"].forEach((key) => {
      payload[key] = Number(payload[key]);
    });
    submitBtn.disabled = true;
    submitBtn.textContent = "Estimating...";
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        const details = (data.details || [data.error]).join(" ");
        throw new Error(details);
      }
      renderResult(data);
      resultPanel.hidden = false;
      resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      errorBox.hidden = false;
      errorBox.textContent = error.message;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Estimate selling price";
    }
  });

  document.getElementById("another-btn").addEventListener("click", () => {
    form.reset();
    fillSelect(model, [], "Select a brand first");
    fillSelect(variant, [], "Select a model first");
    resultPanel.hidden = true;
    form.scrollIntoView({ behavior: "smooth" });
  });
  document.getElementById("back-btn").addEventListener("click", () => {
    resultPanel.hidden = true;
    form.scrollIntoView({ behavior: "smooth" });
  });
}

function applyBounds(id, bound) {
  const input = document.getElementById(id);
  if (!input || !bound) return;
  input.min = bound.min;
  input.max = bound.max;
  input.placeholder = `${bound.min} – ${bound.max}`;
}

function renderResult(data) {
  document.getElementById("result-price").textContent = data.formatted_price;
  const range = document.getElementById("result-range");
  if (data.price_range) {
    range.textContent = `Practical spread using test MAE: ${formatNumber(data.price_range.low)} – ${formatNumber(data.price_range.high)}`;
  } else {
    range.textContent = "";
  }
  document.getElementById("result-note").textContent = `${data.disclaimer} ${data.range_note || ""}`;
  const list = document.getElementById("result-specs");
  list.innerHTML = "";
  Object.entries(data.specifications).forEach(([key, value]) => {
    const wrap = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key.replaceAll("_", " ");
    dd.textContent = value;
    wrap.append(dt, dd);
    list.appendChild(wrap);
  });
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(value);
}
