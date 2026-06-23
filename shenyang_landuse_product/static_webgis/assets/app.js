(function () {
  const data = window.DASHBOARD_DATA;
  const $ = (id) => document.getElementById(id);

  const state = {
    layerType: "landuse",
    group: "2030 未来预测",
    layerId: "rf_ca_2030",
    zoom: 1,
    x: 0,
    y: 0,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    originX: 0,
    originY: 0,
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function fmtNumber(value, digits = 3) {
    const number = Number(value || 0);
    return number.toLocaleString("zh-CN", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function fmtPct(value) {
    return `${fmtNumber(Number(value || 0) * 100, 3)}%`;
  }

  function currentLayers() {
    return state.layerType === "drivers" ? data.drivers : data.layers;
  }

  function layerGroup(layer) {
    return state.layerType === "drivers" ? "驱动因子" : layer.group;
  }

  function groupsForCurrentType() {
    return [...new Set(currentLayers().map(layerGroup))];
  }

  function selectedLayer() {
    return currentLayers().find((layer) => layer.id === state.layerId) || currentLayers()[0];
  }

  function populateSelect(select, options, selectedValue) {
    select.innerHTML = options
      .map((option) => {
        const value = typeof option === "string" ? option : option.value;
        const label = typeof option === "string" ? option : option.label;
        const selected = value === selectedValue ? " selected" : "";
        return `<option value="${escapeHtml(value)}"${selected}>${escapeHtml(label)}</option>`;
      })
      .join("");
  }

  function syncControls() {
    $("layerType").value = state.layerType;
    const groups = groupsForCurrentType();
    if (!groups.includes(state.group)) {
      state.group = groups[0];
    }
    populateSelect($("groupSelect"), groups, state.group);

    const layers = currentLayers()
      .filter((layer) => layerGroup(layer) === state.group)
      .map((layer) => ({ value: layer.id, label: layer.title }));
    if (!layers.some((layer) => layer.value === state.layerId)) {
      state.layerId = layers[0]?.value;
    }
    populateSelect($("layerSelect"), layers, state.layerId);
  }

  function renderLegend() {
    const legend = $("legend");
    if (state.layerType === "drivers") {
      legend.innerHTML = `
        <div class="gradient-legend"></div>
        <div class="legend-item"><span>低值</span><span style="margin-left:auto">高值</span></div>
      `;
      return;
    }
    legend.innerHTML = data.classes
      .map(
        (item) => `
          <div class="legend-item">
            <span class="legend-color" style="background:${item.color}"></span>
            <span>${item.code} ${escapeHtml(item.name)} / ${escapeHtml(item.cn)}</span>
          </div>
        `,
      )
      .join("");
  }

  function fitMapToViewport() {
    const viewport = $("mapViewport");
    const image = $("mapImage");
    if (!image.naturalWidth || !image.naturalHeight) return;
    const fitX = (viewport.clientWidth * 0.92) / image.naturalWidth;
    const fitY = (viewport.clientHeight * 0.92) / image.naturalHeight;
    state.zoom = Math.max(0.18, Math.min(1.2, fitX, fitY));
    state.x = 0;
    state.y = 0;
    $("zoomRange").value = state.zoom.toFixed(2);
    updateMapTransform();
  }

  function updateMapTransform() {
    $("mapImage").style.transform = `translate(-50%, -50%) translate(${state.x}px, ${state.y}px) scale(${state.zoom})`;
  }

  function renderMap() {
    const layer = selectedLayer();
    $("mapTitle").textContent = layer.title;
    $("mapSubtitle").textContent = state.layerType === "drivers" ? "驱动因子预览图，颜色越深表示数值越高。" : "拖拽平移，滚轮或滑块缩放。";
    $("mapYear").textContent = layer.year ? `${layer.year} 年` : "2025 因子";
    $("mapModel").textContent = layer.model || "Driver";
    $("layerDescription").textContent = layer.description || "";
    $("sourceLine").textContent = `数据源：${layer.source || ""}`;
    renderLegend();

    const image = $("mapImage");
    image.onload = fitMapToViewport;
    image.src = layer.src;
    image.alt = layer.title;
  }

  function handleLayerTypeChange() {
    state.layerType = $("layerType").value;
    if (state.layerType === "drivers") {
      state.group = "驱动因子";
      state.layerId = data.drivers[0].id;
    } else {
      state.group = "2030 未来预测";
      state.layerId = "rf_ca_2030";
    }
    syncControls();
    renderMap();
  }

  function initMapInteractions() {
    $("layerType").addEventListener("change", handleLayerTypeChange);
    $("groupSelect").addEventListener("change", () => {
      state.group = $("groupSelect").value;
      const firstLayer = currentLayers().find((layer) => layerGroup(layer) === state.group);
      state.layerId = firstLayer.id;
      syncControls();
      renderMap();
    });
    $("layerSelect").addEventListener("change", () => {
      state.layerId = $("layerSelect").value;
      renderMap();
    });
    $("zoomRange").addEventListener("input", () => {
      state.zoom = Number($("zoomRange").value);
      updateMapTransform();
    });
    $("resetView").addEventListener("click", fitMapToViewport);

    const viewport = $("mapViewport");
    viewport.addEventListener("pointerdown", (event) => {
      state.dragging = true;
      state.dragStartX = event.clientX;
      state.dragStartY = event.clientY;
      state.originX = state.x;
      state.originY = state.y;
      viewport.classList.add("dragging");
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener("pointermove", (event) => {
      if (!state.dragging) return;
      state.x = state.originX + event.clientX - state.dragStartX;
      state.y = state.originY + event.clientY - state.dragStartY;
      updateMapTransform();
    });
    viewport.addEventListener("pointerup", (event) => {
      state.dragging = false;
      viewport.classList.remove("dragging");
      viewport.releasePointerCapture(event.pointerId);
    });
    viewport.addEventListener("wheel", (event) => {
      event.preventDefault();
      const delta = event.deltaY > 0 ? -0.08 : 0.08;
      state.zoom = Math.max(0.18, Math.min(3.4, state.zoom + delta));
      $("zoomRange").value = state.zoom.toFixed(2);
      updateMapTransform();
    }, { passive: false });
  }

  function makeBarRow(label, value, max, options = {}) {
    const width = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
    const colorClass = options.colorClass || "";
    const display = options.display || fmtNumber(value, options.digits ?? 3);
    return `
      <div class="bar-row ${options.rowClass || ""}">
        <div class="bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</div>
        <div class="bar-track"><div class="bar-fill ${colorClass}" style="width:${width}%"></div></div>
        <div class="bar-value">${escapeHtml(display)}</div>
      </div>
    `;
  }

  function makeChangeRow(label, value, maxAbs) {
    const zero = 50;
    const halfWidth = maxAbs > 0 ? Math.min(50, (Math.abs(value) / maxAbs) * 50) : 0;
    const left = value >= 0 ? zero : zero - halfWidth;
    const color = value >= 0 ? "var(--green)" : "var(--red)";
    return `
      <div class="bar-row change-row">
        <div class="bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</div>
        <div class="change-track">
          <div class="change-zero" style="left:${zero}%"></div>
          <div class="change-fill" style="left:${left}%; width:${halfWidth}%; background:${color}"></div>
        </div>
        <div class="bar-value">${value >= 0 ? "+" : ""}${fmtNumber(value, 3)}</div>
      </div>
    `;
  }

  function renderMetricCards() {
    const rows = data.modelComparison;
    const rf = rows.find((row) => row.model === "RF-CA");
    const ca = rows.find((row) => row.model === "CA-Markov");
    const best = [...rows].sort((a, b) => Number(b.overall_accuracy) - Number(a.overall_accuracy))[0];
    const baseline = rows.find((row) => row.model === "Markov baseline");
    const deltaRfCa = Number(rf.overall_accuracy) - Number(ca.overall_accuracy);
    const deltaBaseline = Number(rf.overall_accuracy) - Number(baseline.overall_accuracy);

    const cards = [
      ["最佳模型", best.model, `OA ${fmtPct(best.overall_accuracy)}`],
      ["RF-CA OA", fmtPct(rf.overall_accuracy), `Kappa ${fmtNumber(rf.kappa, 6)}`],
      ["相对 Markov baseline", `+${fmtNumber(deltaBaseline * 100, 3)} p.p.`, "总体精度提升"],
      ["相对 CA-Markov", `+${fmtNumber(deltaRfCa * 100, 4)} p.p.`, "提升很小，适合讨论机制贡献"],
    ];

    $("metricCards").innerHTML = cards
      .map(
        ([label, value, note]) => `
          <div class="metric-card">
            <div class="metric-label">${escapeHtml(label)}</div>
            <div class="metric-value">${escapeHtml(value)}</div>
            <div class="metric-note">${escapeHtml(note)}</div>
          </div>
        `,
      )
      .join("");
  }

  function renderFeatureChart() {
    const top = data.featureImportance.slice(0, 10);
    const max = Math.max(...top.map((row) => Number(row.importance)));
    $("featureChart").innerHTML = top
      .map((row) => makeBarRow(row.feature, Number(row.importance), max, { digits: 4, colorClass: "green" }))
      .join("");
  }

  function renderAccuracy() {
    const sorted = [...data.modelComparison].sort((a, b) => Number(b.overall_accuracy) - Number(a.overall_accuracy));
    const max = Math.max(...sorted.map((row) => Number(row.overall_accuracy)));
    $("accuracyBars").innerHTML = sorted
      .map((row) =>
        makeBarRow(row.model, Number(row.overall_accuracy), max, {
          display: fmtPct(row.overall_accuracy),
          colorClass: row.model === "RF-CA" ? "green" : "",
        }),
      )
      .join("");

    const header = ["模型", "OA", "Kappa", "较 Markov OA 提升", "关键参数"];
    const body = sorted
      .map(
        (row) => `
          <tr>
            <td>${escapeHtml(row.model)}</td>
            <td>${fmtNumber(row.overall_accuracy, 6)}</td>
            <td>${fmtNumber(row.kappa, 6)}</td>
            <td>${fmtNumber(Number(row.delta_oa_vs_markov_baseline) * 100, 3)} p.p.</td>
            <td>${escapeHtml(row.key_parameters)}</td>
          </tr>
        `,
      )
      .join("");
    $("accuracyTable").innerHTML = `<thead><tr>${header.map((item) => `<th>${item}</th>`).join("")}</tr></thead><tbody>${body}</tbody>`;
  }

  function renderArea() {
    const rows = [...data.areaProjection].sort((a, b) => Math.abs(Number(b.change_area_km2)) - Math.abs(Number(a.change_area_km2)));
    const maxAbs = Math.max(...rows.map((row) => Math.abs(Number(row.change_area_km2))));
    $("areaChangeBars").innerHTML = rows.map((row) => makeChangeRow(`${row.class_name} / ${row.class_cn}`, Number(row.change_area_km2), maxAbs)).join("");

    const body = rows
      .map(
        (row) => `
          <tr>
            <td>${row.class_code}</td>
            <td>${escapeHtml(row.class_name)} / ${escapeHtml(row.class_cn)}</td>
            <td>${fmtNumber(row.base_2025_area_km2, 3)}</td>
            <td>${fmtNumber(row.projected_2030_area_km2, 3)}</td>
            <td>${Number(row.change_area_km2) >= 0 ? "+" : ""}${fmtNumber(row.change_area_km2, 3)}</td>
          </tr>
        `,
      )
      .join("");
    $("areaTable").innerHTML = `
      <thead><tr><th>编码</th><th>类别</th><th>2025 面积</th><th>2030 预测面积</th><th>变化量</th></tr></thead>
      <tbody>${body}</tbody>
    `;
  }

  function initClassAccuracy() {
    const models = [...new Set(data.classAccuracy.map((row) => row.model))];
    populateSelect($("modelSelect"), models, "RF-CA");
    $("modelSelect").addEventListener("change", renderClassAccuracy);
    renderClassAccuracy();
  }

  function renderClassAccuracy() {
    const model = $("modelSelect").value;
    const rows = data.classAccuracy.filter((row) => row.model === model);
    $("classF1Bars").innerHTML = rows
      .map((row) =>
        makeBarRow(`${row.class_name} / ${row.class_cn}`, Number(row.f1_score), 1, {
          display: fmtNumber(row.f1_score, 3),
          colorClass: Number(row.f1_score) < 0.5 ? "red" : "green",
        }),
      )
      .join("");
    const body = rows
      .map(
        (row) => `
          <tr>
            <td>${row.class_code}</td>
            <td>${escapeHtml(row.class_name)} / ${escapeHtml(row.class_cn)}</td>
            <td>${fmtNumber(row.precision, 3)}</td>
            <td>${fmtNumber(row.recall, 3)}</td>
            <td>${fmtNumber(row.f1_score, 3)}</td>
          </tr>
        `,
      )
      .join("");
    $("classTable").innerHTML = `
      <thead><tr><th>编码</th><th>类别</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>${body}</tbody>
    `;
  }

  function renderFigures() {
    $("figureGallery").innerHTML = data.figures
      .map(
        (figure) => `
          <article class="figure-item">
            <h3>${escapeHtml(figure.title)}</h3>
            <img src="${escapeHtml(figure.src)}" alt="${escapeHtml(figure.title)}" />
          </article>
        `,
      )
      .join("");
  }

  function initTabs() {
    document.querySelectorAll(".tab-button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        $(button.dataset.tab).classList.add("active");
      });
    });
  }

  function init() {
    syncControls();
    initMapInteractions();
    initTabs();
    renderMap();
    renderMetricCards();
    renderFeatureChart();
    renderAccuracy();
    renderArea();
    initClassAccuracy();
    renderFigures();
  }

  window.addEventListener("resize", () => {
    window.clearTimeout(window.__fitTimer);
    window.__fitTimer = window.setTimeout(fitMapToViewport, 150);
  });

  init();
})();
