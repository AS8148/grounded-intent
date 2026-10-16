"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "scene-select", "overlay-toggle", "scene-canvas", "image-stage", "image-placeholder",
    "scene-loading", "scene-status", "scene-reload", "object-count", "image-dimensions", "object-list",
    "selected-detail", "relations", "instruction-form", "instruction", "character-count",
    "submit-button", "submit-label", "decision-status", "decision-badge", "decision-result",
    "decision-message", "proposed-actions", "proposal-empty", "verified-actions",
    "verified-empty", "forbidden-movement", "target-detail", "check-list", "runtime-stats",
    "official-json", "audit-json", "decision-warnings", "page-error", "decision-instruction",
    "accepted-plan", "check-details", "check-summary", "proposal-details", "warning-details",
    "warning-summary",
  ].map((id) => [id, byId(id)]));
  const state = { generation: 0, graph: null, image: null, selected: null, decision: null,
    pending: false, controller: null };

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  }

  function showError(message) {
    ui["page-error"].textContent = message || "";
    ui["page-error"].hidden = !message;
  }

  async function request(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let message = `The local service could not complete this request (${response.status}).`;
      try {
        const body = await response.json();
        if (typeof body.detail === "string") message = body.detail;
      } catch { /* Non-JSON server errors still have a readable status message. */ }
      throw new Error(message);
    }
    return response.json();
  }

  function updateSubmit() {
    ui["submit-button"].disabled = state.pending || !state.graph || !ui.instruction.value.trim();
    ui["scene-reload"].disabled = ui["scene-select"].disabled
      || ui["image-stage"].getAttribute("aria-busy") === "true";
    ui["submit-label"].textContent = state.pending ? "Interpreting & checking…" : "Run instruction";
    ui.instruction.disabled = state.pending;
    ui["instruction-form"].setAttribute("aria-busy", String(state.pending));
    ui["character-count"].textContent = `${ui.instruction.value.length} / 2000`;
  }

  function resetDecision() {
    state.decision = null;
    ui["decision-result"].hidden = true;
    ui["decision-badge"].textContent = "Not run";
    delete ui["decision-badge"].dataset.kind;
    ui["decision-status"].textContent = "Write an instruction to generate a decision.";
    ui["check-details"].open = false;
    ui["proposal-details"].open = false;
    ui["warning-details"].open = false;
  }

  function placeholder(title, description) {
    ui["image-placeholder"].replaceChildren(node("span", "", "placeholder-line"),
      node("strong", title), node("span", description));
    ui["image-placeholder"].hidden = false;
    ui["scene-canvas"].hidden = true;
    ui["image-stage"].classList.remove("has-image");
  }

  const objectName = (object) => `${object.color} ${object.label}`.replaceAll("_", " ");
  const objectNumber = (index) => String(index + 1).padStart(2, "0");

  function selectObject(id) {
    if (!state.graph) return;
    state.selected = state.graph.objects.some((object) => object.id === id) ? id : null;
    const object = state.graph.objects.find((item) => item.id === state.selected);
    for (const button of ui["object-list"].querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.objectId === state.selected));
    }
    ui["selected-detail"].replaceChildren();
    if (object) {
      const heading = node("div", undefined, "selection-heading");
      heading.append(node("strong", objectName(object)), node("code", object.id, "object-id"));
      const facts = node("p",
        `Pick pixel (${object.pixel.join(", ")}) · ${object.movable ? "reported movable" : "reported fixed"} · ${object.receives_objects ? "receives placements" : "not a receiving surface"}`,
        "object-facts");
      const evidence = node("details");
      evidence.append(
        node("summary", "Observation evidence"),
        node("p", `Model confidence: ${Math.round(object.confidence * 100)}% (not calibrated).`),
        node("p", `Vision-model grasp estimate: ${object.graspable ? "yes" : "no"}. This raw image-only estimate is not the planner's pick-permission result. Physical grasp feasibility has not been verified.`),
        node("p", object.visual_form || object.visual_description || "No visual description reported."),
        node("p", object.evidence || "No additional evidence reported."),
      );
      ui["selected-detail"].append(heading, facts, evidence);
    } else {
      ui["selected-detail"].textContent = "Select an object to inspect its evidence.";
    }
    drawScene();
  }

  function renderObservations() {
    const graph = state.graph;
    ui["object-count"].textContent = `${graph.objects.length} observed`;
    ui["object-list"].replaceChildren();
    if (!graph.objects.length) {
      ui["object-list"].append(node("p", "No confident object observations in this scene.", "empty-text"));
    }
    for (const [index, object] of graph.objects.entries()) {
      const button = node("button", undefined, "object-button");
      button.type = "button";
      button.dataset.objectId = object.id;
      button.title = object.id;
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", `Object ${index + 1}: ${objectName(object)} (${object.id})`);
      button.append(node("span", objectNumber(index), "object-number"), node("span", objectName(object)));
      button.addEventListener("click", () => selectObject(object.id));
      ui["object-list"].append(button);
    }
    ui.relations.replaceChildren();
    const list = node("ul");
    for (const relation of graph.relations) {
      list.append(node("li", `${relation.subject} ${relation.predicate.replaceAll("_", " ")} ${relation.object} · ${Math.round(relation.confidence * 100)}% · ${relation.evidence}`));
    }
    if (graph.relations.length) ui.relations.append(list);
    else ui.relations.append(node("p", "No spatial relations observed.", "empty-text"));
    for (const warning of graph.warnings) ui.relations.append(node("p", warning, "warning-text"));
    ui.relations.append(node("p", `Perception model: ${graph.perception_model}`));
    ui["scene-canvas"].setAttribute("aria-label", `${graph.scene}: ${graph.objects.length} observed objects. Select an instance using the buttons below. Any verified target is marked with a crosshair.`);
  }

  function drawScene() {
    if (!state.image) return;
    const canvas = ui["scene-canvas"];
    const stage = ui["image-stage"];
    const scale = Math.min(stage.clientWidth / state.image.naturalWidth,
      (stage.clientHeight - 2) / state.image.naturalHeight);
    if (scale <= 0) return;
    const width = state.image.naturalWidth * scale;
    const height = state.image.naturalHeight * scale;
    const ratio = window.devicePixelRatio || 1;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(scale * ratio, 0, 0, scale * ratio, 0, 0);
    context.drawImage(state.image, 0, 0);
    const unit = 1 / scale;
    if (state.graph && ui["overlay-toggle"].checked) {
      for (const [index, object] of state.graph.objects.entries()) {
        const selected = object.id === state.selected;
        const [x0, y0, x1, y1] = object.box;
        context.strokeStyle = selected ? "#b7e3ce" : "#b7d5cd88";
        context.fillStyle = selected ? "#b7e3ce26" : "#b7e3ce08";
        context.lineWidth = (selected ? 2 : 1) * unit;
        context.setLineDash([]);
        if (object.polygon.length >= 3) {
          context.beginPath();
          object.polygon.forEach(([x, y], vertex) => vertex ? context.lineTo(x, y) : context.moveTo(x, y));
          context.closePath();
          context.fill();
          context.stroke();
        } else {
          context.fillRect(x0, y0, x1 - x0, y1 - y0);
          context.strokeRect(x0, y0, x1 - x0, y1 - y0);
        }
        const labelWidth = 26 * unit;
        const labelHeight = 22 * unit;
        const labelX = Math.max(0, Math.min(object.pixel[0] - labelWidth / 2, state.image.naturalWidth - labelWidth));
        const labelY = Math.max(0, Math.min(object.pixel[1] - labelHeight / 2, state.image.naturalHeight - labelHeight));
        context.fillStyle = selected ? "#b7e3ce" : "#13211ff2";
        context.fillRect(labelX, labelY, labelWidth, labelHeight);
        context.fillStyle = selected ? "#13211f" : "#e2f1e9";
        context.font = `600 ${11 * unit}px ui-monospace, monospace`;
        context.fillText(objectNumber(index), labelX + 6 * unit, labelY + 15 * unit);
      }
    }
    const selected = state.graph?.objects.find((object) => object.id === state.selected);
    if (selected) drawTarget(context, selected.pixel, "#e7ce99", 18 * unit, 1.5 * unit);
    if (state.decision?.target_pixel) {
      drawTarget(context, state.decision.target_pixel, "#a9e2bc", 12 * unit, 2 * unit);
    }
  }

  function drawTarget(context, [x, y], color, radius, lineWidth) {
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.setLineDash([]);
    context.beginPath();
    context.arc(x, y, radius * 0.6, 0, Math.PI * 2);
    context.moveTo(x - radius, y); context.lineTo(x + radius, y);
    context.moveTo(x, y - radius); context.lineTo(x, y + radius);
    context.stroke();
  }

  async function loadScene(name) {
    const generation = ++state.generation;
    state.controller?.abort();
    state.controller = new AbortController();
    state.graph = null;
    state.image = null;
    state.selected = null;
    resetDecision();
    showError("");
    ui["image-stage"].setAttribute("aria-busy", "true");
    updateSubmit();
    ui["scene-loading"].hidden = false;
    ui["object-count"].textContent = "Analyzing";
    ui["object-list"].replaceChildren(node("p", "Grounding visible objects…", "empty-text"));
    ui["selected-detail"].textContent = "Select an object to inspect its evidence.";
    ui.relations.replaceChildren(node("p", "Scene analysis is pending.", "empty-text"));
    ui["scene-status"].textContent = `Analyzing ${name}…`;
    ui["image-dimensions"].textContent = "Original image coordinates";
    placeholder("Loading scene", "Image analysis may take a moment.");
    const url = `/api/scenes/${encodeURIComponent(name)}`;
    const image = new Image();
    const imageReady = new Promise((resolve, reject) => {
      image.onload = () => {
        if (generation !== state.generation) return resolve();
        state.image = image;
        ui["scene-canvas"].hidden = false;
        ui["image-placeholder"].hidden = true;
        ui["image-stage"].classList.add("has-image");
        ui["image-dimensions"].textContent = `${image.naturalWidth} × ${image.naturalHeight} px`;
        drawScene();
        resolve();
      };
      image.onerror = () => reject(new Error("The scene image could not be loaded."));
      image.src = `${url}/image`;
    });
    try {
      const [graph] = await Promise.all([request(url, { signal: state.controller.signal }), imageReady]);
      if (generation !== state.generation) return;
      if (graph.width !== image.naturalWidth || graph.height !== image.naturalHeight) {
        throw new Error("Scene observations do not match the image dimensions. Reload the scene before planning.");
      }
      state.graph = graph;
      renderObservations();
      drawScene();
      ui["scene-status"].textContent = `${name} · image analysis ready`;
      ui["decision-status"].textContent = "Ready for an instruction.";
    } catch (error) {
      if (generation !== state.generation || error.name === "AbortError") return;
      state.graph = null;
      ui["object-count"].textContent = "Unavailable";
      ui["object-list"].replaceChildren(node("p", "Observations unavailable. Select the scene again to retry.", "empty-text"));
      ui["scene-status"].textContent = "Scene analysis unavailable. No decision can be submitted.";
      showError(error.message);
      if (!state.image) placeholder("Scene unavailable", "Choose another scene or check the local dataset.");
    } finally {
      if (generation === state.generation) {
        ui["scene-loading"].hidden = true;
        ui["image-stage"].setAttribute("aria-busy", "false");
        updateSubmit();
      }
    }
  }

  function renderActions(container, actions, emptyElement) {
    container.replaceChildren();
    emptyElement.hidden = actions.length > 0;
    for (const action of actions) {
      const item = node("li");
      item.append(node("span", action.action.replaceAll("_", " "), "action-verb"));
      const observed = state.graph.objects.find((object) => object.id === action.object);
      if (observed) {
        const button = node("button");
        button.append(node("span", objectName(observed)), node("code", observed.id, "object-id"));
        button.setAttribute("aria-label", `Inspect ${objectName(observed)} (${observed.id})`);
        button.type = "button";
        button.addEventListener("click", () => selectObject(observed.id));
        item.append(button);
      } else item.append(node("span", action.object || action.question || action.reason || ""));
      container.append(item);
    }
  }

  function renderDecision(decision) {
    state.decision = decision;
    const labels = { plan: "Plan ready", ask_user: "Needs clarification", refuse: "Refused" };
    ui["decision-badge"].textContent = labels[decision.kind];
    ui["decision-badge"].dataset.kind = decision.kind;
    ui["decision-status"].textContent = decision.kind === "plan"
      ? "Symbolic checks passed. Review the sequence before use."
      : decision.kind === "ask_user" ? "Clarify the request, then run it again." : "No movement proposed.";
    ui["decision-instruction"].textContent = ui.instruction.value.trim();
    const explanation = decision.proposal.question || decision.proposal.reason;
    ui["decision-message"].textContent = explanation;
    ui["decision-message"].hidden = !explanation;
    ui["accepted-plan"].hidden = decision.kind !== "plan";
    renderActions(ui["proposed-actions"], decision.proposal.actions, ui["proposal-empty"]);
    renderActions(ui["verified-actions"], decision.actions, ui["verified-empty"]);
    ui["forbidden-movement"].hidden = !decision.proposal.forbidden_movement.length;
    ui["forbidden-movement"].textContent = `Protected from movement: ${decision.proposal.forbidden_movement.join(", ")}`;
    ui["target-detail"].textContent = decision.target_pixel
      ? `First pick (${decision.target_pixel.join(", ")}) · green crosshair · image pixels`
      : "No first-pick pixel.";
    ui["check-list"].replaceChildren();
    for (const check of decision.checks) {
      const item = node("li");
      const detail = node("div", `${check.rule}${check.step === null ? "" : ` · step ${check.step}`}`);
      detail.append(node("span", check.detail, "check-detail"));
      item.append(node("span", check.passed ? "PASS" : "FAIL", `check-state ${check.passed ? "pass" : "fail"}`), detail);
      ui["check-list"].append(item);
    }
    if (!decision.checks.length) ui["check-list"].append(node("li", "No checks reported.", "empty-text"));
    const failedChecks = decision.checks.filter((check) => !check.passed).length;
    ui["check-summary"].textContent = `Verification checks · ${decision.checks.length - failedChecks} passed${failedChecks ? ` · ${failedChecks} failed` : ""}`;
    ui["check-summary"].dataset.failed = String(failedChecks > 0);
    ui["check-details"].open = failedChecks > 0;
    ui["runtime-stats"].replaceChildren();
    const usage = Object.entries(decision.usage).map(([key, value]) => `${key.replaceAll("_", " ")}: ${value}`).join(" · ");
    ui["runtime-stats"].append(node("p", `${decision.model} · ${usage || "Token usage not reported"}${decision.repaired ? " · proposal repaired before verification" : ""}`, "runtime-models"));
    ui["decision-warnings"].replaceChildren(...decision.warnings.map((warning) => node("li", warning)));
    ui["decision-warnings"].hidden = !decision.warnings.length;
    ui["warning-details"].hidden = !decision.warnings.length;
    ui["warning-summary"].textContent = `Notes · ${decision.warnings.length}`;
    ui["official-json"].textContent = JSON.stringify(decision.official, null, 2);
    ui["audit-json"].textContent = JSON.stringify(decision, null, 2);
    ui["decision-result"].hidden = false;
    drawScene();
  }

  ui["instruction-form"].addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.pending || !state.graph || !ui.instruction.value.trim()) return;
    const generation = state.generation;
    const scene = ui["scene-select"].value;
    state.pending = true;
    resetDecision();
    showError("");
    drawScene();
    ui["decision-badge"].textContent = "Analyzing";
    ui["decision-status"].textContent = "Apertus is interpreting the request; code will check the resulting sequence.";
    updateSubmit();
    try {
      const decision = await request("/api/decide", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene, instruction: ui.instruction.value.trim() }) });
      if (generation !== state.generation) return;
      renderDecision(decision);
    } catch (error) {
      if (generation !== state.generation) return;
      ui["decision-badge"].textContent = "Unavailable";
      ui["decision-badge"].dataset.kind = "error";
      ui["decision-status"].textContent = "No decision was produced. Check the local service and retry.";
      showError(error.message);
    } finally {
      state.pending = false;
      updateSubmit();
    }
  });

  ui["scene-select"].addEventListener("change", () => loadScene(ui["scene-select"].value));
  ui.instruction.addEventListener("input", updateSubmit);
  ui["overlay-toggle"].addEventListener("change", drawScene);
  ui["scene-canvas"].addEventListener("click", (event) => {
    if (!state.graph || !state.image) return;
    const rect = ui["scene-canvas"].getBoundingClientRect();
    const x = (event.clientX - rect.left) * state.image.naturalWidth / rect.width;
    const y = (event.clientY - rect.top) * state.image.naturalHeight / rect.height;
    const candidates = state.graph.objects.filter(({ box: [x0, y0, x1, y1] }) => x >= x0 && x < x1 && y >= y0 && y < y1);
    candidates.sort((a, b) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]));
    selectObject(candidates[0]?.id || null);
  });
  new ResizeObserver(drawScene).observe(ui["image-stage"]);
  window.addEventListener("resize", drawScene);
  ui["scene-reload"].addEventListener("click", () => loadScene(ui["scene-select"].value));

  async function initialize() {
    try {
      const scenes = await request("/api/scenes");
      ui["scene-select"].replaceChildren();
      if (!scenes.length) {
        ui["scene-select"].append(node("option", "No scenes available"));
        placeholder("No scenes found", "Add scene folders containing a valid rgb.png to the configured dataset.");
        ui["scene-status"].textContent = "No valid local scene images available.";
        ui["image-stage"].setAttribute("aria-busy", "false");
        return;
      }
      for (const scene of scenes) {
        const option = node("option", scene);
        option.value = scene;
        ui["scene-select"].append(option);
      }
      ui["scene-select"].disabled = false;
      await loadScene(scenes[0]);
    } catch (error) {
      ui["scene-select"].replaceChildren(node("option", "Scenes unavailable"));
      placeholder("Local service unavailable", "Reload after checking the dataset and local server.");
      ui["scene-status"].textContent = "Could not load the local scene list.";
      ui["image-stage"].setAttribute("aria-busy", "false");
      showError(error.message);
    }
  }
  initialize();
})();
