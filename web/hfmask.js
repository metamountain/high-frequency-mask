import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Adds a "calculate auto" button and a live mask/overlay preview to the
// High Frequency Mask node.
//
// Both round-trip to the backend: the image only exists there (see
// _LAST_IMAGE in __init__.py), so neither can be computed in the browser.
// The auto button measures once on demand. The live preview re-renders the
// mask on every slider move, debounced and abortable so a quick drag does
// not queue stale responses behind the latest one.

const NODE = "HighFrequencyMask";
const PREVIEW_WIDGETS = [
    "strength", "grow", "feather", "grain_filter", "opacity", "invert",
    "detector", "radius_override", "black_override", "white_override",
];
const PREVIEW_DEBOUNCE_MS = 120;

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function setWidget(node, name, value) {
    const w = widget(node, name);
    if (!w || value === undefined || value === null) return false;
    w.value = value;
    w.callback?.(value);
    return true;
}

const VALID_DETECTORS = ["guided", "high pass"];

// Repair graphs saved before `detector` existed.
//
// ComfyUI restores widget values by position, so a workflow saved when the node
// had one widget fewer feeds every later value into the wrong widget -- which is
// how a numeric 0 lands in a combo that only accepts strings. Anything that is
// not a valid option gets put back to the default.
function repairDetector(node) {
    const w = widget(node, "detector");
    if (!w) return;
    if (!VALID_DETECTORS.includes(w.value)) {
        w.value = VALID_DETECTORS[0];
    }
}

// Fallbacks for the optional numeric widgets, keyed by name.
//
// ComfyUI's frontend can leave an *optional* widget's value at null until the
// user actually touches it -- radius_override, black_override, white_override
// were always exposed to this, and opacity joins them now. Queuing with a
// null value fails server-side validation before build() ever runs
// ("float() argument must be a string or a real number, not 'NoneType'"), so
// this has to be fixed on the widget itself, not inside build().
const OPTIONAL_NUMERIC_DEFAULTS = {
    radius_override: 0,
    black_override: 0.0,
    white_override: 0.0,
    opacity: 1.0,
};

// Make sure an optional widget can never serialize to null in the prompt:
// repair its current value if it is already null/undefined/NaN, and guard
// serializeValue (the hook ComfyUI's frontend actually reads from when
// building the prompt) so a later reset back to null is caught too.
function repairOptionalNumeric(node) {
    for (const [name, fallback] of Object.entries(OPTIONAL_NUMERIC_DEFAULTS)) {
        const w = widget(node, name);
        if (!w) continue;
        if (w.value === null || w.value === undefined || Number.isNaN(w.value)) {
            w.value = fallback;
        }
        if (w._hfmaskGuarded) continue;
        w._hfmaskGuarded = true;
        const origSerialize = w.serializeValue?.bind(w);
        w.serializeValue = async (...args) => {
            let v = origSerialize ? await origSerialize(...args) : w.value;
            if (v === null || v === undefined || Number.isNaN(v)) v = fallback;
            return v;
        };
    }
}

app.registerExtension({
    name: "hfmask.autobutton",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;


        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            repairDetector(this);
            repairOptionalNumeric(this);
        };

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            repairDetector(this);
            repairOptionalNumeric(this);

            const node = this;
            const status = { text: "" };

            // ---- live mask / overlay preview -----------------------------
            //
            // Declared before the button below so the button's callback can
            // trigger an immediate refresh once it has set new values.

            const previewImg = document.createElement("img");
            previewImg.style.width = "100%";
            previewImg.style.display = "none";     // hidden until the first frame arrives
            previewImg.style.borderRadius = "4px";
            previewImg.style.cursor = "pointer";
            previewImg.title = "Click to toggle mask / overlay";

            const previewStatus = document.createElement("div");
            previewStatus.style.fontSize = "10px";
            previewStatus.style.opacity = "0.7";
            previewStatus.style.textAlign = "center";
            previewStatus.style.fontFamily = "monospace";
            previewStatus.style.padding = "2px 0";

            const previewWrap = document.createElement("div");
            previewWrap.style.display = "flex";
            previewWrap.style.flexDirection = "column";
            previewWrap.appendChild(previewImg);
            previewWrap.appendChild(previewStatus);

            let showOverlay = true;
            let lastPreview = null;

            function renderPreview() {
                if (!lastPreview) return;
                previewImg.src = "data:image/png;base64," +
                    (showOverlay ? lastPreview.overlay_png : lastPreview.mask_png);
                previewImg.style.display = "block";
                previewStatus.textContent =
                    `${showOverlay ? "overlay" : "mask"} (click to toggle) — ` +
                    `white ${lastPreview.white}% black ${lastPreview.black}% mean ${lastPreview.mean}`;
            }

            previewImg.addEventListener("click", () => {
                showOverlay = !showOverlay;
                renderPreview();
            });

            let previewTimer = null;
            let previewAbort = null;

            async function fetchPreview() {
                previewAbort?.abort();
                previewAbort = new AbortController();
                const body = { node_id: String(node.id) };
                for (const name of PREVIEW_WIDGETS) {
                    const w = widget(node, name);
                    if (w) body[name] = w.value;
                }
                try {
                    const res = await api.fetchApi("/hfmask/preview", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(body),
                        signal: previewAbort.signal,
                    });
                    const data = await res.json();
                    if (!data.ok) {
                        previewStatus.textContent = data.message || "no preview yet";
                        return;
                    }
                    lastPreview = data;
                    renderPreview();
                } catch (err) {
                    if (err?.name !== "AbortError") {
                        previewStatus.textContent = `preview failed: ${err}`;
                    }
                }
            }

            function schedulePreview(delay = PREVIEW_DEBOUNCE_MS) {
                clearTimeout(previewTimer);
                previewTimer = setTimeout(fetchPreview, delay);
            }

            // ---- calculate auto button -------------------------------------

            const button = this.addWidget("button", "calculate auto", null, async () => {
                button.name = "measuring…";
                node.setDirtyCanvas(true, true);
                try {
                    const res = await api.fetchApi("/hfmask/auto", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            node_id: String(node.id),
                            grow: widget(node, "grow")?.value ?? 0.5,
                            feather: widget(node, "feather")?.value ?? 1.0,
                            detector: widget(node, "detector")?.value ?? "guided",
                        }),
                    });
                    const data = await res.json();

                    if (!data.ok) {
                        status.text = data.message || "could not measure";
                    } else {
                        setWidget(node, "strength", data.strength);
                        setWidget(node, "grain_filter", data.grain_filter);
                        status.text = data.note
                            ? `strength ${data.strength}, grain_filter ${data.grain_filter} — ${data.note}`
                            : `strength ${data.strength}, grain_filter ${data.grain_filter}` +
                              ` — noise ${data.noise}, white ${data.white}%, black ${data.black}%`;
                    }
                } catch (err) {
                    status.text = `failed: ${err}`;
                } finally {
                    button.name = "calculate auto";
                    node.setDirtyCanvas(true, true);
                }
                schedulePreview(0);
            });

            button.serialize = false;

            // A read-only line under the button, so the result is visible without
            // hunting through the console.
            const readout = this.addWidget("text", "", "", () => {}, {
                serialize: false,
                multiline: false,
            });
            readout.disabled = true;
            Object.defineProperty(readout, "value", {
                get: () => status.text,
                set: () => {},
            });

            // Re-render the preview on every slider/combo change, and again once
            // the graph actually runs (a fresh image may have been cached).
            for (const name of PREVIEW_WIDGETS) {
                const w = widget(node, name);
                if (!w) continue;
                const orig = w.callback;
                w.callback = function (...args) {
                    const r = orig?.apply(this, args);
                    schedulePreview();
                    return r;
                };
            }

            const onExecuted = node.onExecuted;
            node.onExecuted = function (...args) {
                const r = onExecuted?.apply(this, args);
                schedulePreview(0);
                return r;
            };

            this.addDOMWidget("hfmask_preview", "preview", previewWrap, { serialize: false });
        };
    },
});
