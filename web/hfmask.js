import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Adds a "calculate auto" button to the High Frequency Mask node.
//
// The button asks the backend to look at the image this node last processed and
// pick strength and grain_filter from it. That image only exists server-side,
// so this is a round trip rather than something the browser can work out. If
// the graph has not run yet there is nothing to measure, and the button says so
// instead of inventing numbers.

const NODE = "HighFrequencyMask";

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

app.registerExtension({
    name: "hfmask.autobutton",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);

            const node = this;
            const status = { text: "" };

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
        };
    },
});
