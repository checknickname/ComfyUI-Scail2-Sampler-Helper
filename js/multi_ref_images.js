import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "SCAIL2MultiRefImages";
const ROW_HEIGHT = 112;
const THUMB_MAX_WIDTH = 96;

function viewUrl(entry) {
    const params = new URLSearchParams({
        filename: entry.name || "",
        subfolder: entry.subfolder || "",
        type: entry.type || "input",
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function readEntries(widget) {
    try {
        const parsed = JSON.parse(widget.value || "[]");
        return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
        return [];
    }
}

function setupNode(node) {
    const dataWidget = node.widgets?.find((w) => w.name === "images_data");
    if (!dataWidget) return;

    dataWidget.type = "hidden";
    dataWidget.hidden = true;
    dataWidget.computeSize = () => [0, -4];

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "4px";
    container.style.overflowY = "auto";
    container.style.padding = "2px";

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.multiple = true;
    fileInput.style.display = "none";
    container.appendChild(fileInput);

    let entries = [];
    let dragIndex = null;

    const sync = () => {
        dataWidget.value = JSON.stringify(
            entries.map((e) => ({
                name: e.name,
                subfolder: e.subfolder || "",
                type: e.type || "input",
                index: Number.isFinite(e.index) ? e.index : 0,
            }))
        );
        render();
        node.setDirtyCanvas(true, true);
    };

    const render = () => {
        Array.from(container.querySelectorAll("[data-scail-row]")).forEach((el) => el.remove());

        entries.forEach((entry, i) => {
            const row = document.createElement("div");
            row.setAttribute("data-scail-row", "1");
            row.draggable = true;
            row.style.display = "flex";
            row.style.alignItems = "center";
            row.style.gap = "6px";
            row.style.height = `${ROW_HEIGHT - 8}px`;
            row.style.cursor = "grab";
            row.style.borderTop = "2px solid transparent";

            row.addEventListener("dragstart", (e) => {
                dragIndex = i;
                e.dataTransfer.effectAllowed = "move";
                row.style.opacity = "0.5";
            });
            row.addEventListener("dragend", () => {
                dragIndex = null;
                row.style.opacity = "";
                container.querySelectorAll("[data-scail-row]").forEach((el) => {
                    el.style.borderTop = "2px solid transparent";
                });
            });
            row.addEventListener("dragover", (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = "move";
                row.style.borderTop = (dragIndex !== null && dragIndex !== i)
                    ? "2px solid #5af" : "2px solid transparent";
            });
            row.addEventListener("dragleave", () => {
                row.style.borderTop = "2px solid transparent";
            });
            row.addEventListener("drop", (e) => {
                e.preventDefault();
                row.style.borderTop = "2px solid transparent";
                if (dragIndex === null || dragIndex === i) return;
                const [moved] = entries.splice(dragIndex, 1);
                entries.splice(i, 0, moved);
                dragIndex = null;
                sync();
            });

            const handle = document.createElement("span");
            handle.textContent = "⠿";
            handle.style.opacity = "0.6";
            handle.style.fontSize = "16px";
            row.appendChild(handle);

            const thumb = document.createElement("img");
            thumb.src = viewUrl(entry);
            thumb.draggable = false;
            thumb.style.height = "100%";
            thumb.style.width = "auto";
            thumb.style.maxWidth = `${THUMB_MAX_WIDTH}px`;
            thumb.style.objectFit = "cover";
            thumb.style.borderRadius = "3px";
            row.appendChild(thumb);

            const idxInput = document.createElement("input");
            idxInput.type = "number";
            idxInput.step = "1";
            idxInput.value = String(entry.index ?? 0);
            idxInput.title = "pose index";
            idxInput.style.width = "64px";
            idxInput.addEventListener("mousedown", (e) => e.stopPropagation());
            idxInput.addEventListener("change", () => {
                const v = parseInt(idxInput.value, 10);
                entries[i].index = Number.isFinite(v) ? v : 0;
                sync();
            });
            row.appendChild(idxInput);

            const del = document.createElement("button");
            del.textContent = "✕";
            del.addEventListener("mousedown", (e) => e.stopPropagation());
            del.addEventListener("click", () => {
                entries.splice(i, 1);
                sync();
            });
            row.appendChild(del);

            container.appendChild(row);
        });
    };

    fileInput.addEventListener("change", async () => {
        const files = Array.from(fileInput.files || []);
        for (const file of files) {
            try {
                const fd = new FormData();
                fd.append("image", file);
                const resp = await api.fetchApi("/upload/image", { method: "POST", body: fd });
                if (resp.status !== 200) continue;
                const data = await resp.json();
                const nextIndex = entries.length
                    ? Math.max(...entries.map((e) => e.index ?? 0)) + 1
                    : 0;
                entries.push({
                    name: data.name,
                    subfolder: data.subfolder || "",
                    type: data.type || "input",
                    index: nextIndex,
                });
            } catch (e) {
                console.error("[SCAIL2MultiRefImages] upload failed", e);
            }
        }
        fileInput.value = "";
        sync();
    });

    node.addWidget("button", "Upload Images", null, () => fileInput.click());

    const domWidget = node.addDOMWidget("multi_ref_preview", "div", container, {
        serialize: false,
    });
    domWidget.computeSize = function () {
        const rows = Math.max(entries.length, 1);
        return [this.parent?.size?.[0] ?? 200, rows * ROW_HEIGHT + 8];
    };

    node.scailRefresh = () => {
        entries = readEntries(dataWidget);
        render();
    };

    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        this.scailRefresh();
    };

    node.scailRefresh();
}

app.registerExtension({
    name: "SCAIL2.MultiRefImages",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            setupNode(this);
            return r;
        };
    },
});
