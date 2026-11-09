const API_URL = "https://genre-classifier-omrz.onrender.com";

const $ = (id) => document.getElementById(id);
const form = $("form"), fileInput = $("file"), drop = $("drop");
const dropTitle = $("drop-title"), dropHint = $("drop-hint");
const stage = $("stage"), canvas = $("wave"), player = $("player");
const submitBtn = $("submit"), statusEl = $("status");
const result = $("result"), genreEl = $("genre");

const ctx = canvas.getContext("2d");
const BARS = 96;
const BASE = "rgba(243, 241, 255, 0.38)";
const LIT = "#ffc93c";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let peaks = new Array(BARS).fill(0.12);
let mode = "idle";            // idle | analyzing | done
let sweep = 0, raf = 0, objectUrl = null;

/* ---------- waveform ---------- */
function fitCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}

function draw() {
    fitCanvas();
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const slot = W / BARS, bw = slot * 0.62;
    for (let i = 0; i < BARS; i++) {
        const h = Math.max(slot * 0.5, peaks[i] * H * 0.92);
        const x = i * slot + (slot - bw) / 2, y = (H - h) / 2;
        const lit = mode === "done" || (mode === "analyzing" && !reduceMotion && i < sweep);
        ctx.fillStyle = lit ? LIT : BASE;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, y, bw, h, bw / 2); else ctx.rect(x, y, bw, h);
        ctx.fill();
    }
}

function tick() {
    sweep = (sweep + 0.9) % (BARS + 14);
    draw();
    raf = requestAnimationFrame(tick);
}

async function computePeaks(file) {
    const AC = window.AudioContext || window.webkitAudioContext;
    const ac = new AC();
    try {
        const audio = await ac.decodeAudioData(await file.arrayBuffer());
        const data = audio.getChannelData(0);
        const block = Math.floor(data.length / BARS);
        const out = [];
        for (let i = 0; i < BARS; i++) {
            let sum = 0;
            for (let j = i * block; j < (i + 1) * block; j++) sum += Math.abs(data[j]);
            out.push(sum / block);
        }
        const max = Math.max(...out) || 1;
        return out.map((v) => Math.max(0.06, v / max));
    } catch (e) {
        return new Array(BARS).fill(0.12);      // browser could not decode: show a flat line
    } finally {
        ac.close();
    }
}

/* ---------- picking a file ---------- */
async function handleFile(file) {
    if (!file) return;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    player.src = objectUrl;
    dropTitle.textContent = file.name;
    dropHint.textContent = "Click to choose a different song";
    result.hidden = true;
    setStatus("");
    mode = "idle";
    stage.hidden = false;
    peaks = await computePeaks(file);
    draw();
}

function setStatus(msg, isError = false) {
    statusEl.textContent = msg;
    statusEl.classList.toggle("error", isError);
}

fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, () => drop.classList.add("dragging")));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("dragging")));
window.addEventListener("resize", draw);

/* ---------- asking the API ---------- */
form.addEventListener("submit", async (e) => {
    e.preventDefault();                          // stop the page from reloading
    const file = fileInput.files[0];
    if (!file) { setStatus("Choose a song first.", true); return; }

    const body = new FormData();
    body.append("file", file);                   // field name must be "file" (matches app.py)

    submitBtn.disabled = true;
    result.hidden = true;
    setStatus("Listening to the first 30 seconds... (the server may take up to a minute to wake up)");
    mode = "analyzing"; sweep = 0;
    if (!reduceMotion) tick(); else draw();

    try {
        const res = await fetch(`${API_URL}/predict`, { method: "POST", body });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "The server could not process this file.");
        mode = "done";
        genreEl.textContent = data.prediction;
        result.hidden = false;
        setStatus("");
    } catch (err) {
        mode = "idle";
        // TypeError = network/CORS failure; SyntaxError = server returned non-JSON (e.g. a 502 while waking up)
        const unreachable = err instanceof TypeError || err instanceof SyntaxError;
        setStatus(unreachable
            ? "Can't reach the server. It may be waking up, so try again in a few seconds."
            : err.message, true);
    } finally {
        cancelAnimationFrame(raf);
        draw();
        submitBtn.disabled = false;
    }
});