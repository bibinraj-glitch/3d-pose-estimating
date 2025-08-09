import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const dom = {
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  exercise: document.getElementById("exercise"),
  stage: document.getElementById("stage"),
  reps: document.getElementById("reps"),
  angle: document.getElementById("angle"),
  fps: document.getElementById("fps"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  resetBtn: document.getElementById("resetBtn"),
};

let landmarker = null;
let running = false;
let lastVideoTime = -1;
let rafId = null;

const counter = {
  name: "squat",
  low: 80,
  high: 160,
  stage: "start",
  reps: 0,
  angle: 0,
  ema: null,
};

class EMA {
  constructor(alpha = 0.2) { this.alpha = alpha; this.value = null; }
  update(x) { this.value = this.value == null ? x : this.alpha * x + (1 - this.alpha) * this.value; return this.value; }
}

function toRad(deg) { return deg * Math.PI / 180; }
function toDeg(rad) { return rad * 180 / Math.PI; }

function angleDegrees(a, b, c) {
  const ba = [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const bc = [c[0]-b[0], c[1]-b[1], c[2]-b[2]];
  const dot = ba[0]*bc[0]+ba[1]*bc[1]+ba[2]*bc[2];
  const nba = Math.hypot(ba[0],ba[1],ba[2]) + 1e-8;
  const nbc = Math.hypot(bc[0],bc[1],bc[2]) + 1e-8;
  let cos = dot/(nba*nbc); cos = Math.max(-1, Math.min(1, cos));
  return toDeg(Math.acos(cos));
}

function setExercise(name) {
  counter.name = name;
  if (name === "squat") { counter.low = 80; counter.high = 160; counter.ema = new EMA(0.2); }
  if (name === "pushup") { counter.low = 70; counter.high = 160; counter.ema = new EMA(0.25); }
  counter.stage = "start"; counter.reps = 0; counter.angle = 0;
  dom.reps.textContent = counter.reps;
  dom.stage.textContent = counter.stage;
  dom.angle.textContent = "-";
}

setExercise(dom.exercise.value);

dom.exercise.addEventListener("change", (e) => setExercise(e.target.value));

dom.resetBtn.addEventListener("click", () => setExercise(counter.name));

dom.startBtn.addEventListener("click", async () => {
  dom.startBtn.disabled = true; dom.stopBtn.disabled = false; running = true;
  await ensureCamera();
  await ensureLandmarker();
  loop();
});

dom.stopBtn.addEventListener("click", () => {
  running = false; dom.startBtn.disabled = false; dom.stopBtn.disabled = true;
  if (rafId) cancelAnimationFrame(rafId);
});

async function ensureCamera() {
  if (dom.video.srcObject) return;
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
  dom.video.srcObject = stream;
  await dom.video.play();
  resizeCanvas();
}

function resizeCanvas() {
  const w = dom.video.videoWidth || 640;
  const h = dom.video.videoHeight || 480;
  dom.overlay.width = w; dom.overlay.height = h;
}

async function ensureLandmarker() {
  if (landmarker) return;
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-assets/pose_landmarker_full.task" },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

function computeAngleAndUpdate(worldLandmarks, visibility) {
  // Visibility fallback: choose side with better joint visibility
  const idx = (name) => ({
    left_hip: 23, right_hip: 24, left_knee: 25, right_knee: 26, left_ankle: 27, right_ankle: 28,
    left_shoulder: 11, right_shoulder: 12, left_elbow: 13, right_elbow: 14, left_wrist: 15, right_wrist: 16,
  })[name];

  const lvisKnee = visibility[25] ?? 0, rvisKnee = visibility[26] ?? 0;
  const lvisElbow = visibility[13] ?? 0, rvisElbow = visibility[14] ?? 0;

  if (counter.name === "squat") {
    const sideLeft = lvisKnee >= rvisKnee;
    const hip = worldLandmarks[sideLeft ? idx("left_hip") : idx("right_hip")];
    const knee = worldLandmarks[sideLeft ? idx("left_knee") : idx("right_knee")];
    const ankle = worldLandmarks[sideLeft ? idx("left_ankle") : idx("right_ankle")];
    const ang = angleDegrees(hip, knee, ankle);
    return ang;
  } else {
    const sideLeft = lvisElbow >= rvisElbow;
    const shoulder = worldLandmarks[sideLeft ? idx("left_shoulder") : idx("right_shoulder")];
    const elbow = worldLandmarks[sideLeft ? idx("left_elbow") : idx("right_elbow")];
    const wrist = worldLandmarks[sideLeft ? idx("left_wrist") : idx("right_wrist")];
    const ang = angleDegrees(shoulder, elbow, wrist);
    return ang;
  }
}

function updateCounter(angle) {
  const a = counter.ema.update(angle);
  counter.angle = a;
  if (counter.stage === "start" || counter.stage === "up") {
    if (a < counter.low) counter.stage = "down";
  } else if (counter.stage === "down") {
    if (a > counter.high) { counter.stage = "up"; counter.reps += 1; }
  }
  dom.stage.textContent = counter.stage;
  dom.reps.textContent = counter.reps.toString();
  dom.angle.textContent = Math.round(a).toString();
}

function draw(results) {
  const ctx = dom.overlay.getContext("2d");
  ctx.clearRect(0, 0, dom.overlay.width, dom.overlay.height);
  if (!results || !results.landmarks?.length) return;
  const utils = new DrawingUtils(ctx);
  const poseLandmarks = results.landmarks[0];
  utils.drawLandmarks(poseLandmarks, { lineWidth: 2, radius: 2, color: "#38bdf8" });
  utils.drawConnectors(poseLandmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "#22c55e", lineWidth: 2 });
}

let fpsHist = [];
function updateFps(start) {
  const dt = (performance.now() - start) / 1000;
  const fps = 1 / Math.max(1e-6, dt);
  fpsHist.push(fps); if (fpsHist.length > 30) fpsHist.shift();
  const avg = fpsHist.reduce((a,b)=>a+b,0)/fpsHist.length;
  dom.fps.textContent = avg.toFixed(1);
}

async function loop() {
  if (!running) return;
  if (dom.video.readyState < 2) { rafId = requestAnimationFrame(loop); return; }
  resizeCanvas();
  const now = dom.video.currentTime;
  if (lastVideoTime !== now) {
    const t0 = performance.now();
    const results = await landmarker.detectForVideo(dom.video, now);
    draw(results);
    if (results.worldLandmarks?.length) {
      const angle = computeAngleAndUpdate(results.worldLandmarks[0], results.poseLandmarks[0].map(l=>l.visibility ?? 0));
      if (angle != null && !Number.isNaN(angle)) updateCounter(angle);
    }
    updateFps(t0);
    lastVideoTime = now;
  }
  rafId = requestAnimationFrame(loop);
}

// iOS Safari requires a user gesture before camera; handled via Start button.