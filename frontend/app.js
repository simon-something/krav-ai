// === State Machine ===
const State = {
  READY: "READY",
  RECORDING: "RECORDING",
  ANALYZING: "ANALYZING",
  FEEDBACK: "FEEDBACK",
};

let currentState = State.READY;
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = null;
let timerInterval = null;
let abortController = null;
const sessionHistory = [];

// === DOM Elements ===
const cameraPreview = document.getElementById("camera-preview");
const videoContainer = document.getElementById("video-container");
const timerDisplay = document.getElementById("timer");
const strikeTypeSelect = document.getElementById("strike-type");
const recordBtn = document.getElementById("record-btn");
const btnLabel = recordBtn.querySelector(".btn-label");
const cameraError = document.getElementById("camera-error");

const statusSection = document.getElementById("status-section");
const statusMessage = document.getElementById("status-message");

const errorSection = document.getElementById("error-section");
const errorMessage = document.getElementById("error-message");
const errorDismiss = document.getElementById("error-dismiss");

const feedbackSection = document.getElementById("feedback-section");
const feedbackSummary = document.getElementById("feedback-summary");
const positivesList = document.getElementById("positives-list");
const correctionsList = document.getElementById("corrections-list");

const historySection = document.getElementById("history-section");
const historyList = document.getElementById("history-list");

// === Camera ===
async function initCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    cameraPreview.srcObject = mediaStream;
    recordBtn.disabled = false;
  } catch (err) {
    console.error("Camera access denied:", err);
    cameraError.classList.remove("hidden");
    recordBtn.disabled = true;
  }
}

// === Recording ===
function startRecording() {
  recordedChunks = [];

  const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
    ? "video/webm;codecs=vp8"
    : "video/webm";

  mediaRecorder = new MediaRecorder(mediaStream, { mimeType });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const duration = (Date.now() - recordingStartTime) / 1000;

    if (duration < 1) {
      showError("Recording too short. Please record at least 1 second.");
      transitionTo(State.READY);
      return;
    }

    const blob = new Blob(recordedChunks, { type: mimeType });
    submitForAnalysis(blob);
  };

  mediaRecorder.start(100); // collect data every 100ms
  recordingStartTime = Date.now();
  startTimer();
  transitionTo(State.RECORDING);
}

function stopRecording() {
  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
  }
  stopTimer();
}

// Auto-stop at max duration (15 seconds)
function checkMaxDuration() {
  if (!recordingStartTime) return;
  const elapsed = (Date.now() - recordingStartTime) / 1000;
  if (elapsed >= 15) {
    stopRecording();
  }
}

// === Timer ===
function startTimer() {
  timerDisplay.classList.remove("hidden");
  updateTimerDisplay(0);
  timerInterval = setInterval(() => {
    const elapsed = (Date.now() - recordingStartTime) / 1000;
    updateTimerDisplay(elapsed);
    checkMaxDuration();
  }, 200);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
  timerDisplay.classList.add("hidden");
}

function updateTimerDisplay(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  timerDisplay.textContent = `${mins}:${secs.toString().padStart(2, "0")}`;
}

// === API Communication ===
async function submitForAnalysis(videoBlob) {
  transitionTo(State.ANALYZING);

  const strikeType = strikeTypeSelect.value;
  const formData = new FormData();
  formData.append("video", videoBlob, "recording.webm");
  formData.append("strike_type", strikeType);

  abortController = new AbortController();

  // Simulate progress steps with timing
  setProgressStep("extracting");

  try {
    const progressTimer = simulateProgress();

    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
      signal: abortController.signal,
    });

    clearTimeout(progressTimer.t1);
    clearTimeout(progressTimer.t2);

    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      throw new Error(errData?.detail || `Server error (${response.status})`);
    }

    // Mark all steps done
    setProgressStep("done-all");

    const feedback = await response.json();
    renderFeedback(feedback, strikeType);
    addToHistory(feedback, strikeType);
    transitionTo(State.FEEDBACK);
  } catch (err) {
    if (err.name === "AbortError") return;
    console.error("Analysis failed:", err);
    showError(err.message || "Analysis failed. Please try again.");
    transitionTo(State.READY);
  } finally {
    abortController = null;
  }
}

function simulateProgress() {
  // These are approximate timings — the real progress depends on server
  const t1 = setTimeout(() => setProgressStep("analyzing"), 3000);
  const t2 = setTimeout(() => setProgressStep("generating"), 8000);
  return { t1, t2 };
}

function setProgressStep(step) {
  const steps = document.querySelectorAll(".step");
  const order = ["extracting", "analyzing", "generating"];

  if (step === "done-all") {
    steps.forEach((s) => {
      s.classList.remove("active");
      s.classList.add("done");
    });
    return;
  }

  const activeIdx = order.indexOf(step);
  steps.forEach((s, i) => {
    s.classList.remove("active", "done");
    if (i < activeIdx) s.classList.add("done");
    else if (i === activeIdx) s.classList.add("active");
  });
}

// === Feedback Rendering ===
function renderFeedback(data, strikeType) {
  // Summary
  feedbackSummary.textContent = data.summary || "No summary available.";

  // Positives
  positivesList.innerHTML = "";
  (data.positives || []).forEach((p) => {
    const li = document.createElement("li");
    li.textContent = p;
    positivesList.appendChild(li);
  });

  // Corrections — sort by priority (high first)
  const priorityOrder = { high: 0, medium: 1, low: 2 };
  const corrections = [...(data.corrections || [])].sort(
    (a, b) => (priorityOrder[a.priority] ?? 3) - (priorityOrder[b.priority] ?? 3)
  );

  correctionsList.innerHTML = "";
  corrections.forEach((c) => {
    const card = document.createElement("div");
    card.className = `correction-card priority-${c.priority || "low"}`;

    card.innerHTML = `
      <div class="correction-header">
        <span class="correction-title">${escapeHtml(c.issue)}</span>
        <span class="priority-badge ${c.priority || "low"}">${c.priority || "low"}</span>
      </div>
      <p class="correction-detail"><strong>Why it matters:</strong> ${escapeHtml(c.why_it_matters)}</p>
      <p class="correction-detail"><strong>How to fix:</strong> ${escapeHtml(c.how_to_fix)}</p>
    `;

    correctionsList.appendChild(card);
  });
}

// === Session History ===
function addToHistory(feedback, strikeType) {
  const attempt = sessionHistory.length + 1;
  sessionHistory.push({ attempt, strikeType, feedback });
  renderHistory();
}

function renderHistory() {
  if (sessionHistory.length === 0) return;
  historySection.classList.remove("hidden");
  historyList.innerHTML = "";

  // Most recent first
  [...sessionHistory].reverse().forEach((entry) => {
    const card = document.createElement("div");
    card.className = "history-card";

    const correctionCount = entry.feedback.corrections?.length || 0;
    const displayType = entry.strikeType.replace("_", "-");

    card.innerHTML = `
      <div class="history-meta">
        <span class="history-attempt">#${entry.attempt}</span>
        <span class="history-strike">${escapeHtml(displayType)}</span>
        <span class="history-corrections-count">${correctionCount} correction${correctionCount !== 1 ? "s" : ""}</span>
      </div>
      <div class="history-summary">${escapeHtml(entry.feedback.summary || "")}</div>
    `;

    historyList.appendChild(card);
  });
}

// === State Transitions ===
function transitionTo(state) {
  currentState = state;

  // Reset visibility
  statusSection.classList.add("hidden");
  errorSection.classList.add("hidden");
  feedbackSection.classList.add("hidden");
  videoContainer.classList.remove("recording");

  switch (state) {
    case State.READY:
      recordBtn.disabled = !mediaStream;
      recordBtn.classList.remove("is-recording");
      recordBtn.setAttribute("aria-label", "Start recording");
      btnLabel.textContent = "Record";
      strikeTypeSelect.disabled = false;
      break;

    case State.RECORDING:
      videoContainer.classList.add("recording");
      recordBtn.disabled = false;
      recordBtn.classList.add("is-recording");
      recordBtn.setAttribute("aria-label", "Stop recording");
      btnLabel.textContent = "Stop";
      strikeTypeSelect.disabled = true;
      break;

    case State.ANALYZING:
      recordBtn.disabled = true;
      recordBtn.classList.remove("is-recording");
      btnLabel.textContent = "Analyzing...";
      strikeTypeSelect.disabled = true;
      statusSection.classList.remove("hidden");
      // Reset progress steps
      document.querySelectorAll(".step").forEach((s) => {
        s.classList.remove("active", "done");
      });
      break;

    case State.FEEDBACK:
      recordBtn.disabled = !mediaStream;
      recordBtn.classList.remove("is-recording");
      recordBtn.setAttribute("aria-label", "Start recording");
      btnLabel.textContent = "Record Again";
      strikeTypeSelect.disabled = false;
      feedbackSection.classList.remove("hidden");
      feedbackSection.scrollIntoView({ behavior: "smooth", block: "start" });
      break;
  }
}

// === Error Handling ===
function showError(message) {
  errorMessage.textContent = message;
  errorSection.classList.remove("hidden");
}

errorDismiss.addEventListener("click", () => {
  errorSection.classList.add("hidden");
});

// === Utility ===
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// === Event Listeners ===
recordBtn.addEventListener("click", () => {
  if (currentState === State.RECORDING) {
    stopRecording();
  } else if (currentState === State.READY || currentState === State.FEEDBACK) {
    startRecording();
  }
});

// Keyboard support: Space/Enter to toggle recording
recordBtn.addEventListener("keydown", (e) => {
  if (e.key === " " || e.key === "Enter") {
    e.preventDefault();
    recordBtn.click();
  }
});

// === Init ===
initCamera();
