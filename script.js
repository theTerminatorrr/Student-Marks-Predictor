const FEATURES = ["study_hours", "attendance_percent", "previous_score", "sleep_hours"];

// The trained Random Forest comes from model.js (window.MODEL)
const forest = (window.MODEL && window.MODEL.trees) || null;
if (!forest) document.getElementById("note").hidden = false;

function predictTree(t, x) {
  let i = 0;
  while (t.l[i] !== -1) i = x[t.f[i]] <= t.t[i] ? t.l[i] : t.r[i];
  return t.v[i];
}

function predict(x) {
  const raw = forest.reduce((sum, t) => sum + predictTree(t, x), 0) / forest.length;
  return Math.max(0, Math.min(100, Math.round(raw)));
}

function verdict(s) {
  if (s >= 85) return ["Excellent", "Strong habits. Keep this routine going."];
  if (s >= 70) return ["Good", "A little more study time could lift this further."];
  if (s >= 50) return ["Average", "Attendance and study time make the biggest difference."];
  return ["Needs work", "Try more study time and more regular attendance."];
}

// Slider and number box stay in sync
FEATURES.forEach(id => {
  const range = document.getElementById(id);
  const num = document.getElementById("n-" + id);
  const paint = () => range.style.setProperty("--fill", (range.value - range.min) / (range.max - range.min) * 100 + "%");
  range.addEventListener("input", () => { num.value = range.value; paint(); });
  num.addEventListener("input", () => {
    if (num.value === "") return;
    const v = Math.min(+range.max, Math.max(+range.min, +num.value));
    range.value = v; paint();
  });
  num.addEventListener("blur", () => { num.value = range.value; });
  paint();
});

document.getElementById("form").addEventListener("submit", e => {
  e.preventDefault();
  if (!forest) {
    document.getElementById("ring").style.setProperty("--p", 0);
    document.getElementById("score").textContent = "--";
    document.getElementById("title").textContent = "Model not loaded";
    document.getElementById("msg").textContent = "model.js is missing or failed to load, so no prediction was made.";
    return;
  }
  const x = FEATURES.map(id => parseFloat(document.getElementById(id).value));
  const score = predict(x);
  const [title, msg] = verdict(score);
  document.getElementById("ring").style.setProperty("--p", score);
  document.getElementById("score").textContent = score;
  document.getElementById("title").textContent = title;
  document.getElementById("msg").textContent = msg;
});
