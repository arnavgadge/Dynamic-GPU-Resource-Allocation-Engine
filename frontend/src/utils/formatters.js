// Pure formatting helpers. None of these compute anything the
// backend hasn't already decided - they only reshape strings/numbers
// the backend already sent for display.

export function formatSimTime(isoString) {
  if (!isoString) return "--:--:--";
  const date = new Date(isoString);
  return date.toISOString().slice(11, 19); // HH:MM:SS, UTC - the simulated clock's own timezone
}

export function formatDate(isoString) {
  if (!isoString) return "----------";
  return new Date(isoString).toISOString().slice(0, 10);
}

export function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return "--";
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m ${String(secs).padStart(2, "0")}s`;
  return `${secs}s`;
}

export function formatPercent(value) {
  if (value == null || Number.isNaN(value)) return "--%";
  return `${value.toFixed(0)}%`;
}

export function formatMemory(usedMb, totalMb) {
  if (usedMb == null || totalMb == null) return "n/a";
  const usedGb = usedMb / 1024;
  const totalGb = totalMb / 1024;
  return `${usedGb.toFixed(1)} / ${totalGb.toFixed(1)} GB`;
}

export function formatScore(score) {
  if (score == null) return "--";
  return score.toFixed(3);
}
