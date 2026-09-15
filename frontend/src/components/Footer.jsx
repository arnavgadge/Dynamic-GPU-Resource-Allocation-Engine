// Connection/backend/engine status shown here reflects the actual
// WebSocket connection state (`connectionStatus`) and the backend's
// own simulation flag - never a value assumed by the frontend.
export default function Footer({ connectionStatus, simulationRunning }) {
  const backendLabel = connectionStatus === "connected" ? "CONNECTED" : "OFFLINE";
  return (
    <footer className="panel footer-panel">
      <div className="footer-item">
        <span className="footer-label">BACKEND</span>
        <span className={`footer-value ${connectionStatus === "connected" ? "ok" : "bad"}`}>{backendLabel}</span>
      </div>
      <div className="footer-item">
        <span className="footer-label">SIMULATION</span>
        <span className="footer-value">{simulationRunning ? "RUNNING" : "PAUSED"}</span>
      </div>
      <div className="footer-item">
        <span className="footer-label">ENGINE</span>
        <span className={`footer-value ${connectionStatus === "connected" ? "ok" : "bad"}`}>
          {connectionStatus === "connected" ? "ONLINE" : "OFFLINE"}
        </span>
      </div>
      <div className="footer-item">
        <span className="footer-label">STREAM</span>
        <span className="footer-value">{connectionStatus.toUpperCase()}</span>
      </div>
    </footer>
  );
}
