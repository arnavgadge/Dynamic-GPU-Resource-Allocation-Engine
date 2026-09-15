import { formatDuration, formatSimTime } from "../utils/formatters.js";

// Issue 5: `realTime`/`uptimeSeconds` come from `useRealClock` - the
// backend's actual system clock and this session's real elapsed
// uptime, polled every second - never the deterministic simulated
// clock and never a value this component invents or increments
// itself. `eventCount` is still read straight off backend state.
export default function Header({ realTime, uptimeSeconds, eventCount }) {
  return (
    <header className="panel header-panel">
      <div>
        <h1 className="header-title">
          GPU Scheduler <span className="dim">– Team Vicimus</span>
        </h1>
        <p className="header-subtitle">DSA Course Project | Real-time Simulation</p>
      </div>
      <div className="header-stats">
        <div className="header-stat">
          <span className="header-stat-label">SIM TIME</span>
          <span className="header-stat-value">{realTime ? formatSimTime(realTime) : "--:--:--"}</span>
        </div>
        <div className="header-stat">
          <span className="header-stat-label">UPTIME</span>
          <span className="header-stat-value">{uptimeSeconds != null ? formatDuration(uptimeSeconds) : "--"}</span>
        </div>
        <div className="header-stat">
          <span className="header-stat-label">EVENTS</span>
          <span className="header-stat-value">{eventCount}</span>
        </div>
      </div>
    </header>
  );
}
