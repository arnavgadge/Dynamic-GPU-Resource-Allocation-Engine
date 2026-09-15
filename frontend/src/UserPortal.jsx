import { usePortalState } from "./hooks/usePortalState.js";
import { useRealClock } from "./hooks/useRealClock.js";
import GPURequestForm from "./components/GPURequestForm.jsx";
import MyJobs from "./components/MyJobs.jsx";
import NotificationsPanel from "./components/NotificationsPanel.jsx";
import ReclamationPrompt from "./components/ReclamationPrompt.jsx";
import { formatSimTime } from "./utils/formatters.js";

// The User Portal: a simple internal GPU request terminal. It shows
// only this user's own jobs/notifications/prompt (Part 24/38) - no
// full GPU pool, no other users, no allocation-engine internals. It
// never decides anything itself; every value comes from
// `usePortalState`, which is backed entirely by `/api/portal/state`
// and the scoped WebSocket stream. The header clock (`useRealClock`)
// is a separate, independent read of the backend's real system clock
// (Issue 5), same as the Admin Console's.
export default function UserPortal({ session, onLogout }) {
  const { state, connectionStatus, bootError, commandError, commands } = usePortalState(session.token);
  const clock = useRealClock();

  if (bootError) {
    return (
      <div className="boot-screen">
        <p className="boot-error">BACKEND OFFLINE</p>
        <p className="boot-detail">{bootError}</p>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="boot-screen">
        <p className="boot-status">CONNECTING...</p>
      </div>
    );
  }

  return (
    <div className="dashboard portal">
      <header className="panel header-panel">
        <div>
          <h1 className="header-title">GPU SCHEDULER <span className="dim">- Employee Portal</span></h1>
          <p className="header-subtitle">USER: {state.user.name}</p>
        </div>
        <div className="header-stats">
          <div className="header-stat">
            <span className="header-stat-label">SIM TIME</span>
            <span className="header-stat-value">{clock?.real_time ? formatSimTime(clock.real_time) : "--:--:--"}</span>
          </div>
          <button className="btn" onClick={onLogout}>LOGOUT</button>
        </div>
      </header>

      {connectionStatus !== "connected" && (
        <div className="banner banner-warning">LIVE STREAM DISCONNECTED - attempting to reconnect...</div>
      )}
      {commandError && <div className="banner banner-error">{commandError}</div>}

      <div className="portal-grid">
        <div className="portal-column">
          <GPURequestForm onSubmit={commands.submitRequest} />
          <NotificationsPanel notifications={state.notifications} />
        </div>
        <div className="portal-column">
          <MyJobs jobs={state.my_jobs} />
        </div>
      </div>

      <ReclamationPrompt
        prompts={state.pending_prompt ? [state.pending_prompt] : []}
        onRespond={commands.respondToPrompt}
        yesLabel="YES, KEEP GPU"
        noLabel="NO, RELEASE GPU"
      />
    </div>
  );
}
