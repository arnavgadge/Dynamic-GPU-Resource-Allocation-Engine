import { useSchedulerState } from "./hooks/useSchedulerState.js";
import { useRealClock } from "./hooks/useRealClock.js";
import Header from "./components/Header.jsx";
import SystemOverview from "./components/SystemOverview.jsx";
import EngineStatus from "./components/EngineStatus.jsx";
import UsersPanel from "./components/UsersPanel.jsx";
import GPUPool from "./components/GPUPool.jsx";
import WaitingQueue from "./components/WaitingQueue.jsx";
import EventLog from "./components/EventLog.jsx";
import DecisionTrace from "./components/DecisionTrace.jsx";
import ScenarioSelector from "./components/ScenarioSelector.jsx";
import SimulationControls from "./components/SimulationControls.jsx";
import ManualAssignPanel from "./components/ManualAssignPanel.jsx";
import ReclamationPrompt from "./components/ReclamationPrompt.jsx";
import PromptStatusList from "./components/PromptStatusList.jsx";
import Footer from "./components/Footer.jsx";

// The engineering control room (Part 7/39) - the pre-existing
// dashboard, unchanged in substance, plus the decision-trace panel
// and a logout control. Every value still comes from
// `useSchedulerState`, backed by the full admin `/api/state` +
// `/ws/state` (no token, or an ADMIN token) - this component decides
// nothing about scheduling, same as before. The header clock
// (`useRealClock`) is a separate, independent read of the backend's
// actual system clock (Issue 5) - never derived from simulated event
// timestamps.
export default function AdminConsole({ onLogout }) {
  const { state, scenarios, connectionStatus, bootError, commandError, commands } = useSchedulerState();
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
    <div className="dashboard">
      <Header
        realTime={clock?.real_time}
        uptimeSeconds={clock?.uptime_seconds}
        eventCount={state.events.length}
      />
      <div className="admin-topbar">
        <span className="dim">ADMIN CONSOLE</span>
        <button className="btn" onClick={onLogout}>LOGOUT</button>
      </div>

      {connectionStatus !== "connected" && (
        <div className="banner banner-warning">LIVE STREAM DISCONNECTED - attempting to reconnect...</div>
      )}
      {commandError && <div className="banner banner-error">{commandError}</div>}

      <div className="dashboard-grid">
        <aside className="dashboard-left">
          <ScenarioSelector
            scenarios={scenarios}
            activeScenarioId={state.simulation.scenario_id}
            onSelect={commands.loadScenario}
          />
          <SimulationControls
            simulation={state.simulation}
            onStart={commands.start}
            onPause={commands.pause}
            onReset={commands.reset}
            onStep={commands.step}
            onSetSpeed={commands.setSpeed}
          />
          <SystemOverview gpus={state.gpus} users={state.users} waitingQueue={state.waiting_queue} />
          <EngineStatus simulation={state.simulation} config={state.config} />
          <ManualAssignPanel gpus={state.gpus} onAssign={commands.manualAssign} />
        </aside>

        <main className="dashboard-main">
          <UsersPanel users={state.users} />
          <GPUPool gpus={state.gpus} />
          <DecisionTrace trace={state.decision_trace} />
          <EventLog events={state.events} />
        </main>

        <aside className="dashboard-right">
          <WaitingQueue jobs={state.waiting_queue} />
        </aside>
      </div>

      <Footer connectionStatus={connectionStatus} simulationRunning={state.simulation.running} />

      {/* Issue 6: a GPU owned by a real logged-in user is answered
          only from that user's own Portal - the Admin Console shows
          it as a read-only status. A classic scenario's synthetic
          user (no portal at all) keeps the interactive admin
          override, exactly as before. */}
      <PromptStatusList prompts={state.pending_prompts.filter((p) => p.owner_has_portal)} />
      <ReclamationPrompt
        prompts={state.pending_prompts.filter((p) => !p.owner_has_portal)}
        onRespond={commands.respondToPrompt}
      />
    </div>
  );
}
