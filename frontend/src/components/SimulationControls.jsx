// Every button here sends a command and renders whatever state the
// backend returns - none of them touch GPU/job state directly. The
// speed slider's allowed values come from the backend
// (`simulation.allowed_speeds`), not a range invented in React.
export default function SimulationControls({ simulation, onStart, onPause, onReset, onStep, onSetSpeed }) {
  const { running, speed, allowed_speeds: allowedSpeeds } = simulation;

  return (
    <section className="panel">
      <h2 className="panel-title">SIMULATION CONTROLS</h2>
      <div className="control-buttons">
        <button className="btn" onClick={onStart} disabled={running}>
          START
        </button>
        <button className="btn" onClick={onPause} disabled={!running}>
          PAUSE
        </button>
        <button className="btn" onClick={() => onStep()}>
          STEP
        </button>
        <button className="btn btn-danger" onClick={onReset}>
          RESET
        </button>
      </div>
      <div className="speed-control">
        <span className="speed-label">SPEED: {speed}x</span>
        <input
          type="range"
          min={0}
          max={allowedSpeeds.length - 1}
          step={1}
          value={allowedSpeeds.indexOf(speed)}
          onChange={(event) => onSetSpeed(allowedSpeeds[Number(event.target.value)])}
        />
        <div className="speed-ticks">
          {allowedSpeeds.map((value) => (
            <span key={value}>{value}x</span>
          ))}
        </div>
      </div>
    </section>
  );
}
