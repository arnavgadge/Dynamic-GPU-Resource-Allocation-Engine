// Every value shown is read from backend state/config - "ENABLED" for
// Reclamation/Balancing reflects that this project's Scheduler always
// wires both engines in (there is no backend toggle to disable
// either), not a value invented on the frontend.
const PILL_CLASS = {
  RUNNING: "status-pill-running",
  ENABLED: "status-pill-enabled",
  STOPPED: "status-pill-stopped",
  PAUSED: "status-pill-paused",
};

export default function EngineStatus({ simulation, config }) {
  const rows = [
    ["SCHEDULER", simulation.running ? "RUNNING" : "STOPPED"],
    ["RECLAMATION", "ENABLED"],
    ["LOAD BALANCING", "ENABLED"],
    ["SIMULATION", simulation.running ? "RUNNING" : "PAUSED"],
    [
      "POLICY",
      `${(config.allocation.priority_weight * 100).toFixed(0)}/${(config.allocation.size_weight * 100).toFixed(0)} ` +
        `· ${(config.allocation.similarity_threshold * 100).toFixed(0)}% FCFS`,
      true,
    ],
  ];

  return (
    <section className="panel">
      <h2 className="panel-title">ENGINE STATUS</h2>
      <dl className="kv-list">
        {rows.map(([label, value, isPlain]) => (
          <div className="kv-row" key={label}>
            <dt>{label}</dt>
            <dd>
              {isPlain ? (
                value
              ) : (
                <span className={`status-pill ${PILL_CLASS[value] ?? "status-pill-neutral"}`}>{value}</span>
              )}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
