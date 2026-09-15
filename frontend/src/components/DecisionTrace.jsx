import { formatPercent, formatScore } from "../utils/formatters.js";

// Every candidate, score, and reason here comes from the real
// `AllocationDecision`/`RoutingDecision` objects the backend already
// produced (`api/serializers.py::serialize_decision_trace`) - this
// panel recomputes nothing; it only shows the last decision made,
// exactly as the backend explains it.
export default function DecisionTrace({ trace }) {
  return (
    <section className="panel decision-trace-panel">
      <h2 className="panel-title">DECISION TRACE (LAST ALLOCATION)</h2>
      {!trace ? (
        <p className="decision-empty">ⓘ No allocation has happened yet.</p>
      ) : (
        <div className="decision-trace-grid">
          <div className="decision-block">
            <h3 className="decision-block-title">ALLOCATION - {trace.allocation.policy}</h3>
            <ul className="decision-candidate-list">
              {trace.allocation.candidates.map((c) => (
                <li key={c.job_id} className={c.job_id === trace.job_id ? "decision-winner" : ""}>
                  <span className="waiting-job-id">{c.job_id}</span>
                  <span>{c.priority}</span>
                  <span>{c.size_minutes}m</span>
                  <span>{c.score != null ? `score ${formatScore(c.score)}` : "-"}</span>
                </li>
              ))}
            </ul>
            <p className="decision-reason">SELECTED: {trace.job_id} - {trace.allocation.reason}</p>
          </div>

          <div className="decision-block">
            <h3 className="decision-block-title">GPU ROUTING - {trace.routing.outcome}</h3>
            <ul className="decision-candidate-list">
              {trace.routing.candidates.map((c) => (
                <li key={c.gpu_id} className={c.gpu_id === trace.gpu_id ? "decision-winner" : ""}>
                  <span className="waiting-job-id">{c.gpu_id}</span>
                  <span>{formatPercent(c.utilization_percent)}</span>
                  <span>{c.available ? "available" : "unavailable"}</span>
                </li>
              ))}
            </ul>
            <p className="decision-reason">SELECTED: {trace.gpu_id} - {trace.routing.reason}</p>
          </div>
        </div>
      )}
    </section>
  );
}
