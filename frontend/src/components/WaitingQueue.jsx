import { formatDuration, formatScore } from "../utils/formatters.js";
import { priorityClassName } from "../utils/eventStyles.js";

// The allocation score column renders `entry.allocation_score`
// exactly as computed by `AllocationEngine.calculate_score` on the
// backend - this component performs no arithmetic on priority or
// job size. See the README's Phase 7 section for why the score shown
// here is relative to the full waiting set, a display simplification
// that never changes what the backend actually allocates.
export default function WaitingQueue({ jobs }) {
  return (
    <section className="panel waiting-queue-panel">
      <h2 className="panel-title">WAITING QUEUE ({jobs.length})</h2>
      {jobs.length === 0 ? (
        <p className="empty-note">No jobs waiting.</p>
      ) : (
        <ul className="waiting-list">
          {jobs.map((job) => (
            <li className="waiting-item" key={job.job_id}>
              <div className="waiting-item-row">
                <span className="waiting-job-id">{job.job_id}</span>
                <span className={`priority-tag ${priorityClassName(job.priority)}`}>{job.priority}</span>
              </div>
              <div className="waiting-item-row waiting-item-meta">
                <span>SIZE: {job.estimated_size_minutes}m</span>
                <span>WAIT: {formatDuration(job.wait_seconds)}</span>
                <span className="waiting-score">SCORE: {formatScore(job.allocation_score)}</span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
