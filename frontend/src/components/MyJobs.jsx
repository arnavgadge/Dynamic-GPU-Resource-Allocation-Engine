import { formatDuration, formatPercent, formatScore, formatSimTime } from "../utils/formatters.js";
import { priorityClassName, gpuStatusClassName } from "../utils/eventStyles.js";

// Every value rendered - status, GPU id, utilization, queue position,
// score, completion time - is read straight from a `my_jobs` entry
// the backend already computed (`api/serializers.py::serialize_portal_state`).
// This component performs no ranking, scoring, or ETA math.
export default function MyJobs({ jobs }) {
  const active = jobs.filter((j) => j.status === "RUNNING");
  const waiting = jobs.filter((j) => j.status === "WAITING");
  const finished = jobs.filter((j) => j.status === "COMPLETED" || j.status === "RECLAIMED");

  return (
    <>
      <section className="panel">
        <h2 className="panel-title">MY CURRENT JOBS ({active.length})</h2>
        {active.length === 0 ? (
          <p className="empty-note">No active jobs.</p>
        ) : (
          active.map((job) => (
            <div className="my-job-card" key={job.job_id}>
              <div className="my-job-header">
                <span className="waiting-job-id">{job.job_id}</span>
                <span className={`badge ${gpuStatusClassName(job.gpu_status)}`}>{job.gpu_status}</span>
              </div>
              {job.gpu_count > 1 && (
                <div className="my-job-row">
                  GPUs: {(job.assigned_gpu_ids ?? []).length}/{job.gpu_count}
                  {job.gpus_still_needed > 0 ? ` (PENDING: ${job.gpus_still_needed})` : ""}
                </div>
              )}
              {/* One row + bar per GPU this job actually holds - never a single
                  averaged number for a multi-GPU job. */}
              {(job.gpus ?? []).map((gpu) => (
                <div key={gpu.gpu_id}>
                  <div className="my-job-row">GPU: {gpu.gpu_id}</div>
                  <div className="my-job-row">UTILIZATION: {formatPercent(gpu.utilization_percent)}</div>
                  <div className="util-bar-track">
                    <div
                      className={`util-bar-fill ${gpuStatusClassName(gpu.status)}`}
                      style={{ width: `${Math.max(0, Math.min(100, gpu.utilization_percent ?? 0))}%` }}
                    />
                  </div>
                </div>
              ))}
              <div className="my-job-row">START: {formatSimTime(job.started_at)}</div>
              <div className="my-job-row">EST. COMPLETION: {formatSimTime(job.estimated_completion)}</div>
              <div className="my-job-row">REMAINING: {formatDuration(job.remaining_seconds)}</div>
            </div>
          ))
        )}
      </section>

      <section className="panel">
        <h2 className="panel-title">MY REQUESTS ({waiting.length + finished.length})</h2>
        {waiting.length === 0 && finished.length === 0 ? (
          <p className="empty-note">No other requests.</p>
        ) : (
          <ul className="waiting-list">
            {waiting.map((job) => (
              <li className="waiting-item" key={job.job_id}>
                <div className="waiting-item-row">
                  <span className="waiting-job-id">{job.job_id}</span>
                  <span className={`priority-tag ${priorityClassName(job.priority)}`}>{job.priority}</span>
                </div>
                <div className="waiting-item-row waiting-item-meta">
                  <span>STATUS: WAITING</span>
                  <span>QUEUE POSITION: #{job.queue_position}</span>
                  <span>WAIT: {formatDuration(job.wait_seconds)}</span>
                  <span className="waiting-score">SCORE: {formatScore(job.allocation_score)}</span>
                </div>
                {/* Issue 7: a partially-satisfied multi-GPU request
                    must never look like "all requested" - the exact
                    assigned/pending split, straight from the backend. */}
                {job.gpu_count > 1 && (
                  <div className="waiting-item-row waiting-item-meta">
                    <span>REQUESTED: {job.gpu_count}</span>
                    <span>ASSIGNED: {(job.assigned_gpu_ids ?? []).length}</span>
                    <span>PENDING: {job.gpus_still_needed}</span>
                  </div>
                )}
                {(job.assigned_gpu_ids ?? []).length > 0 && (
                  <div className="waiting-item-row waiting-item-meta">
                    <span>HELD SO FAR: {job.assigned_gpu_ids.join(", ")}</span>
                  </div>
                )}
              </li>
            ))}
            {finished.map((job) => (
              <li className="waiting-item" key={job.job_id}>
                <div className="waiting-item-row">
                  <span className="waiting-job-id">{job.job_id}</span>
                  <span className={`priority-tag ${priorityClassName(job.priority)}`}>{job.priority}</span>
                </div>
                <div className="waiting-item-row waiting-item-meta">
                  <span>STATUS: {job.status}</span>
                  <span>GPU: {job.assigned_gpu_id ?? "-"}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
