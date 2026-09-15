import StatusBadge from "./StatusBadge.jsx";
import UtilizationBar from "./UtilizationBar.jsx";
import { formatMemory } from "../utils/formatters.js";

// One GPU, entirely from backend fields - no GPU id, status, or
// assignment is ever assumed or hardcoded here.
export default function GPURow({ gpu }) {
  return (
    <div className={`gpu-row${gpu.has_pending_prompt ? " gpu-row-prompt" : ""}`}>
      <div className="gpu-row-header">
        <span className="gpu-id">{gpu.gpu_id}</span>
        <StatusBadge status={gpu.status} />
      </div>
      <UtilizationBar percent={gpu.utilization_percent} status={gpu.status} />
      <dl className="gpu-details">
        <div>
          <dt>USER</dt>
          <dd>{gpu.assigned_user_name ?? "—"}</dd>
        </div>
        <div>
          <dt>JOB</dt>
          <dd>{gpu.assigned_job_name ?? "—"}</dd>
        </div>
        <div>
          <dt>MEMORY</dt>
          <dd>{formatMemory(gpu.memory_used_mb, gpu.total_memory_mb)}</dd>
        </div>
      </dl>
    </div>
  );
}
