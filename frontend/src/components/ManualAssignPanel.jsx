import { useState } from "react";

// Admin-only demo/test control (Phase 10): pick a currently-free GPU
// and a user, and ask the backend to place that user onto it - for
// establishing a starting arrangement before a demonstration begins.
// This component computes nothing: the free-GPU list and the user
// list both come straight from backend state, and the actual
// assignment is entirely `Scheduler.manual_assign_gpu`'s decision -
// this only calls `onAssign` and shows whatever the backend says.
const KNOWN_USERS = [
  { user_id: "user_a", display_name: "User A" },
  { user_id: "user_b", display_name: "User B" },
  { user_id: "user_c", display_name: "User C" },
  { user_id: "user_d", display_name: "User D" },
];

export default function ManualAssignPanel({ gpus, onAssign }) {
  const freeGpus = gpus.filter((g) => g.status === "IDLE" && !g.assigned_user_id);
  const [gpuId, setGpuId] = useState("");
  const [userId, setUserId] = useState(KNOWN_USERS[0].user_id);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const effectiveGpuId = gpuId || freeGpus[0]?.gpu_id || "";

  async function handleSubmit(event) {
    event.preventDefault();
    if (!effectiveGpuId) return;
    setSubmitting(true);
    setError(null);
    try {
      const user = KNOWN_USERS.find((u) => u.user_id === userId);
      await onAssign(effectiveGpuId, userId, user?.display_name ?? userId);
      setGpuId("");
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="panel">
      <h2 className="panel-title">MANUAL ASSIGNMENT (DEMO SETUP)</h2>
      {freeGpus.length === 0 ? (
        <p className="empty-note">No free GPUs left to manually assign.</p>
      ) : (
        <form className="request-form" onSubmit={handleSubmit}>
          <label className="form-row">
            <span>FREE GPU</span>
            <select value={effectiveGpuId} onChange={(e) => setGpuId(e.target.value)}>
              {freeGpus.map((g) => (
                <option key={g.gpu_id} value={g.gpu_id}>{g.gpu_id}</option>
              ))}
            </select>
          </label>
          <label className="form-row">
            <span>ASSIGN TO</span>
            <select value={userId} onChange={(e) => setUserId(e.target.value)}>
              {KNOWN_USERS.map((u) => (
                <option key={u.user_id} value={u.user_id}>{u.display_name}</option>
              ))}
            </select>
          </label>
          <button className="btn btn-primary" type="submit" disabled={submitting}>
            {submitting ? "ASSIGNING..." : "ASSIGN"}
          </button>
          {error && <p className="form-error">{error}</p>}
        </form>
      )}
    </section>
  );
}
