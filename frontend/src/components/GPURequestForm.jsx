import { useState } from "react";

// Collects the four plain input fields the Allocation Engine actually
// takes (workload label, GPU count, estimated size, priority) and
// sends them as-is. This form computes nothing - no score, no
// availability check, no GPU choice; it only calls
// `commands.submitRequest`, and the backend decides everything from there.
const PRIORITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];

export default function GPURequestForm({ onSubmit, disabled }) {
  const [workload, setWorkload] = useState("ML Training");
  const [gpuCount, setGpuCount] = useState(1);
  const [estimatedMinutes, setEstimatedMinutes] = useState(20);
  const [priority, setPriority] = useState("HIGH");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await onSubmit({ workload, gpuCount: Number(gpuCount), estimatedMinutes: Number(estimatedMinutes), priority });
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="panel">
      <h2 className="panel-title">REQUEST GPU RESOURCES</h2>
      <form className="request-form" onSubmit={handleSubmit}>
        <label className="form-row">
          <span>WORKLOAD</span>
          <input value={workload} onChange={(e) => setWorkload(e.target.value)} maxLength={64} required />
        </label>
        <label className="form-row">
          <span>GPUs REQUESTED (1-9)</span>
          <input
            type="number" min={1} max={9} value={gpuCount}
            onChange={(e) => setGpuCount(e.target.value)}
          />
        </label>
        <label className="form-row">
          <span>ESTIMATED SIZE (MIN)</span>
          <input
            type="number" min={1} max={600} value={estimatedMinutes}
            onChange={(e) => setEstimatedMinutes(e.target.value)}
          />
        </label>
        <label className="form-row">
          <span>PRIORITY</span>
          <select value={priority} onChange={(e) => setPriority(e.target.value)}>
            {PRIORITIES.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
        <button className="btn btn-primary" type="submit" disabled={disabled || submitting}>
          {submitting ? "SUBMITTING..." : "REQUEST GPU"}
        </button>
        {Number(gpuCount) > 1 && (
          <p className="form-hint">
            If fewer than {gpuCount} GPUs are free, the backend may ask another user to release one of theirs.
          </p>
        )}
        {error && <p className="form-error">{error}</p>}
      </form>
    </section>
  );
}
