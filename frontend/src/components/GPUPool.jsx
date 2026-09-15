import GPURow from "./GPURow.jsx";

// The GPU pool is however many GPUs the backend reports - it is
// never a fixed count of rows.
export default function GPUPool({ gpus }) {
  return (
    <section className="panel gpu-pool-panel">
      <h2 className="panel-title">GPU POOL ({gpus.length})</h2>
      {gpus.length === 0 ? (
        <p className="empty-note">No GPUs in the current scenario.</p>
      ) : (
        <div className="gpu-pool-grid">
          {gpus.map((gpu) => (
            <GPURow key={gpu.gpu_id} gpu={gpu} />
          ))}
        </div>
      )}
    </section>
  );
}
