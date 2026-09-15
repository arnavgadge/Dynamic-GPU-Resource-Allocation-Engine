// Every number here is a plain count over arrays the backend sent -
// no scheduling decision, just tallying statuses the backend already
// assigned.
export default function SystemOverview({ gpus, users, waitingQueue }) {
  const stats = [
    ["TOTAL GPUs", gpus.length, ""],
    ["ACTIVE", gpus.filter((g) => g.status === "ACTIVE").length, "stat-active"],
    ["IDLE", gpus.filter((g) => g.status === "IDLE").length, "stat-idle"],
    ["WAITING", waitingQueue.length, "stat-waiting"],
    ["USERS", users.length, ""],
  ];

  return (
    <section className="panel">
      <h2 className="panel-title">SYSTEM OVERVIEW</h2>
      <div className="stat-grid">
        {stats.map(([label, value, modifier]) => (
          <div className={`stat-cell ${modifier}`} key={label}>
            <span className="stat-label">{label}</span>
            <span className="stat-value">{String(value).padStart(2, "0")}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
