import { priorityClassName } from "../utils/eventStyles.js";

// Every row and every cell is read straight from `state.users` -
// the workload name is display-only metadata, exactly as the
// backend treats it.
export default function UsersPanel({ users }) {
  return (
    <section className="panel">
      <h2 className="panel-title">USERS ({users.length})</h2>
      <table className="data-table">
        <thead>
          <tr>
            <th>USER</th>
            <th>WORKLOAD</th>
            <th>PRIORITY</th>
            <th>GPU</th>
            <th>STATUS</th>
          </tr>
        </thead>
        <tbody>
          {users.map((user) => (
            <tr key={user.user_id}>
              <td>{user.name}</td>
              <td>{user.workload ?? "—"}</td>
              <td>
                <span className={`priority-tag ${priorityClassName(user.priority)}`}>{user.priority}</span>
              </td>
              <td>{user.gpu_id ?? "—"}</td>
              <td>{user.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
