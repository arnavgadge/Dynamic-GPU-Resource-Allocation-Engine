import { formatSimTime } from "../utils/formatters.js";

// Every notification here is a real backend `Event` about this user
// (`api/serializers.py::serialize_notifications`) - this component
// never synthesizes one locally.
export default function NotificationsPanel({ notifications }) {
  return (
    <section className="panel">
      <h2 className="panel-title">NOTIFICATIONS ({notifications.length})</h2>
      {notifications.length === 0 ? (
        <p className="empty-note">No notifications yet.</p>
      ) : (
        <ul className="event-list">
          {notifications.map((note) => (
            <li className="event-entry notification-entry" key={note.notification_id}>
              <span className="event-time">{formatSimTime(note.timestamp)}</span>
              <span className="event-type">{note.label}</span>
              <span className="event-message">{note.message}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
