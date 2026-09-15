import EventEntry from "./EventEntry.jsx";

// Newest first, keyed by the backend's own `event_id` - the same id
// every engine (Allocation "E...", Reclamation "R...", Balancing
// "B...", Scheduler "S...") already assigns, so React never needs to
// invent a key or guess whether a message is a duplicate.
export default function EventLog({ events }) {
  const newestFirst = [...events].reverse();
  return (
    <section className="panel event-log-panel">
      <h2 className="panel-title">EVENT LOG ({events.length})</h2>
      <ul className="event-list">
        {newestFirst.map((event) => (
          <EventEntry key={event.event_id} event={event} />
        ))}
      </ul>
    </section>
  );
}
