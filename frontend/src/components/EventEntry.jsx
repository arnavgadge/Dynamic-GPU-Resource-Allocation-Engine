import { eventTypeClassName } from "../utils/eventStyles.js";
import { formatSimTime } from "../utils/formatters.js";

// The message and reason are rendered verbatim from the backend
// Event - this component never composes or rewrites event text.
export default function EventEntry({ event }) {
  return (
    <li className={`event-entry ${eventTypeClassName(event.event_type)}`}>
      <span className="event-time">{formatSimTime(event.timestamp)}</span>
      <span className="event-type">{event.event_type}</span>
      <span className="event-message">{event.message}</span>
    </li>
  );
}
