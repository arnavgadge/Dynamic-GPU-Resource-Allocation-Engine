import { gpuStatusClassName } from "../utils/eventStyles.js";

// Renders exactly the status string the backend sent, colored by a
// fixed lookup table - it never infers or guesses a status itself.
export default function StatusBadge({ status }) {
  return <span className={`badge ${gpuStatusClassName(status)}`}>{status}</span>;
}
