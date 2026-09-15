import { formatPercent } from "../utils/formatters.js";
import { gpuStatusClassName } from "../utils/eventStyles.js";

// Renders exactly `percent` as sent by the backend. The CSS
// transition on `width` is what gives a visual "animation" between
// two backend-reported readings - it interpolates, it never invents
// an intermediate value that didn't come from the backend.
export default function UtilizationBar({ percent, status }) {
  const clamped = Math.max(0, Math.min(100, percent ?? 0));
  return (
    <div className="util-bar-track" role="progressbar" aria-valuenow={clamped}>
      <div className={`util-bar-fill ${gpuStatusClassName(status)}`} style={{ width: `${clamped}%` }} />
      <span className="util-bar-label">{formatPercent(percent)}</span>
    </div>
  );
}
