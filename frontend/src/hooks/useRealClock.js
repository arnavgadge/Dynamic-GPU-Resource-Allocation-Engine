import { useEffect, useState } from "react";
import { getClock } from "../services/api.js";

// Issue 5: the header clock/uptime must track the actual system
// clock, not the deterministic simulated clock and not a frontend-
// invented timer. This hook polls the backend's own dedicated
// `/api/clock` endpoint once a second and renders exactly what it
// returns - every tick shown on screen is a fresh read of the real
// backend clock, never a value this component increments itself.
export function useRealClock() {
  const [clock, setClock] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const result = await getClock();
        if (!cancelled) setClock(result);
      } catch {
        // A transient failure just leaves the last known reading on
        // screen - the next tick tries again.
      }
    }

    poll();
    const interval = setInterval(poll, 1000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return clock; // { real_time, uptime_seconds } | null before the first successful poll
}
