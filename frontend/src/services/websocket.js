// Centralized WebSocket client - the ONLY place `WebSocket` is
// constructed from. Reconnects with capped backoff on drop; never
// invents a state update of its own - every message it delivers to
// `onMessage` is exactly what the backend sent.

import { API_BASE_URL } from "./config.js";

const WS_BASE = API_BASE_URL.replace(/^http/, "ws") + "/ws/state";
const RECONNECT_BASE_DELAY_MS = 500;
const RECONNECT_MAX_DELAY_MS = 8000;

// `token` is optional: omitted (or an ADMIN token), the backend
// streams the full admin snapshot exactly as it always has; a USER
// token streams the privacy-scoped portal snapshot instead. The
// identity is resolved from this token server-side - there is no
// other way for a connection to say who it is.
export function createStateSocket({ onMessage, onStatusChange, token }) {
  let socket = null;
  let attempt = 0;
  let closedByCaller = false;
  let reconnectTimer = null;

  const setStatus = (status) => onStatusChange?.(status);

  function connect() {
    setStatus("connecting");
    const url = token ? `${WS_BASE}?token=${encodeURIComponent(token)}` : WS_BASE;
    socket = new WebSocket(url);

    socket.onopen = () => {
      attempt = 0;
      setStatus("connected");
    };

    socket.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        onMessage?.(parsed);
      } catch (err) {
        // A malformed message is dropped, never turned into fabricated state.
        console.error("Malformed WebSocket message", err);
      }
    };

    socket.onclose = () => {
      if (closedByCaller) {
        setStatus("closed");
        return;
      }
      setStatus("disconnected");
      scheduleReconnect();
    };

    socket.onerror = () => {
      setStatus("disconnected");
    };
  }

  function scheduleReconnect() {
    const delay = Math.min(RECONNECT_BASE_DELAY_MS * 2 ** attempt, RECONNECT_MAX_DELAY_MS);
    attempt += 1;
    reconnectTimer = setTimeout(connect, delay);
  }

  connect();

  return {
    close() {
      closedByCaller = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    },
  };
}
