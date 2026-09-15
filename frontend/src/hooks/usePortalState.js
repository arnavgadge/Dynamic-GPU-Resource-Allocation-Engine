// The User Portal's counterpart to `useSchedulerState` - same shape,
// same rule (every field comes from a backend response or a
// WebSocket `portal_state` message; nothing computed here). The
// backend enforces which fields exist at all (Part 24) - this hook
// just renders whatever it's handed.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getPortalState, respondToPrompt as apiRespondToPrompt, submitGpuRequest as apiSubmitGpuRequest } from "../services/api.js";
import { createStateSocket } from "../services/websocket.js";

export function usePortalState(token) {
  const [state, setState] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [bootError, setBootError] = useState(null);
  const [commandError, setCommandError] = useState(null);
  const socketRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const initialState = await getPortalState();
        if (!cancelled) setState(initialState);
      } catch (err) {
        if (!cancelled) setBootError(err.message);
      }
    })();

    const socket = createStateSocket({
      onMessage: (message) => {
        if (message.type === "portal_state") setState(message.payload);
      },
      onStatusChange: setConnectionStatus,
      token,
    });
    socketRef.current = socket;

    return () => {
      cancelled = true;
      socket.close();
    };
  }, [token]);

  const runCommand = useCallback((fn) => {
    setCommandError(null);
    return fn()
      .then((result) => {
        setState(result);
        return result;
      })
      .catch((err) => {
        setCommandError(err.message);
        throw err;
      });
  }, []);

  const commands = useMemo(
    () => ({
      submitRequest: (request) => runCommand(() => apiSubmitGpuRequest(request)),
      respondToPrompt: (gpuId, response) => runCommand(() => apiRespondToPrompt(gpuId, response)),
    }),
    [runCommand]
  );

  return { state, connectionStatus, bootError, commandError, commands };
}
