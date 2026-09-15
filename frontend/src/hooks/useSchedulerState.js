// The one coherent source of frontend scheduler state (per the
// architectural rule: "App / scheduler-state hook -> components
// receive props/state -> render only"). Every field this hook
// exposes under `state` came directly from a backend response or a
// WebSocket `state` message - nothing here computes a GPU's status,
// a job's priority, or an allocation score. Commands are thin
// wrappers over `services/api.js` that apply whatever state the
// backend sends back; they never optimistically guess the result.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getScenarios,
  getState,
  loadScenario as apiLoadScenario,
  startSimulation,
  pauseSimulation,
  resetSimulation,
  stepSimulation,
  setSpeed as apiSetSpeed,
  respondToPrompt as apiRespondToPrompt,
  manualAssignGpu as apiManualAssignGpu,
} from "../services/api.js";
import { createStateSocket } from "../services/websocket.js";
import { getSession } from "../services/auth.js";

export function useSchedulerState() {
  const [state, setState] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [bootError, setBootError] = useState(null);
  const [commandError, setCommandError] = useState(null);
  const socketRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const [scenarioList, initialState] = await Promise.all([getScenarios(), getState()]);
        if (cancelled) return;
        setScenarios(scenarioList);
        setState(initialState);
      } catch (err) {
        if (!cancelled) setBootError(err.message);
      }
    })();

    const socket = createStateSocket({
      onMessage: (message) => {
        if (message.type === "state") {
          setState(message.payload);
        }
      },
      onStatusChange: setConnectionStatus,
      token: getSession()?.token, // an ADMIN token streams the same full snapshot as no token at all
    });
    socketRef.current = socket;

    return () => {
      cancelled = true;
      socket.close();
    };
  }, []);

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
      loadScenario: (scenarioId) => runCommand(() => apiLoadScenario(scenarioId)),
      start: () => runCommand(startSimulation),
      pause: () => runCommand(pauseSimulation),
      reset: () => runCommand(resetSimulation),
      step: (minutes) => runCommand(() => stepSimulation(minutes)),
      setSpeed: (speed) => runCommand(() => apiSetSpeed(speed)),
      respondToPrompt: (gpuId, response) => runCommand(() => apiRespondToPrompt(gpuId, response)),
      manualAssign: (gpuId, userId, displayName) =>
        runCommand(() => apiManualAssignGpu({ gpuId, userId, displayName })),
    }),
    [runCommand]
  );

  return { state, scenarios, connectionStatus, bootError, commandError, commands };
}
