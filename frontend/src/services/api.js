// Centralized REST client - the ONLY place `fetch` is called from.
//
// Every function here sends a request and returns whatever the
// backend decided; none of them contain scheduling logic, and none
// of them fabricate a response on failure - callers are expected to
// handle a rejected promise (see `useSchedulerState`'s error state).
// If a session is logged in, its token is attached automatically -
// the backend resolves identity from that token alone (never from
// anything in a request body).

import { API_BASE_URL } from "./config.js";
import { getSession } from "./auth.js";

async function request(path, options = {}) {
  const session = getSession();
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (session?.token) headers.Authorization = `Bearer ${session.token}`;

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      // response had no JSON body - keep statusText
    }
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json();
}

// -- admin / scenario-driven -------------------------------------------

export function getScenarios() {
  return request("/api/scenarios");
}

export function getState() {
  return request("/api/state");
}

// Issue 5: a cheap, dedicated real-clock read - the actual system
// clock and this session's real elapsed uptime, polled independently
// of the full state snapshot/WebSocket cadence.
export function getClock() {
  return request("/api/clock");
}

export function loadScenario(scenarioId) {
  return request(`/api/scenarios/${encodeURIComponent(scenarioId)}/load`, { method: "POST" });
}

export function startSimulation() {
  return request("/api/control/start", { method: "POST" });
}

export function pauseSimulation() {
  return request("/api/control/pause", { method: "POST" });
}

export function resetSimulation() {
  return request("/api/control/reset", { method: "POST" });
}

export function stepSimulation(minutes) {
  return request("/api/control/step", {
    method: "POST",
    body: JSON.stringify(minutes ? { minutes } : {}),
  });
}

export function setSpeed(speed) {
  return request("/api/control/speed", {
    method: "POST",
    body: JSON.stringify({ speed }),
  });
}

export function respondToPrompt(gpuId, response) {
  return request("/api/prompt/respond", {
    method: "POST",
    body: JSON.stringify({ gpu_id: gpuId, response }),
  });
}

// Admin-only demo/test control (Phase 10): place a user directly onto
// a currently-free GPU, to establish a starting arrangement before a
// demonstration begins. Sends exactly the three plain fields the
// backend needs - the backend, not this form, decides whether the
// GPU is actually free.
export function manualAssignGpu({ gpuId, userId, displayName }) {
  return request("/api/admin/assign", {
    method: "POST",
    body: JSON.stringify({ gpu_id: gpuId, user_id: userId, display_name: displayName }),
  });
}

// -- User Portal ---------------------------------------------------------

export function getPortalState() {
  return request("/api/portal/state");
}

export function submitGpuRequest({ workload, gpuCount, estimatedMinutes, priority }) {
  return request("/api/requests", {
    method: "POST",
    body: JSON.stringify({
      workload,
      gpu_count: gpuCount,
      estimated_minutes: estimatedMinutes,
      priority,
    }),
  });
}

export { API_BASE_URL };
