// Centralized auth client. Holds the current session token in
// memory + sessionStorage (survives a page refresh, cleared when the
// tab closes) - never anything claiming to be a "logged in user" is
// invented client-side; every account fact comes from the backend's
// own login response.

import { API_BASE_URL } from "./config.js";

const STORAGE_KEY = "gpu-scheduler-session";

let currentSession = loadStoredSession();

function loadStoredSession() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function storeSession(session) {
  currentSession = session;
  try {
    if (session) sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // sessionStorage unavailable (e.g. private mode) - in-memory session still works for this tab.
  }
}

export function getSession() {
  return currentSession;
}

export async function login(username) {
  const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `login failed (${response.status})`);
  }
  const session = await response.json();
  storeSession(session);
  return session;
}

export async function logout() {
  const session = currentSession;
  if (session) {
    try {
      await fetch(`${API_BASE_URL}/api/auth/logout`, {
        method: "POST",
        headers: { Authorization: `Bearer ${session.token}` },
      });
    } catch {
      // Backend unreachable - clear the local session anyway.
    }
  }
  storeSession(null);
}
