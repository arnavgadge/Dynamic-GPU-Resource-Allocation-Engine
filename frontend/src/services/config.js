// The backend's base URL - the one place this is read from, so
// `api.js` and `auth.js` can both use it without importing each other.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
