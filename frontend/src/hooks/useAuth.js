// The one place login state lives - the "backend must know the
// logged-in user" rule (Part 29) means this hook only ever stores
// what the backend's `/api/auth/login` response actually said
// (username, role, display_name, token); nothing here decides a
// role or identity on its own.

import { useCallback, useState } from "react";
import { getSession, login as apiLogin, logout as apiLogout } from "../services/auth.js";

export function useAuth() {
  const [session, setSession] = useState(getSession());
  const [loginError, setLoginError] = useState(null);

  const login = useCallback(async (username) => {
    setLoginError(null);
    try {
      const result = await apiLogin(username);
      setSession(result);
      return result;
    } catch (err) {
      setLoginError(err.message);
      throw err;
    }
  }, []);

  const logout = useCallback(async () => {
    await apiLogout();
    setSession(null);
  }, []);

  return { session, loginError, login, logout };
}
