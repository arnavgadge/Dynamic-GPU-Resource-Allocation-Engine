import { useAuth } from "./hooks/useAuth.js";
import Login from "./components/Login.jsx";
import UserPortal from "./UserPortal.jsx";
import AdminConsole from "./AdminConsole.jsx";
import "./index.css";

// The one router this project needs: LOGIN -> USER PORTAL or ADMIN
// CONSOLE, entirely driven by the backend's login response
// (`session.role`) - never a role/identity guessed or hardcoded here.
export default function App() {
  const { session, loginError, login, logout } = useAuth();

  if (!session) {
    return <Login onLogin={login} loginError={loginError} />;
  }

  if (session.role === "ADMIN") {
    return <AdminConsole onLogout={logout} />;
  }

  return <UserPortal session={session} onLogout={logout} />;
}
