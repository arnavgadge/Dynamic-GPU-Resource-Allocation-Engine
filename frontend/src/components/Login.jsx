// The five demo accounts shown here are the exact same fixed set the
// backend seeds in `api/auth.py` - this is legitimate static UI text
// (a login screen listing known accounts), not dynamic scheduler
// state, and it is never used to decide anything: the actual role
// and identity come back from the backend's login response, not from
// this list.
const DEMO_ACCOUNTS = [
  { username: "admin", label: "Admin", hint: "Engine console" },
  { username: "user_a", label: "User A", hint: "Employee portal" },
  { username: "user_b", label: "User B", hint: "Employee portal" },
  { username: "user_c", label: "User C", hint: "Employee portal" },
  { username: "user_d", label: "User D", hint: "Employee portal" },
];

export default function Login({ onLogin, loginError }) {
  return (
    <div className="login-screen">
      <div className="login-box">
        <h1 className="login-title">GPU SCHEDULER</h1>
        <p className="login-subtitle">Team Vicimus | Demonstration Login</p>
        <p className="login-note">Select a demo account - no password, course-project demonstration only.</p>
        <div className="login-accounts">
          {DEMO_ACCOUNTS.map((account) => (
            <button key={account.username} className="login-account-btn" onClick={() => onLogin(account.username)}>
              <span className="login-account-name">{account.label}</span>
              <span className="login-account-hint">{account.hint}</span>
            </button>
          ))}
        </div>
        {loginError && <p className="login-error">{loginError}</p>}
      </div>
    </div>
  );
}
