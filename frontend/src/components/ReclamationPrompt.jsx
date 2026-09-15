// The prompt text and which GPU it's about come entirely from the
// backend's `pending_prompts` list (itself built from the real
// `PROMPT` event `ReclamationEngine` logged, whether raised by a
// utilization tier or an estimated-completion check). YES/NO only
// ever send a command; the backend's Reclamation Engine decides what
// happens. `yesLabel`/`noLabel` only change the button *text* (Admin
// vs User Portal phrasing) - both always send exactly "YES"/"NO".
export default function ReclamationPrompt({ prompts, onRespond, yesLabel = "YES", noLabel = "NO" }) {
  if (!prompts || prompts.length === 0) return null;
  const prompt = prompts[0];

  return (
    <div className="prompt-overlay">
      <div className="prompt-box">
        <div className="prompt-header">{prompt.gpu_id} IDLE WARNING</div>
        <p className="prompt-message">{prompt.message}</p>
        <div className="prompt-buttons">
          <button className="btn btn-yes" onClick={() => onRespond(prompt.gpu_id, "YES")}>
            {yesLabel}
          </button>
          <button className="btn btn-no" onClick={() => onRespond(prompt.gpu_id, "NO")}>
            {noLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
