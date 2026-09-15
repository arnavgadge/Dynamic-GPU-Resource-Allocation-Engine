// Issue 6: a reclamation/resource-request confirmation for a GPU
// owned by a real logged-in demo user is observable here as a plain
// status - never actionable from the Admin Console. The YES/NO
// buttons only ever exist on that user's own User Portal
// (`ReclamationPrompt`); this component never sends a response.
export default function PromptStatusList({ prompts }) {
  if (!prompts || prompts.length === 0) return null;

  return (
    <div className="banner banner-warning prompt-status-list">
      {prompts.map((prompt) => (
        <div key={prompt.gpu_id} className="prompt-status-row">
          PROMPT · {prompt.gpu_id} awaiting {prompt.owner_user_name ?? prompt.owner_user_id} response
          {prompt.requested_by_user_name ? ` (requested by ${prompt.requested_by_user_name})` : ""}
        </div>
      ))}
    </div>
  );
}
