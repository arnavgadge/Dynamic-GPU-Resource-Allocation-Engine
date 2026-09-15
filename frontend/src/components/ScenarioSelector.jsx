// The list of scenarios (id/name/description) comes entirely from
// `GET /api/scenarios` (the backend's ScenarioRegistry) - nothing
// here hardcodes a scenario name or description.
export default function ScenarioSelector({ scenarios, activeScenarioId, onSelect, disabled }) {
  return (
    <section className="panel">
      <h2 className="panel-title">SCENARIO</h2>
      <select
        className="scenario-select"
        value={activeScenarioId ?? ""}
        disabled={disabled}
        onChange={(event) => onSelect(event.target.value)}
      >
        {scenarios.map((scenario) => (
          <option key={scenario.scenario_id} value={scenario.scenario_id}>
            {scenario.name}
          </option>
        ))}
      </select>
      <p className="scenario-description">
        {scenarios.find((s) => s.scenario_id === activeScenarioId)?.description ?? ""}
      </p>
    </section>
  );
}
