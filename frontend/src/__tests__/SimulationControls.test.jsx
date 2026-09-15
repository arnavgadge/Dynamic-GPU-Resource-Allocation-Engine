import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import SimulationControls from "../components/SimulationControls.jsx";

const baseSimulation = { running: false, speed: 1, allowed_speeds: [1, 5, 10, 50, 100] };

describe("SimulationControls", () => {
  it("sends start/pause/reset/step commands, never mutating state itself", () => {
    const onStart = vi.fn();
    const onPause = vi.fn();
    const onReset = vi.fn();
    const onStep = vi.fn();

    render(
      <SimulationControls
        simulation={baseSimulation}
        onStart={onStart}
        onPause={onPause}
        onReset={onReset}
        onStep={onStep}
        onSetSpeed={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText("START"));
    expect(onStart).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByText("RESET"));
    expect(onReset).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByText("STEP"));
    expect(onStep).toHaveBeenCalledTimes(1);
  });

  it("disables START while running and PAUSE while not - reflecting backend state, not local toggling", () => {
    render(
      <SimulationControls
        simulation={{ ...baseSimulation, running: true }}
        onStart={vi.fn()}
        onPause={vi.fn()}
        onReset={vi.fn()}
        onStep={vi.fn()}
        onSetSpeed={vi.fn()}
      />
    );
    expect(screen.getByText("START")).toBeDisabled();
    expect(screen.getByText("PAUSE")).not.toBeDisabled();
  });

  it("the speed range comes from backend-provided allowed_speeds, and picking a value sends it verbatim", () => {
    const onSetSpeed = vi.fn();
    render(
      <SimulationControls
        simulation={baseSimulation}
        onStart={vi.fn()}
        onPause={vi.fn()}
        onReset={vi.fn()}
        onStep={vi.fn()}
        onSetSpeed={onSetSpeed}
      />
    );

    const slider = screen.getByRole("slider");
    expect(slider).toHaveAttribute("max", "4"); // 5 allowed speeds -> indices 0..4

    fireEvent.change(slider, { target: { value: "3" } }); // index 3 -> speed 50
    expect(onSetSpeed).toHaveBeenCalledWith(50);
  });
});
