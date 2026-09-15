import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ReclamationPrompt from "../components/ReclamationPrompt.jsx";

// React YES/NO -> backend command: clicking a button must send exactly
// the GPU id and response the backend's pending prompt is about -
// never a locally-decided GPU status change.
describe("ReclamationPrompt", () => {
  const prompts = [{ gpu_id: "GPU-03", message: "Are you still using this GPU?", reason: null }];

  it("renders nothing when there is no pending backend prompt", () => {
    const onRespond = vi.fn();
    const { container } = render(<ReclamationPrompt prompts={[]} onRespond={onRespond} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the backend-supplied GPU id and message when a prompt is pending", () => {
    render(<ReclamationPrompt prompts={prompts} onRespond={vi.fn()} />);
    expect(screen.getByText("GPU-03 IDLE WARNING")).toBeInTheDocument();
    expect(screen.getByText("Are you still using this GPU?")).toBeInTheDocument();
  });

  it("clicking YES sends exactly (gpu_id, 'YES') to the backend command - never a local status change", () => {
    const onRespond = vi.fn();
    render(<ReclamationPrompt prompts={prompts} onRespond={onRespond} />);

    fireEvent.click(screen.getByText("YES"));

    expect(onRespond).toHaveBeenCalledTimes(1);
    expect(onRespond).toHaveBeenCalledWith("GPU-03", "YES");
  });

  it("clicking NO sends exactly (gpu_id, 'NO') to the backend command", () => {
    const onRespond = vi.fn();
    render(<ReclamationPrompt prompts={prompts} onRespond={onRespond} />);

    fireEvent.click(screen.getByText("NO"));

    expect(onRespond).toHaveBeenCalledWith("GPU-03", "NO");
  });
});
