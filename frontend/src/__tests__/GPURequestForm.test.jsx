import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import GPURequestForm from "../components/GPURequestForm.jsx";

// The most important thing to prove: this form performs NO
// allocation-score/priority/availability logic - it only forwards the
// four plain fields to the backend, exactly as typed.
describe("GPURequestForm", () => {
  it("submits exactly the fields entered, computing nothing itself", async () => {
    const onSubmit = vi.fn().mockResolvedValue({});
    render(<GPURequestForm onSubmit={onSubmit} />);

    fireEvent.change(screen.getByDisplayValue("ML Training"), { target: { value: "Video Editing" } });
    fireEvent.change(screen.getByDisplayValue(20), { target: { value: "15" } });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "MEDIUM" } });

    fireEvent.click(screen.getByText("REQUEST GPU"));

    expect(onSubmit).toHaveBeenCalledWith({
      workload: "Video Editing",
      gpuCount: 1,
      estimatedMinutes: 15,
      priority: "MEDIUM",
    });
  });

  it("shows the backend's rejection message rather than silently failing", async () => {
    const onSubmit = vi.fn().mockRejectedValue(new Error("422 only single-GPU requests are supported"));
    render(<GPURequestForm onSubmit={onSubmit} />);

    fireEvent.click(screen.getByText("REQUEST GPU"));

    expect(await screen.findByText(/only single-GPU requests/)).toBeInTheDocument();
  });
});
