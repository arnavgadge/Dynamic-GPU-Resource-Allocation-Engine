import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ManualAssignPanel from "../components/ManualAssignPanel.jsx";

const gpus = [
  { gpu_id: "GPU-1", status: "IDLE", assigned_user_id: null },
  { gpu_id: "GPU-2", status: "ACTIVE", assigned_user_id: "user_a" },
  { gpu_id: "GPU-3", status: "IDLE", assigned_user_id: null },
];

describe("ManualAssignPanel", () => {
  it("only offers genuinely free GPUs, never an already-assigned one", () => {
    render(<ManualAssignPanel gpus={gpus} onAssign={vi.fn()} />);
    expect(screen.getByRole("option", { name: "GPU-1" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "GPU-3" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "GPU-2" })).not.toBeInTheDocument();
  });

  it("submits exactly the chosen GPU and user - decides nothing itself", async () => {
    const onAssign = vi.fn().mockResolvedValue({});
    render(<ManualAssignPanel gpus={gpus} onAssign={onAssign} />);

    fireEvent.click(screen.getByText("ASSIGN"));

    expect(onAssign).toHaveBeenCalledWith("GPU-1", "user_a", "User A");
  });

  it("shows an empty note instead of a form when no GPU is free", () => {
    const allTaken = gpus.map((g) => ({ ...g, status: "ACTIVE", assigned_user_id: "someone" }));
    render(<ManualAssignPanel gpus={allTaken} onAssign={vi.fn()} />);
    expect(screen.getByText("No free GPUs left to manually assign.")).toBeInTheDocument();
  });
});
