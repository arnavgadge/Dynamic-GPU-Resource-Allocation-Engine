import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import GPUPool from "../components/GPUPool.jsx";

// Backend state -> React state -> GPU Pool: the count and every
// value rendered must trace back to the fixture below, standing in
// for a real backend payload - none of it is hardcoded in the component.
const backendGpus = [
  { gpu_id: "GPU-1", utilization_percent: 92, status: "ACTIVE", assigned_user_name: "Alice", assigned_job_name: "ML Training", memory_used_mb: 22000, total_memory_mb: 24576, has_pending_prompt: false },
  { gpu_id: "GPU-2", utilization_percent: 7, status: "IDLE_WARNING", assigned_user_name: "Bob", assigned_job_name: "Excel", memory_used_mb: 2000, total_memory_mb: 24576, has_pending_prompt: true },
  { gpu_id: "GPU-3", utilization_percent: 0, status: "IDLE", assigned_user_name: null, assigned_job_name: null, memory_used_mb: 0, total_memory_mb: 24576, has_pending_prompt: false },
];

describe("GPUPool", () => {
  it("renders exactly the GPUs the backend sent, dynamically - not a fixed count", () => {
    render(<GPUPool gpus={backendGpus} />);

    expect(screen.getByText("GPU POOL (3)")).toBeInTheDocument();
    expect(screen.getByText("GPU-1")).toBeInTheDocument();
    expect(screen.getByText("GPU-2")).toBeInTheDocument();
    expect(screen.getByText("GPU-3")).toBeInTheDocument();
  });

  it("renders a different GPU count when the backend reports a different pool size", () => {
    render(<GPUPool gpus={backendGpus.slice(0, 1)} />);
    expect(screen.getByText("GPU POOL (1)")).toBeInTheDocument();
    expect(screen.queryByText("GPU-2")).not.toBeInTheDocument();
  });

  it("renders the exact utilization percentage the backend reported", () => {
    render(<GPUPool gpus={backendGpus} />);
    expect(screen.getByText("92%")).toBeInTheDocument();
    expect(screen.getByText("7%")).toBeInTheDocument();
  });

  it("renders assigned user/job from backend state, or a dash when unassigned", () => {
    render(<GPUPool gpus={backendGpus} />);
    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("ML Training")).toBeInTheDocument();
    // GPU-3 is unassigned.
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThan(0);
  });

  it("maps each backend status to its fixed semantic color class", () => {
    render(<GPUPool gpus={backendGpus} />);
    expect(document.querySelector(".status-active")).not.toBeNull();
    expect(document.querySelector(".status-warning")).not.toBeNull();
    expect(document.querySelector(".status-idle")).not.toBeNull();
  });

  it("shows an empty-pool message rather than fabricating a GPU when there are none", () => {
    render(<GPUPool gpus={[]} />);
    expect(screen.getByText("GPU POOL (0)")).toBeInTheDocument();
    expect(screen.getByText(/No GPUs/)).toBeInTheDocument();
  });
});
