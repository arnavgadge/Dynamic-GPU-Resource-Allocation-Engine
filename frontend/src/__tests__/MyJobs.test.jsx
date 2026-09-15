import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import MyJobs from "../components/MyJobs.jsx";

const backendJobs = [
  {
    job_id: "REQ-1", status: "RUNNING", gpu_count: 1,
    assigned_gpu_id: "GPU-2", assigned_gpu_ids: ["GPU-2"], gpu_status: "ACTIVE",
    gpu_utilization_percent: 82, gpus: [{ gpu_id: "GPU-2", utilization_percent: 82, status: "ACTIVE" }],
    started_at: "2026-01-01T09:00:00+00:00",
    estimated_completion: "2026-01-01T09:20:00+00:00", remaining_seconds: 600,
  },
  {
    job_id: "REQ-2", status: "WAITING", priority: "HIGH", wait_seconds: 180,
    queue_position: 2, allocation_score: 0.82,
  },
];

const partialMultiGpuJob = {
  job_id: "REQ-3", status: "WAITING", priority: "HIGH", wait_seconds: 60,
  queue_position: 1, allocation_score: 0.9,
  gpu_count: 3, assigned_gpu_ids: ["GPU-1", "GPU-2"], gpus_still_needed: 1,
};

describe("MyJobs", () => {
  it("renders the backend's exact GPU/utilization/completion values for an active job", () => {
    render(<MyJobs jobs={backendJobs} />);
    expect(screen.getByText("GPU: GPU-2")).toBeInTheDocument();
    expect(screen.getByText("UTILIZATION: 82%")).toBeInTheDocument();
  });

  it("renders the backend's exact queue position and score for a waiting job, never recomputed", () => {
    render(<MyJobs jobs={backendJobs} />);
    expect(screen.getByText("QUEUE POSITION: #2")).toBeInTheDocument();
    expect(screen.getByText("SCORE: 0.820")).toBeInTheDocument();
  });

  it("shows the real assigned/pending split for a partially-satisfied multi-GPU request", () => {
    render(<MyJobs jobs={[partialMultiGpuJob]} />);
    expect(screen.getByText("REQUESTED: 3")).toBeInTheDocument();
    expect(screen.getByText("ASSIGNED: 2")).toBeInTheDocument();
    expect(screen.getByText("PENDING: 1")).toBeInTheDocument();
    expect(screen.getByText("HELD SO FAR: GPU-1, GPU-2")).toBeInTheDocument();
  });

  it("shows empty-state messages rather than fabricating jobs when there are none", () => {
    render(<MyJobs jobs={[]} />);
    expect(screen.getByText("No active jobs.")).toBeInTheDocument();
    expect(screen.getByText("No other requests.")).toBeInTheDocument();
  });
});
