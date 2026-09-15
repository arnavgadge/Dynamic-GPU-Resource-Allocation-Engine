import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import WaitingQueue from "../components/WaitingQueue.jsx";

// The score shown must be exactly the backend-computed value -
// this component performs no arithmetic on priority or size at all.
const backendWaitingQueue = [
  { job_id: "JOB-07", priority: "HIGH", estimated_size_minutes: 20, wait_seconds: 480, allocation_score: 0.821 },
];

describe("WaitingQueue", () => {
  it("renders the backend's exact allocation score, not a recomputed one", () => {
    render(<WaitingQueue jobs={backendWaitingQueue} />);
    expect(screen.getByText("SCORE: 0.821")).toBeInTheDocument();
  });

  it("renders backend-provided size and wait time", () => {
    render(<WaitingQueue jobs={backendWaitingQueue} />);
    expect(screen.getByText(/SIZE: 20m/)).toBeInTheDocument();
    expect(screen.getByText(/WAIT: 8m/)).toBeInTheDocument();
  });

  it("shows an empty message rather than a fabricated entry when nothing is waiting", () => {
    render(<WaitingQueue jobs={[]} />);
    expect(screen.getByText("No jobs waiting.")).toBeInTheDocument();
  });
});
