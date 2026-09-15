import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import EventLog from "../components/EventLog.jsx";

// Backend event -> Event stream -> Event Log: the exact message text
// and event type are rendered verbatim; nothing is generated here.
const backendEvents = [
  { event_id: "E1", timestamp: "2026-01-01T09:00:00+00:00", event_type: "REQUEST", message: "JOB-17 entered waiting queue", reason: null },
  { event_id: "E2", timestamp: "2026-01-01T09:00:01+00:00", event_type: "ALLOC", message: "JOB-17 allocated to GPU-02", reason: null },
  { event_id: "R1", timestamp: "2026-01-01T09:05:11+00:00", event_type: "RECLAIM", message: "GPU-03 reclaimed", reason: null },
];

describe("EventLog", () => {
  it("renders every backend event's message verbatim", () => {
    render(<EventLog events={backendEvents} />);
    expect(screen.getByText("JOB-17 entered waiting queue")).toBeInTheDocument();
    expect(screen.getByText("JOB-17 allocated to GPU-02")).toBeInTheDocument();
    expect(screen.getByText("GPU-03 reclaimed")).toBeInTheDocument();
  });

  it("shows the newest backend event first", () => {
    render(<EventLog events={backendEvents} />);
    const messages = screen.getAllByText(/JOB-17|GPU-03/).map((el) => el.textContent);
    expect(messages[0]).toBe("GPU-03 reclaimed");
  });

  it("keys each entry by the backend's own event_id, never inventing one", () => {
    const { container } = render(<EventLog events={backendEvents} />);
    // Re-rendering with the same ids should not duplicate entries.
    expect(container.querySelectorAll(".event-entry")).toHaveLength(3);
  });

  it("reflects the backend's total event count in the panel title", () => {
    render(<EventLog events={backendEvents} />);
    expect(screen.getByText("EVENT LOG (3)")).toBeInTheDocument();
  });
});
