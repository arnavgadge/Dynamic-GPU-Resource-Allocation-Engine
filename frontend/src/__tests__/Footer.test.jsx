import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Footer from "../components/Footer.jsx";

describe("Footer", () => {
  it("shows CONNECTED/ONLINE when the WebSocket is connected", () => {
    render(<Footer connectionStatus="connected" simulationRunning={true} />);
    expect(screen.getAllByText("CONNECTED").length).toBeGreaterThan(0);
    expect(screen.getByText("ONLINE")).toBeInTheDocument();
    expect(screen.getByText("RUNNING")).toBeInTheDocument();
  });

  it("shows OFFLINE rather than silently displaying stale data when disconnected", () => {
    render(<Footer connectionStatus="disconnected" simulationRunning={true} />);
    expect(screen.getAllByText("OFFLINE").length).toBeGreaterThan(0);
    expect(screen.getByText("DISCONNECTED")).toBeInTheDocument();
  });

  it("reflects the backend's paused simulation flag", () => {
    render(<Footer connectionStatus="connected" simulationRunning={false} />);
    expect(screen.getByText("PAUSED")).toBeInTheDocument();
  });
});
