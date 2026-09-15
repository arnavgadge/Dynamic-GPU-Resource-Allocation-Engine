import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Header from "../components/Header.jsx";

describe("Header", () => {
  it("renders exactly the real-clock/uptime values it was given - never the simulated clock", () => {
    render(<Header realTime="2026-05-01T21:45:32+00:00" uptimeSeconds={5} eventCount={3} />);
    expect(screen.getByText("21:45:32")).toBeInTheDocument();
    expect(screen.getByText("5s")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("shows a neutral placeholder before the first clock reading arrives", () => {
    render(<Header realTime={undefined} uptimeSeconds={undefined} eventCount={0} />);
    expect(screen.getByText("--:--:--")).toBeInTheDocument();
    expect(screen.getByText("--")).toBeInTheDocument();
  });
});
