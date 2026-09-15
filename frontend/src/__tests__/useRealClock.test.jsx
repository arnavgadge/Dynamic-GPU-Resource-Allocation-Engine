import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../services/api.js", () => ({
  getClock: vi.fn(),
}));

import { getClock } from "../services/api.js";
import { useRealClock } from "../hooks/useRealClock.js";

function Probe() {
  const clock = useRealClock();
  return <div>{clock ? `${clock.real_time}|${clock.uptime_seconds}` : "none"}</div>;
}

describe("useRealClock", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders exactly what the backend's /api/clock endpoint returns", async () => {
    getClock.mockResolvedValue({ real_time: "2026-05-01T21:45:32+00:00", uptime_seconds: 12.5 });

    await act(async () => {
      render(<Probe />);
    });

    expect(await screen.findByText("2026-05-01T21:45:32+00:00|12.5")).toBeInTheDocument();
  });

  it("polls again and reflects a later backend reading - never a locally-incremented value", async () => {
    getClock
      .mockResolvedValueOnce({ real_time: "2026-05-01T21:45:32+00:00", uptime_seconds: 12.5 })
      .mockResolvedValueOnce({ real_time: "2026-05-01T21:45:33+00:00", uptime_seconds: 13.5 });

    vi.useFakeTimers({ shouldAdvanceTime: true });
    await act(async () => {
      render(<Probe />);
    });
    await screen.findByText("2026-05-01T21:45:32+00:00|12.5");

    await act(async () => {
      vi.advanceTimersByTime(1000);
    });

    expect(await screen.findByText("2026-05-01T21:45:33+00:00|13.5")).toBeInTheDocument();
    vi.useRealTimers();
  });
});
