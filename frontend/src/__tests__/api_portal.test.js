import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../services/auth.js", () => ({
  getSession: vi.fn(),
}));

import { getSession } from "../services/auth.js";
import { getPortalState, submitGpuRequest } from "../services/api.js";

function mockFetchOnce(responseBody, ok = true) {
  global.fetch = vi.fn().mockResolvedValue({
    ok,
    status: ok ? 200 : 422,
    statusText: ok ? "OK" : "Unprocessable Entity",
    json: async () => responseBody,
  });
}

describe("api service - Phase 9 additions", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("attaches the session's bearer token automatically when logged in", async () => {
    getSession.mockReturnValue({ token: "abc123", role: "USER" });
    mockFetchOnce({ my_jobs: [] });

    await getPortalState();

    const [, options] = fetch.mock.calls[0];
    expect(options.headers.Authorization).toBe("Bearer abc123");
  });

  it("sends no Authorization header when not logged in", async () => {
    getSession.mockReturnValue(null);
    mockFetchOnce({ my_jobs: [] });

    await getPortalState();

    const [, options] = fetch.mock.calls[0];
    expect(options.headers.Authorization).toBeUndefined();
  });

  it("submitGpuRequest sends exactly the four fields the backend expects, snake_cased", async () => {
    getSession.mockReturnValue({ token: "abc123" });
    mockFetchOnce({ my_jobs: [] });

    await submitGpuRequest({ workload: "ML Training", gpuCount: 1, estimatedMinutes: 20, priority: "HIGH" });

    const [url, options] = fetch.mock.calls[0];
    expect(url).toContain("/api/requests");
    expect(JSON.parse(options.body)).toEqual({
      workload: "ML Training",
      gpu_count: 1,
      estimated_minutes: 20,
      priority: "HIGH",
    });
  });
});
