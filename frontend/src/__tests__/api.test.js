import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { respondToPrompt, loadScenario, setSpeed, getState } from "../services/api.js";

// React command -> Backend: verifies the exact HTTP request each API
// function sends - method, path, and JSON body - matches the FastAPI
// contract `api/app.py` actually implements. No scheduling logic is
// exercised here; this only proves the wire format is correct.

function mockFetchOnce(responseBody, ok = true) {
  global.fetch = vi.fn().mockResolvedValue({
    ok,
    status: ok ? 200 : 400,
    statusText: ok ? "OK" : "Bad Request",
    json: async () => responseBody,
  });
}

describe("api service", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("respondToPrompt POSTs the exact gpu_id and response the backend expects", async () => {
    mockFetchOnce({ ok: true });

    await respondToPrompt("GPU-03", "NO");

    expect(fetch).toHaveBeenCalledTimes(1);
    const [url, options] = fetch.mock.calls[0];
    expect(url).toContain("/api/prompt/respond");
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual({ gpu_id: "GPU-03", response: "NO" });
  });

  it("loadScenario POSTs to the scenario-specific load endpoint", async () => {
    mockFetchOnce({ ok: true });

    await loadScenario("idle_user");

    const [url, options] = fetch.mock.calls[0];
    expect(url).toContain("/api/scenarios/idle_user/load");
    expect(options.method).toBe("POST");
  });

  it("setSpeed sends the numeric speed value untouched - no client-side clamping/logic", async () => {
    mockFetchOnce({ ok: true });

    await setSpeed(50);

    const [, options] = fetch.mock.calls[0];
    expect(JSON.parse(options.body)).toEqual({ speed: 50 });
  });

  it("getState performs a plain GET against /api/state", async () => {
    mockFetchOnce({ gpus: [] });

    await getState();

    const [url, options] = fetch.mock.calls[0];
    expect(url).toContain("/api/state");
    expect(options.method).toBeUndefined(); // defaults to GET
  });

  it("a non-ok backend response rejects with the backend's own error detail", async () => {
    mockFetchOnce({ detail: "GPU 'GPU-99' has no pending confirmation prompt" }, false);

    await expect(respondToPrompt("GPU-99", "YES")).rejects.toThrow(/no pending confirmation prompt/);
  });
});
