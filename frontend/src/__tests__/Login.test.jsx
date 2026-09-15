import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import Login from "../components/Login.jsx";

describe("Login", () => {
  it("clicking a demo account sends exactly that username to the backend, nothing else", () => {
    const onLogin = vi.fn();
    render(<Login onLogin={onLogin} loginError={null} />);

    fireEvent.click(screen.getByText("User A"));

    expect(onLogin).toHaveBeenCalledWith("user_a");
  });

  it("clicking Admin sends the admin username", () => {
    const onLogin = vi.fn();
    render(<Login onLogin={onLogin} loginError={null} />);

    fireEvent.click(screen.getByText("Admin"));

    expect(onLogin).toHaveBeenCalledWith("admin");
  });

  it("shows the backend's own login error, not a fabricated one", () => {
    render(<Login onLogin={vi.fn()} loginError="401 unknown demo account" />);
    expect(screen.getByText("401 unknown demo account")).toBeInTheDocument();
  });
});
