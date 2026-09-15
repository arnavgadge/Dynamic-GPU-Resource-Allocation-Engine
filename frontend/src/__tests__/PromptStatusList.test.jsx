import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PromptStatusList from "../components/PromptStatusList.jsx";

describe("PromptStatusList", () => {
  it("shows the affected user's name as read-only status - no buttons", () => {
    render(
      <PromptStatusList
        prompts={[{ gpu_id: "GPU-3", owner_user_id: "user_c", owner_user_name: "User C" }]}
      />
    );
    expect(screen.getByText(/GPU-3 awaiting User C response/)).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("renders nothing when there are no such prompts", () => {
    const { container } = render(<PromptStatusList prompts={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows who requested it, for a resource-request prompt", () => {
    render(
      <PromptStatusList
        prompts={[{
          gpu_id: "GPU-7", owner_user_id: "user_d", owner_user_name: "User D",
          requested_by_user_id: "user_b", requested_by_user_name: "User B",
        }]}
      />
    );
    expect(screen.getByText(/requested by User B/)).toBeInTheDocument();
  });
});
