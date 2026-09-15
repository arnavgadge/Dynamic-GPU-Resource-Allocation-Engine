// Maps a backend-reported enum value to a CSS class name only - the
// mapping itself communicates state, it never decides state. Colors
// must stay consistent with the project's fixed semantics:
// green = active, yellow = idle warning, red = reclaiming,
// blue = reallocating/allocation activity.

const GPU_STATUS_CLASS = {
  ACTIVE: "status-active",
  IDLE: "status-idle",
  IDLE_WARNING: "status-warning",
  RECLAIMING: "status-reclaiming",
  REALLOCATING: "status-reallocating",
};

export function gpuStatusClassName(status) {
  return GPU_STATUS_CLASS[status] ?? "status-unknown";
}

const EVENT_TYPE_CLASS = {
  ALLOC: "event-alloc",
  BALANCE: "event-balance",
  MONITOR: "event-monitor",
  PROMPT: "event-prompt",
  RESPONSE: "event-response",
  RECLAIM: "event-reclaim",
  STATUS: "event-status",
  REQUEST: "event-request",
  SYSTEM: "event-system",
};

export function eventTypeClassName(eventType) {
  return EVENT_TYPE_CLASS[eventType] ?? "event-system";
}

const PRIORITY_CLASS = {
  CRITICAL: "priority-critical",
  HIGH: "priority-high",
  MEDIUM: "priority-medium",
  LOW: "priority-low",
};

export function priorityClassName(priority) {
  return PRIORITY_CLASS[priority] ?? "priority-medium";
}
