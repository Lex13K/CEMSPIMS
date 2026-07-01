import type { Node, Edge } from "@xyflow/react";
import { Position } from "@xyflow/react";
import type { GraphNode } from "../api";

export function statusBadgeColor(status: string): string {
  switch (status) {
    case "complete":
      return "#22c55e";
    case "stale":
      return "#eab308";
    case "running":
      return "#3b82f6";
    default:
      return "#64748b";
  }
}

export function toFlowNodes(nodes: GraphNode[]): Node[] {
  return nodes.map((n) => {
    const stripe = n.color || "#64748b";
    const badge = statusBadgeColor(n.status);
    return {
      id: n.id,
      position: n.position,
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      data: { label: n.label, node: n, badge, stripe },
      style: {
        background: n.kind === "terminal" ? "#334155" : "#1e293b",
        color: "#e2e8f0",
        border: "1px solid #475569",
        borderLeft: `5px solid ${stripe}`,
        borderRadius: n.kind === "terminal" ? 4 : 8,
        padding: "8px 8px 8px 10px",
        fontSize: 11,
        width: n.kind === "terminal" ? 120 : 180,
        minHeight: n.kind === "terminal" ? 48 : 44,
      },
      className: n.status === "running" ? "node-running" : undefined,
    };
  });
}

export function statusLabel(status: string): string {
  return status.charAt(0).toUpperCase() + status.slice(1);
}
