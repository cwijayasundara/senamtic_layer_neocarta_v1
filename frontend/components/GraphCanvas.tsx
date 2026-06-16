/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";
import dynamic from "next/dynamic";
import { useMemo } from "react";
import type { GraphData } from "@/lib/types";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const KIND_COLOR: Record<string, string> = {
  source: "#76b900", // NVIDIA green
  table: "#3b82f6",
  document: "#f59e0b",
};

function endId(end: any): string {
  return typeof end === "object" && end !== null ? end.id : end;
}

export function GraphCanvas({ graph, highlight }: { graph: GraphData; highlight: string[] }) {
  const hi = useMemo(() => new Set(highlight), [highlight]);
  const data = useMemo(
    () => ({
      nodes: graph.nodes.map((n) => ({ ...n })),
      links: graph.edges.map((e) => ({ source: e.source, target: e.target, type: e.type })),
    }),
    [graph],
  );

  const linkLit = (l: any) => hi.has(endId(l.source)) && hi.has(endId(l.target));

  return (
    <ForceGraph2D
      graphData={data as any}
      backgroundColor="#0b0f14"
      nodeRelSize={5}
      linkColor={(l: any) => (linkLit(l) ? "#76b900" : "#1f2937")}
      linkWidth={(l: any) => (linkLit(l) ? 3 : 1)}
      nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, scale: number) => {
        const active = hi.has(node.id);
        const r = active ? 7 : 4;
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = KIND_COLOR[node.kind] ?? "#9ca3af";
        ctx.globalAlpha = active || hi.size === 0 ? 1 : 0.25;
        ctx.fill();
        if (active) {
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        if (active || scale > 2) {
          ctx.globalAlpha = 1;
          ctx.fillStyle = "#e5e7eb";
          ctx.font = `${10 / scale}px sans-serif`;
          ctx.fillText(node.label, node.x + r + 1, node.y + 3);
        }
        ctx.globalAlpha = 1;
      }}
    />
  );
}
