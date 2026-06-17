/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GraphData, GraphNode } from "@/lib/types";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

const KIND_COLOR: Record<string, string> = {
  source: "#76b900", // NVIDIA green
  table: "#3b82f6", // blue
  document: "#f59e0b", // amber
  chunk: "#a855f7", // purple — PDF passages
  entity: "#ec4899", // pink — extracted entities
  value: "#14b8a6", // teal — canonical bridge values
};

// The document layer is dense (many chunks/entities); render those smaller and
// only label them when zoomed in or emphasized, so PDFs read as a constellation.
const SMALL_KINDS = new Set(["chunk", "entity", "value"]);

function endId(end: any): string {
  return typeof end === "object" && end !== null ? end.id : end;
}

// A minimal d3-force that pulls every node toward the origin, so disconnected
// clusters drift together instead of spreading across the canvas.
function centerPull(strength: number) {
  let nodes: any[] = [];
  const force = (alpha: number) => {
    for (const n of nodes) {
      n.vx -= n.x * strength * alpha;
      n.vy -= n.y * strength * alpha;
    }
  };
  force.initialize = (n: any[]) => {
    nodes = n;
  };
  return force;
}

export function GraphCanvas({
  graph,
  highlight,
  selectedId,
  onSelect,
}: {
  graph: GraphData;
  highlight: string[];
  selectedId?: string | null;
  onSelect?: (node: GraphNode | null) => void;
}) {
  const fgRef = useRef<any>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const forcesSet = useRef(false);
  const [size, setSize] = useState({ width: 0, height: 0 });

  // Size the canvas to its container; without this react-force-graph defaults to
  // the full window size and overflows the layout.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const hi = useMemo(() => new Set(highlight), [highlight]);
  const data = useMemo(
    () => ({
      nodes: graph.nodes.map((n) => ({ ...n })),
      links: graph.edges.map((e) => ({ source: e.source, target: e.target, type: e.type })),
    }),
    [graph],
  );

  // Neighbours of the selected node — used to highlight its connections.
  const neighborIds = useMemo(() => {
    const s = new Set<string>();
    if (!selectedId) return s;
    for (const e of graph.edges) {
      if (e.source === selectedId) s.add(e.target);
      else if (e.target === selectedId) s.add(e.source);
    }
    return s;
  }, [graph, selectedId]);

  const linkLit = (l: any) => hi.has(endId(l.source)) && hi.has(endId(l.target));
  const linkSel = (l: any) =>
    !!selectedId && (endId(l.source) === selectedId || endId(l.target) === selectedId);

  // react-force-graph rebuilds the simulation on data change; reapply our forces.
  useEffect(() => {
    forcesSet.current = false;
  }, [data]);

  // Pull clusters closer together and zoom to fill the viewport once settled.
  const onEngineStop = useCallback(() => {
    fgRef.current?.zoomToFit(400, 60);
  }, []);

  const configureForces = useCallback(() => {
    const fg = fgRef.current;
    if (!fg || forcesSet.current) return;
    fg.d3Force("charge")?.strength(-120).distanceMax(300);
    fg.d3Force("link")?.distance(40);
    fg.d3Force("centerPull", centerPull(0.12));
    forcesSet.current = true;
  }, []);

  return (
    <div ref={wrapRef} className="absolute inset-0 overflow-hidden">
    <ForceGraph2D
      ref={fgRef}
      width={size.width}
      height={size.height}
      graphData={data as any}
      backgroundColor="#0b0f14"
      nodeRelSize={12}
      cooldownTicks={120}
      onEngineStop={onEngineStop}
      onEngineTick={configureForces}
      onNodeClick={(node: any) => onSelect?.(node as GraphNode)}
      onBackgroundClick={() => onSelect?.(null)}
      linkColor={(l: any) => (linkLit(l) || linkSel(l) ? "#76b900" : "#1f2937")}
      linkWidth={(l: any) => (linkLit(l) || linkSel(l) ? 3 : 1)}
      nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, scale: number) => {
        const active = hi.has(node.id);
        const selected = node.id === selectedId;
        const neighbor = neighborIds.has(node.id);
        const emphasized = active || selected || neighbor;
        const dimContext = hi.size > 0 || !!selectedId;
        const small = SMALL_KINDS.has(node.kind);
        const baseR = small ? 5 : 12;
        const r = active || selected ? baseR + 4 : baseR;
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = KIND_COLOR[node.kind] ?? "#9ca3af";
        ctx.globalAlpha = emphasized || !dimContext ? 1 : 0.25;
        ctx.fill();
        if (selected) {
          ctx.strokeStyle = "#76b900";
          ctx.lineWidth = 4;
          ctx.stroke();
        } else if (neighbor) {
          ctx.strokeStyle = "#76b900";
          ctx.lineWidth = 2.5;
          ctx.stroke();
        } else if (active) {
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 2.5;
          ctx.stroke();
        }
        if (emphasized || scale > (small ? 2.5 : 1.2)) {
          ctx.globalAlpha = 1;
          ctx.fillStyle = "#e5e7eb";
          ctx.font = `${12 / scale}px sans-serif`;
          ctx.fillText(node.label, node.x + r + 1, node.y + 3);
        }
        ctx.globalAlpha = 1;
      }}
    />
    </div>
  );
}
