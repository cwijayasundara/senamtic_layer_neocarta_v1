"use client";
import { useEffect, useState } from "react";
import { fetchGraph } from "@/lib/api";
import { useChatStream } from "@/lib/useChatStream";
import { GraphCanvas } from "@/components/GraphCanvas";
import { NodeDetails } from "@/components/NodeDetails";
import { ChatPanel } from "@/components/ChatPanel";
import { TracePanel } from "@/components/TracePanel";
import type { GraphData, GraphNode } from "@/lib/types";

export default function Home() {
  const [graph, setGraph] = useState<GraphData>({ nodes: [], edges: [] });
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const { events, answer, highlight, busy, ask, reset } = useChatStream();

  useEffect(() => {
    fetchGraph()
      .then(setGraph)
      .catch(() => {});
  }, []);

  return (
    <main className="h-screen w-screen grid grid-cols-[420px_1fr] bg-[#0b0f14] text-gray-100 overflow-hidden">
      <section className="border-r border-gray-800 flex flex-col min-h-0">
        <header className="px-4 py-3 border-b border-gray-800 font-semibold">
          NeoCarta-Local <span className="text-[#76b900]">semantic layer</span>
        </header>
        <div className="flex-1 min-h-0">
          <ChatPanel answer={answer} busy={busy} onAsk={ask} onReset={reset} />
        </div>
        <div className="h-48 border-t border-gray-800">
          <TracePanel events={events} />
        </div>
      </section>
      <section className="min-h-0 relative">
        <GraphCanvas
          graph={graph}
          highlight={highlight}
          selectedId={selected?.id ?? null}
          onSelect={setSelected}
        />
        {selected && (
          <NodeDetails
            node={selected}
            graph={graph}
            onClose={() => setSelected(null)}
            onSelect={setSelected}
          />
        )}
      </section>
    </main>
  );
}
