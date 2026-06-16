"use client";
import type { ChatEvent } from "@/lib/types";

export function TracePanel({ events }: { events: ChatEvent[] }) {
  return (
    <div className="h-full overflow-auto p-3 text-xs font-mono space-y-1">
      {events.length === 0 && <div className="text-gray-600">agent trace appears here…</div>}
      {events.map((e, i) => {
        if (e.type === "tool_call")
          return (
            <div key={i} className="text-blue-300">
              → {e.name}({JSON.stringify(e.args)})
            </div>
          );
        if (e.type === "tool_result")
          return (
            <div key={i} className="text-gray-400 truncate">
              {e.name}: {e.content}
            </div>
          );
        return (
          <div key={i} className="text-[#76b900]">
            ✓ answer ({e.highlight.length} nodes lit)
          </div>
        );
      })}
    </div>
  );
}
