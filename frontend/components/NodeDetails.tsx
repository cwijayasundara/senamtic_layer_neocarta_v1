"use client";
import type { GraphData, GraphNode } from "@/lib/types";

const KIND_COLOR: Record<string, string> = {
  source: "#76b900",
  table: "#3b82f6",
  document: "#f59e0b",
  chunk: "#a855f7",
  entity: "#ec4899",
  value: "#14b8a6",
};

function neighbors(graph: GraphData, id: string): { node: GraphNode; type: string }[] {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const out: { node: GraphNode; type: string }[] = [];
  for (const e of graph.edges) {
    if (e.source === id && byId.has(e.target)) out.push({ node: byId.get(e.target)!, type: e.type });
    else if (e.target === id && byId.has(e.source)) out.push({ node: byId.get(e.source)!, type: e.type });
  }
  return out;
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-sm text-gray-100 break-words">{value}</div>
    </div>
  );
}

export function NodeDetails({
  node,
  graph,
  onClose,
  onSelect,
}: {
  node: GraphNode;
  graph: GraphData;
  onClose: () => void;
  onSelect: (node: GraphNode) => void;
}) {
  const links = neighbors(graph, node.id);
  return (
    <aside className="absolute top-3 right-3 w-72 max-h-[calc(100%-1.5rem)] overflow-y-auto rounded-lg border border-gray-800 bg-[#11161d]/95 shadow-xl backdrop-blur">
      <div className="flex items-start justify-between gap-2 px-4 py-3 border-b border-gray-800">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="h-3 w-3 shrink-0 rounded-full"
            style={{ background: KIND_COLOR[node.kind] ?? "#9ca3af" }}
          />
          <span className="font-semibold text-sm truncate">{node.label}</span>
        </div>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-200 text-lg leading-none"
          aria-label="Close details"
        >
          ×
        </button>
      </div>
      <div className="px-4 py-3 space-y-3">
        <Field label="Kind" value={node.kind} />
        <Field label="Source" value={node.source} />
        {node.platform && <Field label="Platform" value={node.platform} />}
        {node.entityType && <Field label="Entity type" value={node.entityType} />}
        {node.text && (
          <div>
            <div className="text-[11px] uppercase tracking-wide text-gray-500">Passage</div>
            <div className="text-sm text-gray-300 break-words italic">“{node.text}…”</div>
          </div>
        )}
        <Field label="ID" value={node.id} />
        <div>
          <div className="text-[11px] uppercase tracking-wide text-gray-500 mb-1">
            Connections ({links.length})
          </div>
          {links.length === 0 ? (
            <div className="text-sm text-gray-500">None</div>
          ) : (
            <ul className="space-y-1">
              {links.map(({ node: nb, type }, i) => (
                <li key={`${nb.id}-${i}`}>
                  <button
                    onClick={() => onSelect(nb)}
                    className="w-full text-left text-sm text-gray-300 hover:text-[#76b900] flex items-center gap-2"
                  >
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ background: KIND_COLOR[nb.kind] ?? "#9ca3af" }}
                    />
                    <span className="truncate">{nb.label}</span>
                    <span className="ml-auto text-[10px] text-gray-600 shrink-0">{type}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </aside>
  );
}
