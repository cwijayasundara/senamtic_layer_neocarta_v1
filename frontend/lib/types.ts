export type GraphNode = {
  id: string;
  label: string;
  kind: "source" | "table" | "document" | "chunk" | "entity" | "value";
  source: string;
  platform?: string;
  text?: string; // chunk passage preview
  entityType?: string; // POLE+O label for entity nodes
};
export type GraphEdge = { source: string; target: string; type: string };
export type GraphData = { nodes: GraphNode[]; edges: GraphEdge[] };
export type Source = { name: string; platform: string; kind: "sql" | "api" };

export type ChatEvent =
  | { type: "tool_call"; scope: string; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; scope: string; name: string; content: string }
  | { type: "answer"; content: string; highlight: string[] };
