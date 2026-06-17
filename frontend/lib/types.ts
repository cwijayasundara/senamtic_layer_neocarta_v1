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
  | {
      type: "answer";
      content: string;
      highlight: string[];
      sql_runs: SqlRun[];
      api_calls: ApiCall[];
      doc_citations: DocCitation[];
      caveats: string[];
    };

export type AnswerEvent = Extract<ChatEvent, { type: "answer" }>;

export type SqlRun = {
  source: string;
  sql: string;
  columns: string[];
  rows: (string | number | null)[][];
  row_count: number;
  error: string | null;
};
export type ApiCall = {
  source: string;
  path: string;
  params: Record<string, unknown>;
  status: number | null;
  row_count: number;
  data: unknown;
};
export type DocCitation = {
  doc_id: string;
  chunk_id: string;
  quote: string;
  score: number | null;
};
