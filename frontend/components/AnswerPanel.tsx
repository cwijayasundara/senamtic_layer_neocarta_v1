"use client";
import type { AnswerEvent, SqlRun } from "@/lib/types";

function ResultTable({ run }: { run: SqlRun }) {
  if (run.error)
    return <div className="text-xs text-red-400 font-mono">error: {run.error}</div>;
  const rows = run.rows.slice(0, 50);
  return (
    <div className="overflow-auto max-h-64 border border-gray-800 rounded">
      <table className="text-xs w-full">
        <thead className="bg-gray-900 text-gray-400 sticky top-0">
          <tr>
            {run.columns.map((c) => (
              <th key={c} className="text-left px-2 py-1 font-medium">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-gray-800">
              {row.map((cell, j) => (
                <td key={j} className="px-2 py-1 text-gray-200">{String(cell)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {run.row_count > rows.length && (
        <div className="text-[10px] text-gray-500 px-2 py-1">
          showing {rows.length} of {run.row_count} rows
        </div>
      )}
    </div>
  );
}

export function AnswerPanel({
  answer,
  onSelectNode,
}: {
  answer: AnswerEvent;
  onSelectNode?: (id: string) => void;
}) {
  return (
    <div className="space-y-3">
      <div className="rounded-lg bg-gray-800 p-3 text-gray-100 whitespace-pre-wrap">
        {answer.content}
      </div>

      {answer.caveats.length > 0 && (
        <div className="rounded border border-amber-700 bg-amber-950/40 p-2 text-xs text-amber-300 space-y-1">
          <div className="font-medium">Groundedness check</div>
          {answer.caveats.map((c, i) => (
            <div key={i}>⚠ {c}</div>
          ))}
        </div>
      )}

      {answer.sql_runs.map((run, i) => (
        <details key={i} className="rounded border border-gray-800" open={i === 0}>
          <summary className="cursor-pointer px-2 py-1 text-xs text-blue-300">
            Generated SQL · <span className="text-gray-500">{run.source}</span>
          </summary>
          <pre className="text-[11px] font-mono text-gray-300 px-2 py-1 overflow-auto whitespace-pre-wrap">
            {run.sql}
          </pre>
          <div className="px-2 pb-2">
            <ResultTable run={run} />
          </div>
        </details>
      ))}

      {answer.api_calls.map((call, i) => (
        <div key={i} className="rounded border border-gray-800 px-2 py-1 text-xs text-gray-300">
          API · <span className="text-blue-300">{call.source}{call.path}</span>{" "}
          <span className="text-gray-500">({call.row_count} rows)</span>
        </div>
      ))}

      {answer.doc_citations.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {answer.doc_citations.map((c) => (
            <button
              key={c.chunk_id}
              title={c.quote}
              onClick={() => onSelectNode?.(c.chunk_id)}
              className="text-[11px] rounded border border-gray-700 px-2 py-1 text-gray-300 hover:border-[#76b900]"
            >
              {c.doc_id}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
