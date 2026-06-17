"use client";
import { useState } from "react";

const EXAMPLES = [
  "Which business segment has the highest total revenue?",
  "How many open support tickets are there?",
  "According to the press releases, what drove Data Center growth?",
  "In FY2025, which EMEA Cloud customers bought Blackwell Data Center products, and what was each customer's total revenue by quarter?",
  "Compare the Data Center revenue we recorded for Blackwell products with what the NVIDIA press releases say drove Data Center growth.",
];

export function ChatPanel({
  answer,
  busy,
  onAsk,
  onReset,
}: {
  answer: string;
  busy: boolean;
  onAsk: (q: string) => void;
  onReset: () => void;
}) {
  const [q, setQ] = useState("");
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto p-4 space-y-3">
        {answer && !busy && (
          <button
            onClick={onReset}
            className="text-sm text-gray-400 hover:text-[#76b900]"
          >
            ← Back to questions
          </button>
        )}
        {!answer && !busy && (
          <div className="space-y-2">
            <p className="text-sm text-gray-400">Try a question:</p>
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => onAsk(ex)}
                className="block w-full text-left text-sm rounded border border-gray-800 bg-gray-900 px-3 py-2 text-gray-300 hover:border-[#76b900]"
              >
                {ex}
              </button>
            ))}
          </div>
        )}
        {answer && (
          <div className="rounded-lg bg-gray-800 p-3 text-gray-100 whitespace-pre-wrap">{answer}</div>
        )}
        {busy && <div className="text-sm text-gray-400 animate-pulse">thinking…</div>}
      </div>
      <form
        className="p-3 border-t border-gray-800 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (q.trim()) {
            onAsk(q);
            setQ("");
          }
        }}
      >
        <input
          className="flex-1 rounded bg-gray-900 border border-gray-700 px-3 py-2 text-gray-100"
          placeholder="Ask across databases, APIs, and documents…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          disabled={busy}
        />
        <button
          className="rounded bg-[#76b900] px-4 py-2 font-medium text-black disabled:opacity-50"
          disabled={busy}
        >
          Ask
        </button>
      </form>
    </div>
  );
}
