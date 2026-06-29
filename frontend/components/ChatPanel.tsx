"use client";
import { useState } from "react";
import type { AnswerEvent } from "@/lib/types";
import { AnswerPanel } from "./AnswerPanel";

type Level = "Simple" | "Moderate" | "Complex" | "Max" | "Extreme";

// Complexity ramps by how many source TYPES (and databases) a question must fuse.
const LEVEL_STYLE: Record<Level, string> = {
  Simple: "border-emerald-700 bg-emerald-950/50 text-emerald-300",
  Moderate: "border-sky-700 bg-sky-950/50 text-sky-300",
  Complex: "border-amber-700 bg-amber-950/50 text-amber-300",
  Max: "border-fuchsia-600 bg-fuchsia-950/50 text-fuchsia-300",
  Extreme: "border-rose-600 bg-rose-950/50 text-rose-300",
};

function sourceClass(s: string): string {
  if (s.startsWith("SQL")) return "border-blue-800 text-blue-300";
  if (s.startsWith("API")) return "border-violet-800 text-violet-300";
  if (s.startsWith("Ontology")) return "border-rose-800 text-rose-300";
  if (s.startsWith("Facts")) return "border-orange-800 text-orange-300";
  return "border-green-800 text-green-300"; // Docs
}

const EXAMPLES: { q: string; level: Level; sources: string[]; name?: string }[] = [
  {
    q: "Which business segment has the highest total revenue?",
    level: "Simple",
    sources: ["SQL"],
  },
  {
    q: "How many open support tickets are there?",
    level: "Simple",
    sources: ["API"],
  },
  {
    q: "According to the press releases, what drove Data Center growth?",
    level: "Simple",
    sources: ["Docs"],
  },
  {
    q: "In FY2025, which EMEA Cloud customers bought Blackwell Data Center products, and what was each customer's total revenue by quarter?",
    level: "Moderate",
    sources: ["SQL"],
  },
  {
    q: "Compare the Data Center revenue we recorded for Blackwell products with what the NVIDIA press releases say drove Data Center growth.",
    level: "Complex",
    sources: ["SQL", "Docs"],
  },
  {
    // The showcase: needs TWO databases (sales_pg + financials), TWO APIs
    // (DGX + ITSM), and the press-release documents — all fused in one answer.
    q: "The full picture: for our top EMEA Cloud customers by Blackwell Data Center revenue (sales database), how much DGX Cloud GPU usage have they consumed and how many support tickets are open for them (DGX + ITSM APIs), and how does our company-wide quarterly revenue and gross margin (financials database) compare with the exact Data Center revenue figure NVIDIA quotes in its latest press release (search the documents and cite it)?",
    level: "Max",
    sources: ["SQL ×2", "API ×2", "Docs"],
  },
  {
    name: "Extremely complex query",
    q: "For the Data Center segment and Blackwell product architecture, combine SQL sales revenue by fiscal period, relevant NVIDIA document evidence, DGX/API usage or support signals, extracted fact triplets, and POLE+O ontology context. Explain whether the ontology classifies Blackwell as a ProductArchitecture, cite the documents or facts that support the narrative, and compare the structured revenue signal with the API signal.",
    level: "Extreme",
    sources: ["SQL", "API", "Docs", "Ontology", "Facts"],
  },
];

export function ChatPanel({
  answerEvent,
  busy,
  onAsk,
  onReset,
}: {
  answerEvent: AnswerEvent | null;
  busy: boolean;
  onAsk: (q: string) => void;
  onReset: () => void;
}) {
  const [q, setQ] = useState("");
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-auto p-4 space-y-3">
        {answerEvent && !busy && (
          <button
            onClick={onReset}
            className="text-sm text-gray-400 hover:text-[#76b900]"
          >
            ← Back to questions
          </button>
        )}
        {!answerEvent && !busy && (
          <div className="space-y-2">
            <p className="text-sm text-gray-400">
              Try a question — tagged by complexity and the sources it must fuse:
            </p>
            {EXAMPLES.map((ex) => (
              <button
                key={ex.q}
                onClick={() => onAsk(ex.q)}
                className={`block w-full text-left rounded border bg-gray-900 px-3 py-2 hover:border-[#76b900] ${
                  ex.level === "Max" ? "border-fuchsia-700/60" : "border-gray-800"
                }`}
              >
                <div className="mb-1 flex flex-wrap items-center gap-1">
                  <span
                    className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${LEVEL_STYLE[ex.level]}`}
                  >
                    {ex.level}
                  </span>
                  {ex.sources.map((s) => (
                    <span
                      key={s}
                      className={`rounded border bg-gray-950 px-1.5 py-0.5 text-[10px] ${sourceClass(s)}`}
                    >
                      {s}
                    </span>
                  ))}
                </div>
                {ex.name && (
                  <div className="mb-1 text-xs font-semibold text-gray-100">
                    {ex.name}
                  </div>
                )}
                <span className="text-sm text-gray-300">{ex.q}</span>
              </button>
            ))}
          </div>
        )}
        {answerEvent && <AnswerPanel answer={answerEvent} />}
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
