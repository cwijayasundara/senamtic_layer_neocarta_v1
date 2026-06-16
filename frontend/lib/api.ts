import type { GraphData, Source } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function fetchSources(): Promise<Source[]> {
  const r = await fetch(`${BASE}/sources`);
  return r.json();
}

export async function fetchGraph(): Promise<GraphData> {
  const r = await fetch(`${BASE}/graph`);
  return r.json();
}

export const API_BASE = BASE;
