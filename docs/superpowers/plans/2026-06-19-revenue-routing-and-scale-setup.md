# Revenue Fact-Table Routing + Scale-Aware Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "revenue by X" questions route to the real sales fact (`order_line`) via curated table-embedding descriptions, and add `setup.sh --scale` so provisioning the 1,000-table catalog doesn't get wiped.

**Architecture:** Fix 1 folds a small curated `TABLE_DESCRIPTIONS` map into the table embedding text at `embed_tables` time, so "revenue" vector-matches `order_line` (and `income_statement` is pushed away from regional/transactional revenue). Fix 2 adds a `--scale` flag to `setup.sh` that runs the existing `scale-seed`/`scale-ingest` flow instead of the baseline non-scale ingest.

**Tech Stack:** Python 3.11+, Neo4j vector index, OpenAI `text-embedding-3-small`, bash, pytest.

## Global Constraints

- Python `>=3.11`; run all commands from `backend/` with `backend/.venv/bin/python`. Run only the FOCUSED test files named in each task — the full backend `pytest` wipes document embeddings.
- `_table_embed_text` stays backward-compatible: with `description=""` it must produce the exact prior output (`"<name> — columns: <cols>"`, or just `<name>` when no cols).
- `TABLE_DESCRIPTIONS` is the ONLY curated set; every other table falls back to name+columns. The two load-bearing entries are `table:sales_pg.sales.order_line` and `table:financials.main.income_statement`.
- `embed_tables` remains always-real (no `fake_embeddings` branch) and keeps the `table_embeddings` index.
- `setup.sh` default behavior (no flag) must be byte-for-byte unchanged; `--scale` runs the scale flow mirroring `Makefile` `scale-seed`/`scale-ingest`.
- Commit after every task with the message shown in its final step.

---

### Task 1: Curated table descriptions in the embedding text

**Files:**
- Create: `backend/semantic_layer/ingest/table_descriptions.py`
- Modify: `backend/semantic_layer/ingest/embeddings.py` (`_table_embed_text`, `embed_tables`)
- Test: `backend/tests/test_table_descriptions.py`

**Interfaces:**
- Produces: `TABLE_DESCRIPTIONS: dict[str, str]` (table id → one-line description); updated
  `_table_embed_text(name: str, cols: list[str], description: str = "") -> str`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_table_descriptions.py
from semantic_layer.ingest.embeddings import _table_embed_text
from semantic_layer.ingest.table_descriptions import TABLE_DESCRIPTIONS


def test_embed_text_without_description_is_unchanged():
    assert _table_embed_text("order_line", ["line_id", "amount"]) == \
        "order_line — columns: line_id, amount"
    assert _table_embed_text("region", []) == "region"


def test_embed_text_folds_in_description():
    out = _table_embed_text("order_line", ["amount"], "sales revenue line items")
    assert "sales revenue line items" in out
    assert out == "order_line — sales revenue line items — columns: amount"


def test_descriptions_cover_the_load_bearing_tables():
    assert "table:sales_pg.sales.order_line" in TABLE_DESCRIPTIONS
    assert "table:financials.main.income_statement" in TABLE_DESCRIPTIONS
    # order_line description must mention revenue (the whole point)
    assert "revenue" in TABLE_DESCRIPTIONS["table:sales_pg.sales.order_line"].lower()
    # income_statement must be marked NOT regional/per-order to push it away
    assert "not" in TABLE_DESCRIPTIONS["table:financials.main.income_statement"].lower()
    assert all(isinstance(v, str) and v for v in TABLE_DESCRIPTIONS.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_table_descriptions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_layer.ingest.table_descriptions'`

- [ ] **Step 3: Create the descriptions module**

```python
# backend/semantic_layer/ingest/table_descriptions.py
"""Curated one-line descriptions for the answerable-core tables, folded into the
table embedding text so semantic routing maps business words to the right table.

Only these tables are described; everything else (incl. the scale_* distractors)
falls back to name + column names. The two load-bearing entries disambiguate the
"revenue" trap: sales revenue lives in order_line.amount, NOT income_statement."""

TABLE_DESCRIPTIONS: dict[str, str] = {
    "table:sales_pg.sales.order_line":
        "sales revenue line items; amount is the line revenue (quantity x unit_price); "
        "the source for revenue by region, industry, segment, product, or period",
    "table:sales_pg.sales.sales_order":
        "customer sales orders; one row per order, links order lines to a customer and fiscal period",
    "table:sales_pg.sales.customer":
        "customers (accounts) that place sales orders; linked to country and industry",
    "table:sales_pg.sales.product":
        "products sold; each belongs to a product line",
    "table:sales_pg.sales.product_line":
        "product lines grouping products into a business segment",
    "table:sales_pg.sales.segment":
        "business segments (e.g. Data Center, Gaming) products belong to",
    "table:sales_pg.sales.region":
        "geographic sales regions (e.g. EMEA, Americas) reached via customer country",
    "table:sales_pg.sales.country":
        "countries, each mapped to a sales region",
    "table:sales_pg.sales.industry":
        "customer industries (verticals)",
    "table:sales_pg.sales.fiscal_period":
        "fiscal quarters/years used to scope sales by period",
    "table:financials.main.income_statement":
        "company-level reported quarterly financial statements (total revenue, net income); "
        "NOT per-order, per-customer, or regional — do not use for revenue by region/segment",
    "table:financials.main.stock_price":
        "daily company stock prices (open, high, low, close, volume)",
}
```

- [ ] **Step 4: Update `_table_embed_text` and `embed_tables`**

In `backend/semantic_layer/ingest/embeddings.py`, replace `_table_embed_text` (the
function whose body is `if cols: return f"{name} — columns: ..."; return name`) with:

```python
def _table_embed_text(name: str, cols: list[str], description: str = "") -> str:
    """Text embedded per table: name, an optional curated description, plus column
    names. With description='' the output is the prior name+columns form."""
    parts = [name]
    if description:
        parts.append(description)
    if cols:
        parts.append(f"columns: {', '.join(cols)}")
    return " — ".join(parts)
```

Add the import near the top of `embeddings.py` (after the existing imports):

```python
from semantic_layer.ingest.table_descriptions import TABLE_DESCRIPTIONS
```

In `embed_tables`, the loop currently builds
`texts = [_table_embed_text(r["name"], r["cols"]) for r in window]`. Change it to look up
the curated description by table id:

```python
            texts = [
                _table_embed_text(r["name"], r["cols"], TABLE_DESCRIPTIONS.get(r["id"], ""))
                for r in window
            ]
```

(The Cypher already returns `t.id AS id`, so `r["id"]` is available.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_table_descriptions.py -v`
Expected: PASS (3 tests). Backward-compat test confirms `description=""` is unchanged.

- [ ] **Step 6: Commit**

```bash
git add backend/semantic_layer/ingest/table_descriptions.py backend/semantic_layer/ingest/embeddings.py backend/tests/test_table_descriptions.py
git commit -m "feat(routing): curated core-table descriptions in table embeddings"
```

---

### Task 2: `setup.sh --scale` flag

**Files:**
- Modify: `setup.sh` (arg parsing, seed step, ingest step)
- Test: manual/`bash -n` checks (documented below; shell wiring over already-tested modules)

**Interfaces:**
- Consumes: existing `data.seed_scale`, the scale pipeline env (`SCALE_MODE`,
  `SCHEMA_ROUTING_ENABLED`, `FAKE_EMBEDDINGS`).
- Produces: `./setup.sh --scale` provisions the scale catalog; `./setup.sh` unchanged.

- [ ] **Step 1: Read `setup.sh` and locate the three edit points**

Read `setup.sh`. Find: (a) the usage comment near the top and where vars are initialized
(before the heavy steps), (b) the seed step
`( cd "$BACKEND" && "$PY" -m data.seed_postgres && "$PY" -m data.seed_sqlite )`, and (c) the
ingest `if [ "$HAVE_KEY" = "1" ]; then … else … fi` block. Make the edits by matching these
exact strings.

- [ ] **Step 2: Add `--scale` arg parsing**

After the usage/var setup near the top of `setup.sh` (before the provisioning steps), insert:

```bash
SCALE=false
for arg in "$@"; do
  case "$arg" in
    --scale) SCALE=true ;;
    -h|--help)
      printf "Usage: %s [--scale]\n" "$(basename "$0")"
      printf "  --scale   provision the scale catalog (1000 distractor tables + scaled core)\n"
      printf "            instead of the baseline core; needs OPENAI_API_KEY for routing embeddings\n"
      exit 0 ;;
    *) printf "Unknown option: %s (try --help)\n" "$arg" >&2; exit 2 ;;
  esac
done
```

- [ ] **Step 3: Make the seed step scale-aware**

Replace the seed step
`( cd "$BACKEND" && "$PY" -m data.seed_postgres && "$PY" -m data.seed_sqlite )` (and its
preceding `say` line) with:

```bash
if [ "$SCALE" = "true" ]; then
  say "Seeding the SCALE catalog (core at scale volume + 1000 distractor tables) + SQLite"
  ( cd "$BACKEND" && SCALE_MODE=true "$PY" -m data.seed_scale )
  ( cd "$BACKEND" && "$PY" -m data.seed_sqlite )
else
  say "Seeding databases (Postgres sales schema + SQLite financials/org)"
  ( cd "$BACKEND" && "$PY" -m data.seed_postgres && "$PY" -m data.seed_sqlite )
fi
```

- [ ] **Step 4: Make the ingest step scale-aware**

Replace the ingest `if [ "$HAVE_KEY" = "1" ]; then … else … fi` block with a scale branch in
front of the existing two branches:

```bash
if [ "$SCALE" = "true" ]; then
  if [ "$HAVE_KEY" = "1" ]; then
    say "Ingesting the SCALE catalog (1072 tables + table embeddings; schema routing on)"
    ( cd "$BACKEND" && SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true FAKE_EMBEDDINGS=true \
        "$PY" -m semantic_layer.ingest.pipeline )
  else
    warn "No OPENAI_API_KEY — scale routing needs table embeddings; ingesting metadata only (keyword routing fallback)."
    ( cd "$BACKEND" && SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true \
        "$PY" -c "from semantic_layer.ingest.pipeline import run_ingest; print(run_ingest(with_llm=False, reset=True))" )
  fi
elif [ "$HAVE_KEY" = "1" ]; then
  say "Ingesting the knowledge graph (metadata + documents + entities + glossary + embeddings)"
  ( cd "$BACKEND" && "$PY" -m semantic_layer.ingest.pipeline )
else
  say "Ingesting metadata + documents only (no OPENAI_API_KEY — entities/glossary/embeddings skipped)"
  ( cd "$BACKEND" && "$PY" -c "from semantic_layer.ingest.pipeline import run_ingest; print(run_ingest(with_llm=False, reset=True))" )
fi
```

- [ ] **Step 5: Verify the script (no heavy run)**

```bash
cd /Users/chamindawijayasundara/Documents/context_graphs/sementic_layer_neocarta_v1
bash -n setup.sh && echo "syntax OK"
./setup.sh --help | grep -q -- "--scale" && echo "help shows --scale"
( ./setup.sh --bogus; echo "exit=$?" ) 2>&1 | grep -q "Unknown option" && echo "rejects unknown args"
```
Expected: `syntax OK`, `help shows --scale`, `rejects unknown args`. Do NOT run
`./setup.sh --scale` here (it rebuilds the graph — that is Task 3's controller-run validation).

- [ ] **Step 6: Commit**

```bash
git add setup.sh
git commit -m "feat(scale): setup.sh --scale provisions the scale catalog (no baseline wipe)"
```

---

### Task 3: Validation — re-embed with descriptions + re-eval revenue questions (controller-run)

Not a code change and not for an implementer subagent — the controller runs it after Tasks 1–2
are merged, because it re-ingests the live graph and calls OpenAI. The dev env should be in
scale state already; if not, `./setup.sh --scale` (now available) provisions it.

- [ ] **Step 1: Re-ingest so `embed_tables` re-embeds with the new descriptions**

```bash
cd backend && SCALE_MODE=true SCHEMA_ROUTING_ENABLED=true FAKE_EMBEDDINGS=true \
  .venv/bin/python -m semantic_layer.ingest.pipeline
```
Expect a counts dict with `scale_sources` and no error.

- [ ] **Step 2: Confirm `order_line` is now routed for "revenue by region"**

```bash
cd backend && SCHEMA_ROUTING_ENABLED=true .venv/bin/python -c "
from semantic_layer.agent import routing
out = routing.retrieve_candidate_tables('What is total revenue by region?', k_ret=20)
ids = {c['table_id'] for c in out}
print('order_line routed:', 'table:sales_pg.sales.order_line' in ids)
print('income_statement routed:', 'table:financials.main.income_statement' in ids)
"
```
**Success:** `order_line routed: True`. (Ideally `income_statement` is no longer the revenue
pick.) If `order_line` is still absent, the description isn't strong enough — strengthen the
`order_line`/`income_statement` entries in `table_descriptions.py` and re-run Step 1.

- [ ] **Step 3: Re-run the answer eval and compare the revenue/multi-join questions**

```bash
cd backend && SCHEMA_ROUTING_ENABLED=true .venv/bin/python -m eval.run_eval --out scorecard-answers-v3.json
```
**Success:** `join-revenue-by-industry` and `join-top-customer` now `answer_ok: true`, and
overall accuracy holds or improves vs the 91–96% baseline (`scorecard-answers.json` /
`scorecard-answers-mini.json`). Note the before/after in the branch summary. No commit (scorecards
are run artifacts).

---

## Notes for the implementer

- **The fix is in retrieval, not planning.** `select_fact_table` already prefers the
  sales-schema table with the most FKs; the bug is that `order_line` was never *routed*. The
  curated description gets it into the candidate set; the existing planner does the rest.
- **Do not run the full pytest suite** — project policy: it wipes document embeddings. Run the
  focused file in Task 1. Task 3's re-ingest restores embeddings.
- **`setup.sh --scale` needs `OPENAI_API_KEY`** for real table embeddings (routing quality);
  without it the script warns and falls back to keyword routing (metadata-only ingest).
- Re-embedding is required for Task 1 to take effect on a live graph — embeddings are computed
  at ingest, so the description change only lands after a re-ingest (Task 3 Step 1).
