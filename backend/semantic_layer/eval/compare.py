"""A/B compare answer quality with schema routing OFF vs ON over the golden evalset.

Run before flipping `schema_routing_enabled` on by default: confirm ON >= OFF."""

import json

from semantic_layer.config import settings
from semantic_layer.eval.evalset import load_evalset
from semantic_layer.eval.run import run_eval


def compare_routing(evalset: list[dict], run_fn=run_eval) -> dict:
    """Run the evalset with routing forced OFF then ON; restore the original setting."""
    original = settings.schema_routing_enabled
    try:
        settings.schema_routing_enabled = False
        off = run_fn(evalset)
        settings.schema_routing_enabled = True
        on = run_fn(evalset)
    finally:
        settings.schema_routing_enabled = original
    return {"off": off, "on": on,
            "delta_mean": round(on["mean_score"] - off["mean_score"], 2)}


def main() -> None:
    print(json.dumps(compare_routing(load_evalset()), indent=2))


if __name__ == "__main__":
    main()
