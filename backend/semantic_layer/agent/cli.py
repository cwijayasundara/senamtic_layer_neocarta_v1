"""CLI: ask the semantic-layer agent a question.

Usage: python -m semantic_layer.agent.cli "your question"
"""

import sys

from semantic_layer.agent.build import ask


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m semantic_layer.agent.cli "question"')
        raise SystemExit(2)
    print(ask(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
