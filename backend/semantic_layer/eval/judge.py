"""LLM-as-judge: score an agent answer 1-4 against the expectation."""

from pydantic import BaseModel, Field

from semantic_layer.config import settings
from semantic_layer.ingest.llm import get_chat_model


class _Verdict(BaseModel):
    score: int = Field(ge=1, le=4)
    reason: str


_JUDGE_PROMPT = (
    "You are grading an answer from an enterprise data agent against a description of "
    "what a correct answer must contain. Score 1-4: 4 = fully correct AND complete; "
    "3 = mostly correct, minor omission; 2 = partially correct or missing a required "
    "part; 1 = wrong, unsupported, or non-answer. Judge only against the expectation; "
    "do not reward extra unverifiable claims. Return score and a one-sentence reason."
)


def judge_answer(question: str, answer: str, expect: str) -> dict:
    """One structured LLM call scoring `answer` 1-4 against `expect`."""
    model = get_chat_model(settings.synthesis_model_resolved).with_structured_output(_Verdict)
    v = model.invoke([
        ("system", _JUDGE_PROMPT),
        ("human", f"Question:\n{question}\n\nExpectation:\n{expect}\n\nAnswer:\n{answer}"),
    ])
    return {"score": v.score, "reason": v.reason}
