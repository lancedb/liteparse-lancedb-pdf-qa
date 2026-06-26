"""Answer-generation and LLM-judge evaluation over hybrid retrieval bundles."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryImage, UserContent

from benchmark import load_questions
from embeddings import TextEmbedder
from eval import score_result
from retrieval import Retriever


class AnswerOutput(BaseModel):
    """Structured answer produced from retrieved evidence."""

    answer: str = Field(description="Answer to the benchmark question, grounded only in evidence.")
    cited_pages: list[int] = Field(description="PDF page numbers used to answer.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-rated confidence from 0 to 1.")


class JudgeOutput(BaseModel):
    """Structured LLM-as-judge correctness result."""

    is_correct: bool = Field(description="True when the actual answer is materially correct.")
    score: float = Field(ge=0.0, le=1.0, description="Correctness score from 0 to 1.")
    verdict: str = Field(description="Short verdict such as correct, partially_correct, incorrect.")
    rationale: str = Field(description="Brief explanation of the score.")
    missing_or_wrong: list[str] = Field(description="Important omissions or mistakes, if any.")


ANSWER_SYSTEM_PROMPT = """\
You are an ESG report analyst answering Climate Finance Bench questions.

Use only the supplied retrieved evidence bundle. Do not use outside knowledge.
If the evidence is insufficient, say what can be answered and what is missing.
When the question asks for numbers, preserve units and fiscal-year context.
Return a concise answer plus the page numbers you relied on.
"""


JUDGE_SYSTEM_PROMPT = """\
You judge ESG benchmark answers.

Compare the actual answer against the expected answer for the same verbatim question.
Use semantic correctness, not exact wording. Award partial credit for answers that capture the
main facts but miss details, units, years, or caveats. Penalize unsupported contradictions.

Scoring guide:
- 1.0: fully correct or equivalent to expected answer.
- 0.75: mostly correct with minor omissions.
- 0.5: partially correct but missing important information.
- 0.25: small relevant fragment only.
- 0.0: incorrect, contradictory, or non-answer.
"""


def run_answer_eval(
    *,
    db_path: Path,
    questions_path: Path,
    out_path: Path,
    metrics_path: Path,
    answer_model: str,
    judge_model: str,
    embedding_backend: str,
    text_model: str,
    text_dimensions: int | None,
    top_k: int,
    limit: int | None,
    max_evidence_images: int,
    concurrency: int = 6,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    if limit is not None:
        questions = questions[:limit]

    text_embedder = TextEmbedder(embedding_backend, text_model, text_dimensions)
    retriever = Retriever(db_path, text_embedder)
    answer_agent = Agent(
        f"openai-chat:{answer_model}", output_type=AnswerOutput, instructions=ANSWER_SYSTEM_PROMPT
    )
    judge_agent = Agent(
        f"openai-chat:{judge_model}", output_type=JudgeOutput, instructions=JUDGE_SYSTEM_PROMPT
    )

    rows = asyncio.run(
        _run_questions(
            questions,
            retriever=retriever,
            text_embedder=text_embedder,
            answer_agent=answer_agent,
            judge_agent=judge_agent,
            answer_model=answer_model,
            judge_model=judge_model,
            top_k=top_k,
            max_evidence_images=max_evidence_images,
            concurrency=concurrency,
        )
    )

    write_jsonl(out_path, rows)
    metrics = summarize_answer_eval(rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


async def _run_questions(
    questions: list[dict[str, Any]],
    *,
    retriever: Retriever,
    text_embedder: TextEmbedder,
    answer_agent: Agent,
    judge_agent: Agent,
    answer_model: str,
    judge_model: str,
    top_k: int,
    max_evidence_images: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Process questions concurrently, bounded by a semaphore. Order is preserved."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _answer_one(
            question,
            semaphore=semaphore,
            retriever=retriever,
            text_embedder=text_embedder,
            answer_agent=answer_agent,
            judge_agent=judge_agent,
            answer_model=answer_model,
            judge_model=judge_model,
            top_k=top_k,
            max_evidence_images=max_evidence_images,
        )
        for question in questions
    ]
    return await asyncio.gather(*tasks)


async def _answer_one(
    question: dict[str, Any],
    *,
    semaphore: asyncio.Semaphore,
    retriever: Retriever,
    text_embedder: TextEmbedder,
    answer_agent: Agent,
    judge_agent: Agent,
    answer_model: str,
    judge_model: str,
    top_k: int,
    max_evidence_images: int,
) -> dict[str, Any]:
    async with semaphore:
        # Retrieval embeds the query (network) and reads blobs (disk), so run that
        # synchronous work in a thread to keep the event loop free for the model calls.
        retrieval_scored, answer_content, evidence_images = await asyncio.to_thread(
            _prepare_question, question, retriever, text_embedder, top_k, max_evidence_images
        )

        answer_started = time.perf_counter()
        answer_result = await answer_agent.run(answer_content)
        answer_latency_ms = (time.perf_counter() - answer_started) * 1000
        answer = answer_result.output

        judge_started = time.perf_counter()
        judge_result = await judge_agent.run(build_judge_prompt(question, answer))
        judge_latency_ms = (time.perf_counter() - judge_started) * 1000
        judge = judge_result.output

    return {
        "question_id": question["question_id"],
        "company": question["company"],
        "question_type": question.get("question_type"),
        "difficulty": question.get("difficulty"),
        "required_modality": question.get("required_modality"),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "expected_pages": question["expected_pages"],
        "retrieval": retrieval_scored,
        "answer": answer.model_dump(),
        "judge": judge.model_dump(),
        "answer_model": answer_model,
        "judge_model": judge_model,
        "evidence_images": [identifier for identifier, _ in evidence_images],
        "answer_latency_ms": answer_latency_ms,
        "judge_latency_ms": judge_latency_ms,
    }


def _prepare_question(
    question: dict[str, Any],
    retriever: Retriever,
    text_embedder: TextEmbedder,
    top_k: int,
    max_evidence_images: int,
) -> tuple[dict[str, Any], list[UserContent], list[tuple[str, bytes]]]:
    query_vector = text_embedder.embed(question["question"])
    retrieval = retriever.retrieve(
        question, mode="hybrid_bundle", top_k=top_k, query_vector=query_vector
    )
    retrieval_scored = score_result(question, retrieval)
    answer_prompt = build_answer_prompt(question, format_evidence_bundle(retrieval["top_k"]))
    evidence_images = retriever.page_screenshots(
        [hit["page_id"] for hit in retrieval["top_k"]], max_evidence_images
    )
    return retrieval_scored, build_answer_content(answer_prompt, evidence_images), evidence_images


def format_evidence_bundle(hits: list[dict[str, Any]], *, max_text_chars: int = 3000) -> str:
    sections = []
    for index, hit in enumerate(hits, start=1):
        text = (hit.get("text") or hit.get("text_preview") or "").strip()
        if len(text) > max_text_chars:
            text = text[:max_text_chars].rstrip() + "\n[truncated]"
        header = (
            f"Evidence {index}: source={hit.get('source_table')}, page={hit.get('page_num')}, "
            f"asset_type={hit.get('asset_type')}, score={hit.get('score'):.4f}"
        )
        provenance = []
        if hit.get("screenshot_path"):
            provenance.append(f"screenshot_path={hit['screenshot_path']}")
        if hit.get("path"):
            provenance.append(f"asset_path={hit['path']}")
        if hit.get("source_tables"):
            provenance.append(f"source_tables={hit['source_tables']}")
        if provenance:
            header += "\n" + "\n".join(provenance)
        sections.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(sections)


def build_answer_prompt(question: dict[str, Any], evidence_text: str) -> str:
    return f"""\
Question ID: {question["question_id"]}
Company: {question["company"]}
Source PDF: {question["source_pdf"]}
Question, verbatim:
{question["question"]}

Retrieved hybrid evidence bundle:
{evidence_text}
"""


def build_answer_content(
    answer_prompt: str, evidence_images: list[tuple[str, bytes]]
) -> list[UserContent]:
    content: list[UserContent] = [answer_prompt]
    for identifier, data in evidence_images:
        content.append(BinaryImage(data=data, media_type="image/png", identifier=identifier))
    return content


def build_judge_prompt(question: dict[str, Any], answer: AnswerOutput) -> str:
    return f"""\
Question, verbatim:
{question["question"]}

Expected answer:
{question["expected_answer"]}

Actual answer:
{answer.answer}

Actual cited pages:
{answer.cited_pages}
"""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_answer_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["judge"]["score"]) for row in rows]
    correct = [bool(row["judge"]["is_correct"]) for row in rows]
    answer_latencies = [float(row["answer_latency_ms"]) for row in rows]
    judge_latencies = [float(row["judge_latency_ms"]) for row in rows]
    return {
        "questions": len(rows),
        "answer_correct_rate": sum(correct) / len(correct) if correct else 0.0,
        "mean_judge_score": sum(scores) / len(scores) if scores else 0.0,
        "median_judge_score": statistics.median(scores) if scores else 0.0,
        "retrieval_any_page_hit_rate": _mean(row["retrieval"]["any_page_hit"] for row in rows),
        "retrieval_page_coverage": _mean(row["retrieval"]["page_coverage"] for row in rows),
        "retrieval_all_pages_hit_rate": _mean(row["retrieval"]["all_pages_hit"] for row in rows),
        "answer_latency_ms_p50": _median(answer_latencies),
        "judge_latency_ms_p50": _median(judge_latencies),
        "by_company": _group_summary(rows, "company"),
        "by_question_type": _group_summary(rows, "question_type"),
        "by_modality": _group_summary(rows, "required_modality"),
        "by_difficulty": _group_summary(rows, "difficulty"),
    }


def _group_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(field)), []).append(row)
    output = []
    for value, items in sorted(groups.items()):
        scores = [float(row["judge"]["score"]) for row in items]
        output.append(
            {
                field: value,
                "questions": len(items),
                "answer_correct_rate": _mean(row["judge"]["is_correct"] for row in items),
                "mean_judge_score": sum(scores) / len(scores) if scores else 0.0,
                "retrieval_any_page_hit_rate": _mean(
                    row["retrieval"]["any_page_hit"] for row in items
                ),
                "retrieval_page_coverage": _mean(
                    row["retrieval"]["page_coverage"] for row in items
                ),
            }
        )
    return output


def _mean(values) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0
