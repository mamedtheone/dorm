"""
app.py - CLI entrypoint for Dorm-Net.
"""

from __future__ import annotations

import argparse
import os

from modules.brain_module import RAGManager
from modules.persona_module import Message, PersonaManager
from modules.tutor_controller import TutorController


DB_PATH = os.getenv("DORM_NET_DB_PATH", "./dorm_net_db")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dorm-Net offline tutor")
    parser.add_argument("--question", help="Ask a single question and exit.")
    parser.add_argument("--ingest", nargs="*", help="PDF files to index before use.")
    parser.add_argument(
        "--persona",
        default="software",
        choices=["software", "mechanical", "electrical", "math", "eli12"],
    )
    parser.add_argument(
        "--mode",
        default="answer",
        choices=["answer", "concept_breakdown", "diagnosis", "notes"],
    )
    parser.add_argument("--model", default="mistral:latest")
    parser.add_argument("--subject", default="engineering")
    parser.add_argument("--concise", action="store_true")
    return parser


def print_sources(sources: list[dict]):
    if not sources:
        print("Sources: none")
        return
    print("Sources:")
    for source in sources:
        score = source.get("score")
        score_text = f" | score={score:.3f}" if isinstance(score, float) else ""
        print(f"- {source['source']} p.{source['page']}{score_text}")


def run_turn(
    controller: TutorController,
    history: list[Message],
    prompt: str,
    args,
):
    turn = controller.complete(
        question=prompt,
        history=history,
        model=args.model,
        persona_key=args.persona,
        step_by_step=not args.concise,
        subject_hint=args.subject,
        mode=args.mode,
        user_level=None,
        ocr_text=None,
        debug_mode=False,
        top_k=4,
    )
    if not turn.response.success:
        raise RuntimeError(turn.response.error or "Tutor turn failed.")

    history.append(Message(role="user", content=prompt))
    history.append(Message(role="assistant", content=turn.response.answer))

    print("\nTutor:\n")
    print(turn.response.answer)
    print()
    if turn.response.follow_up_questions:
        print("Follow-up Questions:")
        for follow_up in turn.response.follow_up_questions:
            print(f"- {follow_up}")
        print()
    print_sources(turn.rag_sources)


def main():
    args = build_parser().parse_args()
    rag = RAGManager(db_path=DB_PATH)
    persona = PersonaManager(ollama_url=OLLAMA_URL, default_model=args.model)
    controller = TutorController(rag, persona)

    if args.ingest:
        for pdf_path in args.ingest:
            report = rag.ingest_pdf(pdf_path)
            if report.success:
                status = "skipped" if report.skipped else "indexed"
                print(f"[{status}] {report.source} | pages={report.total_pages} chunks={report.total_chunks}")
            else:
                print(f"[error] {pdf_path} | {report.error}")

    if args.question:
        run_turn(controller, [], args.question, args)
        return

    history: list[Message] = []
    print("Dorm-Net CLI")
    print("Type 'exit' to quit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("Exiting.")
            return
        try:
            run_turn(controller, history, prompt, args)
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
