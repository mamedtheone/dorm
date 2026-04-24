"""
tutor_controller.py - Orchestrates Dorm-Net's offline tutor workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from modules.brain_module import QuizQuestion, RAGManager
from modules.persona_module import Message, PersonaManager, TutorRequest, TutorResponse

logger = logging.getLogger(__name__)


@dataclass
class TutorTurn:
    response: TutorResponse
    rag_sources: list[dict] = field(default_factory=list)
    raw_context: list[str] = field(default_factory=list)


class TutorController:
    def __init__(self, rag: RAGManager, persona: PersonaManager):
        self.rag = rag
        self.persona = persona

    def build_request(
        self,
        question: str,
        history: list[Message],
        model: str,
        persona_key: str,
        step_by_step: bool,
        subject_hint: str,
        mode: str = "answer",
        user_level: Optional[str] = None,
        ocr_text: Optional[str] = None,
        debug_mode: bool = False,
        top_k: int = 4,
    ) -> tuple[TutorRequest, list[dict], list[str]]:
        retrieval = self.rag.query(question, top_k=top_k) if self.rag.get_chunk_count() else None
        rag_context: list[str] = []
        rag_sources: list[dict] = []

        if retrieval and retrieval.success and retrieval.chunks:
            rag_context = [chunk.text for chunk in retrieval.chunks]
            rag_sources = []
            for index, chunk in enumerate(retrieval.chunks):
                score = retrieval.distances[index] if index < len(retrieval.distances) else None
                rag_sources.append(
                    {
                        "source": chunk.source,
                        "page": chunk.page,
                        "snippet": chunk.text,
                        "score": score,
                    }
                )

        request = TutorRequest(
            user_question=question,
            rag_context=rag_context,
            history=history[-8:],
            model=model,
            persona_key=persona_key,
            step_by_step=step_by_step,
            user_level=user_level,
            subject_hint=subject_hint,
            ocr_text=ocr_text,
            mode=mode,
            debug_mode=debug_mode,
        )
        return request, rag_sources, rag_context

    def complete(self, **kwargs) -> TutorTurn:
        request, rag_sources, rag_context = self.build_request(**kwargs)
        response = self.persona.complete(request)
        return TutorTurn(response=response, rag_sources=rag_sources, raw_context=rag_context)

    def parse_streamed_turn(self, request: TutorRequest, raw_text: str, rag_sources: list[dict], raw_context: list[str]) -> TutorTurn:
        response = self.persona.parse_streamed_response(request, raw_text)
        return TutorTurn(response=response, rag_sources=rag_sources, raw_context=raw_context)

    def generate_quiz(self, topic: str) -> list[QuizQuestion]:
        return self.rag.generate_quiz(topic, n_questions=5)
