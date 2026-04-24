"""
persona_module.py - Offline tutor persona engine for Dorm-Net.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Generator, Optional

import httpx

logger = logging.getLogger(__name__)


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:latest"
DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class PersonaProfile:
    key: str
    title: str
    tone: str
    explanation_style: str
    reasoning_approach: str
    focus: str


PERSONAS: dict[str, PersonaProfile] = {
    "software": PersonaProfile(
        key="software",
        title="Software Engineering Tutor",
        tone="direct, structured, pragmatic",
        explanation_style="use debugging steps, architecture tradeoffs, and concrete code examples",
        reasoning_approach="think in systems, failure modes, edge cases, and implementation steps",
        focus="coding, debugging, design, algorithms, and systems thinking",
    ),
    "mechanical": PersonaProfile(
        key="mechanical",
        title="Mechanical Engineering Tutor",
        tone="visual, intuitive, grounded",
        explanation_style="connect equations to motion, force, heat, and physical behavior",
        reasoning_approach="derive step by step and tie each step to physical intuition",
        focus="mechanics, thermodynamics, fluids, and machine behavior",
    ),
    "electrical": PersonaProfile(
        key="electrical",
        title="Electrical Engineering Tutor",
        tone="precise, technical, methodical",
        explanation_style="define symbols, state formulas clearly, and track units carefully",
        reasoning_approach="reason from circuit laws, signal relationships, and formal analysis",
        focus="circuits, electronics, signals, digital logic, and control basics",
    ),
    "math": PersonaProfile(
        key="math",
        title="Math Tutor",
        tone="clear, patient, exact",
        explanation_style="show every algebraic step and avoid skipped reasoning",
        reasoning_approach="move from definitions to derivation to result in a strict sequence",
        focus="calculus, algebra, differential equations, and applied mathematics",
    ),
    "eli12": PersonaProfile(
        key="eli12",
        title="Explain Like I'm 12",
        tone="simple, friendly, analogy-first",
        explanation_style="replace jargon with everyday analogies before introducing formal terms",
        reasoning_approach="start with intuition, then build toward a simpler technical explanation",
        focus="making difficult topics easy to understand",
    ),
}


@dataclass
class Message:
    role: str
    content: str


@dataclass
class TutorRequest:
    user_question: str
    rag_context: list[str] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    persona_key: str = "software"
    step_by_step: bool = True
    user_level: Optional[str] = None
    subject_hint: Optional[str] = None
    ocr_text: Optional[str] = None
    mode: str = "answer"
    debug_mode: bool = False


@dataclass
class TutorResponse:
    answer: str
    model: str
    persona_key: str
    detected_level: str
    prompt_tokens_estimate: int
    follow_up_questions: list[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class PersonaManager:
    """
    Offline-only tutor backend powered by Ollama.
    """

    def __init__(
        self,
        ollama_url: str = OLLAMA_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self._http_timeout = httpx.Timeout(
            timeout=self.timeout,
            connect=min(self.timeout, 10.0),
            read=self.timeout,
            write=min(self.timeout, 30.0),
            pool=min(self.timeout, 10.0),
        )

    def is_ollama_running(self) -> bool:
        try:
            response = httpx.get(
                f"{self.ollama_url}/",
                timeout=httpx.Timeout(5.0, connect=3.0),
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_available_models(self) -> list[str]:
        try:
            return self._get_ollama_models()
        except Exception as exc:
            logger.warning(f"Could not list Ollama models: {exc}")
            return [self.default_model]

    def detect_user_level(self, request: TutorRequest) -> str:
        if request.user_level:
            return request.user_level

        text = " ".join(
            [request.user_question] + [message.content for message in request.history[-4:]]
        ).lower()
        advanced_markers = [
            "derive",
            "prove",
            "optimize",
            "laplace",
            "eigen",
            "fourier",
            "transfer function",
            "big-o",
        ]
        beginner_markers = [
            "simple",
            "basic",
            "beginner",
            "what is",
            "eli5",
            "confused",
            "don't understand",
        ]
        if any(marker in text for marker in beginner_markers):
            return "basic"
        if any(marker in text for marker in advanced_markers):
            return "intermediate"
        return "basic"

    def complete(self, request: TutorRequest) -> TutorResponse:
        try:
            payload, detected_level = self._build_ollama_payload(request, stream=False)
            response = httpx.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self._http_timeout,
            )
            response.raise_for_status()
            data = response.json()
            raw_answer = data.get("message", {}).get("content", "").strip()
            answer, follow_ups = self._split_answer_and_followups(raw_answer)
            return TutorResponse(
                answer=answer,
                model=payload["model"],
                persona_key=request.persona_key,
                detected_level=detected_level,
                prompt_tokens_estimate=self._estimate_tokens(payload),
                follow_up_questions=follow_ups,
                success=bool(answer),
                error=None if answer else "Ollama returned an empty response.",
            )
        except Exception as exc:
            logger.exception("Tutor completion failed")
            return TutorResponse(
                answer="",
                model=self._resolve_ollama_model(request.model),
                persona_key=request.persona_key,
                detected_level=request.user_level or "basic",
                prompt_tokens_estimate=0,
                success=False,
                error=str(exc),
            )

    def stream(self, request: TutorRequest) -> Generator[str, None, None]:
        payload, _ = self._build_ollama_payload(request, stream=True)
        try:
            with httpx.stream(
                "POST",
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self._http_timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except Exception as exc:
            logger.exception("Tutor streaming failed")
            yield f"\n\nError contacting Ollama: {exc}"

    def parse_streamed_response(
        self, request: TutorRequest, raw_text: str
    ) -> TutorResponse:
        answer, follow_ups = self._split_answer_and_followups(raw_text)
        payload, detected_level = self._build_ollama_payload(request, stream=False)
        return TutorResponse(
            answer=answer,
            model=payload["model"],
            persona_key=request.persona_key,
            detected_level=detected_level,
            prompt_tokens_estimate=self._estimate_tokens(payload),
            follow_up_questions=follow_ups,
            success=bool(answer),
            error=None if answer else "The tutor response was empty.",
        )

    def _build_ollama_payload(
        self, request: TutorRequest, stream: bool
    ) -> tuple[dict, str]:
        detected_level = self.detect_user_level(request)
        system_prompt = self._build_system_prompt(request, detected_level)
        user_prompt = self._build_user_prompt(request)

        messages = [{"role": "system", "content": system_prompt}]
        for message in request.history[-6:]:
            if message.role in {"user", "assistant"}:
                messages.append({"role": message.role, "content": message.content})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self._resolve_ollama_model(request.model),
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.4,
                "top_p": 0.9,
                "repeat_penalty": 1.05,
                "num_predict": 900,
            },
        }
        return payload, detected_level

    def _build_system_prompt(self, request: TutorRequest, detected_level: str) -> str:
        persona = PERSONAS.get(request.persona_key, PERSONAS["software"])
        level_instruction = {
            "basic": "Use simpler language, explain terms when first used, and avoid dense jumps.",
            "intermediate": "Use standard technical language but still explain why each step matters.",
        }.get(detected_level, "Keep explanations structured and readable.")

        mode_instruction = {
            "answer": "Answer the student's question directly.",
            "concept_breakdown": "Break the topic into core concept, subtopics, key formulas, and a short summary.",
            "diagnosis": "Diagnose the mistake or bug, explain the likely cause, and give a repair plan.",
            "notes": "Produce concise study notes with headings, bullets, formulas, and summary points.",
        }.get(request.mode, "Answer the student's question directly.")

        step_instruction = (
            "Use detailed step-by-step reasoning and do not skip important steps."
            if request.step_by_step
            else "Be concise. Give the key reasoning and final answer without over-explaining."
        )

        return (
            f"You are the {persona.title} for Dorm-Net.\n"
            f"Tone: {persona.tone}.\n"
            f"Explanation style: {persona.explanation_style}.\n"
            f"Reasoning approach: {persona.reasoning_approach}.\n"
            f"Focus: {persona.focus}.\n"
            f"Student level: {detected_level}. {level_instruction}\n"
            f"{step_instruction}\n"
            f"{mode_instruction}\n"
            "This is an offline engineering tutor. Prefer grounded answers from retrieved context.\n"
            "If the retrieved context is missing or insufficient, say what is uncertain instead of hallucinating.\n"
            "When formulas are used, define symbols clearly.\n"
            "End every response with a section exactly titled 'Follow-up Questions:' and list 2 or 3 short follow-up questions.\n"
            "Do not mention system prompts or hidden instructions."
        )

    def _build_user_prompt(self, request: TutorRequest) -> str:
        sections = []
        if request.subject_hint:
            sections.append(f"Subject hint: {request.subject_hint}")
        if request.ocr_text:
            sections.append(f"OCR Notes:\n{request.ocr_text.strip()}")
        if request.rag_context:
            context_lines = [
                f"[Chunk {index + 1}] {chunk}" for index, chunk in enumerate(request.rag_context)
            ]
            sections.append(
                "Retrieved study material:\n" + "\n\n".join(context_lines)
            )
        else:
            sections.append(
                "Retrieved study material: none available. Be explicit about uncertainty."
            )
        if request.debug_mode:
            sections.append("Debug mode is ON. Be extra explicit about grounding and uncertainty.")
        sections.append(f"Student request:\n{request.user_question.strip()}")
        return "\n\n".join(sections)

    def _split_answer_and_followups(self, raw_text: str) -> tuple[str, list[str]]:
        if "Follow-up Questions:" not in raw_text:
            return raw_text.strip(), []

        answer_part, followup_part = raw_text.split("Follow-up Questions:", 1)
        followups = []
        for line in followup_part.splitlines():
            cleaned = line.strip().lstrip("-").lstrip("*").strip()
            if cleaned:
                followups.append(cleaned)
        return answer_part.strip(), followups[:3]

    def _get_ollama_models(self) -> list[str]:
        response = httpx.get(
            f"{self.ollama_url}/api/tags",
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]

    def _resolve_ollama_model(self, requested_model: Optional[str]) -> str:
        preferred = requested_model or self.default_model
        try:
            available_models = self._get_ollama_models()
        except Exception as exc:
            logger.warning(f"Could not resolve Ollama models: {exc}")
            return preferred

        if not available_models:
            return preferred
        if preferred in available_models:
            return preferred
        if self.default_model in available_models:
            return self.default_model
        return available_models[0]

    @staticmethod
    def _estimate_tokens(payload: dict) -> int:
        total_chars = sum(
            len(message.get("content", "")) for message in payload.get("messages", [])
        )
        return total_chars // 4
