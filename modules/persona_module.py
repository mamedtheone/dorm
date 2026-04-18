"""
persona_module.py — Dorm-Net Persona Engine
=============================================
Manages the "AASTU Senior Tutor" persona and builds Chain-of-Thought
prompts that guide the LLM to explain engineering concepts step-by-step.

Supports:
  - CoT System Prompt construction
  - Context injection (RAG chunks + OCR text)
  - Streaming response handler
  - Conversation history formatting
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Generator
import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────── #
#  Constants                                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"   # swap to llama3.2:1b on low-RAM machines

AASTU_SYSTEM_PROMPT = """You are **Kebede**, a 4th-year Electrical Engineering student
at Addis Ababa Science and Technology University (AASTU). You are the smartest,
most patient senior tutor in the department.

Your teaching style:
- You explain everything using the **Chain-of-Thought** method: think out loud,
  break the problem into numbered steps, and reason through each step before
  giving the final answer.
- You speak in clear, friendly English — occasionally mixing in Amharic
  encouragements (e.g., "እናንተ ትችላላችሁ!" = "You can do it!").
- You relate abstract theory to real-world Ethiopian engineering contexts
  (power grids, hydro dams, telecom, construction).
- You always check for misunderstandings by asking "Does that make sense so far?"
  at the end of a complex explanation.
- You NEVER give the final answer before showing your reasoning.
- When you use a formula, you define every symbol.
- If the student provides a photo of their notes (OCR text), acknowledge it
  and correct any errors you spot.

Formatting rules:
- Use numbered steps for derivations.
- Use **bold** for key terms on first use.
- Use LaTeX-style inline math between $...$ symbols.
- Keep each step ≤ 3 sentences so it's easy to follow on a phone screen.
"""

SUBJECTS = {
    "mathematics": "Calculus, Linear Algebra, Differential Equations",
    "circuits": "Circuit Analysis, Electronics, Signals & Systems",
    "programming": "C, Python, Data Structures, Algorithms",
    "physics": "Mechanics, Electromagnetism, Thermodynamics",
    "engineering": "Thermodynamics, Fluid Mechanics, Structural Analysis",
    "general": "Any AASTU undergraduate topic",
}


# ─────────────────────────────────────────────────────────────────────────── #
#  Data structures                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class Message:
    role: str   # "user" | "assistant" | "system"
    content: str


@dataclass
class TutorRequest:
    user_question: str
    rag_context: list[str] = field(default_factory=list)
    ocr_text: Optional[str] = None
    history: list[Message] = field(default_factory=list)
    subject_hint: Optional[str] = None
    model: str = DEFAULT_MODEL


@dataclass
class TutorResponse:
    answer: str
    model: str
    prompt_tokens_estimate: int
    success: bool
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────── #
#  Persona Manager                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class PersonaManager:
    """
    Builds and fires prompts to the local Ollama LLM.

    Two modes:
      complete()  → returns full string (good for non-streaming UIs)
      stream()    → yields token strings for real-time display
    """

    def __init__(
        self,
        ollama_url: str = OLLAMA_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    # ── Public API ───────────────────────────────────────────────────── #

    def complete(self, request: TutorRequest) -> TutorResponse:
        """
        Single-shot completion. Returns the full answer as a string.
        Use this for simple queries or when streaming is not needed.
        """
        payload = self._build_payload(request, stream=False)
        try:
            resp = httpx.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("message", {}).get("content", "")
            return TutorResponse(
                answer=answer,
                model=request.model,
                prompt_tokens_estimate=self._estimate_tokens(payload),
                success=True,
            )
        except Exception as exc:
            logger.exception("Ollama completion failed")
            return TutorResponse(
                answer="",
                model=request.model,
                prompt_tokens_estimate=0,
                success=False,
                error=str(exc),
            )

    def stream(self, request: TutorRequest) -> Generator[str, None, None]:
        """
        Streaming completion. Yields text tokens as they arrive.
        Use with Streamlit's st.write_stream() for real-time output.

        Yields:
            str — token chunk (may contain spaces/newlines)
        """
        payload = self._build_payload(request, stream=True)
        try:
            with httpx.stream(
                "POST",
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json as _json
                    try:
                        chunk = _json.loads(line)
                    except Exception:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except Exception as exc:
            logger.exception("Ollama streaming failed")
            yield f"\n\n⚠️ Error contacting Ollama: {exc}"

    def build_cot_prompt(self, question: str, subject: Optional[str] = None) -> str:
        """
        Standalone utility: wraps a raw question in a CoT instruction frame.
        Useful for testing prompts outside the full pipeline.
        """
        subject_desc = SUBJECTS.get(subject or "general", SUBJECTS["general"])
        return (
            f"Subject area: {subject_desc}\n\n"
            f"Student question: {question}\n\n"
            "Please answer using the Chain-of-Thought method:\n"
            "1. Restate the problem in your own words.\n"
            "2. Identify the key concepts and formulas needed.\n"
            "3. Solve step-by-step, showing all working.\n"
            "4. State the final answer clearly.\n"
            "5. Give one real-world example from an Ethiopian engineering context.\n"
        )

    def list_available_models(self) -> list[str]:
        """Returns models currently installed in the local Ollama instance."""
        try:
            resp = httpx.get(f"{self.ollama_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        except Exception as exc:
            logger.warning(f"Could not list Ollama models: {exc}")
            return []

    def is_ollama_running(self) -> bool:
        try:
            resp = httpx.get(f"{self.ollama_url}/", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Internal helpers ─────────────────────────────────────────────── #

    def _build_payload(self, request: TutorRequest, stream: bool) -> dict:
        """
        Constructs the full Ollama /api/chat payload.

        Message order:
          [system]  → persona + CoT instruction
          [history] → prior conversation turns
          [user]    → current question + RAG context + OCR text
        """
        system_content = AASTU_SYSTEM_PROMPT
        if request.subject_hint:
            system_content += (
                f"\n\nThe student is studying: "
                f"{SUBJECTS.get(request.subject_hint, request.subject_hint)}"
            )

        messages = [{"role": "system", "content": system_content}]

        # Inject conversation history
        for msg in request.history[-10:]:  # keep last 10 turns to stay within context
            messages.append({"role": msg.role, "content": msg.content})

        # Build the current user turn
        user_content_parts = []

        if request.ocr_text and request.ocr_text.strip():
            user_content_parts.append(
                "📷 **I photographed my handwritten notes. Here is the OCR text:**\n"
                f"```\n{request.ocr_text.strip()}\n```\n"
            )

        if request.rag_context:
            ctx_block = "\n\n---\n".join(
                f"[Source {i+1}]: {chunk}"
                for i, chunk in enumerate(request.rag_context)
            )
            user_content_parts.append(
                f"📚 **Relevant textbook excerpts for context:**\n{ctx_block}\n"
            )

        cot_question = self.build_cot_prompt(
            request.user_question, request.subject_hint
        )
        user_content_parts.append(cot_question)

        user_content = "\n\n".join(user_content_parts)
        messages.append({"role": "user", "content": user_content})

        return {
            "model": request.model or self.default_model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }

    @staticmethod
    def _estimate_tokens(payload: dict) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(
            len(m.get("content", "")) for m in payload.get("messages", [])
        )
        return total_chars // 4
