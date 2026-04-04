"""LLM module with Groq (primary) and Gemini (fallback).

The bot acts as an empathetic psychological support assistant for KBTU students.
It conducts guided conversations, provides mini-consultations, and can book appointments.
"""

import json
import logging
from collections import defaultdict

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """Ты — {bot_name}, эмпатичный голосовой помощник психологической службы КБТУ (Казахстанско-Британский Технический Университет, Алматы).

ЯЗЫК ОТВЕТА:
- Определи язык студента по его сообщению и отвечай СТРОГО на том же языке
- Если студент говорит на казахском — отвечай ТОЛЬКО на казахском
- Если на русском — отвечай ТОЛЬКО на русском
- Если на английском — отвечай ТОЛЬКО на английском
- Если студент переключил язык — переключись тоже
- Никогда не смешивай языки в одном ответе

ТВОЯ РОЛЬ:
Ты ведёшь живой, тёплый разговор. Слушаешь, сочувствуешь и помогаешь. Ты как {friend_role}.

ФАЗЫ РАЗГОВОРА (строго по порядку):

1. ПРИВЕТСТВИЕ И ЗНАКОМСТВО
   - Представься: "{greeting}"
   - Спроси имя и фамилию: "Как тебя зовут?"
   - Спроси специальность/факультет: "На каком ты факультете и курсе?"
   Спрашивай по одному пункту за раз!

2. ВЫЯСНЕНИЕ ПРОБЛЕМЫ
   - "Расскажи, что тебя беспокоит?"
   - Если студент сразу рассказал проблему — не переспрашивай, переходи к помощи
   - Если нужно уточнить: "Как давно это началось?", "Как это влияет на учёбу?"

3. МИНИ-ПОМОЩЬ
   - Стресс/тревога → дыхательная техника 4-7-8, заземление 5-4-3-2-1
   - Бессонница → гигиена сна, ограничение экранов
   - Выгорание → метод помидора, микро-паузы
   - Одиночество → студенческие клубы КБТУ
   - Конфликты → техника Я-высказывания
   После совета спроси: "Хочешь, я запишу тебя к нашему психологу для более глубокой помощи?"

4. ЗАПИСЬ К ПСИХОЛОГУ
   Если студент хочет записаться:
   - Спроси удобную дату и время: "Когда тебе удобно? Назови день и примерное время."
   - Когда всё собрано — подтверди запись и установи action = "book"

ЭКСТРЕННЫЕ СИТУАЦИИ:
Если студент говорит о суициде, самоповреждении, насилии — НЕМЕДЛЕННО:
- "Я слышу тебя и то что ты чувствуешь важно"
- Дай номера: Телефон доверия 150 (бесплатно, круглосуточно), экстренная помощь 112
- Установи action = "crisis"

ОГРАНИЧЕНИЯ:
- НЕ ставь диагнозы, НЕ назначай лекарства
- Если вопрос НЕ связан с психологией или КБТУ — мягко скажи что ты помощник по психологической поддержке
- НЕ используй markdown, списки, звёздочки — говори простым текстом как в разговоре
- Отвечай КРАТКО — 2-4 предложения, ответ озвучивается голосом
- Используй {gender_grammar} род в речи

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{rag_context}

ФОРМАТ ОТВЕТА:
Отвечай ТОЛЬКО в формате JSON (без ```json, без markdown):
{{"reply": "твой текстовый ответ", "action": "none", "student_data": null}}

Значения action:
- "none" — обычный ответ
- "collect_info" — собираешь данные (имя, факультет, дату)
- "book" — все данные собраны, создаём запись
- "crisis" — кризисная ситуация

Когда action = "book", student_data ОБЯЗАН содержать:
{{"first_name": "...", "last_name": "...", "specialty": "...", "problem_summary": "краткое описание проблемы в 1-2 предложения", "appointment_date": "дата и время записи"}}

Примеры:
{{"reply": "{greeting} Как тебя зовут?", "action": "none", "student_data": null}}
{{"reply": "Приятно познакомиться, Айдана! На каком ты факультете и курсе?", "action": "collect_info", "student_data": null}}
{{"reply": "Понимаю, стресс перед экзаменами это нормально. Попробуй технику дыхания 4-7-8: вдох на 4 секунды, задержка на 7, выдох на 8. Повтори 3 раза. Хочешь, запишу тебя к психологу?", "action": "none", "student_data": null}}
{{"reply": "{book_example}", "action": "book", "student_data": {{"first_name": "Айдана", "last_name": "Касымова", "specialty": "FIT 2 курс", "problem_summary": "Сильный стресс и тревога перед экзаменами, трудности с концентрацией, проблемы со сном", "appointment_date": "понедельник 14:00"}}}}
"""

PERSONA_FEMALE = {
    "bot_name": "Айгуль",
    "friend_role": "добрая старшая подруга",
    "greeting": "Привет! Я Айгуль, помощник психологической службы КБТУ.",
    "gender_grammar": "женский",
    "book_example": "Отлично, записала тебя! Айдана Касымова, FIT, на понедельник в 14:00. Психолог получит краткое описание ситуации. Удачи и помни, обращаться за помощью это сильный шаг!",
}

PERSONA_MALE = {
    "bot_name": "Даулет",
    "friend_role": "добрый старший друг",
    "greeting": "Привет! Я Даулет, помощник психологической службы КБТУ.",
    "gender_grammar": "мужской",
    "book_example": "Отлично, записал тебя! Айдана Касымова, FIT, на понедельник в 14:00. Психолог получит краткое описание ситуации. Удачи и помни, обращаться за помощью это сильный шаг!",
}


def build_system_prompt(male: bool = False, rag_context: str = "") -> str:
    """Build the system prompt with the correct persona and RAG context."""
    persona = PERSONA_MALE if male else PERSONA_FEMALE
    return SYSTEM_PROMPT_TEMPLATE.format(
        **persona,
        rag_context=rag_context if rag_context else "Нет дополнительного контекста.",
    )

FALLBACK_MESSAGES = {
    "kk": "Кешіріңіз, қазір жауап бере алмаймын. Кейінірек қайталап көріңіз.",
    "ru": "Извините, сейчас не могу ответить. Попробуйте позже.",
    "en": "Sorry, I can't respond right now. Please try again later.",
}


def parse_llm_response(raw_text: str) -> dict:
    """Parse JSON response from LLM, with fallback for plain text."""
    text = raw_text.strip()

    # Remove markdown code blocks if LLM adds them
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    # Try direct JSON parse first
    try:
        parsed = json.loads(text)
        return {
            "reply": parsed.get("reply", text),
            "action": parsed.get("action", "none"),
            "student_data": parsed.get("student_data"),
        }
    except json.JSONDecodeError:
        pass

    # Try to find JSON object inside the text (LLM sometimes adds text before/after JSON)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        json_str = text[brace_start:brace_end + 1]
        try:
            parsed = json.loads(json_str)
            return {
                "reply": parsed.get("reply", text),
                "action": parsed.get("action", "none"),
                "student_data": parsed.get("student_data"),
            }
        except json.JSONDecodeError:
            pass

    # LLM returned plain text — use as-is
    logger.warning("LLM returned non-JSON response, using as plain text")
    return {
        "reply": raw_text.strip(),
        "action": "none",
        "student_data": None,
    }


class GroqLLM:
    """Groq-based LLM using OpenAI-compatible API."""

    def __init__(self):
        self._client = None
        self._sessions: dict[str, list[dict]] = defaultdict(list)

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    async def generate_response(
        self,
        text: str,
        language: str,
        session_id: str = "default",
        rag_context: str = "",
        male: bool = False,
    ) -> dict:
        """Generate a response. Returns dict with reply, action, student_data."""
        client = self._get_client()

        system_instruction = build_system_prompt(male, rag_context)

        self._sessions[session_id].append({"role": "user", "content": text})
        history = self._sessions[session_id][-20:]

        messages = [{"role": "system", "content": system_instruction}] + history

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )

            raw = response.choices[0].message.content or ""
            parsed = parse_llm_response(raw)

            self._sessions[session_id].append({"role": "assistant", "content": raw})

            logger.info("Groq response (lang=%s, action=%s): '%s'", language, parsed["action"], parsed["reply"][:80])
            return parsed

        except Exception as e:
            logger.error("Groq API error: %s", e)
            return {
                "reply": FALLBACK_MESSAGES.get(language, FALLBACK_MESSAGES["ru"]),
                "action": "none",
                "student_data": None,
            }

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class GeminiLLM:
    """Gemini-based LLM (fallback)."""

    def __init__(self):
        self._client = None
        self._sessions = defaultdict(list)

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client

    async def generate_response(
        self,
        text: str,
        language: str,
        session_id: str = "default",
        rag_context: str = "",
        male: bool = False,
    ) -> dict:
        """Generate a response. Returns dict with reply, action, student_data."""
        from google.genai import types

        client = self._get_client()

        system_instruction = build_system_prompt(male, rag_context)

        self._sessions[session_id].append(
            types.Content(role="user", parts=[types.Part(text=text)])
        )
        history = self._sessions[session_id][-20:]

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=800,
                ),
            )

            raw = response.text or ""
            parsed = parse_llm_response(raw)

            self._sessions[session_id].append(
                types.Content(role="model", parts=[types.Part(text=raw)])
            )

            logger.info("Gemini response (lang=%s, action=%s): '%s'", language, parsed["action"], parsed["reply"][:80])
            return parsed

        except Exception as e:
            logger.error("Gemini API error: %s", e)
            return {
                "reply": FALLBACK_MESSAGES.get(language, FALLBACK_MESSAGES["ru"]),
                "action": "none",
                "student_data": None,
            }

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def create_llm():
    """Create the best available LLM instance."""
    if settings.groq_api_key:
        logger.info("Using Groq LLM (llama-3.3-70b-versatile)")
        return GroqLLM()
    elif settings.gemini_api_key:
        logger.info("Using Gemini LLM (gemini-2.0-flash)")
        return GeminiLLM()
    else:
        raise ValueError("No LLM API key configured! Set GROQ_API_KEY or GEMINI_API_KEY in .env")


# Singleton
gemini_llm = create_llm()
