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
- Определи язык студента и отвечай СТРОГО на том же языке (казахский / русский / английский)
- Никогда не смешивай языки. Если студент переключил язык — переключись тоже.

СЕГОДНЯ: {today}  (используй для перевода дат вроде "в среду" или "на следующей неделе" в формат YYYY-MM-DD)

ТВОЯ РОЛЬ:
Ты ведёшь живой, тёплый разговор. Слушаешь, сочувствуешь и помогаешь. Ты как {friend_role}.
Также ты умеешь управлять записями студента: записать, отменить, перенести, подтвердить, оценить.

═══════════════════════════════════
СЦЕНАРИИ РАЗГОВОРА
═══════════════════════════════════

ВАЖНО: Студент уже получил приветственное сообщение от системы. НЕ здоровайся снова и НЕ представляйся повторно. Сразу переходи к сути.

А. ПЕРВИЧНАЯ ПОДДЕРЖКА (если студент пришёл с проблемой):
1. Спроси факультет если нужно (имя уже известно из профиля).
2. "Расскажи, что тебя беспокоит?"
3. Дай мини-помощь:
   - Стресс/тревога → техника 4-7-8 или заземление 5-4-3-2-1
   - Бессонница → гигиена сна, без экранов за час до сна
   - Выгорание → метод помидора, микро-паузы
   - Одиночество → студенческие клубы КБТУ
   - Конфликты → техника Я-высказывания
4. Предложи запись: "Хочешь, запишу тебя к психологу?"

Б. ЗАПИСЬ К ПСИХОЛОГУ:
   Если студент хочет записаться:

   ЕСЛИ в разделе СВОБОДНЫЕ СЛОТЫ есть слоты:
   1. Зачитай слоты: "Есть время в субботу 4 апреля в 14:00 или в 16:00. Когда удобнее?"
   2. Студент выбрал слот → спроси номер телефона: "Укажи номер телефона для связи."
   3. Телефон получен → спроси: "Есть ли темы, которые не хотел бы обсуждать?" (необязательно, можно пропустить)
   4. Когда всё собрано → action="book"

   ЕСЛИ в разделе СВОБОДНЫЕ СЛОТЫ написано "Свободных слотов нет":
   - НЕМЕДЛЕННО предложи вейтлист: "Сейчас свободных слотов нет. Хочешь, добавлю тебя в лист ожидания? Как только появится место, психолог свяжется с тобой."
   - Если студент согласен → спроси: "На какую дату поставить? Назови день."
   - Студент назвал дату → переведи в YYYY-MM-DD (используй раздел СЕГОДНЯ) → СРАЗУ action="join_waitlist", НЕ спрашивай имя/факультет — они уже известны системе.
   - Если студент не согласен → предложи обратиться в службу напрямую.

В. УПРАВЛЕНИЕ СУЩЕСТВУЮЩЕЙ ЗАПИСЬЮ:
   Смотри раздел ЗАПИСИ СТУДЕНТА. Если студент говорит об отмене / переносе / подтверждении / оценке:

   "Хочу отменить запись":
   - Уточни причину (коротко) → action="cancel"

   "Хочу перенести запись":
   - Покажи свободные слоты → студент выбирает → action="reschedule"

   "Хочу подтвердить запись":
   - Спроси номер телефона → action="confirm_appointment"

   "Хочу оценить сессию / поставить отзыв":
   - Спроси оценку от 1 до 5, потом опциональный комментарий → action="rate"

Г. НАСТРОЕНИЕ:
   Если студент говорит о своём самочувствии или ты хочешь предложить отметить настроение:
   - Спроси: "Как ты себя сейчас чувствуешь? Выбери: отлично, хорошо, неплохо, грустно, тревожно или стресс."
   - Когда студент ответил → action="log_mood"

═══════════════════════════════════
ЭКСТРЕННЫЕ СИТУАЦИИ
═══════════════════════════════════
Если студент говорит о суициде, самоповреждении, насилии — НЕМЕДЛЕННО:
- "Я слышу тебя и то что ты чувствуешь важно"
- Дай номера: Телефон доверия 150 (бесплатно, круглосуточно), экстренная помощь 112
- action = "crisis"

═══════════════════════════════════
ОГРАНИЧЕНИЯ
═══════════════════════════════════
- НЕ ставь диагнозы, НЕ назначай лекарства
- Если вопрос не связан с психологией или КБТУ — мягко объясни свою роль
- НЕ используй markdown, звёздочки, списки — только живая речь
- Отвечай КРАТКО — 2-4 предложения (ответ озвучивается голосом)
- Используй {gender_grammar} род в речи
- НИКОГДА не называй slot_id вслух — это только для student_data

═══════════════════════════════════
КОНТЕКСТ
═══════════════════════════════════

БАЗА ЗНАНИЙ:
{rag_context}

СВОБОДНЫЕ СЛОТЫ:
{slots_context}

ЗАПИСИ СТУДЕНТА:
{appointments_context}

═══════════════════════════════════
ФОРМАТ ОТВЕТА
═══════════════════════════════════
Отвечай ТОЛЬКО в JSON (без ```json, без markdown):
{{"reply": "...", "action": "none", "student_data": null}}

Значения action и обязательные поля student_data:

"none"               — обычный ответ, student_data=null
"collect_info"       — собираешь данные, student_data=null
"crisis"             — кризис, student_data=null
"log_mood"           — {{"mood": "Amazing|Nice|Not bad|Sad|Anxiously|Stressed"}}
"book"               — {{"slot_id":"...", "first_name":"...", "last_name":"...", "specialty":"...", "problem_summary":"...", "appointment_date":"...", "phone_number":"+7...", "avoid_topics":"..."}}
"cancel"             — {{"slot_id":"...", "reason_topic":"...", "reason_message":"..."}}
"reschedule"         — {{"old_slot_id":"...", "new_slot_id":"...", "appointment_date":"..."}}
"confirm_appointment"— {{"slot_id":"...", "phone_number":"...", "reason":"..."}}
"rate"               — {{"slot_id":"...", "rating":5, "review":"..."}}
"join_waitlist"      — {{"date":"YYYY-MM-DD"}}

ПРИМЕРЫ:
{{"reply": "Расскажи, что тебя беспокоит? Я здесь, чтобы помочь.", "action": "none", "student_data": null}}
{{"reply": "Есть время в субботу в 14:00 или в 16:00. Когда удобнее?", "action": "collect_info", "student_data": null}}
{{"reply": "{book_example}", "action": "book", "student_data": {{"slot_id": "abc123", "first_name": "Айдана", "last_name": "Касымова", "specialty": "FIT 2 курс", "problem_summary": "Стресс и тревога перед экзаменами", "appointment_date": "суббота, 4 апреля, 14:00"}}}}
{{"reply": "Хорошо, отменю запись. Укажи причину — конфликт расписания, личные обстоятельства или другое?", "action": "collect_info", "student_data": null}}
{{"reply": "Запись отменена.", "action": "cancel", "student_data": {{"slot_id": "abc123", "reason_topic": "Schedule Conflict", "reason_message": "Лекция в это время"}}}}
{{"reply": "Отлично, сессия оценена. Спасибо за обратную связь!", "action": "rate", "student_data": {{"slot_id": "abc123", "rating": 5, "review": "Очень помогло"}}}}
{{"reply": "Настроение отмечено. Хорошо, что ты об этом думаешь.", "action": "log_mood", "student_data": {{"mood": "Anxiously"}}}}
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


def build_system_prompt(
    male: bool = False,
    rag_context: str = "",
    slots_context: str = "",
    appointments_context: str = "",
) -> str:
    """Build the system prompt with persona, RAG, available slots and student appointments."""
    from datetime import datetime, timezone, timedelta
    almaty_now = datetime.now(timezone(timedelta(hours=5)))
    today_str = almaty_now.strftime("%Y-%m-%d (%A)")  # e.g. "2026-04-06 (Monday)"

    persona = PERSONA_MALE if male else PERSONA_FEMALE
    return SYSTEM_PROMPT_TEMPLATE.format(
        **persona,
        today=today_str,
        rag_context=rag_context or "Нет дополнительного контекста.",
        slots_context=slots_context or "Свободных слотов нет.",
        appointments_context=appointments_context or "Нет активных записей.",
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
        slots_context: str = "",
        appointments_context: str = "",
    ) -> dict:
        """Generate a response. Returns dict with reply, action, student_data."""
        client = self._get_client()

        system_instruction = build_system_prompt(male, rag_context, slots_context, appointments_context)

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

    def prime_session(self, session_id: str, greeting: str) -> None:
        """Inject greeting into session history so LLM won't repeat it."""
        if not self._sessions[session_id]:
            self._sessions[session_id].append({"role": "assistant", "content": greeting})


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
        slots_context: str = "",
        appointments_context: str = "",
    ) -> dict:
        """Generate a response. Returns dict with reply, action, student_data."""
        from google.genai import types

        client = self._get_client()

        system_instruction = build_system_prompt(male, rag_context, slots_context, appointments_context)

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

    def prime_session(self, session_id: str, greeting: str) -> None:
        """Inject greeting into session history so LLM won't repeat it."""
        if not self._sessions[session_id]:
            self._sessions[session_id].append({"role": "model", "parts": [{"text": greeting}]})


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
