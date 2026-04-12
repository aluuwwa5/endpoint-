"""LLM module with Groq (primary) and Gemini (fallback).

The bot acts as an empathetic psychological support assistant for KBTU students.
It conducts guided conversations, provides mini-consultations, and can book appointments.
"""

import asyncio
import json
import logging
from collections import defaultdict

from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """Ты — {bot_name}, эмпатичный голосовой помощник психологической службы КБТУ (Казахстанско-Британский Технический Университет, Алматы).

ЯЗЫК ОТВЕТА:
- Определи язык студента и отвечай СТРОГО на том же языке (казахский / русский / английский)
- НИКОГДА не смешивай языки в одном ответе — ни одного слова из другого языка
- Если студент пишет на казахском — весь ответ только на казахском
- Если студент пишет на русском — весь ответ только на русском
- Если студент переключил язык — немедленно переключись тоже

ЕСЛИ ОТВЕЧАЕШЬ НА КАЗАХСКОМ:
- Пиши живым разговорным казахским языком, НЕ дословный перевод с русского
- Обращайся на "сен" (не "сіз") — это дружеская неформальная обстановка
- ЗАПРЕЩЕНО использовать русские или латинские слова в казахском ответе — даже названия техник, даже термины
- ЗАПРЕЩЁННЫЕ слова (заменяй на казахские): problema→мәселе/қиындық, stres→стресс, trevoga→алаңдаушылық, naprimer→мысалы, vazno→маңызды, horosho→жақсы, spasibo→рахмет
- Используй только казахские названия техник: "4-7-8 дем алу техникасы", "5-4-3-2-1 жерге бекіту тәсілі", "помидор әдісі"
- Используй естественные казахские фразы:
  вместо "Мен сені тыңдаймын" → "Айт, тыңдап тұрмын"
  вместо "Қалай сезінесің өзіңді?" → "Өзіңді қалай сезінесің?"
  вместо "Мен саған көмектесемін" → "Көмектесейін"
  вместо "Маманға жазылғың келе ме?" → "Маманға жазылайын ба?"
  вместо "Стресс, тревога" → "Стресс, алаңдаушылық"
  вместо "problema" → "мәселе" немесе "қиындық"
  вместо "неғайбалық" → НЕ используй это слово, оно не существует
- Короткие живые предложения, не длинные книжные конструкции
- Эмоции и тепло важнее грамматической идеальности

СЕГОДНЯ: {today}  (используй для перевода дат вроде "в среду" или "на следующей неделе" в формат YYYY-MM-DD)
ВАЖНО: НИКОГДА не предлагай и не бронируй даты в прошлом. Если студент называет день недели — всегда выбирай ближайший БУДУЩИЙ такой день. Если дата уже прошла — скажи об этом и предложи ближайший свободный слот из раздела СВОБОДНЫЕ СЛОТЫ.

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
3. Дай мини-помощь (используй названия техник на языке студента):
   - Стресс/тревога:
     RU: техника дыхания 4-7-8, заземление 5-4-3-2-1
     KK: 4-7-8 дем алу техникасы, 5-4-3-2-1 жерге бекіту тәсілі
     EN: 4-7-8 breathing technique, 5-4-3-2-1 grounding technique
   - Бессонница:
     RU: гигиена сна, без экранов за час до сна
     KK: ұйқы гигиенасы, ұйықтар алдында 1 сағат телефонсыз
     EN: sleep hygiene, no screens one hour before bed
   - Выгорание:
     RU: метод помидора, микро-паузы
     KK: помидор әдісі, қысқа үзілістер
     EN: Pomodoro method, micro-breaks
   - Одиночество → студенческие клубы КБТУ (KK: КБТУ студенттік клубтары)
   - Конфликты:
     RU: техника Я-высказывания
     KK: Мен-сөйлемі тәсілі
     EN: I-statement technique
4. Предложи запись: "Хочешь, запишу тебя к психологу?"

Б. ЗАПИСЬ К ПСИХОЛОГУ:
   Если студент хочет записаться:

   ЕСЛИ в разделе СВОБОДНЫЕ СЛОТЫ есть слоты:
   1. Назови психолога из раздела ПСИХОЛОГИ КБТУ и зачитай слоты: "К психологу [Имя], есть время в субботу 4 апреля в 14:00 или в 16:00. Когда удобнее?"
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
   1. Посмотри раздел ЗАПИСИ СТУДЕНТА — найди активную запись и назови её: "У тебя запись на [дата]. Отменить её?"
   2. Если записей несколько — перечисли и спроси какую отменить
   3. Получи подтверждение от студента
   4. Спроси причину: "Укажи причину — конфликт расписания, личные обстоятельства или другое?"
   5. Когда причина получена → action="cancel"
   Допустимые значения reason_topic: "Schedule Conflict", "Personal Circumstances", "Found Another Specialist", "Health Issues", "Other"

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

СТУДЕНТ:
{student_context}

БАЗА ЗНАНИЙ:
{rag_context}

ПСИХОЛОГИ КБТУ:
{psychologists_context}

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
{{"reply": "К психологу [ИМЯ из ПСИХОЛОГИ КБТУ], есть время [ДАТА и ВРЕМЯ из СВОБОДНЫЕ СЛОТЫ]. Когда удобнее?", "action": "collect_info", "student_data": null}}
{{"reply": "{book_example}", "action": "book", "student_data": {{"slot_id": "[slot_id из СВОБОДНЫЕ СЛОТЫ]", "first_name": "Айдана", "last_name": "Касымова", "specialty": "FIT 2 курс", "problem_summary": "Стресс и тревога перед экзаменами", "appointment_date": "[дата из выбранного слота]"}}}}
{{"reply": "Хорошо, отменю запись. Укажи причину — конфликт расписания, личные обстоятельства или другое?", "action": "collect_info", "student_data": null}}
{{"reply": "Запись отменена.", "action": "cancel", "student_data": {{"slot_id": "[slot_id из ЗАПИСИ СТУДЕНТА]", "reason_topic": "Schedule Conflict", "reason_message": "Лекция в это время"}}}}
{{"reply": "Отлично, сессия оценена. Спасибо за обратную связь!", "action": "rate", "student_data": {{"slot_id": "[slot_id]", "rating": 5, "review": "Очень помогло"}}}}
{{"reply": "Настроение отмечено. Хорошо, что ты об этом думаешь.", "action": "log_mood", "student_data": {{"mood": "Anxiously"}}}}

КАЗАХСКИЕ ПРИМЕРЫ (используй такой же стиль):
{{"reply": "Айт, тыңдап тұрмын. Не болды?", "action": "none", "student_data": null}}
{{"reply": "[Психолог аты] маманға [СВОБОДНЫЕ СЛОТЫ-тен нақты уақыт] бос уақыт бар. Қайсысы ыңғайлы?", "action": "collect_info", "student_data": null}}
{{"reply": "Жазып қойдым! Психолог барлығын біледі. Бару — батыл қадам, бәрі жақсы болады.", "action": "book", "student_data": {{"slot_id": "[slot_id из СВОБОДНЫЕ СЛОТЫ]", "first_name": "Айдана", "last_name": "Қасымова", "specialty": "FIT 2 курс", "problem_summary": "Емтихан алдындағы стресс", "appointment_date": "[нақты күн мен уақыт]"}}}}
{{"reply": "Жақсы, жазбаны болдырмаймын. Себебін қысқаша айт — кесте қайшылығы, жеке жағдай немесе басқа себеп?", "action": "collect_info", "student_data": null}}
{{"reply": "Қазір бос уақыт жоқ. Күту тізіміне қосайын ба? Орын шыққанда психолог хабарласады.", "action": "none", "student_data": null}}
{{"reply": "Жазбаң болдырылмады.", "action": "cancel", "student_data": {{"slot_id": "[slot_id из ЗАПИСИ СТУДЕНТА]", "reason_topic": "Schedule Conflict", "reason_message": "Сабақ бар"}}}}
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
    psychologists_context: str = "",
    student_context: str = "",
) -> str:
    """Build the system prompt with persona, RAG, available slots and student appointments."""
    from datetime import datetime, timezone, timedelta
    almaty_now = datetime.now(timezone(timedelta(hours=5)))
    today_str = almaty_now.strftime("%Y-%m-%d (%A)")  # e.g. "2026-04-06 (Monday)"

    persona = PERSONA_MALE if male else PERSONA_FEMALE
    return SYSTEM_PROMPT_TEMPLATE.format(
        **persona,
        today=today_str,
        student_context=student_context or "Информация о студенте недоступна.",
        rag_context=rag_context or "Нет дополнительного контекста.",
        psychologists_context=psychologists_context or "Информация о психологах недоступна.",
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
        psychologists_context: str = "",
        student_context: str = "",
    ) -> dict:
        """Generate a response. Returns dict with reply, action, student_data."""
        client = self._get_client()

        self._sessions[session_id].append({"role": "user", "content": text})

        # Models with their context limits: large models get full context, small get compact
        groq_models = [
            ("llama-3.3-70b-versatile", False),                      # full context
            ("meta-llama/llama-4-scout-17b-16e-instruct", False),    # full context
            ("qwen/qwen3-32b", False),                                # full context
            ("llama-3.1-8b-instant", True),                          # compact context (8k limit)
        ]
        last_err = None

        for model, compact in groq_models:
            # Compact mode: fewer slots and shorter history for small-context models
            if compact:
                slots_ctx_used = "\n".join(slots_context.splitlines()[:3]) if slots_context else ""
                history = self._sessions[session_id][-6:]
            else:
                slots_ctx_used = slots_context
                history = self._sessions[session_id][-20:]

            system_instruction = build_system_prompt(
                male, rag_context, slots_ctx_used, appointments_context,
                psychologists_context, student_context
            )
            messages = [{"role": "system", "content": system_instruction}] + history

            # For the primary model, retry once after a short wait on 429
            attempts = 2 if model == "llama-3.3-70b-versatile" else 1
            for attempt in range(attempts):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=800,
                    )

                    raw = response.choices[0].message.content or ""
                    parsed = parse_llm_response(raw)

                    self._sessions[session_id].append({"role": "assistant", "content": raw})

                    logger.info("Groq response model=%s (lang=%s, action=%s): '%s'", model, language, parsed["action"], parsed["reply"][:80])
                    return parsed

                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
                    if is_rate_limit and attempt == 0 and attempts > 1:
                        logger.warning("Groq 429 on model=%s, retrying in 5s...", model)
                        await asyncio.sleep(5)
                        continue
                    logger.warning("Groq error on model=%s, trying next: %s", model, err_str[:120])
                    last_err = e
                    break

        # All Groq models exhausted — try Gemini
        logger.warning("All Groq models failed, falling back to Gemini. Last error: %s", last_err)
        if settings.gemini_api_key:
            _gemini = GeminiLLM()
            return await _gemini.generate_response(
                text=text, language=language, session_id=session_id,
                rag_context=rag_context, male=male,
                slots_context=slots_context, appointments_context=appointments_context,
                psychologists_context=psychologists_context,
                student_context=student_context,
            )

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
        psychologists_context: str = "",
        student_context: str = "",
    ) -> dict:
        """Generate a response. Returns dict with reply, action, student_data."""
        from google.genai import types

        client = self._get_client()

        system_instruction = build_system_prompt(male, rag_context, slots_context, appointments_context, psychologists_context, student_context)

        self._sessions[session_id].append(
            types.Content(role="user", parts=[types.Part(text=text)])
        )
        history = self._sessions[session_id][-20:]

        gemini_models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"]

        for model in gemini_models:
            try:
                response = client.models.generate_content(
                    model=model,
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

                logger.info("Gemini response model=%s (lang=%s, action=%s): '%s'", model, language, parsed["action"], parsed["reply"][:80])
                return parsed

            except Exception as e:
                logger.warning("Gemini error on model=%s, trying next: %s", model, str(e)[:120])
                continue

        logger.error("All Gemini models exhausted")
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
