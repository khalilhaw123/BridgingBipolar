import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

import requests

from graphrag.ollama_env import ollama_generate_url

SYSTEM_PROMPT = """You are a warm, caring companion for people living with bipolar disorder — like a knowledgeable friend who truly understands what they're going through. Your role is to support, validate, and gently educate, not to lecture or diagnose.

TONE & PERSONALITY:

Speak like a calm, empathetic person — not a manual. Use "I", "we", "you" naturally.

Always acknowledge feelings before offering information. A person needs to feel heard first.

Normalize their experience: what they feel is real and makes sense.

Be warm but never over-enthusiastic. Avoid hollow phrases like "That's great!" or "Absolutely!".

Use simple, everyday words. Explain clinical terms in one plain sentence if used.

Keep answers short and digestible — under 180 words.

NATURAL CONVERSATION (No Repetition):

No Robotic Greetings (all languages): Never start your reply with a salutation plus the user’s name. Forbidden openings include:
French: “Bonjour [name]”, “Salut [name]”, “Bonsoir [name]”, “Coucou [name]” (with or without comma);
English: “Hello [name]”, “Hi [name]”, “Hey [name]”;
Arabic: “مرحبا” + name patterns — same rule.
The user’s first name appears in USER PROFILE only for rare, natural use inside the message — never in the first sentence, and never as a greeting formula.

Your first words must react to what they just said (feelings, facts, question) — not a hello.

Continuous Flow: Sound like a continuous chat. Refer back to what was just said without being repetitive.

Memory: Check the recent history. Do not ask a question you have already asked in this session. If you already know they are sleeping well, move on to a different topic like their creative energy or daily routine.

QUESTIONS — NO RE-ASKING (critical, all languages):

Read every User line in RECENT CONVERSATION before you write. If the user already answered a topic, you must NEVER ask that again — not in other words, not as a “check-in”, not paired with another question. This applies for French, English, and Arabic.

Never reuse the same question wording (or the same two-question combo) that already appears in a previous Assistant turn. Examples of overused stock questions to ban after the first use (same idea in any language):
French: « Comment s’annonce ta journée ? », « … aujourd’hui ? », « Peux-tu partager si ton état affecte tes relations professionnelles ou familiales ? »
English: “How is your day (today)?”, “How’s your day?”, “Does your mood affect your work or family (relationships)?”, “Could you share if … affects your work or family?”
Arabic: « كيف يومك؟ », « كيف حالك اليوم؟ », « هل يؤثر ذلك على عملك أو عائلتك؟ » — if any of these already appear in Assistant lines above, never ask them again, even with tiny wording changes.

Do not ask two questions in one reply. Prefer zero questions: react, validate, or give one concrete idea. At most one new question, only if you still lack information they have not already given.

If the user says you repeated yourself (e.g. “why do you keep asking that?” / « pourquoi tu répètes ? » / « ليش تعيد نفس السؤال؟ »), answer briefly: acknowledge, restate what you understood in new words, and do not repeat any prior question.

VARIETY — NO BOILERPLATE (critical):

Each reply must feel fresh and spoken, not copy-pasted. Read RECENT CONVERSATION: never repeat the same closing paragraph, the same stock sentence, or the same “see a professional / talk to someone” advice you already used in a previous Assistant turn — pick new words or omit that advice entirely.

Do not end most messages with generic “see a therapist / talk to a professional” fillers in any language (French examples: “Si besoin, il peut être utile de discuter…”, English: “You might want to speak with a mental health professional…”, Arabic: “يفضل أن تتحدث مع مختص…”) unless there is a real safety concern, acute confusion about treatment, or the user explicitly asks for medical referral. In normal supportive chat, stay with them in the moment instead of defaulting to that script.

Avoid hammering the same structures every time (“Il est important de…”, “C'est bien que…”, “N'oublie pas que…”). Vary how you open and how you transition. Short, concrete, spontaneous reactions beat a polished brochure tone.

INTERACTION STYLE:

Spontaneous Curiosity: Ask questions only when they feel like a natural "next step." If the user provides a lot of detail, you don't need to ask anything at all—just react.

Limit Questions: Default to no question. At most one short question per reply, only if it seeks genuinely new information and does not duplicate anything already asked or answered in RECENT CONVERSATION. Never use a questionnaire tone.

Human Feel: Reflect back what the user said in your own words. Show gentle curiosity without sounding like a clinical checklist — one angle per reply, not a full survey.

BALANCE & GROUNDING:

Offer thoughtful, experience-based support first — stay specific to what they just shared.

Only suggest professional help (clinicians/emergency services) if there is a safety risk, extreme confusion about treatment, or a direct medical question. Do not tack on a “consult a professional” paragraph out of habit.

Draw only from provided context. If information is missing, say it briefly once in varied wording — do not reuse the same “raise with your clinician” line every turn.

Never invent facts, dosages, or clinical guidance.

RESPONSE SHAPE:

Start with 1–2 sentences of validation that echo their message (no “Bonjour”, no “Hi”, no name-dropping at the start).

Offer 2–5 short points OR a short natural paragraph.

Match the user's language and formality level.

No markdown headings, no source tags. Light use of "-" for lists is fine.

SAFETY:

If the user mentions self-harm or immediate danger: Respond briefly with warmth, encourage contacting emergency services or their care team immediately, and do not provide long explanations
"""


def build_context_blocks(chunks: List[Dict]) -> str:
    """Number passages for the model's internal use; instruct model not to echo these labels."""
    lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        cid = c.get("chunk_id", f"p{i}")
        section = (c.get("metadata") or {}).get("section_title") or c.get("section_title") or ""
        sec = f" | section: {section}" if section else ""
        text = (c.get("text") or "").strip()
        lines.append(f"--- Passage {i}{sec} (internal_id={cid}) ---\n{text}")
    return "\n\n".join(lines) if lines else "(no passages retrieved)"


def _strip_chunk_tokens(text: str) -> str:
    """Remove any [chunk_...] the model may still emit."""
    text = re.sub(r"\[chunk_[^\]]+\]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[passage\s*\d+\]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _strip_leading_salutation_with_name(text: str, first_name: str) -> str:
    """Remove robotic 'Bonjour Prénom' / 'Hi Name' style openings the model still emits."""
    name = (first_name or "").strip()
    if len(name) < 2:
        return text
    # Match greeting + full stored name, or + first token only (e.g. profile "Youssef Benali" vs "Bonjour Youssef").
    name_first = name.split()[0] if name.split() else name
    variants = []
    for n in {name, name_first}:
        if len(n) >= 2 and n not in variants:
            variants.append(n)
    for n in variants:
        pat = (
            rf"(?i)^(?:bonjour|salut|bonsoir|coucou|hello|hi|hey|good\s+morning|good\s+evening)\s*,?\s*"
            rf"{re.escape(n)}\b\s*[,.!?…]*\s*"
        )
        prev = None
        while prev != text:
            prev = text
            text = re.sub(pat, "", text, count=1).lstrip()
    # Second pass: salutation alone at the very start (no name), one line only
    text = re.sub(
        r"(?i)^(?:bonjour|salut|bonsoir|coucou|hello|hi|hey)\s*[,.!?…]*\s+",
        "",
        text,
        count=1,
    ).lstrip()
    return text


def _history_signals_day_relations_already_covered(
    conversation_history: Optional[Sequence[Dict[str, str]]],
    current_query: str,
) -> bool:
    """True when user/history already touched topics the stock questionnaire asks about (FR/EN/AR)."""
    raw_parts: List[str] = [current_query or ""]
    for m in conversation_history or []:
        raw_parts.append(m.get("content") or "")
    raw_blob = " ".join(raw_parts)
    blob = raw_blob.lower()

    if re.search(
        r"familial|famille|relation|professionnel|travail|boulot|"
        r"n\s*['']?\s*affect|affecte|stable|ça\s+va|mon\s+état|motiv|fatigu|dorm|"
        r"family|relationships?|professional|work\s+life|tired|exhausted|sleepy|burnout|"
        r"عائلة|أسرة|عمل|متعب|تعب|نوم|يوم|مزاج|حال",
        raw_blob,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"comment\s+s['\u2019]?annonce\s+ta\s+journ", blob, re.IGNORECASE):
        return True
    if re.search(r"how\s+is\s+your\s+day|how's\s+your\s+day|how\s+was\s+your\s+day", blob, re.IGNORECASE):
        return True
    if re.search(r"does\s+your\s+(?:mood|state|mental\s+health)\s+affect", blob, re.IGNORECASE):
        return True
    if re.search(r"can\s+you\s+share\s+if\s+your", blob, re.IGNORECASE) and (
        "work" in blob or "family" in blob or "relationship" in blob
    ):
        return True
    # Arabic stock questions / topics (no full lowercasing of script needed for these substrings)
    if any(
        s in raw_blob
        for s in (
            "كيف يومك",
            "كيف حالك",
            "كيف حالك اليوم",
            "هل يؤثر",
            "عملك",
            "عائلتك",
            "العمل",
            "العائلة",
        )
    ):
        return True
    if "relations professionnelles" in blob or "familiales" in blob:
        return True
    return False


def _strip_stock_day_relations_questions(text: str) -> str:
    """Remove robotic questionnaire tails the model repeats (French, English, Arabic)."""
    if not text:
        return text
    # French
    _fr_day = (
        r"(?is)[\s.!?…]*Comment\s+s['\u2019]?annonce\s+ta\s+journ[ée]e"
        r"(?:\s+aujourd['\u2019]?hui)?\s*\?\s*"
    )
    _fr_share = (
        r"(?is)\s*Peux-tu\s+partager\s+si\s+ton\s+[ée]tat\s+affecte[^.?…\u061f]*"
        r"(?:professionnelles|familiales)\s*[\?\u061f]?\s*"
    )
    # English
    _en_day = r"(?is)[\s.!?…]*How\s+is\s+your\s+day(?:\s+today)?\s*[\?\u061f]?\s*"
    _en_day2 = r"(?is)[\s.!?…]*How['\u2019]?s\s+your\s+day(?:\s+today)?\s*[\?\u061f]?\s*"
    _en_day3 = r"(?is)[\s.!?…]*How\s+was\s+your\s+day(?:\s+today)?\s*[\?\u061f]?\s*"
    _en_affect = (
        r"(?is)[\s.!?…]*(?:Does|Do)\s+your\s+(?:mood|state|mental\s+health)\s+affect"
        r"[^.?…\u061f]{0,120}?[\?\u061f]\s*"
    )
    _en_share = (
        r"(?is)[\s.!?…]*Could\s+you\s+share\s+if\s+your\s+(?:mood|state)[^.?…\u061f]{0,120}?"
        r"(?:work|family|relationships)[^.?…\u061f]{0,40}?[\?\u061f]\s*"
    )
    # Arabic (Latin ? or Arabic ؟ U+061F)
    _ar_day = r"(?s)[\s.!?…\u061f]*كيف\s+يومك\s*[\?\u061f]+\s*"
    _ar_how = r"(?s)[\s.!?…\u061f]*كيف\s+حالك(?:\s+اليوم)?\s*[\?\u061f]+\s*"
    _ar_affect = r"(?s)[\s.!?…\u061f]*هل\s+يؤثر[^؟\?\n]{0,120}?[\?\u061f]+\s*"

    patterns = (_fr_day, _fr_share, _en_day, _en_day2, _en_day3, _en_affect, _en_share, _ar_day, _ar_how, _ar_affect)
    changed = True
    while changed:
        changed = False
        for pat in patterns:
            new_t = re.sub(pat, " ", text, count=1)
            if new_t != text:
                text = new_t
                changed = True
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([,.!?…\u061f])", r"\1", text)
    return text.strip()


def _rag_debug_log(system_prompt: str, user_prompt: str, chunks: List[Dict]) -> None:
    sep = "=" * 72
    print(f"\n{sep}\nRAG_DEBUG_PROMPT — full prompt to LLM\n{sep}", file=sys.stderr)
    print("\n--- SYSTEM ---\n", file=sys.stderr)
    print(system_prompt, file=sys.stderr)
    print("\n--- USER (instruction + context + question) ---\n", file=sys.stderr)
    print(user_prompt, file=sys.stderr)
    print(f"\n{sep}\nRAG_DEBUG_PROMPT — retrieved chunks (raw)\n{sep}", file=sys.stderr)
    for i, c in enumerate(chunks, start=1):
        rec = {
            "chunk_id": c.get("chunk_id"),
            "section_title": (c.get("metadata") or {}).get("section_title"),
            "text": c.get("text"),
        }
        print(json.dumps(rec, ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"{sep}\n", file=sys.stderr)


def build_recent_history_block(conversation_history: Sequence[Dict[str, str]]) -> str:
    if not conversation_history:
        return "(no recent conversation)"
    lines: List[str] = []
    for msg in conversation_history:
        role = (msg.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines) if lines else "(no recent conversation)"


def build_user_prompt(
    query: str,
    retrieval_result: Dict,
    user_profile: Optional[Dict[str, str]] = None,
    conversation_history: Optional[Sequence[Dict[str, str]]] = None,
    *,
    eval_grounding: bool = False,
) -> str:
    chunks = retrieval_result.get("chunks") or []
    context = build_context_blocks(chunks)
    recent = build_recent_history_block(conversation_history or [])
    lang = retrieval_result.get("lang", "en")
    intent = retrieval_result.get("intent", "general")
    display_name = ((user_profile or {}).get("name") or "").strip()
    if display_name:
        profile_line = (
            f"USER PROFILE: first_name={display_name} (for context only — do not greet them by name; "
            f"do not start with Bonjour/Salut/Hi/Hey/Hello or Arabic salutation + this name.)\n"
        )
    else:
        profile_line = "USER PROFILE: name=(unknown)\n"
    critical = ""
    if _history_signals_day_relations_already_covered(conversation_history, query):
        critical = (
            "\nCRITICAL (must follow): The conversation already covers how the day is going and/or "
            "family or work / mood impact — or the user answered those angles (in any language). "
            "You must NOT ask again the same stock questions: e.g. French « comment s'annonce ta journée » "
            "(with or without « aujourd'hui »), English “How is your day / How’s your day?”, Arabic « كيف يومك / كيف حالك », "
            "nor “does it affect work or family” style questions. Prefer zero questions; respond only to the latest User message.\n"
        )
    base = (
        f"{profile_line}"
        f"User language (ISO): {lang}\n"
        f"Detected intent: {intent}\n\n"
        "RECENT CONVERSATION (for continuity; prioritize current user message):\n\n"
        f"{recent}\n\n"
        f"CONTEXT (do not quote verbatim; do not repeat passage headers or internal_id in your answer):\n\n"
        f"{context}\n\n"
        f"{critical}"
        f"User message: {query}\n\n"
        "OUTPUT RULE: Your first sentence must respond to the substance of the User message — no salutation line, "
        "no 'Bonjour …', no 'Hi …', and do not use their first name at the beginning of the reply.\n"
        "Do not repeat phrases, stock closings, or any question you (Assistant) already asked above if the User already answered — sound spontaneous and specific to this turn.\n"
        "Remember: no chunk IDs or [brackets] in the reply."
    )
    if eval_grounding:
        base += (
            "\n\nGROUNDING (benchmark mode): Base every factual claim on the CONTEXT passages above. "
            "If they do not contain enough detail for a precise answer, say so briefly and avoid inventing "
            "clinical facts not supported by the context."
        )
    return base


def generate_with_ollama(
    query: str,
    retrieval_result: Dict,
    user_profile: Optional[Dict[str, str]] = None,
    conversation_history: Optional[Sequence[Dict[str, str]]] = None,
    model: str = "qwen2.5:3b-instruct",
    ollama_url: Optional[str] = None,
    strip_citations: bool = True,
    debug_prompt: Optional[bool] = None,
    *,
    eval_grounding: bool = False,
    temperature: Optional[float] = None,
) -> str:
    ollama_url = ollama_url or ollama_generate_url()
    if debug_prompt is None:
        debug_prompt = os.getenv("RAG_DEBUG_PROMPT", "").strip().lower() in ("1", "true", "yes")

    user_prompt = build_user_prompt(
        query,
        retrieval_result,
        user_profile=user_profile,
        conversation_history=conversation_history,
        eval_grounding=eval_grounding,
    )
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    if debug_prompt:
        _rag_debug_log(SYSTEM_PROMPT, user_prompt, retrieval_result.get("chunks") or [])

    gen_temp = 0.15 if temperature is None else float(temperature)
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": gen_temp,
            "num_predict": 400,
            "num_ctx": 4096,
        },
    }
    response = requests.post(ollama_url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    text = (data.get("response") or "").strip()
    if not text:
        return "I do not have enough evidence in the materials to answer safely. Please check with a mental health professional."
    if strip_citations:
        text = _strip_chunk_tokens(text)
    display_name = ((user_profile or {}).get("name") or "").strip()
    if display_name:
        text = _strip_leading_salutation_with_name(text, display_name)
    if _history_signals_day_relations_already_covered(conversation_history, query):
        text = _strip_stock_day_relations_questions(text)
    return text
