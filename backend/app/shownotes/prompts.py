"""Show notes prompts, English and Spanish (docs/decisions/0011 §2).

`map_messages` asks for bullet notes on one chunk; `reduce_messages`
asks for the final JSON from those notes (or from the whole transcript
when it is short). Bump `PROMPT_VERSION` whenever a prompt changes: it
ends up in `show_notes_drafts.llm_model_used`.
"""

PROMPT_VERSION = "shownotes-v1"

# JSON schema for the reduce step's grammar-constrained output. String
# lengths aren't constrained here (that blows up the grammar); the
# validator cuts them instead.
DRAFT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title_suggestions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 15},
        "summary": {"type": "string"},
    },
    "required": ["title_suggestions", "description", "tags", "summary"],
    "additionalProperties": False,
}

_MAP = {
    "en": (
        "You take notes on a podcast episode.",
        "Here is part of the episode's transcript{about}:\n\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Summarize what is said in 3 to 5 short bullet points, in English. Keep names, numbers "
        "and concrete claims. Use only what the transcript says. Reply with the bullet points only.",
        ' (chapter "{title}")',
    ),
    "es": (
        "Tomas notas sobre un episodio de podcast.",
        "Esta es una parte de la transcripción del episodio{about}:\n\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Resume lo que se dice en 3 a 5 viñetas cortas, en español. Conserva nombres, cifras y "
        "afirmaciones concretas. Usa solo lo que dice la transcripción. Responde solo con las viñetas.",
        ' (capítulo "{title}")',
    ),
}

_REDUCE = {
    "en": (
        "You write YouTube metadata for a podcast episode.",
        "{source}\n\n\"\"\"\n{material}\n\"\"\"\n\n{chapters}"
        "Write, in English, a JSON object with:\n"
        '- "title_suggestions": 5 different YouTube titles for the whole episode, each under 70 '
        "characters, engaging but faithful to the content\n"
        '- "description": a YouTube description of 80 to 150 words: one hook sentence, then what '
        "the episode covers. No timestamps, hashtags or links.\n"
        '- "tags": 10 to 15 short search tags of 1 to 3 words, lowercase\n'
        '- "summary": 2 or 3 sentences summarizing the episode for its producer\n'
        "Use only what the {kind} says. Don't invent names, guests or facts. Reply with the JSON only.",
        "Here are notes on the episode, in order:",
        "Here is the episode's transcript:",
        ("notes", "transcript"),
        "Chapters: {titles}\n\n",
    ),
    "es": (
        "Escribes los metadatos de YouTube de un episodio de podcast.",
        "{source}\n\n\"\"\"\n{material}\n\"\"\"\n\n{chapters}"
        "Escribe, en español, un objeto JSON con:\n"
        '- "title_suggestions": 5 títulos de YouTube distintos para el episodio completo, de menos '
        "de 70 caracteres cada uno, atractivos pero fieles al contenido\n"
        '- "description": una descripción de YouTube de 80 a 150 palabras: una frase gancho y luego '
        "de qué trata el episodio. Sin marcas de tiempo, hashtags ni enlaces.\n"
        '- "tags": de 10 a 15 etiquetas de búsqueda cortas, de 1 a 3 palabras, en minúsculas\n'
        '- "summary": 2 o 3 frases que resuman el episodio para su productor\n'
        "Usa solo lo que dicen {kind}. No inventes nombres, invitados ni datos. Responde solo con el JSON.",
        "Estas son notas sobre el episodio, en orden:",
        "Esta es la transcripción del episodio:",
        ("las notas", "la transcripción"),
        "Capítulos: {titles}\n\n",
    ),
}


def _lang(lang: str) -> str:
    return lang if lang in _MAP else "en"


def map_messages(text: str, lang: str, title: str | None = None) -> list[dict]:
    system, user, about = _MAP[_lang(lang)]
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(text=text, about=about.format(title=title) if title else "")}]


def reduce_messages(material: str, lang: str, chapter_titles: list[str] | None = None,
                    from_transcript: bool = False) -> list[dict]:
    system, user, notes_src, transcript_src, kinds, chapters_tpl = _REDUCE[_lang(lang)]
    chapters = chapters_tpl.format(titles="; ".join(chapter_titles)) if chapter_titles else ""
    return [{"role": "system", "content": system},
            {"role": "user", "content": user.format(
                source=transcript_src if from_transcript else notes_src,
                material=material, chapters=chapters, kind=kinds[from_transcript])}]
