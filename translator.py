from __future__ import annotations

from dataclasses import dataclass

import httpx
from openai import OpenAI

from epub_processor import TranslationUnit, parse_segmented_translation
from cache import TranslationCache


PROMPT_VERSION = "epub-translategemma-v2"


# Map language names to ISO 639-1 codes for translation models.
LANGUAGE_CODES = {
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
    "Turkish": "tr",
}


def build_segmented_prompt(
    unit: TranslationUnit,
    source_language: str,
    target_language: str,
) -> str:
    source_code = LANGUAGE_CODES.get(
        source_language,
        source_language.lower()[:2],
    )
    target_code = LANGUAGE_CODES.get(
        target_language,
        target_language.lower()[:2],
    )

    return f"""You are a professional translator translating from
{source_language} ({source_code}) into {target_language} ({target_code}).

Translate the text accurately while preserving its meaning, tone,
grammar, and formatting.

CRITICAL MARKER RULES:
1. Every segment has an opening marker such as ⟦SEG_0⟧
   and a closing marker such as ⟦/SEG_0⟧.
2. Preserve EVERY marker exactly as written.
3. Never translate, remove, modify, or reorder markers.
4. Translate ONLY the natural-language text between markers.
5. Keep each translated segment between its original markers.
6. Return all segments in their original order.
7. Do not add explanations, introductions, or commentary.
8. Return the complete marked translation, even when there is
   only one segment.

Example input:
⟦SEG_0⟧Good morning.⟦/SEG_0⟧

Example output:
⟦SEG_0⟧Guten Morgen.⟦/SEG_0⟧

Translate from {source_language} into {target_language}.

INPUT:
{unit.combined_text}

OUTPUT:
"""


@dataclass
class TranslatorConfig:
    source_language: str
    target_language: str
    model_name: str
    temperature: float = 0.0
    cache_dir: str = "cache"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_api_key: str = "ollama"


def fetch_ollama_models(
    base_url: str = "http://127.0.0.1:11434/v1",
) -> list[str]:
    """Query Ollama for available models."""

    try:
        with httpx.Client(
            trust_env=False,
            timeout=5.0,
        ) as http_client:
            client = OpenAI(
                base_url=base_url,
                api_key="ollama",
                http_client=http_client,
            )

            models = client.models.list()
            model_names = [model.id for model in models.data]

            return model_names or ["gemma3", "llama3.2"]

    except Exception:
        return ["gemma3", "llama3.2", "qwen2.5"]


class BaseTranslator:
    def translate_unit(self, unit: TranslationUnit) -> str:
        raise NotImplementedError


class CachedTranslator(BaseTranslator):
    def __init__(
        self,
        backend: BaseTranslator,
        config: TranslatorConfig,
    ):
        self.backend = backend
        self.config = config
        self.cache = TranslationCache(config.cache_dir)

    def translate_unit(self, unit: TranslationUnit) -> str:
        key = self.cache.key(
            unit.combined_text,
            self.config.source_language,
            self.config.target_language,
            self.config.model_name,
            PROMPT_VERSION,
        )

        cached = self.cache.get(key)

        if cached is not None:
            # Never reuse an invalid cached translation.
            if parse_segmented_translation(
                cached,
                len(unit.slots),
            ) is not None:
                return cached

        result = self.backend.translate_unit(unit)

        # Cache only valid translations.
        if parse_segmented_translation(
            result,
            len(unit.slots),
        ) is None:
            raise ValueError(
                f"Invalid SEG placeholders returned for {unit.unit_id}. "
                "Translation was not cached."
            )

        self.cache.put(
            key,
            result,
            {
                "source_language": self.config.source_language,
                "target_language": self.config.target_language,
                "model_name": self.config.model_name,
                "prompt_version": PROMPT_VERSION,
            },
        )

        return result


class OllamaTranslator(BaseTranslator):
    """Translate through Ollama's OpenAI-compatible API."""

    def __init__(self, config: TranslatorConfig):
        self.config = config

        self.client = OpenAI(
            base_url=config.ollama_base_url,
            api_key=config.ollama_api_key,
            http_client=httpx.Client(
                trust_env=False,
                timeout=300.0,
            ),
        )

    def translate_unit(self, unit: TranslationUnit) -> str:
        prompt = build_segmented_prompt(
            unit,
            self.config.source_language,
            self.config.target_language,
        )

        expected_count = len(unit.slots)
        last_response = ""

        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
            )

            last_response = (
                response.choices[0].message.content or ""
            ).strip()

            if parse_segmented_translation(
                last_response,
                expected_count,
            ) is not None:
                return last_response

            # Retry with stronger instructions if markers are missing.
            prompt += f"""

RETRY {attempt + 1} OF 3:

Your previous response was INVALID because it did not preserve
the required segment markers.

You must return exactly {expected_count} segments.

Original marked text:
{unit.combined_text}

Rules:
- Copy every opening and closing marker exactly.
- Do not omit any marker.
- Do not translate the markers.
- Translate only the text between each pair of markers.
- Return ONLY the corrected marked translation.
"""

        raise ValueError(
            f"Ollama model '{self.config.model_name}' failed to preserve "
            f"SEG placeholders for {unit.unit_id} after 3 attempts.\n"
            f"Expected segments: {expected_count}\n"
            f"Last response: {last_response[:1000]}"
        )


def create_translator(
    config: TranslatorConfig,
) -> CachedTranslator:
    backend = OllamaTranslator(config)
    return CachedTranslator(backend, config)
