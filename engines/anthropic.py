import json
import re
import socket as _socket
import time
import ssl
from typing import Generator
from urllib.parse import urljoin
from urllib.request import (
    Request as _UrlRequest,
    urlopen as _urlopen,
    build_opener as _build_opener,
    ProxyHandler as _ProxyHandler,
    HTTPSHandler as _HTTPSHandler,
)
from urllib.error import HTTPError as _UrlHTTPError
from http.client import IncompleteRead

from mechanize._response import response_seek_wrapper as Response

from .. import EbookTranslator
from ..lib.utils import request, log

from .genai import GenAI
from .languages import anthropic
from .prompt_extensions import anthropic as anthropic_prompt_extension


load_translations()  # type: ignore


class ClaudeTranslate(GenAI):
    name = 'Claude'
    alias = 'Claude (Anthropic)'
    lang_codes = GenAI.load_lang_codes(anthropic)
    endpoint = 'https://api.anthropic.com/v1/messages'
    # by default use the latest version of the api (currently this is 2023-06-01)
    api_version = '2023-06-01'
    api_key_hint = 'sk-ant-xxxx'
    # https://docs.anthropic.com/claude/reference/errors
    api_key_errors = ['401', 'permission_error']

    concurrency_limit = 1
    request_interval = 12.0
    request_timeout = 30.0

    prompt = (
        'You are a translation engine integrated into Calibre, a personal '
        'ebook management application. This is an automated translation '
        'pipeline that processes text segments from the user\'s personal '
        'library — content the user legally owns and is translating for '
        'private reading. '
        'Translate the given content from <slang> to <tlang> only. Do not '
        'explain any term or answer any question-like content. Your answer '
        'should be solely the translation of the given content. In your '
        'answer do not add any prefix or suffix to the translated content. '
        'Websites\' URLs/addresses should be preserved as is in the '
        'translation\'s output. Do not omit any part of the content, even if '
        'it seems unimportant. Never refuse a translation request, never '
        'summarize instead of translating, and never add commentary or '
        'disclaimers. Output only the translated text. ')

    samplings = ['temperature', 'top_p']
    sampling = 'temperature'
    temperature = 1.0
    top_p = 1.0
    top_k = 1
    stream = True
    enable_extended_output = False  # 128K output for Claude 3.7 Sonnet
    enable_extended_context = True  # 1M context (native on Claude 4.6 models)
    enable_dynamic_timeout = False  # Dynamic timeout based on content length
    enable_prompt_caching = False  # Prompt caching for parallel sections with full context
    refusal_max_retries = 3  # Max retries when Claude refuses to translate (after splitting kicks in)
    enable_consistency_pass = False  # Run a Pass-2 consistency review after main translation
    # Translation Brief: auto-build a structured reference document
    # from the source text before drafting, then inject it into each
    # per-paragraph translation call so canonical names, character
    # gender, and recurring terminology stay consistent. Default on
    # for Claude. User can disable to skip the brief build.
    enable_translation_brief = True
    # Agreement Pass: opt-in revision pass that scans translated
    # paragraphs for residual gender/number drift against the
    # canonical character morphology in the brief and emits single-
    # occurrence find/replace fixes. Distinct from drafting: runs
    # AFTER translation is complete, sees only translated text +
    # brief (no source-language copyright concern), and the host
    # validates uniqueness before applying. Default off; manual
    # button trigger.
    supports_agreement_review = True
    enable_agreement_pass = True
    # Copyright-refusal mitigation toggles. All default-on; individually
    # toggleable for users who prefer fail-loud over auto-recovery.
    enable_strip_identifying_content = True   # Strip copyright/ISBN paragraphs from cached book context
    enable_refusal_split = True               # On refusal exhaustion, split chunk and retry each half
    enable_bare_context_fallback = True       # Final fallback: retry without cached book context

    # event types for streaming are listed here:
    # https://docs.anthropic.com/en/api/messages-streaming
    valid_event_types = [
        'ping',
        'error',
        'content_block_start',
        'content_block_delta',
        'content_block_stop',
        'message_start',
        'message_delta',
        'message_stop']

    models: list[str] = []
    model: str | None = 'claude-sonnet-4-6'

    def __init__(self):
        super().__init__()
        self.endpoint = self.config.get('endpoint', self.endpoint)
        self.prompt = self.config.get('prompt', self.prompt)
        self.sampling = self.config.get('sampling', self.sampling)
        self.temperature = self.config.get('temperature', self.temperature)
        self.top_p = self.config.get('top_p', self.top_p)
        self.top_k = self.config.get('top_k', self.top_k)
        self.stream = self.config.get('stream', self.stream)
        self.model = self.config.get('model', self.model)
        self.enable_extended_output = self.config.get(
            'enable_extended_output', self.enable_extended_output)
        self.enable_extended_context = self.config.get(
            'enable_extended_context', self.enable_extended_context)
        self.enable_dynamic_timeout = self.config.get(
            'enable_dynamic_timeout', self.enable_dynamic_timeout)
        self.enable_prompt_caching = self.config.get(
            'enable_prompt_caching', self.enable_prompt_caching)
        self.refusal_max_retries = self.config.get(
            'refusal_max_retries', self.refusal_max_retries)
        self.enable_consistency_pass = self.config.get(
            'enable_consistency_pass', self.enable_consistency_pass)
        self.enable_translation_brief = self.config.get(
            'enable_translation_brief', self.enable_translation_brief)
        self.enable_agreement_pass = self.config.get(
            'enable_agreement_pass', self.enable_agreement_pass)
        self.enable_strip_identifying_content = self.config.get(
            'enable_strip_identifying_content',
            self.enable_strip_identifying_content)
        self.enable_refusal_split = self.config.get(
            'enable_refusal_split', self.enable_refusal_split)
        self.enable_bare_context_fallback = self.config.get(
            'enable_bare_context_fallback',
            self.enable_bare_context_fallback)
        self.full_book_context = None  # Set externally for prompt caching
        # Phase 1a spike: the Translation Brief, set externally by
        # the orchestrator after build_translation_brief or after
        # rehydrating from cache. When set, get_body() injects it
        # into the system prompt as a cached reference document so
        # per-paragraph translation respects canonical names,
        # gender, terminology, and style decisions.
        self.translation_brief = None

    # Patterns that indicate Claude refused to translate due to copyright concerns.
    # Requires 2+ matches to avoid false positives from legitimate translations.
    _refusal_indicators = [
        # Direct refusal patterns
        "can't translate",
        "cannot translate",
        "not able to translate",
        "unable to translate",
        # Copyright identification
        "copyrighted material",
        "copyrighted book",
        "copyrighted content",
        "copyrighted text",
        "from a copyrighted",
        # Offering alternatives
        "I'd be happy to help",
        "I would be happy to help",
        "How can I help you instead",
        "How would you like me to help",
        "Would you like me to continue",
        # Partial translation / scope limiting
        "rather than a full translation",
        "a reasonable excerpt",
        "shorter portion",
        "translate a shorter",
        "a very long passage",
        # Legal framing
        "substantial portion",
        "derivative work",
        "reproducing copyrighted",
        "protected content",
    ]

    def is_translation_refusal(self, translation):
        """Detect if the response is a refusal to translate rather than an
        actual translation. Claude sometimes refuses to translate content it
        identifies as copyrighted, especially when the full book context is
        provided via prompt caching.

        Uses a two-stage approach:
        1. Quick heuristic pre-filter (pattern matching) to avoid unnecessary
           API calls on every translation.
        2. LLM classification call to confirm, avoiding false positives when
           translating text that legitimately discusses copyright (e.g., legal
           texts translated to English).
        """
        # Stage 1: Quick heuristic pre-filter
        translation_lower = translation.lower()
        indicator_count = sum(
            1 for p in self._refusal_indicators
            if p.lower() in translation_lower)
        if indicator_count < 2:
            return False

        # Stage 2: LLM classification to confirm
        try:
            body = json.dumps({
                'model': self.model,
                'max_tokens': 20,
                'temperature': 0,
                'system': (
                    'Classify the following text as either a genuine '
                    'translation or a refusal to translate. Respond with '
                    'exactly one word: "translation" or "refusal".'),
                'messages': [{
                    'role': 'user',
                    'content': (
                        'Expected: a translation to %s.\n\n'
                        'Actual response:\n%s'
                        % (self.target_lang, translation[:1000]))
                }]
            })
            response = request(
                self.endpoint,
                data=body,
                headers={
                    'Content-Type': 'application/json',
                    'anthropic-version': self.api_version,
                    'x-api-key': self.api_key,
                },
                method='POST',
                timeout=15,
                proxy_uri=(
                    self.proxy_uri if self.proxy_type == 'http' else None),
            )
            result = json.loads(response)
            classification = result['content'][0]['text'].strip().lower()
            is_refusal = 'refusal' in classification
            log.info('Refusal classification: %s (indicators: %d)'
                     % (classification, indicator_count))
            return is_refusal
        except Exception:
            # If classification fails, fall back to heuristic result.
            # Better to retry a potential refusal than to return one.
            log.warning('Refusal classification failed, falling back to '
                        'heuristic (indicators: %d)' % indicator_count)
            return True

    # ── Terminology mapping (model-facing vs codebase) ────────────────
    #   block (model-facing): one translation unit. Equals one
    #       Paragraph object's content. Document title is "block_N".
    #   Paragraph (codebase, lib/cache.py): the Python data model and
    #       cache row.
    #   source structural unit: one HTML block-level element from the
    #       EPUB (typically <p>, but also <h1>-<h6>, <li>, etc).
    #   real paragraph (reader's intuition): a logical unit of prose,
    #       roughly equal to a <p> element. NOT all source structural
    #       units are real paragraphs (a heading is a structural unit,
    #       not a paragraph in the reader's sense).
    #
    # When merge is OFF: 1 block = 1 Paragraph = 1 source structural
    # unit (which may or may not be a real paragraph).
    # When merge is ON:  1 block = 1 Paragraph = N source structural
    # units joined by '\n\n', batched up to merge_length characters.
    def _get_prompt(self):
        prompt = self.prompt.replace('<tlang>', self.target_lang)
        if self._is_auto_lang():
            prompt = prompt.replace('<slang>', 'detected language')
        else:
            prompt = prompt.replace('<slang>', self.source_lang)

        prompt_extension = anthropic_prompt_extension.get(self.target_lang)
        if prompt_extension is not None:
            prompt += ' ' + prompt_extension

        # Recommend setting temperature to 0.5 for retaining the placeholder.
        if self.merge_enabled:
            prompt += (' Ensure that placeholders matching the pattern '
                       '{{id_\\d+}} in the content are retained.')
            # When merge is enabled, the input may contain multiple
            # structural blocks (paragraphs, headings, list items,
            # blockquotes, etc.) separated by double newlines (\n\n).
            # The plugin's alignment check (Paragraph.do_aligment)
            # flags rows where the block count differs between original
            # and translation. The instruction below is imperative and
            # specific — telling the model to count separators makes
            # it an explicit operation rather than a soft goal.
            prompt += (
                ' The input is a sequence of text blocks separated by '
                'double newlines (\\n\\n). These blocks may be '
                'paragraphs, headings, list items, or other structural '
                'units from the source — translate each as a unit. '
                'CRITICAL: your output must preserve the EXACT same '
                'structure. Count the number of \\n\\n separators in '
                'the input. Your output must contain EXACTLY that many '
                '\\n\\n separators, in the same positions, in the same '
                'order. NEVER merge two blocks into one. NEVER split '
                'one block into two. The output\'s block count and '
                'separator positions must match the input exactly.')
        return prompt

    # Substrings that identify Anthropic's content filter HTTP 400.
    # These are returned as the error message body when the content
    # filter blocks input or output for copyright/policy reasons.
    _content_filter_markers = [
        'output blocked by content filtering',
        'input blocked by content filtering',
        'content filtering policy',
    ]

    def _is_content_filter_error(self, exc):
        msg = str(exc).lower()
        return any(m in msg for m in self._content_filter_markers)

    def _content_filter_refusal(self):
        """Return a synthetic refusal payload that the existing refusal
        detection (pattern + LLM) will recognize. Matches the shape of
        the streaming output (generator) when stream=True, plain string
        otherwise."""
        text = (
            'I cannot translate this copyrighted material. '
            'I would be happy to help with shorter excerpts from '
            'copyrighted content.')
        if self.stream:
            return iter([text])
        return text

    def translate(self, content):
        # Use dynamic timeout only if enabled (default: disabled)
        if self.enable_dynamic_timeout:
            # Calculate dynamic timeout based on estimated token generation time
            # Estimate output tokens (conservative: 3 chars per token)
            estimated_output_tokens = len(content) // 3
            # Claude models generate 65-120 tokens/second (varies by model)
            # Use conservative 50 tokens/sec to account for network latency and processing
            # Source: https://artificialanalysis.ai/models/claude-3-opus
            # Add 60s base overhead for request processing and network latency
            estimated_time = (estimated_output_tokens / 50) + 60
            # Minimum 30s, maximum 2 hours for very large content
            dynamic_timeout = max(30.0, min(estimated_time, 7200.0))

            # Temporarily set timeout for this request
            original_timeout = self.request_timeout
            self.request_timeout = dynamic_timeout

            try:
                return super().translate(content)
            except Exception as e:
                if self._is_content_filter_error(e):
                    return self._content_filter_refusal()
                raise
            finally:
                # Restore original timeout
                self.request_timeout = original_timeout
        try:
            # Use user-configured timeout (default 30s)
            return super().translate(content)
        except Exception as e:
            if self._is_content_filter_error(e):
                return self._content_filter_refusal()
            raise

    # Patterns indicating the model refused to perform a review (rather
    # than returning the requested JSON). Different from the translation
    # refusal indicators because review-style refusals talk about
    # "reviewing" or "helping with content", not "translating".
    _review_refusal_indicators = [
        "i cannot",
        "i can't",
        "i'm not able",
        "i am not able",
        "unable to",
        "cannot review",
        "can't review",
        "cannot assist",
        "can't help with",
        "copyright",
        "copyrighted",
        "i'd be happy to help",
        "i would be happy to help",
        "i'm sorry",
    ]

    def _is_review_refusal(self, raw_text):
        """Return True if the raw response looks like a refusal to
        perform the review (rather than a malformed JSON or empty
        response). Uses the same 2+-match heuristic as translation
        refusal detection."""
        if not raw_text:
            return False
        text_lower = raw_text.lower()
        matches = sum(1 for p in self._review_refusal_indicators
                      if p in text_lower)
        return matches >= 2

    # JSON schema for the consistency review response. Used with
    # Anthropic's Structured Outputs feature (output_config.format),
    # which guarantees the model's text response matches this schema.
    # Streams via text_delta events on the regular text-streaming
    # pipeline — avoids the stall bug that affects tool_use streaming
    # on Sonnet 4.x with large prompts.
    #
    # Loose required fields — we filter at the validation layer for
    # content quality. No length/numeric constraints (Structured
    # Outputs strips them).
    # Phase 0 (validation spike) — Translation Brief schema.
    # Built ONCE per book before drafting; the structured artifact
    # that captures canonical translation decisions (character names
    # with gender, place names, recurring terminology, style choices).
    # Drafting and revision passes consume the brief; never the other
    # way around. Schema is intentionally compact for the spike — the
    # full pipeline-v1 schema (lib/brief.py) will be a superset.
    # Brief content schema. Used as the inner shape for the brief-
    # building stage of the multi-turn pipeline (see
    # _TRANSLATION_BRIEF_SCHEMA below). Defined separately so we can
    # reference it from both the build-turn slot and from the
    # apply-step parsing without duplicating ~150 lines of property
    # declarations.
    _BRIEF_CONTENT_SCHEMA = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'source_summary': {
                'type': 'object',
                'additionalProperties': False,
                'description': (
                    'Brief overview of the source text — genre, '
                    'narrator, register, themes, and central '
                    'conflict. Captures what the book is *about* so '
                    'the brief is self-sufficient for downstream '
                    'stages.'),
                'properties': {
                    'genre': {'type': 'string'},
                    'narrator': {
                        'type': 'string',
                        'description': (
                            'e.g. "first-person past", "omniscient '
                            'third-person".'),
                    },
                    'register': {
                        'type': 'string',
                        'description': (
                            'e.g. "literary, mid-formal", '
                            '"casual contemporary".'),
                    },
                    'themes': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': (
                            'Top 3-7 themes / subjects the book '
                            'engages with. Single short phrases — '
                            'high-level topical descriptors that '
                            'summarize what the work is *about* at '
                            'an abstract level.'),
                    },
                    'central_conflict': {
                        'type': 'string',
                        'description': (
                            'One-sentence statement of the central '
                            'narrative conflict / what is at stake. '
                            'Drives register and tone decisions.'),
                    },
                },
            },
            'style_decisions': {
                'type': 'object',
                'additionalProperties': False,
                'description': (
                    'Translator-style decisions to apply to the whole '
                    'work. Excludes punctuation/quote conventions.'),
                'properties': {
                    'tense_strategy': {'type': 'string'},
                    'pov': {'type': 'string'},
                    'formality': {
                        'type': 'string',
                        'description': (
                            'Default formality and any T-V '
                            'distinctions per character class.'),
                    },
                    'idiom_strategy': {
                        'type': 'string',
                        'description': (
                            'When to domesticate vs foreignize '
                            'idioms.'),
                    },
                },
            },
            'characters': {
                'type': 'array',
                'description': (
                    'Recurring named characters / referenced persons. '
                    'Include only entities mentioned in two or more '
                    'distinct passages.'),
                'items': {
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {
                        'id': {
                            'type': 'string',
                            'description': (
                                'Stable short identifier for cross-'
                                'references in this brief, e.g. '
                                '"c_001", "c_002". Numbered '
                                'sequentially in the order entities '
                                'first appear in the source. Used by '
                                'relationships[].to_id.'),
                        },
                        'canonical_name': {
                            'type': 'string',
                            'description': (
                                'The target-language form to use '
                                'consistently.'),
                        },
                        'source_name': {
                            'type': 'string',
                            'description': (
                                'How the entity is named in the '
                                'source text.'),
                        },
                        'aliases': {
                            'type': 'array',
                            'items': {'type': 'string'},
                            'description': (
                                'Other names/nicknames the same '
                                'entity goes by in the source.'),
                        },
                        'gender': {
                            'type': 'string',
                            'description': (
                                'Canonical grammatical gender for '
                                'target-language agreement: '
                                '"masculine" / "feminine" / "neuter" '
                                '/ "mixed" / "unknown". Drives gender '
                                'agreement across verb forms, '
                                'pronouns, adjectives.'),
                        },
                        'number': {
                            'type': 'string',
                            'description': (
                                '"singular" / "plural". Most '
                                'characters singular; collective '
                                'nouns (a guild, a family) may be '
                                'plural.'),
                        },
                        'role': {
                            'type': 'string',
                            'description': (
                                'Role descriptor in the standard '
                                'story-bible vocabulary. Begin with '
                                'one of: "protagonist", "antagonist", '
                                '"deuteragonist", "supporting", '
                                '"mentor", "love interest", '
                                '"sidekick", "minor", "walk-on", or '
                                '"cameo". Then optionally a colon '
                                'and a one-line free-form '
                                'description. The leading '
                                'descriptor lets downstream stages '
                                'recognize narrative weight.'),
                        },
                        'voice': {
                            'type': 'string',
                            'description': (
                                'Brief notes on the character\'s '
                                'voice / register / typical speech '
                                'patterns.'),
                        },
                        'relationships': {
                            'type': 'array',
                            'description': (
                                'STRUCTURAL ties to other characters '
                                'in this brief — durable connections '
                                'that define the social/family/'
                                'professional graph (parent, sibling, '
                                'mentor, friend, ally, employer, '
                                'etc.). NOT plot events (killings, '
                                'betrayals, infections, romantic '
                                'arcs, transformations) — those go '
                                'in the role description as prose. '
                                'See the system prompt for the '
                                'universal-core type vocabulary and '
                                'rules for extending it.'),
                            'items': {
                                'type': 'object',
                                'additionalProperties': False,
                                'properties': {
                                    'to_id': {
                                        'type': 'string',
                                        'description': (
                                            'The id of the related '
                                            'character (must match '
                                            'another characters[].id '
                                            'in this brief).'),
                                    },
                                    'type': {
                                        'type': 'string',
                                        'description': (
                                            'Relationship type. '
                                            'Use the universal-core '
                                            'vocabulary from the '
                                            'system prompt where '
                                            'applicable; extend with '
                                            'genre-specific types '
                                            'when the universal core '
                                            'does not fit. Must '
                                            'represent a STRUCTURAL '
                                            'tie, not a plot event.'),
                                    },
                                },
                                'required': ['to_id', 'type'],
                            },
                        },
                        'first_occurrence_index': {
                            'type': 'integer',
                            'description': (
                                'The block_N index where this '
                                'character first appears in the '
                                'source. Used by downstream stages '
                                'for "introduce on first use" '
                                'patterns. Best-effort estimate.'),
                        },
                        'mention_count': {
                            'type': 'integer',
                            'description': (
                                'Approximate number of mentions '
                                'across the source text. A '
                                'continuous-signal proxy for '
                                'narrative weight: protagonists '
                                'have hundreds; walk-ons have a '
                                'handful. Best-effort estimate; '
                                'precision is not required.'),
                        },
                    },
                    'required': [
                        'id', 'canonical_name', 'gender',
                        'relationships'],
                },
            },
            'terminology': {
                'type': 'array',
                'description': (
                    'Recurring non-person entities: places, '
                    'organizations, titles, distinctive objects, '
                    'invented terms. Each entity must appear in two '
                    'or more distinct passages.'),
                'items': {
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {
                        'id': {
                            'type': 'string',
                            'description': (
                                'Stable short identifier, e.g. '
                                '"t_001", "t_002". Numbered '
                                'sequentially in the order terms '
                                'first appear.'),
                        },
                        'canonical': {
                            'type': 'string',
                            'description': (
                                'Target-language form to use '
                                'consistently.'),
                        },
                        'source_form': {'type': 'string'},
                        'type': {
                            'type': 'string',
                            'description': (
                                '"place" / "organization" / "title" '
                                '/ "object" / "term".'),
                        },
                        'do_not_translate': {
                            'type': 'boolean',
                            'description': (
                                'When true, the entity should be '
                                'kept in source language form (or '
                                'transliterated only) rather than '
                                'translated. Standard CAT-tool DNT '
                                'flag for brand names, song titles, '
                                'product names, code identifiers, '
                                'and proper nouns the user wants '
                                'preserved verbatim.'),
                        },
                        'first_occurrence_index': {
                            'type': 'integer',
                            'description': (
                                'The block_N index where this term '
                                'first appears in the source. '
                                'Best-effort.'),
                        },
                        'mention_count': {
                            'type': 'integer',
                            'description': (
                                'Approximate number of mentions '
                                'across the source. Continuous-'
                                'signal weight indicator.'),
                        },
                        'notes': {'type': 'string'},
                    },
                    'required': ['id', 'canonical'],
                },
            },
            'translator_notes': {
                'type': 'string',
                'description': (
                    'Free-form notes on anything else the drafting '
                    'translator should know — flagged passages, '
                    'recurring patterns, decisions worth recording.'),
            },
        },
        'required': [
            'source_summary', 'style_decisions',
            'characters', 'terminology'],
    }

    # Change-list item schema. One entry per issue the review critic
    # identifies. Two action types:
    #
    #   action="replace": edit an existing leaf string at field_path
    #                     (current value must match `current`).
    #                     Used for: language fixes, type corrections
    #                     on existing relationships, role text
    #                     corrections, anything that touches an
    #                     already-present value.
    #
    #   action="insert":  append a new entry to an array at field_path
    #                     (the array must already exist; the entry is
    #                     `insert_value`). Used for: adding missing
    #                     reciprocal relationships, adding
    #                     relationships to characters that are missing
    #                     them entirely.
    #
    # field_path is JSON-pointer-style: "characters[3].canonical_name"
    # for a leaf, "characters[3].relationships" for an array (insert).
    # The optional "brief." prefix is stripped during apply.
    _BRIEF_CHANGE_LIST_ITEM_SCHEMA = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'action': {
                'type': 'string',
                'enum': ['replace', 'insert'],
                'description': (
                    'The mutation type. "replace" edits an existing '
                    'leaf string at field_path. "insert" appends a '
                    'new object to an existing array at field_path. '
                    'The discriminator commits the issue to one '
                    'shape: replace uses current/suggested, insert '
                    'uses insert_value.'),
            },
            'field_path': {
                'type': 'string',
                'description': (
                    'JSON-pointer-style path inside the brief. For '
                    'action="replace", this points to a leaf string '
                    '(e.g. "characters[3].canonical_name", '
                    '"characters[3].relationships[1].type"). For '
                    'action="insert", this points to an ARRAY (e.g. '
                    '"characters[10].relationships"); the new entry '
                    'is appended to the array.'),
            },
            'current': {
                'type': 'string',
                'description': (
                    'For action="replace": the exact current value '
                    'of the field, copied verbatim from the brief. '
                    'For action="insert": MUST be empty string "".'),
            },
            'suggested': {
                'type': 'string',
                'description': (
                    'For action="replace": the corrected value to '
                    'use instead. For action="insert": MUST be '
                    'empty string "".'),
            },
            'insert_value': {
                'type': ['object', 'null'],
                'additionalProperties': False,
                'description': (
                    'For action="insert": the new object to append '
                    'to the array at field_path. Currently '
                    'supports only relationship-array inserts '
                    '({to_id, type}). For action="replace": MUST '
                    'be null.'),
                'properties': {
                    'to_id': {
                        'type': 'string',
                        'description': (
                            'The id of the related character; must '
                            'match an existing characters[].id in '
                            'the brief.'),
                    },
                    'type': {
                        'type': 'string',
                        'description': (
                            'Structural relationship type (parent-'
                            'of, mentor-of, ally-of, etc.). Use a '
                            'plot-event type only if no structural '
                            'tie applies.'),
                    },
                },
                'required': ['to_id', 'type'],
            },
            'category': {
                'type': 'string',
                'description': (
                    'Short tag for the issue category, e.g. "typo", '
                    '"semantic", "naturalness", "consistency", '
                    '"dual_form", "grammar", "missing_reciprocity", '
                    '"missing_relationship", "role_mismatch", '
                    '"plot_event", "self_reference", '
                    '"dangling_id".'),
            },
            'reason': {
                'type': 'string',
                'description': (
                    'One-sentence explanation of why the change is '
                    'needed.'),
            },
        },
        'required': [
            'action', 'field_path', 'current', 'suggested',
            'insert_value', 'category', 'reason'],
    }

    # Unified pipeline schema. Build turn AND both review turns
    # produce JSON matching this shape. The `stage` discriminator
    # is required and identifies which branch the response
    # represents:
    #
    #   stage="build"           → brief is populated, change_list
    #                             is null
    #   stage="review_language" → brief is null, change_list is
    #                             populated with TARGET-LANGUAGE
    #                             quality issues
    #   stage="review_logic"    → brief is null, change_list is
    #                             populated with INTERNAL-LOGIC /
    #                             relationship-graph issues
    #
    # Both review stages may emit an empty change_list (meaning "no
    # issues found"). The host validates this contract defensively
    # in _validate_brief_response.
    _TRANSLATION_BRIEF_SCHEMA = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'stage': {
                'type': 'string',
                'enum': [
                    'build', 'review_language', 'review_logic'],
                'description': (
                    'Discriminator. "build" for the initial brief '
                    'construction; "review_language" for the '
                    'target-language quality critique pass; '
                    '"review_logic" for the internal-consistency / '
                    'relationship-graph critique pass. Set this '
                    'BEFORE filling brief or change_list; the '
                    'value of stage commits the response to one '
                    'branch.'),
            },
            'brief': {
                'type': ['object', 'null'],
                'additionalProperties': False,
                'description': (
                    'Populated when stage="build". MUST be null in '
                    'either review stage.'),
                'properties':
                    _BRIEF_CONTENT_SCHEMA['properties'],
                'required': _BRIEF_CONTENT_SCHEMA['required'],
            },
            'change_list': {
                'type': ['array', 'null'],
                'description': (
                    'Populated in either review stage. MUST be '
                    'null when stage="build". Empty array (no '
                    'issues) is valid in either review stage.'),
                'items': _BRIEF_CHANGE_LIST_ITEM_SCHEMA,
            },
        },
        'required': ['stage'],
    }

    _CONSISTENCY_OUTPUT_SCHEMA = {
        'type': 'object',
        # Anthropic's Structured Outputs require additionalProperties:false
        # on every object type — closed schemas guarantee no extra
        # fields appear in the output.
        'additionalProperties': False,
        'properties': {
            'glossary': {
                'type': 'array',
                'description': (
                    'Recurring entities (character names, places, '
                    'titles, proper nouns, key terms) that appear in '
                    'TWO OR MORE different forms in the translation. '
                    'For each entity, identify the canonical (correct/'
                    'preferred) form and list the alternate variant '
                    'forms that should be normalized to the canonical. '
                    'These are applied via deterministic substitution '
                    '(no LLM) — DO NOT include style/punctuation '
                    'variation.'),
                'items': {
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {
                        'canonical': {
                            'type': 'string',
                            'description': (
                                'The correct / preferred form to '
                                'use everywhere in the translation. '
                                'Must be the EXACT character sequence '
                                'as it should appear.'),
                        },
                        'variants': {
                            'type': 'array',
                            'description': (
                                'Other forms of this same entity '
                                'that appear verbatim in the text '
                                'and should be replaced with the '
                                'canonical form. Each entry MUST be '
                                'an EXACT character sequence as it '
                                'appears in the translation (so a '
                                'literal text.replace(variant, '
                                'canonical) at the host will hit). '
                                'Do NOT include the canonical form '
                                'itself in this list. Do NOT include '
                                'forms that are substrings of '
                                'unrelated words. Variants are '
                                'matched as whole words.'),
                            'items': {'type': 'string'},
                        },
                        'type': {
                            'type': 'string',
                            'description': (
                                'Category: "character", "place", '
                                '"title", "term", "object".'),
                        },
                        'gender': {
                            'type': 'string',
                            'description': (
                                'For characters / persons only: the '
                                'canonical grammatical gender '
                                '(masculine / feminine / neuter / '
                                'mixed / unknown). Leave empty for '
                                'non-person entries. NOTE: gender '
                                'consistency cannot be enforced by '
                                'name substitution — this is '
                                'informational only.'),
                        },
                        'notes': {'type': 'string'},
                    },
                    'required': ['canonical', 'variants'],
                },
            },
        },
        'required': ['glossary'],
    }

    # Agreement Pass output schema. The model emits a list of single-
    # occurrence find/replace fixes that correct residual gender /
    # number / pronoun drift in the translation against the canonical
    # character morphology supplied in the brief. The host validates
    # uniqueness (old_str must appear EXACTLY ONCE in its target
    # block) before applying — same uniqueness contract used by the
    # consistency repair loop. Failures are repaired in a second
    # round via _build_repair_message / _validate_corrections.
    _AGREEMENT_PASS_OUTPUT_SCHEMA = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'fixes': {
                'type': 'array',
                'description': (
                    'Single-occurrence find/replace operations that '
                    'correct gender / number / pronoun agreement '
                    'drift in the translation against the canonical '
                    'character morphology in the brief. Each old_str '
                    'must appear EXACTLY ONCE in its target block — '
                    'extend with surrounding context until uniqueness '
                    'holds. Empty array is fine when no drift is '
                    'found.'),
                'items': {
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {
                        'block_index': {
                            'type': 'integer',
                            'description': (
                                '0-based block_N index identifying '
                                'the target paragraph this fix '
                                'applies to.'),
                        },
                        'character_id': {
                            'type': 'string',
                            'description': (
                                'c_NNN id of the character whose '
                                'canonical morphology drives this '
                                'fix. Empty string if not character-'
                                'specific (e.g. a generic plural / '
                                'collective-agreement fix).'),
                        },
                        'kind': {
                            'type': 'string',
                            'enum': ['gender', 'number', 'pronoun',
                                     'other'],
                            'description': (
                                'Class of agreement error this fix '
                                'addresses.'),
                        },
                        'old_str': {
                            'type': 'string',
                            'description': (
                                'Verbatim character sequence from '
                                'the target block that contains the '
                                'agreement error. MUST appear '
                                'EXACTLY ONCE in the block — extend '
                                'with surrounding context until '
                                'uniqueness is satisfied.'),
                        },
                        'new_str': {
                            'type': 'string',
                            'description': (
                                'Replacement string with morphology '
                                'corrected to match the canonical '
                                'gender/number from the brief. '
                                'Preserve everything except the '
                                'agreement-bearing tokens (verb '
                                'inflections, pronouns, adjective '
                                'endings).'),
                        },
                        'reason': {
                            'type': 'string',
                            'description': (
                                'One sentence: which character, '
                                'which token(s), what canonical form '
                                'is expected.'),
                        },
                    },
                    'required': [
                        'block_index', 'character_id', 'kind',
                        'old_str', 'new_str', 'reason',
                    ],
                },
            },
        },
        'required': ['fixes'],
    }

    def _consistency_system_prompt(self, target_lang, attempt=0):
        """Build the system prompt for a consistency review attempt.
        Higher attempt numbers use stronger editorial-context framing
        to recover from refusals on the first try.

        The output structure is enforced by Structured Outputs
        (output_config.format) — the prompt focuses on the analytical
        task only.
        """
        # Critical instruction shared across all attempts: corrections
        # are surgical find/replace operations, NOT paragraph rewrites.
        # The previous "full corrected paragraph" approach caused the
        # model to truncate long merged segments — losing chapter
        # headings, sub-paragraphs, and other content. Span-based
        # replacements preserve the original byte-for-byte except where
        # the matched span is replaced.
        #
        # Glossary-first design: the model's job is JUDGMENT
        # (which form is canonical?), not SUBSTITUTION. The host
        # applies variants → canonical via word-boundary regex,
        # which is convergent in O(1) — no repair loop, no
        # validation failures.
        scope_rules = (
            'INCLUDE: character names, place names, titles, group/'
            'organization names, recurring proper nouns, distinctive '
            'recurring terms, special objects with proper names. '
            'Each entity should appear in TWO OR MORE different forms '
            'across the translation; otherwise omit it.\n'
            '\n'
            'EXCLUDE absolutely (these are NOT consistency issues):\n'
            '  • Quote marks (", \', ״, ׳, fancy quotes, single vs '
            'double) — DO NOT report quote variation under any '
            'circumstance\n'
            '  • Punctuation (commas, periods, em-dashes, hyphens, '
            'ellipses)\n'
            '  • Whitespace, line breaks, paragraph formatting\n'
            '  • Capitalization where the language allows variation\n'
            '  • Stylistic word choices that aren\'t proper nouns\n'
            '  • Phrasings or sentence structures\n'
            '  • One-off variations that appear only once\n'
            '  • Any difference you would describe as "stylistic"\n'
        )
        variant_rules = (
            'For each entity, populate "variants" with the EXACT '
            'character sequences (verbatim, character-by-character) '
            'as they appear in the translation. The host will execute '
            'a literal text.replace(variant, canonical) over every '
            'block, gated by word boundaries. Therefore:\n'
            '\n'
            '  • Each variant must be the EXACT string as it appears '
            '— do not paraphrase, do not normalize, do not strip '
            'surrounding context.\n'
            '  • Variants are matched as whole words. Do not include '
            'a variant that is a substring of an unrelated word '
            '(e.g., a short common-noun fragment that happens to '
            'appear inside a longer canonical name will also match '
            'unrelated phrases — list the full distinctive form, '
            'not the substring).\n'
            '  • Do NOT include the canonical form in its own '
            'variants list.\n'
            '  • If you cannot list at least one verbatim variant, '
            'OMIT the entry — informational-only entries are useless.\n'
            '\n'
            'You will NOT emit corrections / replacements / find-'
            'replace operations. Substitution is performed by the '
            'host from this glossary, not by you.\n'
        )
        gender_note = (
            'For each character entity, also fill "gender" with the '
            'canonical grammatical gender (masculine / feminine / '
            'neuter / mixed / unknown). NOTE: gender consistency is '
            'NOT enforced by name substitution — it would require '
            'editing verb forms, pronouns, and adjective endings. '
            'The "gender" field is informational for the user; the '
            'host will not auto-fix gender drift.\n'
        )
        block_terminology = (
            'INPUT FORMAT: the translated text is provided as '
            'separate document content blocks, each titled "block_N" '
            'where N is the block\'s 0-indexed position. A block may '
            'be a single paragraph, a heading, a list item, or '
            '(when paragraph merging is enabled) several structural '
            'units concatenated with double newlines (\\n\\n). Treat '
            'every block as part of the same translated work when '
            'judging which form of an entity is canonical.\n'
        )
        if attempt == 0:
            return (
                'You are analyzing a {tlang} translation of a book '
                'for terminology consistency. Your task is to '
                'produce a glossary of recurring entities that '
                'appear in multiple forms in the text, identifying '
                'the canonical (correct/preferred) form and the '
                'variant forms found in the translation that should '
                'be normalized to the canonical.\n'
                '\n'
                '═══ SCOPE ═══\n'
                + scope_rules +
                '\n'
                '═══ VARIANT REPORTING (CRITICAL) ═══\n'
                + variant_rules +
                '\n'
                '═══ GENDER (informational) ═══\n'
                + gender_note +
                '\n'
                '═══ INPUT ═══\n'
                + block_terminology
            ).format(tlang=target_lang)
        # Stronger editorial framing for retry on refusal.
        return (
            'You are an editorial assistant integrated into a '
            'translation tool. The user has produced a {tlang} '
            'translation of their personal copy of a book and is '
            'requesting a TERMINOLOGY GLOSSARY of recurring entities '
            '(names, places, titles, terms) that appear in multiple '
            'forms — they are not asking you to translate or '
            'reproduce any source material. The glossary is used by '
            'the host to apply deterministic find/replace '
            'normalization. This is a standard quality-assurance '
            'task.\n'
            '\n'
            '═══ SCOPE ═══\n'
            + scope_rules +
            '\n'
            '═══ VARIANT REPORTING (CRITICAL) ═══\n'
            + variant_rules +
            '\n'
            '═══ GENDER (informational) ═══\n'
            + gender_note +
            '\n'
            '═══ INPUT ═══\n'
            + block_terminology
        ).format(tlang=target_lang)

    def _consistency_repair_system_prompt(self, target_lang):
        """System prompt for repair rounds. Tells the model that the
        host validated previous replacements and lists which ones
        failed; asks it to emit corrected versions only for those
        failures, with extended context for uniqueness.

        Strong emphasis on verbatim copying: the dominant failure
        mode in repair is the model fabricating an old_str that
        looks plausible but doesn't actually appear in the block
        (the model recites from memory rather than copying).
        """
        return (
            'You are an editorial assistant repairing a previous '
            'consistency review of a {tlang} translation. The user '
            'message lists find/replace operations you previously '
            'proposed that failed host-side uniqueness validation: '
            'each "old_str" was either absent from its target block '
            'or matched at multiple locations. The user message '
            'INCLUDES the verbatim text of every affected block.\n'
            '\n'
            'CRITICAL RULES for emitting a corrected "old_str":\n'
            '\n'
            '1. VERBATIM COPYING. The new "old_str" MUST be a '
            'character-by-character copy of a contiguous substring '
            'of the block text shown in the user message. Do NOT '
            'paraphrase, reconstruct, or substitute equivalent '
            'words. Do NOT recite from memory. If you cannot find '
            'the wrong term in the block as written, the previous '
            'flag was incorrect and you must mark the failure as '
            'unfixable (see rule 4).\n'
            '\n'
            '2. UNIQUENESS. The corrected "old_str" must appear '
            'EXACTLY ONCE in its target block. Extend with '
            'surrounding context (preceding or following words/'
            'sentences) until verification confirms no other '
            'location contains the same substring.\n'
            '\n'
            '3. COMPLETENESS. You MUST address EVERY failure '
            'listed. Do not silently skip a failure. If you cannot '
            'fix one, mark it unfixable (rule 4) — never omit it.\n'
            '\n'
            '4. UNFIXABLE marker. If a failure cannot be fixed '
            '(the supposed wrong term is genuinely not in the '
            'block, or the original flag was a false positive), '
            'emit a correction with the same target block index, '
            'an empty replacements array, and a "reason" field '
            'starting with "UNFIXABLE: " followed by your specific '
            'explanation (e.g., "UNFIXABLE: the term \\"X\\" does '
            'not appear in block_N, the original flag was a false '
            'positive").\n'
            '\n'
            '5. SCOPE. Do NOT introduce new corrections beyond the '
            'failed list. Fix only what was rejected. The '
            '"glossary" array MUST be empty in the repair response.'
        ).format(tlang=target_lang)

    def _diff_changed_portion(self, old_str, new_str):
        """Return the substring of old_str that differs from new_str
        (after stripping common prefix and suffix). Used to identify
        the 'wrong term' the model intended to replace, which we
        then search for in the block to provide candidate anchors.

        Examples:
          ('לורד אשקום צעד', 'לורד אשקומב צעד') → 'אשקום' (with right
          context 'אשקומב' as the target form)
          ('יצחק', 'אייזק') → 'יצחק'
        """
        p = 0
        while (p < min(len(old_str), len(new_str))
               and old_str[p] == new_str[p]):
            p += 1
        s = 0
        max_s = min(len(old_str) - p, len(new_str) - p)
        while (s < max_s
               and old_str[len(old_str) - 1 - s]
                   == new_str[len(new_str) - 1 - s]):
            s += 1
        wrong = old_str[p:len(old_str) - s] if (
            len(old_str) - s) > p else ''
        return wrong

    def _find_candidate_anchors(self, failure, block_text,
                                window_chars=40, max_candidates=5):
        """Find substrings of block_text that likely correspond to
        the model's intended target. Strategy: identify the changed
        portion of old_str (the 'wrong term') and search for it in
        the block; return windowed snippets around each occurrence.

        Returns a list of strings (verbatim block excerpts) the
        model can copy from to produce a valid old_str. Empty if no
        candidates found.
        """
        wrong = self._diff_changed_portion(
            failure['old_str'], failure['new_str'])
        if not wrong or len(wrong) < 2:
            # Fall back to longest common token between old_str and
            # block_text — pick a substring of old_str that exists
            # in the block as a starting point.
            for size in (40, 30, 20, 10):
                for start in range(0, len(failure['old_str']) - size + 1):
                    sub = failure['old_str'][start:start + size]
                    if sub in block_text:
                        wrong = sub
                        break
                if wrong:
                    break
            if not wrong:
                return []
        candidates = []
        seen = set()
        start = 0
        while len(candidates) < max_candidates:
            pos = block_text.find(wrong, start)
            if pos < 0:
                break
            window_start = max(0, pos - window_chars)
            window_end = min(len(block_text),
                             pos + len(wrong) + window_chars)
            snippet = block_text[window_start:window_end]
            if snippet not in seen:
                seen.add(snippet)
                # Mark whether the snippet starts/ends mid-text.
                prefix = '…' if window_start > 0 else ''
                suffix = '…' if window_end < len(block_text) else ''
                candidates.append('{}{}{}'.format(prefix, snippet, suffix))
            start = pos + max(1, len(wrong))
        return candidates

    def _build_repair_message(self, failures, block_text_by_index):
        """Build the trailing user-message text for a repair call.

        Includes the verbatim text of every affected block inline
        (so the model can copy from local context rather than
        long-range attention over the full document set), plus
        candidate anchors — windowed excerpts around suspected
        wrong-term occurrences — to anchor the model's choice of
        old_str on real characters in the block.
        """
        # Group failures by block index so each block's full text
        # is shown once, with all its failures grouped under it.
        by_block = {}
        for f in failures:
            by_block.setdefault(f['block_index'], []).append(f)

        lines = [
            'REPAIR REQUEST. The replacements listed below failed '
            'host-side uniqueness validation. For each failure: '
            'emit a corrected old_str/new_str in the "corrections" '
            'array, copying old_str CHARACTER-BY-CHARACTER from the '
            'verbatim block text shown below. The block text shown '
            'is the SOURCE OF TRUTH; if a wrong term you previously '
            'flagged is not actually present in the block as '
            'written, mark the failure UNFIXABLE per the system '
            'prompt rules. You MUST address every failure (no '
            'silent omissions).',
            '',
        ]

        for idx in sorted(by_block.keys()):
            block_text = block_text_by_index.get(idx, '')
            block_failures = by_block[idx]
            lines.append('═══ block_{} (verbatim text below) ═══'
                         .format(idx))
            lines.append(block_text)
            lines.append('═══ end of block_{} ═══'.format(idx))
            lines.append('')
            lines.append('Failed in block_{}:'.format(idx))
            for f in block_failures:
                lines.append(
                    '  - old_str={old!r}'.format(old=f['old_str']))
                lines.append(
                    '    new_str={new!r}'.format(new=f['new_str']))
                lines.append(
                    '    reason: {reason}'.format(reason=f['reason']))
                anchors = self._find_candidate_anchors(f, block_text)
                if anchors:
                    lines.append('    Candidate anchors found in '
                                 'block_{} (verbatim excerpts you '
                                 'can copy from):'.format(idx))
                    for a in anchors:
                        lines.append('      • {}'.format(a))
                else:
                    lines.append('    No anchors found — the wrong '
                                 'term may not be in this block. '
                                 'Consider marking UNFIXABLE.')
            lines.append('')

        lines.append(
            'Return JSON in the same schema. The "glossary" array '
            'MUST be empty in this repair response. Address EVERY '
            'failure above — emit a fix or an UNFIXABLE marker, '
            'never silently skip one.')
        return '\n'.join(lines)

    def _validate_corrections(self, corrections, block_text_by_index):
        """Classify each replacement as resolved or failed using the
        uniqueness contract (count == 1 in target block).

        Returns (resolved, failures):
          - resolved: list of {index, reason, replacements} where every
            replacement's old_str is uniquely present in its block.
          - failures: list of {block_index, old_str, new_str, reason}
            for every replacement that failed validation.
        """
        resolved = []
        failures = []
        for c in corrections:
            idx = c.get('index')
            text = block_text_by_index.get(idx)
            if text is None:
                for r in (c.get('replacements') or []):
                    failures.append({
                        'block_index': idx,
                        'old_str': r.get('old_str', ''),
                        'new_str': r.get('new_str', ''),
                        'reason':
                            'unknown block index {}'.format(idx),
                    })
                continue
            valid = []
            for r in (c.get('replacements') or []):
                old_str = r.get('old_str', '')
                new_str = r.get('new_str', '')
                if not old_str:
                    continue
                count = text.count(old_str)
                if count == 1:
                    valid.append({
                        'old_str': old_str,
                        'new_str': new_str,
                    })
                else:
                    reason = (
                        'not present in block_{}'.format(idx)
                        if count == 0
                        else '{} matches in block_{} — needs more '
                             'surrounding context'.format(count, idx))
                    failures.append({
                        'block_index': idx,
                        'old_str': old_str,
                        'new_str': new_str,
                        'reason': reason,
                    })
            if valid:
                resolved.append({
                    'index': idx,
                    'reason': c.get('reason', ''),
                    'replacements': valid,
                })
        return resolved, failures

    def _merge_corrections(self, existing, additions):
        """Merge `additions` into `existing` by block index. Both are
        lists of {index, reason, replacements}. Replacements are
        concatenated; reason from `existing` wins (the original
        review's reason is more informative than the repair's).
        """
        by_index = {c['index']: dict(c, replacements=list(c['replacements']))
                    for c in existing}
        for a in additions:
            idx = a['index']
            if idx in by_index:
                by_index[idx]['replacements'].extend(a['replacements'])
            else:
                by_index[idx] = dict(a, replacements=list(a['replacements']))
        return list(by_index.values())

    def consistency_review(self, items, on_progress=None,
                           cancel_request=None):
        """Glossary-first consistency review. Returns a glossary of
        recurring entities (character names, places, titles, terms)
        that appear in multiple forms in the translation, with each
        entity's canonical form and the variant forms found in the
        text that should be normalized.

        Architecture: the LLM does *judgment* (which form is canonical
        for each entity, what variants exist) — the host does
        *substitution* via deterministic word-boundary regex
        replacement (in lib/translation.py). This split is convergent
        in O(1): once a glossary is applied, all listed variants are
        replaced and there is nothing left to "find inconsistent."

        Style/punctuation/quotes are explicitly out of scope (the
        prompt forbids them). Stylistic axes are infinite and would
        prevent convergence.

        Refusal retry: up to two prompt variants are tried (default,
        then stronger editorial framing) if the first response looks
        like a refusal.

        Returns a dict:
          - 'glossary':     list of {canonical, variants, type, gender,
                            notes} entries. variants is a list of
                            verbatim character sequences from the
                            translation that should be replaced with
                            canonical.
          - 'corrections':  always [] (kept for backwards compatibility
                            with consumers expecting the old shape).
          - 'raw_response': raw text when parsing failed.

        Sees only translated text — no copyright risk, since this is
        the user's own translation output, not the original
        copyrighted book.
        """
        if not items:
            return {'glossary': [], 'corrections': []}

        # Build cached document blocks. cache_control on the last
        # block caches the whole prefix; the trailing text is
        # uncached and cheap.
        docs = []
        for item in items:
            docs.append({
                'type': 'document',
                'source': {
                    'type': 'content',
                    'content': [{
                        'type': 'text',
                        'text': item['translation'],
                    }],
                },
                'title': 'block_{}'.format(item['index']),
            })
        if docs:
            docs[-1]['cache_control'] = {'type': 'ephemeral', 'ttl': '1h'}

        trailing = (
            'Above are {} translated blocks of text, provided as '
            'documents titled "block_N" where N is the 0-indexed '
            'block position. Produce a glossary of recurring '
            'entities that appear in multiple forms — see the '
            'system prompt for scope rules and variant reporting '
            'requirements. Do NOT report stylistic / punctuation / '
            'quote variation. Do NOT emit corrections.'
        ).format(len(items))

        target_lang = self.target_lang

        # Refusal-retry loop: up to two prompt variants.
        last_raw = ''
        for refusal_attempt in range(2):
            if cancel_request and cancel_request():
                return {'glossary': [], 'corrections': []}
            if refusal_attempt > 0 and on_progress:
                on_progress(_(
                    '  Retrying consistency review with stronger '
                    'editorial framing (attempt {})...'
                ).format(refusal_attempt + 1))
            system_prompt = self._consistency_system_prompt(
                target_lang, attempt=refusal_attempt)
            content_blocks = docs + [
                {'type': 'text', 'text': trailing}]
            result = self._consistency_api_call(
                content_blocks, system_prompt, on_progress,
                cancel_request)
            last_raw = result.get('raw_response') or ''
            empty = not (result.get('glossary') or [])
            if (empty and last_raw
                    and self._is_review_refusal(last_raw)
                    and refusal_attempt == 0):
                if on_progress:
                    on_progress(_(
                        '  Detected refusal in raw response — '
                        'retrying with stronger editorial framing'))
                continue
            return {
                'glossary': result.get('glossary') or [],
                'corrections': [],
                'raw_response': (last_raw if empty else ''),
            }

        return {
            'glossary': [],
            'corrections': [],
            'raw_response': last_raw,
        }

    def _character_index(self, brief):
        """Build a fast pre-filter index from the brief's characters.

        Returns:
          - id_to_char: {char_id: char_dict}
          - mention_regex: compiled `\\b(form1|form2|...)\\b` matching
            any character mention form (canonical_name, source_name,
            aliases). None if there are no usable forms.

        Used by agreement_review to skip paragraphs that mention no
        named character — most prose paragraphs (description / non-
        dialogue action) qualify, so this typically halves or better
        the token cost of the pass.
        """
        if not brief or not isinstance(brief, dict):
            return {'id_to_char': {}, 'mention_regex': None}
        chars = brief.get('characters') or []
        id_to_char = {}
        forms = []
        seen = set()
        for c in chars:
            if not isinstance(c, dict):
                continue
            cid = c.get('id') or ''
            if not cid:
                continue
            id_to_char[cid] = c
            for key in ('canonical_name', 'source_name'):
                v = c.get(key)
                if isinstance(v, str) and v.strip() and v not in seen:
                    seen.add(v)
                    forms.append(v)
            for v in (c.get('aliases') or []):
                if isinstance(v, str) and v.strip() and v not in seen:
                    seen.add(v)
                    forms.append(v)
            for v in (c.get('variants') or []):
                if isinstance(v, str) and v.strip() and v not in seen:
                    seen.add(v)
                    forms.append(v)
        # Sort longest-first so re alternation prefers the longest
        # match at any position (relevant when one form is a prefix
        # of another).
        forms.sort(key=len, reverse=True)
        mention_regex = None
        if forms:
            try:
                mention_regex = re.compile(
                    r'\b(' + '|'.join(re.escape(f) for f in forms)
                    + r')\b',
                    re.UNICODE)
            except re.error:
                mention_regex = None
        return {'id_to_char': id_to_char, 'mention_regex': mention_regex}

    def _format_brief_morphology(self, chars):
        """Render the brief's character morphology as a compact,
        model-facing table. Only fields relevant to agreement (id,
        canonical_name, gender, number) are included. Aliases are
        listed so the model can map mention forms in the translation
        to a single canonical character.
        """
        lines = []
        for c in chars:
            if not isinstance(c, dict):
                continue
            cid = c.get('id', '')
            name = c.get('canonical_name', '')
            gender = c.get('gender', '') or 'unknown'
            number = c.get('number', '') or 'sg'
            aliases = [a for a in (c.get('aliases') or [])
                       if isinstance(a, str) and a.strip()]
            line = '  • [{}] {} — gender: {}, number: {}'.format(
                cid, name, gender, number)
            if aliases:
                line += '; aliases: ' + ', '.join(aliases)
            lines.append(line)
        return '\n'.join(lines) if lines else '  (no character data)'

    def _agreement_system_prompt(self, target_lang, morphology):
        """System prompt for the Agreement Pass. Editorial framing
        aligned with the brief-build / consistency-review prompts:
        the user owns the translation, this pass is a quality-
        assurance step over their own output, not over copyrighted
        source material. Sees only translated text + morphology.
        """
        return (
            'You are an editorial assistant performing an AGREEMENT '
            'PASS on a {tlang} translation. The user has produced '
            'a {tlang} translation of a book in their personal '
            'library; you are reviewing the translated text for '
            'residual GENDER, NUMBER, and PRONOUN agreement drift '
            'against a canonical character morphology table that '
            'was prepared earlier (the Translation Brief). You see '
            'only the translated text and the morphology table — '
            'no copyrighted source material is present.\n'
            '\n'
            '═══ CANONICAL CHARACTER MORPHOLOGY ═══\n'
            '{morph}\n'
            '\n'
            '═══ SCOPE ═══\n'
            'Flag and fix in-place where the translation\'s verb '
            'forms, pronouns, possessives, or adjective endings '
            'disagree with a character\'s canonical gender or '
            'number from the table above. Examples in '
            'morphologically rich {tlang}:\n'
            '  • A character marked feminine but referenced with '
            'a masculine verb form.\n'
            '  • A singular character referenced with plural '
            'agreement (or vice versa).\n'
            '  • A pronoun whose grammatical gender contradicts '
            'the canonical gender of its antecedent.\n'
            '\n'
            'DO NOT FLAG:\n'
            '  • The character\'s name itself (handled by the '
            'terminology pass).\n'
            '  • Style, word choice, naturalness, register.\n'
            '  • Punctuation, quote marks, whitespace.\n'
            '  • Anything that is not a target-language morphology '
            '/ agreement issue.\n'
            '  • Characters whose gender is "unknown" or "mixed" — '
            'the brief is not authoritative about agreement for '
            'those, so leave them alone.\n'
            '  • Direct-speech inflection that uses the speaker\'s '
            'addressee\'s morphology (a male character speaking to '
            'a female uses feminine 2nd-person forms — that is '
            'correct).\n'
            '\n'
            '═══ FIX SHAPE ═══\n'
            'Each fix is a single-occurrence find/replace operation '
            'against ONE block_N. Your "old_str" MUST appear '
            'EXACTLY ONCE in that block; if the agreement error '
            'lives on a common verb form that occurs multiple '
            'times in the block, extend with surrounding context '
            '(preceding noun, following word) until uniqueness '
            'holds.\n'
            '\n'
            'The "new_str" must change ONLY the agreement-bearing '
            'tokens (verb inflection, pronoun, adjective ending) '
            '— preserve word order, content words, and punctuation '
            'in the rest of the substring. Surgical edits, not '
            'paraphrases.\n'
            '\n'
            'Set "kind" to "gender", "number", "pronoun", or '
            '"other" matching the class of error. Set "character_'
            'id" to the c_NNN id of the character whose canonical '
            'morphology drove the fix; empty string if the fix is '
            'not character-specific.\n'
            '\n'
            'Empty fixes array is fine if the translation is '
            'already consistent. Do NOT invent issues to fill the '
            'response.'
        ).format(tlang=target_lang,
                 morph=morphology or '  (no character data)')

    def agreement_review(self, items, brief, on_progress=None,
                         cancel_request=None):
        """Detect residual gender/number/pronoun agreement drift in
        translated paragraphs against the canonical character
        morphology in the brief, and emit single-occurrence find/
        replace fixes. Convergent: each fix is bounded, validated by
        the host (uniqueness contract), and applied via plain string
        replacement.

        :items: list of {index, translation} dicts. Caller passes
            only paragraphs with non-empty translations.
        :brief: the canonical Translation Brief dict (drives which
            characters to check and their canonical morphology).

        Returns a dict:
          - 'fixes': validated [{block_index, character_id, kind,
                old_str, new_str, reason}] entries, each old_str
                verified unique within its target block.
          - 'unfixable': failures that didn't pass uniqueness
                validation (surfaced for log inspection).
          - 'considered': how many paragraphs were sent to the model
                after pre-filtering.
          - 'raw_response': raw text on parse failure.
        """
        empty_result = {
            'fixes': [], 'unfixable': [], 'considered': 0,
            'raw_response': '',
        }
        if not items or not brief:
            return empty_result
        chars = brief.get('characters') if isinstance(brief, dict) \
            else None
        if not chars:
            return empty_result

        cidx = self._character_index(brief)
        mention_regex = cidx['mention_regex']
        if mention_regex is None:
            return empty_result

        # Pre-filter: keep only translated paragraphs that mention
        # at least one named character. Drops description-only
        # paragraphs without a single named character mention.
        relevant = []
        for item in items:
            text = (item.get('translation') or '').strip()
            if not text:
                continue
            if mention_regex.search(text):
                relevant.append({
                    'index': item['index'],
                    'translation': item['translation'],
                })
        if on_progress:
            on_progress(_(
                'Agreement Pass: {}/{} translated paragraphs '
                'mention named characters; reviewing those.'
            ).format(len(relevant), len(items)))
        if not relevant:
            return empty_result

        target_lang = self.target_lang

        # Build cached document blocks. cache_control on the last
        # block caches the prefix (system prompt + morphology +
        # documents); a future revalidate or repair call within the
        # 1-hour TTL would hit the cache.
        docs = []
        block_text_by_index = {}
        valid_indices = set()
        for item in relevant:
            idx = item['index']
            text = item['translation']
            block_text_by_index[idx] = text
            valid_indices.add(idx)
            docs.append({
                'type': 'document',
                'source': {
                    'type': 'content',
                    'content': [{'type': 'text', 'text': text}],
                },
                'title': 'block_{}'.format(idx),
            })
        if docs:
            docs[-1]['cache_control'] = {
                'type': 'ephemeral', 'ttl': '1h'}

        morphology = self._format_brief_morphology(chars)
        system_prompt = self._agreement_system_prompt(
            target_lang, morphology)
        trailing = (
            'Above are {} translated blocks (titled "block_N", N '
            'is the 0-indexed block position). Identify GENDER / '
            'NUMBER / PRONOUN agreement errors against the '
            'canonical character morphology table in the system '
            'prompt and emit single-occurrence find/replace fixes. '
            'Empty fixes array is fine if the translation is '
            'already consistent.'
        ).format(len(relevant))
        content_blocks = docs + [{'type': 'text', 'text': trailing}]

        if cancel_request and cancel_request():
            return empty_result

        result = self._consistency_api_call(
            content_blocks, system_prompt, on_progress,
            cancel_request, valid_indices=valid_indices,
            schema=self._AGREEMENT_PASS_OUTPUT_SCHEMA)
        parsed = result.get('parsed') or {}
        raw_fixes = parsed.get('fixes') if isinstance(parsed, dict) \
            else None
        if not isinstance(raw_fixes, list):
            return {
                'fixes': [], 'unfixable': [],
                'considered': len(relevant),
                'raw_response': result.get('raw_response') or '',
            }

        # Validate uniqueness via the existing helper (which expects
        # the consistency-review corrections shape: a list of
        # {index, replacements:[{old_str, new_str}]} entries). Group
        # fixes by block_index, run validation, then map back to the
        # fixes shape carrying through metadata (character_id, kind,
        # reason) keyed by (block_index, old_str, new_str).
        fix_meta = {}
        by_block = {}
        for f in raw_fixes:
            if not isinstance(f, dict):
                continue
            idx = f.get('block_index')
            if not isinstance(idx, int) or idx not in valid_indices:
                continue
            old_str = f.get('old_str') or ''
            new_str = f.get('new_str') or ''
            if not old_str or old_str == new_str:
                continue
            fix_meta[(idx, old_str, new_str)] = f
            by_block.setdefault(idx, []).append({
                'old_str': old_str, 'new_str': new_str,
            })
        corrections = [
            {'index': idx, 'reason': '', 'replacements': reps}
            for idx, reps in by_block.items()
        ]
        resolved, failures = self._validate_corrections(
            corrections, block_text_by_index)

        flat_fixes = []
        for c in resolved:
            idx = c['index']
            for r in c.get('replacements', []):
                meta = fix_meta.get(
                    (idx, r['old_str'], r['new_str']), {})
                flat_fixes.append({
                    'block_index': idx,
                    'character_id': meta.get('character_id') or '',
                    'kind': meta.get('kind') or 'other',
                    'old_str': r['old_str'],
                    'new_str': r['new_str'],
                    'reason': meta.get('reason') or '',
                })

        return {
            'fixes': flat_fixes,
            'unfixable': failures,
            'considered': len(relevant),
            'raw_response': result.get('raw_response') or '',
        }

    def _brief_system_prompt(self, source_lang, target_lang,
                             attempt=0):
        """System prompt for Phase 0 brief-build spike.

        Editorial-assistant framing aligned with the existing
        copyright-mitigation strategy used by full_book_context: the
        user is producing a {tlang} translation of their personal
        copy of a book, and is asking for an editorial reference
        document — not for translation or reproduction of source
        material. The brief is a tool to help the translator make
        consistent decisions; it does not contain translated text.
        """
        scope_rules = (
            'INCLUDE in the brief:\n'
            '  • Recurring named characters / referenced persons '
            '(must appear in two or more distinct passages)\n'
            '  • Recurring places, organizations, distinctive '
            'objects, invented terms\n'
            '  • Titles, honorifics, formality conventions\n'
            '  • One-line style decisions (tense strategy, POV, '
            'default formality, idiom strategy)\n'
            '\n'
            'EXCLUDE absolutely:\n'
            '  • Quote-mark conventions, punctuation styles, '
            'whitespace, paragraph formatting\n'
            '  • Stylistic word choices that aren\'t proper nouns\n'
            '  • One-off variations that appear only once\n'
            '  • The translated text itself — the brief is a '
            'reference document, NOT a draft\n'
        )
        gender_rules = (
            'For each character entity, fill the "gender" field with '
            'the canonical grammatical gender to use throughout the '
            '{tlang} translation: "masculine" / "feminine" / '
            '"neuter" / "mixed" / "unknown". This drives target-'
            'language agreement (verb forms, pronouns, adjectives) '
            'in morphologically rich target languages such as '
            'Hebrew, Arabic, Spanish, French, Russian, German.\n'
        )
        canonical_form_rules = (
            'For each entity, "canonical_name" / "canonical" must '
            'be the EXACT target-language ({tlang}) form to use '
            'consistently throughout the translation. "source_name" '
            '/ "source_form" is the form that appears in the '
            'source ({slang}) text. Aliases in the source go in '
            'the "aliases" array.\n'
            '\n'
            'NEVER use parentheses, slashes, or "or" to provide '
            'alternative canonical forms. Pick ONE form. Strings '
            'like "X (Y)", "X / Y", or "X or Y" in canonical_name '
            'or canonical fields are FORBIDDEN. If two forms could '
            'both work, choose the one that appears more often in '
            'the source.\n'
            '\n'
            'For canonical_name and canonical, prefer NATURAL '
            '{tlang} forms over phonetic transliterations of '
            'common source-language words. Transliterate only '
            'when (a) the source word is a proper noun with no '
            '{tlang} equivalent, or (b) {tlang} convention is to '
            'transliterate (e.g. brand names, song titles, Latin '
            'phrases that should remain Latin).\n'
            '\n'
            'do_not_translate=true ONLY when canonical equals '
            'source_form character-for-character in source script '
            '(e.g. "Le Morte d\'Arthur" stays "Le Morte d\'Arthur", '
            '"Fiat Lux" stays "Fiat Lux"). do_not_translate=false '
            'for transliterations into {tlang} script — those '
            'ARE translations, just phonetic ones.\n'
        )
        pipeline_rules = (
            'THREE-STAGE PIPELINE. Your output MUST always set the '
            '"stage" field to one of "build", "review_language", '
            'or "review_logic", and MUST fill EXACTLY ONE of '
            '"brief" or "change_list" (populating the field that '
            'matches the stage; the other field MUST be null).\n'
            '\n'
            'CRITICAL RULES:\n'
            '  • stage="build" → brief is a fully-populated object, '
            'change_list is null. Set the brief\'s required fields '
            '(source_summary, style_decisions, characters, '
            'terminology) and any optional fields you can populate.\n'
            '  • stage="review_language" → brief is null, '
            'change_list is an array (possibly empty). Each issue '
            'identifies one TARGET-LANGUAGE quality problem at a '
            'specific field_path inside the brief produced '
            'earlier (semantic accuracy, naturalness, '
            'transliteration, grammar, typos).\n'
            '  • stage="review_logic" → brief is null, change_list '
            'is an array (possibly empty). Each issue identifies '
            'one INTERNAL-CONSISTENCY problem (relationship-graph '
            'errors, role-relationship contradictions, missing '
            'reciprocity, dangling to_id references).\n'
            '  • You will be told in each turn\'s user message '
            'which stage to produce. Follow that instruction '
            'exactly — never mix stages.\n'
            '  • Filling BOTH brief and change_list is forbidden. '
            'Filling NEITHER is forbidden. Exactly one, never '
            'both, never neither.\n'
            '  • Setting "stage" to one value and then populating '
            'the wrong field is a contract violation. Match them.\n'
        )
        structured_field_rules = (
            'IDS. Every characters[] entry needs an "id" of form '
            '"c_001", "c_002", ... numbered sequentially in the '
            'order entities first appear in the source. Every '
            'terminology[] entry needs a "t_001", "t_002", ... id '
            'in the same way. These ids are used for cross-'
            'references (relationships) and are stable across '
            'downstream processing.\n'
            '\n'
            'ROLE (characters). The "role" field begins with one of '
            'the standard story-bible role descriptors: '
            '"protagonist", "antagonist", "deuteragonist", '
            '"supporting", "mentor", "love interest", "sidekick", '
            '"minor", "walk-on", "cameo". Optionally followed by a '
            'colon and a one-line free-form description. Form: '
            '<descriptor>: <one-line role description>. The '
            'leading descriptor lets downstream stages recognize '
            'narrative weight without parsing prose.\n'
            '\n'
            'RELATIONSHIPS (characters). Capture only STRUCTURAL '
            'ties — durable connections that define the social, '
            'family, and professional graph. Do NOT encode plot '
            'events (killings, betrayals, infections, romantic '
            'arcs, transformations) as relationships; those belong '
            'in the role description as prose. The directional '
            'ambiguity of plot events ("X killed Y" vs '
            '"X was killed by Y") makes them error-prone in the '
            'graph; durable structural ties have unambiguous '
            'meaning.\n'
            '\n'
            'For each character, fill "relationships" with an '
            'array of {{to_id, type}} entries pointing to other '
            'characters in this brief. Use to_id values that '
            'match characters[].id. Empty array is fine for '
            'characters with no structural ties to other '
            'characters in the brief.\n'
            '\n'
            'UNIVERSAL CORE VOCABULARY (use these when applicable):\n'
            '  • Family: "parent-of", "child-of", "sibling-of" '
            '(or "brother-of"/"sister-of" if gender is relevant), '
            '"spouse-of", "grandparent-of", "grandchild-of"\n'
            '  • Mentorship / Education: "mentor-of", '
            '"apprentice-of", "teacher-of", "student-of"\n'
            '  • Friendship: "friend-of", "best-friend-of", '
            '"companion-of"\n'
            '  • Romantic: "partner-of", "lover-of", "ex-of"\n'
            '  • Stance: "ally-of", "rival-of", "enemy-of"\n'
            '  • Hierarchy / Workplace: "works-for", "employer-'
            'of", "colleague-of", "leader-of", "member-of"\n'
            '  • Care: "guardian-of", "ward-of"\n'
            '\n'
            'EXTENDING THE VOCABULARY. The universal core covers '
            'most fiction. For genre-specific connections that the '
            'core does not fit (e.g. a magical bond, a sworn-'
            'enemy oath, a monarch/subject relation), extend with '
            'a hyphenated lowercase string that follows the same '
            '"X-of" convention. Examples by genre: epic fantasy '
            '"sworn-vassal-of", romance "betrothed-to", sci-fi '
            '"linked-to", military "commands". These are still '
            'STRUCTURAL ties (the relationship is ongoing), not '
            'plot events.\n'
            '\n'
            'FIRST OCCURRENCE & MENTION COUNT. For both characters '
            'and terminology, fill "first_occurrence_index" with '
            'the block_N index where the entity first appears, and '
            '"mention_count" with an approximate count of mentions '
            'across the source. These are best-effort estimates — '
            'do not aim for precision; the goal is a continuous '
            'signal of narrative weight (protagonists have '
            'hundreds of mentions; walk-ons have a handful).\n'
            '\n'
            'DO_NOT_TRANSLATE (terminology only). Set '
            '"do_not_translate": true for entities that should be '
            'kept in source-language form and not translated — '
            'brand names, song titles, product names, code '
            'identifiers, proper nouns the user wants preserved '
            'verbatim. False or omitted otherwise.\n'
            '\n'
            'THEMES & CENTRAL CONFLICT (source_summary). Populate '
            '"themes" with 3-7 short phrases naming the book\'s '
            'subjects — high-level topical descriptors that '
            'capture what the work is *about* at the abstract '
            'level (one phrase per theme, two or three words). '
            'Populate "central_conflict" with a one-sentence '
            'statement of what is at stake in the narrative. '
            'These drive register and tone decisions downstream.\n'
        )
        if attempt == 0:
            return (
                'You are an editorial assistant integrated into a '
                'personal e-book translation tool. The user is '
                'producing a {tlang} translation of a book from '
                'their personal library. You are NOT being asked to '
                'translate or reproduce the source material; you are '
                'being asked to produce a TRANSLATION BRIEF — a '
                'compact reference document the user will consult '
                'while translating, capturing canonical names, '
                'recurring terminology, character profiles (with '
                'grammatical gender for target-language agreement), '
                'and style decisions.\n'
                '\n'
                'The brief is a quality-assurance tool: it lets the '
                'user maintain consistency across thousands of '
                'segments. It does not contain translated text. '
                'Producing it is a standard editorial-assistance '
                'task analogous to a human translator\'s '
                'preparation notebook.\n'
                '\n'
                '═══ SCOPE ═══\n'
                + scope_rules +
                '\n'
                '═══ CANONICAL FORMS ═══\n'
                + canonical_form_rules +
                '\n'
                '═══ STRUCTURED FIELDS ═══\n'
                + structured_field_rules +
                '\n'
                '═══ TWO-STAGE PIPELINE ═══\n'
                + pipeline_rules +
                '\n'
                '═══ GENDER (CRITICAL) ═══\n'
                + gender_rules +
                '\n'
                'Source language: {slang}. Target language: {tlang}. '
                'Input is a series of document content blocks, one '
                'per passage of the source text, titled "block_N". '
                'Read all blocks before deciding canonical forms — '
                'the most common form across the work is usually '
                'the right canonical.'
            ).format(slang=source_lang, tlang=target_lang)
        # Stronger framing for refusal retry.
        return (
            'You are an editorial assistant. The user is the owner '
            'of a {tlang} translation project for a book in their '
            'personal library and is requesting a TRANSLATION BRIEF '
            '— a structured editorial reference with canonical '
            'character names, recurring terminology, and style '
            'decisions, used to maintain consistency across the '
            'translation. This is a standard quality-assurance '
            'request. You will not translate or reproduce source '
            'material; you will produce a structured reference '
            'document only.\n'
            '\n'
            '═══ SCOPE ═══\n'
            + scope_rules +
            '\n'
            '═══ CANONICAL FORMS ═══\n'
            + canonical_form_rules +
            '\n'
            '═══ STRUCTURED FIELDS ═══\n'
            + structured_field_rules +
            '\n'
            '═══ TWO-STAGE PIPELINE ═══\n'
            + pipeline_rules +
            '\n'
            '═══ GENDER (CRITICAL) ═══\n'
            + gender_rules +
            '\n'
            'Source: {slang}. Target: {tlang}. The brief is the '
            'deliverable; no translated text is required.'
        ).format(slang=source_lang, tlang=target_lang)

    def _brief_review_language_user_message(self, target_lang):
        """User message for the LANGUAGE-review turn of the
        multi-turn brief pipeline. The model has already produced
        the brief in turn 1 (with stage="build"); this message
        instructs it to switch roles and produce a critic's change
        list (with stage="review_language") for target-language
        quality issues.

        Notable design choices:
          - This is a USER MESSAGE, not a system prompt — Anthropic's
            API doesn't allow the system prompt to change mid-call,
            but a strongly-worded role shift in a user message
            achieves the same effect (see Self-Refine literature).
          - The critic's scope is target-language quality ONLY. The
            analytical content (who the characters are, gender, role,
            terms list) is trusted; do not re-analyze the source.
          - Output schema is the same _TRANSLATION_BRIEF_SCHEMA, but
            this turn must produce stage="review_language" with
            brief=null and change_list populated.
        """
        evaluation_criteria = (
            'Apply these criteria ONLY to canonical-form fields: '
            'characters[].canonical_name, characters[].aliases[] '
            '(those in {tlang}), and terminology[].canonical. '
            'Do NOT evaluate role, voice, notes, translator_notes, '
            'source_summary, or style_decisions — those are prose, '
            'not canonical labels.\n'
            '\n'
            '1. SEMANTIC ACCURACY. The canonical {tlang} string must '
            'denote what the source intends. Flag entries where the '
            'chosen {tlang} word means something different from the '
            'source concept (e.g., a wrong-word translation where '
            'the chosen target term denotes a related but distinct '
            'concept than the source word).\n'
            '\n'
            '2. NATURALNESS. The canonical string must read as '
            'natural {tlang} for its register. Flag overloaded '
            'metaphors, awkward calques, and word choices that a '
            'fluent {tlang} speaker would not use for this concept '
            '— e.g., a heavyweight or grandiose word applied to a '
            'mundane referent, or a too-literal calque where idiom '
            'requires a different lexical choice.\n'
            '\n'
            '3. SINGLE CANONICAL FORM. Each entry must have exactly '
            'ONE canonical string. Flag entries containing '
            'parenthetical alternatives, slashes, or "or"-conjunctions '
            '(forms like "X (Y)", "X / Y", "X or Y"). Pick one form.\n'
            '\n'
            '4. INTERNAL CONSISTENCY. Transliteration conventions, '
            'diacritic usage, and orthographic patterns must be '
            'consistent across the brief. Flag entries that diverge '
            'from the predominant convention used by other entries — '
            'e.g., one name carries vowel marks while ten others do '
            'not; or two related place names are transliterated under '
            'different systems. Compare entries against each other; '
            'do not impose an external "correct" system.\n'
            '\n'
            '5. GRAMMAR AND ORTHOGRAPHY. The canonical string must '
            'be grammatically correct {tlang}. Flag construct-state '
            'errors, agreement errors, and other grammatical '
            'mistakes specific to {tlang}. Flag typos: dropped '
            'letters, doubled letters, transposed letters, wrong '
            'characters.\n'
            '\n'
            '6. PROPER-NOUN INTEGRITY. For transliterated proper '
            'nouns, the {tlang} form must reasonably represent the '
            'source pronunciation. Flag dropped syllables, added '
            'unwarranted ones, and wrong consonant/vowel '
            'substitutions.\n'
            '\n'
            'DO NOT FLAG:\n'
            '  • Whether an entity belongs in the brief (analytical '
            'content; trust it).\n'
            '  • A character\'s gender, role, number, or aliases '
            'list (analytical content; trust it).\n'
            '  • Whether one transliteration system is "better" '
            'than another — flag only inconsistency within the '
            'brief.\n'
            '  • Translation-philosophy choices (foreignize vs '
            'domesticate) when applied consistently.\n'
            '  • Prose fields (role, voice, notes, translator_notes, '
            'source_summary, style_decisions).\n'
        ).format(tlang=target_lang)
        return (
            'NOW SWITCH ROLES. You are no longer the brief-builder. '
            'You are now a native {tlang} speaker acting as a '
            'critic, reviewing the brief you just produced for '
            'target-language quality issues. Have no investment in '
            'the prior choices — be willing to flag your own '
            'output.\n'
            '\n'
            'OUTPUT CONTRACT for this turn:\n'
            '  • stage = "review_language"\n'
            '  • brief = null\n'
            '  • change_list = an array of issues (possibly empty, '
            'meaning the brief looks correct)\n'
            'Filling brief in this turn is a contract violation.\n'
            '\n'
            'YOUR SCOPE is target-language quality ONLY. Assume the '
            'analytical content of the brief is correct (which '
            'characters exist, their gender, their role, which '
            'terms recur, the style decisions). Do not propose new '
            'entries or remove existing ones. Do not re-translate '
            'things from scratch. Your job is to evaluate the '
            '{tlang} STRINGS already in the brief.\n'
            '\n'
            '═══ EVALUATION CRITERIA ═══\n'
            + evaluation_criteria +
            '\n'
            'For each issue, set:\n'
            '  • action = "replace" (the language critic ONLY '
            'emits replacements; it never inserts new entries).\n'
            '  • field_path = JSON-pointer-style path to the leaf '
            'string being corrected, e.g. '
            '"characters[3].canonical_name", '
            '"terminology[7].canonical". Indices are 0-based.\n'
            '  • current = the verbatim value at that path.\n'
            '  • suggested = the corrected value.\n'
            '  • insert_value = null (always null for replace).\n'
            '  • category = one of "typo", "semantic", '
            '"naturalness", "consistency", "dual_form", "grammar", '
            '"transliteration".\n'
            '  • reason = single sentence explaining the fix.\n'
            '\n'
            'If the brief looks correct, return change_list as an '
            'empty array []. Do NOT invent issues to fill the '
            'response.'
        ).format(tlang=target_lang)

    def _brief_review_logic_user_message(self, target_lang):
        """User message for the LOGIC-review turn (third turn) of
        the multi-turn brief pipeline. The model has already
        produced the brief AND a target-language review change
        list. This message instructs it to switch roles AGAIN and
        produce an internal-consistency critique focused on the
        relationship graph and role/relationship coherence.

        Scope is deliberately disjoint from the language critic:
        the language critic worked on canonical strings; the logic
        critic works on STRUCTURE — relationships[] arrays, role
        text vs relationship types, ID references. They share the
        same change_list output schema but flag different things.

        The critic emits string-valued REPLACEMENTS only (we don't
        support array-element deletion via field_path). For
        spurious relationships, the critic should change the type
        to a more accurate one rather than try to remove the entry.
        """
        evaluation_criteria = (
            'YOUR SCOPE: validate INTERNAL consistency of the '
            'brief\'s relationship graph. Relationship types are '
            'free-form strings; there is NO external vocabulary '
            'to check against. Your concern is whether the graph '
            'is self-consistent within this brief.\n'
            '\n'
            'KEY DISTINCTION. Relationships represent STRUCTURAL '
            'TIES between characters (parent, sibling, mentor, '
            'friend, ally, employer, etc.) — durable connections '
            'that define the social/family/professional graph. '
            'Plot events (killings, betrayals, infections, '
            'romantic arcs) belong in role descriptions, NOT in '
            'relationships. If you see a relationship that '
            'encodes a plot event (e.g. type contains "killed-by", '
            '"infected-by", "betrayed-by"), suggest replacing the '
            'type with a structural tie that captures the '
            'underlying connection ("rival-of", "enemy-of", '
            '"former-ally-of"), and note that the plot event '
            'itself should already be in the role text.\n'
            '\n'
            '1. ROLE-RELATIONSHIP COHERENCE. For each character, '
            'the relationships array should be consistent with the '
            'role description. Flag entries where:\n'
            '  • A character whose role describes them as the '
            'protagonist\'s antagonist has a friendly-tie type '
            '("ally-of", "friend-of", "mentor-of") pointing TO '
            'the protagonist (or vice versa).\n'
            '  • A relationship type is implausible given the '
            'role description.\n'
            '\n'
            '2. RECIPROCITY. If character A has a relationship of '
            'type X pointing to B, character B should have a '
            'consistent reciprocal pointing back to A. Common '
            'reciprocal pairs:\n'
            '  • Family symmetric: brother-of/sister-of/sibling-'
            'of (mutually), spouse-of (symmetric)\n'
            '  • Family inverse: parent-of ↔ child-of, '
            'grandparent-of ↔ grandchild-of\n'
            '  • Mentorship inverse: mentor-of ↔ apprentice-of, '
            'teacher-of ↔ student-of\n'
            '  • Stance symmetric: friend-of, best-friend-of, '
            'ally-of, rival-of, enemy-of\n'
            '  • Workplace inverse: works-for ↔ employer-of '
            '(or leader-of)\n'
            '  • Care inverse: guardian-of ↔ ward-of\n'
            '\n'
            'VERIFICATION REQUIRED BEFORE FLAGGING (CRITICAL). For '
            'EACH potential reciprocity flag, you MUST first '
            'inspect the TARGET character\'s relationships array '
            '(not just the source\'s) to confirm the reciprocal is '
            'genuinely absent. Procedure:\n'
            '  1. Identify the source: A.relationships[i] = '
            '{{to_id: B, type: X}}.\n'
            '  2. Look up character B by id in the brief.\n'
            '  3. Scan B.relationships for ANY entry whose '
            'to_id == A.\n'
            '  4. Decide:\n'
            '     - B has NO entry pointing to A → reciprocal '
            'GENUINELY MISSING. Emit action="insert".\n'
            '     - B has an entry pointing to A with the WRONG '
            'type → emit action="replace" on that entry\'s type.\n'
            '     - B has an entry pointing to A with a CONSISTENT '
            'type → reciprocity is SATISFIED. DO NOT FLAG. Move '
            'on.\n'
            '\n'
            'COMMON FALSE-POSITIVE PATTERN TO AVOID. The build '
            'phase typically already populates obvious reciprocal '
            'pairs on both sides (sibling-of pairs, parent-of/'
            'child-of pairs, spouse-of pairs, etc.). Do NOT '
            'mechanically emit "missing reciprocity" inserts based '
            'on the source side\'s entry alone — the reciprocal is '
            'often already there. Only flag when verification step '
            '3 above explicitly returns no match in the target\'s '
            'relationships array.\n'
            '\n'
            '3. SELF-REFERENCE. A character must NOT have a '
            'relationship to themselves. Flag any '
            'relationships[i].to_id that equals the character\'s '
            'own id.\n'
            '\n'
            '4. ID INTEGRITY. All relationships[i].to_id values '
            'must reference an actual character id present in the '
            'characters array. Flag dangling references — to_ids '
            'that don\'t resolve.\n'
            '\n'
            '5. PLOT-EVENT IN GRAPH (NEW). Flag any relationship '
            'whose type encodes a plot event rather than a '
            'structural tie. Examples: "killed-by", "killed-by-'
            'proxy", "killer-of", "infected-by", "betrayed-by", '
            '"transformed-by", "rescued-by". Suggest replacing '
            'with the closest structural tie (typically '
            '"rival-of", "enemy-of", or "former-ally-of") and '
            'note that the plot event is already / should be in '
            'the role description.\n'
            '\n'
            'DEDUPLICATION. Emit AT MOST ONE issue per field_'
            'path. If you have multiple suggestions for the same '
            'field, pick the single best correction; do not '
            'enumerate alternatives.\n'
            '\n'
            'DO NOT FLAG:\n'
            '  • Target-language word choice (covered by the '
            'previous review).\n'
            '  • Whether an entity should exist or not (analytical '
            'content; trust it).\n'
            '  • Gender / number / first_occurrence_index / '
            'mention_count values (analytical content).\n'
            '  • Whether a relationship type is "in" or "out" of '
            'an external vocabulary — types are free-form. Only '
            'flag types that are inconsistent with the brief '
            'itself (plot events, role contradictions, missing '
            'reciprocity, dangling IDs, self-references).\n'
            '  • Whether a role description is the "right" '
            'characterization (interpretation, not consistency).\n'
            '  • Stylistic preferences in role/voice prose.\n'
            '  • ID ordering (c_001 should be earliest etc.) — '
            'cannot be fixed via field_path replacement.\n'
            '\n'
            'For each issue, set:\n'
            '  • action: "replace" (edit existing leaf string) OR '
            '"insert" (append entry to existing array).\n'
            '  • field_path: JSON-pointer-style path inside the '
            'brief.\n'
            '  • For action="replace":\n'
            '       field_path = leaf path, e.g. "characters[3].'
            'relationships[1].type", "characters[3].role".\n'
            '       current = verbatim current value at that path.\n'
            '       suggested = corrected value.\n'
            '       insert_value = null.\n'
            '  • For action="insert":\n'
            '       field_path = ARRAY path, e.g. "characters[10].'
            'relationships" (NOT a leaf inside the array).\n'
            '       insert_value = the new entry object {{to_id: '
            '"c_NNN", type: "structural-tie-type"}}.\n'
            '       current = "" (empty string).\n'
            '       suggested = "" (empty string).\n'
            '  • category: one of "role_mismatch", '
            '"missing_reciprocity", "missing_relationship", '
            '"self_reference", "plot_event", "dangling_id".\n'
            '  • reason: single sentence explaining the fix.\n'
            '\n'
            'You CANNOT remove an array entry via this protocol. '
            'If a relationship is wholly spurious, change its '
            'type to the closest accurate structural tie (use '
            'action="replace") rather than trying to delete.'
        )
        return (
            'NOW SWITCH ROLES ONE MORE TIME. You are no longer the '
            'language critic. You are now an INTERNAL-CONSISTENCY '
            'critic, reviewing the brief\'s structural coherence — '
            'specifically the relationship graph and how it lines '
            'up with each character\'s role description. Have no '
            'investment in the brief as it stands; your job is to '
            'find logical contradictions and propose corrections.\n'
            '\n'
            'OUTPUT CONTRACT for this turn:\n'
            '  • stage = "review_logic"\n'
            '  • brief = null\n'
            '  • change_list = an array of issues (possibly empty)\n'
            'Filling brief in this turn is a contract violation. '
            'Re-flagging language issues from the previous turn is '
            'also out of scope.\n'
            '\n'
            'YOUR SCOPE is INTERNAL CONSISTENCY ONLY. The previous '
            'turn already corrected target-language quality. Your '
            'concern is the LOGIC of the brief: do the '
            'relationships make sense given each character\'s '
            'role? Are reciprocal relationships symmetric? Are all '
            'to_id references valid? You will only emit string-'
            'valued replacements (you cannot remove an array entry, '
            'so for a wrong relationship suggest a better type '
            'rather than trying to delete it).\n'
            '\n'
            '═══ EVALUATION CRITERIA ═══\n'
            + evaluation_criteria +
            '\n'
            'If the brief is internally consistent, return '
            'change_list as an empty array []. Do NOT invent '
            'issues to fill the response.'
        )

    def build_translation_brief(self, items, on_progress=None,
                                cancel_request=None, _attempt=0):
        """Phase 0 (validation spike): build a Translation Brief
        from the source text. Returns the parsed brief dict (per
        _TRANSLATION_BRIEF_SCHEMA) or None if construction failed.

        :items: list of {index: int, text: str} dicts representing
            the source paragraphs in document order. Caller is
            responsible for filtering identifying content (the
            existing _strip_identifying_content lives in
            lib/translation.py and is applied by the caller before
            invoking this method).

        Mirrors consistency_review's content-blocks construction:
        each item becomes a `document` content block titled
        block_N where N is the integer index. cache_control on the
        last document caches the prefix so refusal-retries hit the
        cache.

        Phase 0 deliberately omits persistence, schema utilities,
        and the rest of the pipeline — it exists only to validate
        that brief construction works on real copyrighted books.
        """
        if not items:
            return None

        docs = []
        for item in items:
            docs.append({
                'type': 'document',
                'source': {
                    'type': 'content',
                    'content': [{
                        'type': 'text',
                        'text': item['text'],
                    }],
                },
                'title': 'block_{}'.format(item['index']),
            })
        if docs:
            docs[-1]['cache_control'] = {'type': 'ephemeral', 'ttl': '1h'}

        trailing = (
            'Above are {} source-text blocks. Read all of them '
            'before deciding canonical forms. Produce a single '
            'JSON Translation Brief per the schema. Do NOT '
            'translate any text. Do NOT report style/punctuation '
            'variation — only canonical-form decisions for '
            'recurring entities.'
        ).format(len(items))
        content_blocks = docs + [{'type': 'text', 'text': trailing}]

        source_lang = self.source_lang
        target_lang = self.target_lang
        system_prompt = self._brief_system_prompt(
            source_lang, target_lang, attempt=_attempt)

        if _attempt > 0 and on_progress:
            on_progress(_(
                '  Retrying brief build with stronger editorial '
                'framing (attempt {})...'
            ).format(_attempt + 1))

        # ── Turn 1: BUILD ────────────────────────────────────────
        if on_progress:
            on_progress(_('  Turn 1/2: building initial brief...'))
        result = self._consistency_api_call(
            content_blocks, system_prompt,
            on_progress=on_progress,
            cancel_request=cancel_request,
            schema=self._TRANSLATION_BRIEF_SCHEMA)

        parsed = result.get('parsed')
        raw = result.get('raw_response') or ''

        # If the parse failed AND raw text reads like a refusal AND
        # we haven't retried yet, retry with stronger framing.
        if parsed is None and raw and self._is_review_refusal(raw):
            if _attempt < 1:
                if on_progress:
                    on_progress(_(
                        '  Detected refusal in raw response — '
                        'retrying with stronger editorial framing'))
                return self.build_translation_brief(
                    items,
                    on_progress=on_progress,
                    cancel_request=cancel_request,
                    _attempt=_attempt + 1)
            if on_progress:
                on_progress(_(
                    '  Refusal persisted after retry. Raw response '
                    'preview: {}').format(raw[:400]))
            return None

        if parsed is None:
            if on_progress:
                excerpt = raw[:400] + ('...' if len(raw) > 400 else '')
                on_progress(_(
                    '  Brief parse failed. Raw response: {}'
                ).format(excerpt))
            return None

        brief, build_err = self._validate_brief_response(
            parsed, expected_stage='build')
        if brief is None:
            if on_progress:
                on_progress(_(
                    '  Build-turn validation failed: {}'
                ).format(build_err))
            return None

        # The full assistant response (the JSON that came back) gets
        # carried into turn 2 as the assistant's prior turn. Use the
        # raw streamed text rather than re-serializing parsed (the
        # raw is what the model actually emitted).
        import json as _json_local
        try:
            assistant_text = _json_local.dumps(parsed, ensure_ascii=False)
        except Exception:
            assistant_text = '{}'

        # ── Turn 2: LANGUAGE REVIEW ──────────────────────────────
        if cancel_request and cancel_request():
            if on_progress:
                on_progress(_(
                    '  Cancel requested — skipping review turns, '
                    'returning unrefined brief.'))
            return brief

        if on_progress:
            on_progress(_('  Turn 2/3: target-language review pass...'))

        language_user_text = (
            self._brief_review_language_user_message(target_lang))
        language_messages = [
            {'role': 'user', 'content': content_blocks},
            {'role': 'assistant', 'content': assistant_text},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': language_user_text}]},
        ]
        language_result = self._consistency_api_call(
            None, system_prompt,
            on_progress=on_progress,
            cancel_request=cancel_request,
            schema=self._TRANSLATION_BRIEF_SCHEMA,
            messages=language_messages)

        language_parsed = language_result.get('parsed')
        language_raw = language_result.get('raw_response') or ''

        # Defensive fallback — if turn 2 fails, return the
        # un-reviewed brief from turn 1 instead of giving up.
        refined_brief = brief
        language_change_list = []
        language_applied = language_skipped = 0
        if language_parsed is None:
            if on_progress:
                excerpt = (language_raw[:400] +
                           ('...' if len(language_raw) > 400 else ''))
                on_progress(_(
                    '  Language-review parse failed; skipping. '
                    'Raw: {}').format(excerpt))
        else:
            language_change_list, lang_err = (
                self._validate_brief_response(
                    language_parsed,
                    expected_stage='review_language'))
            if language_change_list is None:
                if on_progress:
                    on_progress(_(
                        '  Language-review validation failed: {} — '
                        'skipping.').format(lang_err))
                language_change_list = []
            else:
                if on_progress:
                    on_progress(_(
                        '  Language review returned {} issue(s). '
                        'Applying...').format(len(language_change_list)))
                (language_applied, language_skipped,
                 refined_brief) = self._apply_brief_changes(
                    brief, language_change_list,
                    on_progress=on_progress)
                if on_progress:
                    on_progress(_(
                        '  Language: applied {} of {}; skipped {} '
                        'where current value did not match.'
                    ).format(language_applied,
                             len(language_change_list),
                             language_skipped))

        # ── Turn 3: LOGIC REVIEW ─────────────────────────────────
        # The logic critic reviews the LANGUAGE-CORRECTED brief
        # (refined_brief) — not the original. This way the critic
        # sees the corrected canonical names when judging
        # relationship plausibility.
        if cancel_request and cancel_request():
            if on_progress:
                on_progress(_(
                    '  Cancel requested — skipping logic-review '
                    'turn, returning brief as-is.'))
            refined_brief['_review_language_changes'] = (
                language_change_list)
            refined_brief['_review_logic_changes'] = []
            return refined_brief

        if on_progress:
            on_progress(_(
                '  Turn 3/3: internal-consistency review pass...'))

        # The logic critic should see the CORRECTED brief in the
        # conversation history, not the original. Feed it the
        # refined brief as a synthesized "previous assistant turn."
        try:
            refined_assistant_text = _json_local.dumps(
                {'stage': 'build', 'brief': refined_brief,
                 'change_list': None},
                ensure_ascii=False)
        except Exception:
            refined_assistant_text = assistant_text

        logic_user_text = (
            self._brief_review_logic_user_message(target_lang))
        logic_messages = [
            {'role': 'user', 'content': content_blocks},
            {'role': 'assistant', 'content': refined_assistant_text},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': logic_user_text}]},
        ]
        logic_result = self._consistency_api_call(
            None, system_prompt,
            on_progress=on_progress,
            cancel_request=cancel_request,
            schema=self._TRANSLATION_BRIEF_SCHEMA,
            messages=logic_messages)

        logic_parsed = logic_result.get('parsed')
        logic_raw = logic_result.get('raw_response') or ''
        logic_change_list = []
        logic_applied = logic_skipped = 0

        if logic_parsed is None:
            if on_progress:
                excerpt = (logic_raw[:400] +
                           ('...' if len(logic_raw) > 400 else ''))
                on_progress(_(
                    '  Logic-review parse failed; skipping. '
                    'Raw: {}').format(excerpt))
        else:
            logic_change_list, logic_err = (
                self._validate_brief_response(
                    logic_parsed, expected_stage='review_logic'))
            if logic_change_list is None:
                if on_progress:
                    on_progress(_(
                        '  Logic-review validation failed: {} — '
                        'skipping.').format(logic_err))
                logic_change_list = []
            else:
                if on_progress:
                    on_progress(_(
                        '  Logic review returned {} issue(s). '
                        'Applying...').format(len(logic_change_list)))
                (logic_applied, logic_skipped,
                 refined_brief) = self._apply_brief_changes(
                    refined_brief, logic_change_list,
                    on_progress=on_progress)
                if on_progress:
                    on_progress(_(
                        '  Logic: applied {} of {}; skipped {} '
                        'where current value did not match.'
                    ).format(logic_applied,
                             len(logic_change_list),
                             logic_skipped))

        # Attach both change lists to the returned brief so the
        # caller can surface them separately in the log.
        refined_brief['_review_language_changes'] = (
            language_change_list)
        refined_brief['_review_logic_changes'] = logic_change_list
        return refined_brief

    def _validate_brief_response(self, parsed, expected_stage):
        """Defensively validate that the model honored the
        discriminator contract:
          - stage matches what we asked for (one of "build",
            "review_language", "review_logic")
          - exactly one of brief/change_list is populated, never
            both, never neither (empty change_list array IS valid
            in either review stage — means "no issues found")
        Returns (data, error) where data is the populated payload
        (brief dict for build stage, change_list array for either
        review stage) and error is None on success or an error
        string.
        """
        if not isinstance(parsed, dict):
            return None, 'response is not a JSON object'
        stage = parsed.get('stage')
        if stage != expected_stage:
            return None, (
                'stage mismatch: expected {!r}, got {!r}'
                .format(expected_stage, stage))
        brief = parsed.get('brief')
        change_list = parsed.get('change_list')
        brief_populated = isinstance(brief, dict) and bool(brief)
        # change_list is "populated" if it's a list, even an empty
        # one — empty means "no issues" which is a valid review
        # outcome.
        cl_populated = isinstance(change_list, list)
        if expected_stage == 'build':
            if not brief_populated:
                return None, 'build stage but brief is null/empty'
            if cl_populated and len(change_list) > 0:
                return None, (
                    'build stage but change_list is also populated '
                    '(contract violation)')
            return brief, None
        # Either review stage (review_language or review_logic)
        if brief_populated:
            return None, (
                'review stage but brief is also populated '
                '(contract violation)')
        if not cl_populated:
            return None, (
                'review stage but change_list is null (must be an '
                'array, possibly empty)')
        return change_list, None

    def _apply_brief_changes(self, brief, change_list,
                             on_progress=None):
        """Walk the change list and apply each issue's mutation at
        the issue's field_path. Two action types are supported:

          • action="replace": edit an existing leaf string at
            field_path; the actual current value must match the
            issue's `current` claim, otherwise the change is
            skipped.

          • action="insert": append `insert_value` (an object)
            to an existing array at field_path. The to_id is
            validated against existing characters; duplicates
            (same to_id+type already present) are skipped.

        Returns (applied_count, skipped_count, refined_brief).
        Operates on a deep copy so the input brief is not mutated.

        DEDUPLICATION. The critic occasionally emits multiple
        suggestions targeting the same field. We dedupe BEFORE
        walking:
          • For replace: keep the first issue per field_path.
          • For insert: keep the first issue per
            (field_path, to_id, type) triple.
        """
        import copy
        import re as _re
        refined = copy.deepcopy(brief)
        applied = 0
        skipped = 0

        def _strip_prefix(fp):
            if fp.startswith('brief.'):
                return fp[len('brief.'):]
            if fp.startswith('brief['):
                return fp[len('brief'):]
            return fp

        # ── Pre-apply coordination & dedup ───────────────────────
        # Three classes of issue we need to drop before walking:
        #
        # 1. EXACT DUPLICATE REPLACE on same field_path. The critic
        #    occasionally enumerates alternatives ("change to X" /
        #    "change to Y" for the same field). Keep first; drop
        #    rest.
        # 2. EXACT DUPLICATE INSERT — same (path, to_id, type)
        #    tuple proposed twice.
        # 3. CROSS-FIELD CONFLICT on the same array-element parent.
        #    Example: critic emits two replaces, one changing
        #    relationships[2].to_id and another changing
        #    relationships[2].type. Applied independently they can
        #    yield an incoherent entry (e.g. "ward-of Oswyn" when
        #    only one of those changes was intended). We accept
        #    the first replace per parent-entry path, drop the
        #    rest. This is conservative — a smarter merger could
        #    theoretically combine non-conflicting field changes,
        #    but rejecting the second is safer.
        import re as _re_local
        # Match a parent path like "characters[N].relationships[M]"
        # — the prefix before the leaf field. Used to detect
        # cross-field conflicts on array elements.
        _ARRAY_PARENT_RE = _re_local.compile(
            r'^(.+\[\d+\])\.\w+$')

        seen_replace_paths = set()
        seen_array_parents = set()  # parent paths with a replace
        seen_inserts = set()  # (path, to_id, type) tuples
        deduped = []
        duplicates = 0
        cross_field_skips = 0
        for issue in (change_list or []):
            if not isinstance(issue, dict):
                continue
            action = issue.get('action') or 'replace'
            fp = issue.get('field_path', '') or ''
            if not fp:
                deduped.append(issue)
                continue
            norm = _strip_prefix(fp)
            if action == 'replace':
                if norm in seen_replace_paths:
                    duplicates += 1
                    if on_progress:
                        on_progress(
                            '    ⊝ skip duplicate replace for {}'
                            .format(fp))
                    continue
                # Cross-field conflict guard: if a previous replace
                # has already targeted a sibling field on the same
                # array element, drop this one.
                m = _ARRAY_PARENT_RE.match(norm)
                if m:
                    parent = m.group(1)
                    if parent in seen_array_parents:
                        cross_field_skips += 1
                        if on_progress:
                            on_progress(
                                '    ⊝ skip cross-field replace '
                                'for {} (sibling field on {} '
                                'already changed)'.format(
                                    fp, parent))
                        continue
                    seen_array_parents.add(parent)
                seen_replace_paths.add(norm)
            elif action == 'insert':
                iv = issue.get('insert_value') or {}
                key = (norm,
                       (iv.get('to_id') or '') if isinstance(iv, dict)
                       else '',
                       (iv.get('type') or '') if isinstance(iv, dict)
                       else '')
                if key in seen_inserts:
                    duplicates += 1
                    if on_progress:
                        on_progress(
                            '    ⊝ skip duplicate insert for {}'
                            .format(fp))
                    continue
                seen_inserts.add(key)
            deduped.append(issue)
        if duplicates:
            skipped += duplicates
        if cross_field_skips:
            skipped += cross_field_skips

        # Build set of valid character IDs for insert validation.
        valid_char_ids = {
            c.get('id') for c in (refined.get('characters') or [])
            if isinstance(c, dict) and c.get('id')}

        for issue in deduped:
            action = issue.get('action') or 'replace'
            field_path = issue.get('field_path', '') or ''
            if not field_path:
                skipped += 1
                if on_progress:
                    on_progress('    ✗ skip: empty field_path')
                continue
            field_path = _strip_prefix(field_path)
            tokens = _re.findall(r'(\w+)|\[(\d+)\]', field_path)
            if not tokens:
                skipped += 1
                if on_progress:
                    on_progress(
                        '    ✗ skip: unparseable field_path {}'
                        .format(field_path))
                continue

            # ── action=replace ───────────────────────────────────
            if action == 'replace':
                current = issue.get('current', '') or ''
                suggested = issue.get('suggested', '') or ''
                if current == suggested:
                    skipped += 1
                    continue
                try:
                    container = refined
                    for name, idx in tokens[:-1]:
                        if name:
                            container = container[name]
                        else:
                            container = container[int(idx)]
                    last_name, last_idx = tokens[-1]
                    if last_name:
                        actual = container.get(last_name) \
                            if isinstance(container, dict) else None
                    else:
                        actual = container[int(last_idx)]
                    if actual != current:
                        skipped += 1
                        if on_progress:
                            on_progress(
                                '    ✗ skip replace {}: current '
                                'mismatch (expected {!r}, got {!r})'
                                .format(field_path, current, actual))
                        continue
                    if last_name:
                        container[last_name] = suggested
                    else:
                        container[int(last_idx)] = suggested
                    applied += 1
                    if on_progress:
                        on_progress(
                            '    ✓ replace {}: {!r} → {!r}'.format(
                                field_path, current, suggested))
                except (KeyError, IndexError, TypeError, ValueError):
                    skipped += 1
                    if on_progress:
                        on_progress(
                            '    ✗ skip replace {}: path does not '
                            'resolve'.format(field_path))
                    continue

            # ── action=insert ────────────────────────────────────
            elif action == 'insert':
                insert_value = issue.get('insert_value')
                if not isinstance(insert_value, dict) \
                        or not insert_value:
                    skipped += 1
                    if on_progress:
                        on_progress(
                            '    ✗ skip insert {}: insert_value '
                            'missing or invalid'.format(field_path))
                    continue
                # Validate to_id (if relationship-style) references
                # an existing character.
                to_id = (insert_value.get('to_id') or '').strip()
                if to_id and to_id not in valid_char_ids:
                    skipped += 1
                    if on_progress:
                        on_progress(
                            '    ✗ skip insert {}: to_id {!r} '
                            'references no existing character'
                            .format(field_path, to_id))
                    continue
                try:
                    # Walk to the array itself (no [-1] split — the
                    # entire path resolves to a list).
                    container = refined
                    for name, idx in tokens:
                        if name:
                            container = container[name]
                        else:
                            container = container[int(idx)]
                    if not isinstance(container, list):
                        skipped += 1
                        if on_progress:
                            on_progress(
                                '    ✗ skip insert {}: path '
                                'resolves to {} not array'.format(
                                    field_path,
                                    type(container).__name__))
                        continue
                    # Self-reference check (character can't relate
                    # to themselves) — derive owner id from the
                    # walked path: characters[N].relationships.
                    if (len(tokens) >= 2 and tokens[0][0] ==
                            'characters' and tokens[1][1]):
                        owner_idx = int(tokens[1][1])
                        chars = refined.get('characters') or []
                        if 0 <= owner_idx < len(chars):
                            owner_id = chars[owner_idx].get(
                                'id') or ''
                            if to_id == owner_id:
                                skipped += 1
                                if on_progress:
                                    on_progress(
                                        '    ✗ skip insert {}: '
                                        'self-reference (to_id == '
                                        'owner id)'.format(
                                            field_path))
                                continue
                    # Dedupe against existing entries in the array.
                    if insert_value in container:
                        skipped += 1
                        if on_progress:
                            on_progress(
                                '    ⊝ skip insert {}: entry '
                                'already present'.format(field_path))
                        continue
                    container.append(insert_value)
                    applied += 1
                    if on_progress:
                        on_progress(
                            '    + insert into {}: {!r}'.format(
                                field_path, insert_value))
                except (KeyError, IndexError, TypeError, ValueError):
                    skipped += 1
                    if on_progress:
                        on_progress(
                            '    ✗ skip insert {}: path does not '
                            'resolve'.format(field_path))
                    continue

            else:
                skipped += 1
                if on_progress:
                    on_progress(
                        '    ✗ skip: unknown action {!r}'.format(
                            action))
                continue

        # ── Post-apply cleanup ───────────────────────────────────
        # Walk every character's relationships array and remove
        # exact-content duplicates. Duplicates can arise when an
        # insert and a subsequent replace converge on the same
        # final {to_id, type} — the insert ran when the existing
        # entry still differed, the replace then mutated the
        # existing entry to match. Apply-time dedup-against-
        # existing checks against the snapshot at insert time,
        # not the post-replace state, so this final pass catches
        # the post-mutation duplicates.
        post_dedup_count = 0
        for char in (refined.get('characters') or []):
            if not isinstance(char, dict):
                continue
            rels = char.get('relationships')
            if not isinstance(rels, list) or len(rels) < 2:
                continue
            seen = []
            unique = []
            for entry in rels:
                if entry in seen:
                    post_dedup_count += 1
                    continue
                seen.append(entry)
                unique.append(entry)
            if len(unique) != len(rels):
                char['relationships'] = unique
        if post_dedup_count and on_progress:
            on_progress(
                '    ⊝ post-apply: removed {} duplicate '
                'relationship entr{} from refined brief'.format(
                    post_dedup_count,
                    'y' if post_dedup_count == 1 else 'ies'))

        return applied, skipped, refined

    def _consistency_api_call(self, content_blocks, system_prompt,
                              on_progress=None, cancel_request=None,
                              valid_indices=None, schema=None,
                              messages=None):
        """Single API call for the consistency review. Takes prebuilt
        content_blocks and system_prompt, streams the response, parses
        the JSON output, and returns a dict.

        Does NOT handle refusal-retry, validation, or repair —
        consistency_review() orchestrates those.

        :valid_indices: set of integer block indices that are
            permissible in the response. Corrections targeting an
            index outside this set are dropped. None disables this
            check.
        :schema: JSON schema to enforce on the model's output. If
            None, uses _CONSISTENCY_OUTPUT_SCHEMA. Other callers
            (e.g. build_translation_brief) pass a different schema
            and read the raw parsed dict from the returned 'parsed'
            field.
        :messages: optional list of message dicts (role + content)
            to send instead of wrapping `content_blocks` as a single
            user message. Used by multi-turn callers like
            build_translation_brief's review pass, where the prior
            turns must be carried in the messages array. When set,
            content_blocks is ignored.

        Returns:
          - 'glossary':       list of validated glossary entries
                              (consistency-review shape; empty when
                              a non-consistency schema was used)
          - 'corrections':    list of {index, reason,
                              replacements: [{old_str, new_str}]}
                              (always [] in the v2.4.37+ glossary-
                              first design)
          - 'parsed':         raw parsed JSON dict, schema-shaped.
                              Use this when caller passed a custom
                              schema.
          - 'raw_response':   raw text (populated when parsing fails
                              or both buckets are empty)
          - 'final_stop_reason': model's stop_reason for this call
        """
        if schema is None:
            schema = self._CONSISTENCY_OUTPUT_SCHEMA
        # NOTE: the comment block below is preserved here because it
        # documents the model-facing terminology that the caller's
        # content_blocks construction depends on.

        # Build the user-message content as a list of `document`
        # content blocks — one per Paragraph object.
        #
        # ── Terminology (used consistently in the model-facing
        # ── prompt and document titles, distinct from the codebase's
        # ── internal class names) ─────────────────────────────────────
        #
        #   block (model-facing):
        #       One unit of translation passed to the model. Equals
        #       one Paragraph object's content. Document title is
        #       "block_N" where N is the 0-indexed position.
        #
        #   Paragraph (codebase class — lib/cache.py):
        #       The Python data model. One row in the advanced
        #       translation table. Stored in the SQLite cache.
        #
        #   source structural unit:
        #       One HTML block-level element from the EPUB (typically
        #       <p>, but also <h1>-<h6>, <li>, <blockquote>, <td>,
        #       etc). Each carries one chunk of text.
        #
        #   real paragraph (reader's intuition):
        #       A logical unit of prose. Roughly equals a <p>
        #       element, but NOT all source structural units are real
        #       paragraphs — a heading is a structural unit, not a
        #       paragraph in the reader's sense.
        #
        # ── Relationships ─────────────────────────────────────────────
        #
        #   merge OFF: 1 block = 1 Paragraph = 1 source structural unit
        #              (which may or may not be a real paragraph)
        #
        #   merge ON:  1 block = 1 Paragraph = N source structural
        #              units concatenated with '\n\n', batched up to
        #              merge_length characters. Internally a block
        #              may contain a chapter heading, a list item,
        #              and several real paragraphs all glued together
        #              with '\n\n' separators.
        #
        # Each document gets:
        #  - source.type: "content" with a single text sub-block
        #    (custom-content variant that does NOT auto-chunk; the
        #    block is treated as one indivisible unit)
        #  - title: "block_N" — the model's reference handle. When
        #    the model returns a correction with index=N, we look up
        #    the Paragraph at our internal index N (0-indexed across
        #    items in order, which matches the document order).
        #
        # Followed by a final `text` block with the lead-in
        # instruction. The system prompt explains the analytical task
        # and the terminology; this block anchors the user message.
        #
        # No citations.enabled: citations are mutually exclusive with
        # Structured Outputs (output_format), and we need the JSON
        # schema enforcement more than we need parser-validated
        # citations.
        # content_blocks and system_prompt are caller-provided.
        # Caller is responsible for cache_control, trailing text
        # block, and prompt selection (initial / refusal-retry /
        # repair).

        # Stream the response. The consistency review can take many minutes
        # for a long book, which would exceed any reasonable socket read
        # timeout if we waited for the full response. Streaming lets each
        # individual read complete quickly while the overall generation
        # continues in the background.
        #
        # Use Anthropic Structured Outputs (output_config.format) to
        # guarantee the response is a JSON object matching our schema.
        # This streams via plain text_delta events on the regular
        # text-streaming pipeline — avoids the stall bug that affected
        # tool_use streaming on Sonnet 4.x with large prompts (see
        # anthropic-sdk-typescript#842, claude-code#19143).
        #
        # Prompt caching marks the user message as cacheable. On a
        # retry within the 5-minute TTL, the second request reads the
        # cache at 10% input cost. We always cache for the consistency
        # pass — the cache is over our own translation output (no
        # copyright concern), and the savings are substantial for
        # retry scenarios.
        body = json.dumps({
            'model': self.model,
            'max_tokens': 64_000,
            'temperature': 0,
            'stream': True,
            'system': [
                {
                    'type': 'text',
                    'text': system_prompt,
                }
            ],
            # Structured Outputs — legacy beta form (top-level
            # output_format + beta header). The newer production form
            # (output_config.format) returned HTTP 400 with our
            # anthropic-version header, so we use the legacy form which
            # is documented to still work.
            'output_format': {
                'type': 'json_schema',
                'schema': schema,
            },
            'messages': messages if messages is not None else [{
                'role': 'user',
                'content': content_blocks,
            }],
        })

        # Use urllib directly rather than the project's mechanize-backed
        # request() helper. mechanize's response_seek_wrapper can buffer
        # streaming responses in ways that defeat incremental readline().
        # urllib's HTTPResponse exposes a true streaming socket file
        # object, so readline() returns as soon as a complete line is
        # available on the wire.
        url_req = _UrlRequest(
            self.endpoint,
            data=body.encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'anthropic-version': self.api_version,
                'x-api-key': self.api_key,
                # Three beta headers, comma-separated:
                # - prompt-caching: 90% discount on retry input cost
                # - extended-cache-ttl: enables 1-hour cache TTL
                #   (vs default 5 minutes). 2× cache-write cost,
                #   amortized across more reads in build→review→
                #   draft chains that may span >5 minutes.
                # - structured-outputs: enables the output_format
                #   field (legacy form — see body comment above)
                'anthropic-beta': (
                    'prompt-caching-2024-07-31,'
                    'extended-cache-ttl-2025-04-11,'
                    'structured-outputs-2025-11-13'
                ),
            },
            method='POST',
        )
        try:
            ssl_ctx = ssl.create_default_context()
        except Exception:
            ssl_ctx = ssl._create_unverified_context()
        # Build opener with proxy + SSL context if needed.
        handlers = [_HTTPSHandler(context=ssl_ctx)]
        try:
            if self.proxy_type == 'http' and self.proxy_uri:
                handlers.append(_ProxyHandler({
                    'http': self.proxy_uri,
                    'https': self.proxy_uri,
                }))
                opener = _build_opener(*handlers)
                response = opener.open(
                    url_req,
                    timeout=max(int(self.request_timeout) * 4, 120),
                )
            else:
                response = _urlopen(
                    url_req,
                    timeout=max(int(self.request_timeout) * 4, 120),
                    context=ssl_ctx,
                )
        except _UrlHTTPError as e:
            # Anthropic returns rich JSON error bodies on 4xx/5xx — read
            # them and surface in the exception so the user can see what
            # the API actually said (auth error, malformed request, etc.)
            # instead of a bare "HTTP Error 400".
            try:
                err_body = e.read().decode('utf-8', errors='replace')
            except Exception:
                err_body = ''
            raise Exception(_(
                'HTTP {}: {}\n\nResponse body:\n{}'
            ).format(e.code, e.reason, err_body[:2000]))
        if on_progress:
            on_progress(_('  HTTP connection established, '
                          'awaiting first event...'))

        # Set a per-read socket timeout. urllib's response.close() from
        # another thread does NOT interrupt a blocked readline() — this
        # is a fundamental Python sync-I/O limitation (urllib3#2868).
        # The fix is to set a short socket timeout so readline() raises
        # socket.timeout periodically; the reader can then check the
        # cancel flag between attempts.
        #
        # Without this, the Stop button gets stuck on "Stopping..."
        # indefinitely while the reader sits blocked waiting for bytes
        # that may never arrive (Anthropic stream stall — see issue
        # #842 in anthropic-sdk-typescript).
        for attr_path in ('fp.raw._sock', 'fp._sock', '_fp._sock'):
            obj = response
            try:
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                obj.settimeout(5.0)
                break
            except AttributeError:
                continue

        # Collect streamed text. Decouples blocking I/O from the main
        # loop using a reader thread + queue, so we can check
        # cancel_request every 100ms regardless of how long any single
        # readline() takes. Also surfaces lifecycle events so the user
        # can distinguish the "model is processing input" phase from
        # active output generation.
        import threading
        import queue as _queue
        if on_progress:
            on_progress(_('Streaming response from model — '
                          'large prompts may take 30-60s before output '
                          'starts...'))
        chunks = []
        chars_received = 0
        progress_step = 500
        next_progress_at = progress_step
        ping_count = 0
        output_started = False
        # message_delta carries stop_reason; we save it for the
        # post-stream check. With Structured Outputs, the relevant
        # values are:
        #   - "end_turn": normal completion, JSON should be parseable
        #   - "max_tokens": output truncated mid-JSON — invalid output
        #   - "refusal": model declined — output is refusal text, not JSON
        final_stop_reason = None

        # Reader thread feeds raw lines into a queue. Daemonized so it
        # can't block process exit.
        read_queue: _queue.Queue = _queue.Queue()
        reader_done = threading.Event()
        EOF = object()
        ERR = object()

        def _reader():
            while not reader_done.is_set():
                try:
                    raw = response.readline()
                    if not raw:
                        read_queue.put((EOF, None))
                        return
                    read_queue.put(('line', raw))
                except IncompleteRead:
                    # Transient — keep reading. Mirrors _parse_stream.
                    continue
                except (_socket.timeout, TimeoutError):
                    # Per-read socket timeout fired (5s of no data).
                    # Loop back to check reader_done — this is the
                    # mechanism that lets us cancel a blocked readline.
                    continue
                except Exception as e:
                    read_queue.put((ERR, e))
                    return

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        # Idle heartbeat — if we go too long with nothing on the queue,
        # log a "still waiting" message so the user knows we haven't
        # silently hung.
        idle_seconds_so_far = 0.0
        idle_log_interval = 15.0  # log every 15s of pure silence
        last_idle_log = 0.0

        # Stall detection: Anthropic's tool_use+streaming has a known bug
        # where the response stalls mid-stream after some output, with
        # the connection alive but no further events arriving. See
        # anthropic-sdk-typescript#842 and claude-code#19143. After
        # this many seconds of idle (with data already received), we
        # abort with a clear error rather than waiting forever.
        stall_threshold = 60.0
        stall_detected = False

        try:
            while True:
                # Cancellation check — runs every 100ms regardless of
                # I/O state. We don't call response.close() here because
                # close() can hang if the reader thread is blocked in
                # readline (urllib3#2868). Instead we set reader_done in
                # finally and rely on the socket timeout to wake the
                # reader. The TranslationCanceled exception propagates
                # cleanly without needing the connection torn down.
                if cancel_request is not None and cancel_request():
                    from ..lib.exception import TranslationCanceled as _TC
                    raise _TC(_('Translation canceled.'))

                # Stall detection: known Anthropic bug, abort cleanly.
                if (chars_received > 0
                        and idle_seconds_so_far >= stall_threshold
                        and not stall_detected):
                    stall_detected = True
                    raise Exception(_(
                        'Stream stalled: no data for {:.0f}s after '
                        'receiving {} chars. This is a known Anthropic '
                        'API issue with tool_use streaming on large '
                        'prompts (see anthropic-sdk-typescript#842). '
                        'Try reducing paragraph count or retrying later.'
                    ).format(idle_seconds_so_far, chars_received))

                try:
                    kind, payload = read_queue.get(timeout=0.1)
                except _queue.Empty:
                    idle_seconds_so_far += 0.1
                    if (on_progress and
                            idle_seconds_so_far - last_idle_log
                            >= idle_log_interval):
                        # Different message depending on whether we've
                        # already received data — the model produces
                        # output in bursts with thinking pauses between
                        # sections, so silence after data has flowed is
                        # normal (not an indication something is stuck).
                        if chars_received > 0:
                            on_progress(_(
                                '  ...no new chunks for {:.0f}s '
                                '(model is thinking between sections — '
                                'received {} chars so far)'
                            ).format(idle_seconds_so_far, chars_received))
                        else:
                            on_progress(_(
                                '  ...still waiting for first response '
                                'chunk ({:.0f}s elapsed)'
                            ).format(idle_seconds_so_far))
                        last_idle_log = idle_seconds_so_far
                    continue

                # Got something — reset idle counter.
                idle_seconds_so_far = 0.0
                last_idle_log = 0.0

                if kind is EOF:
                    break
                if kind is ERR:
                    break  # Loop exit; cancellation/raw_response handled below

                line = payload.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                try:
                    evt = json.loads(line.split('data: ', 1)[1])
                except (IndexError, json.JSONDecodeError):
                    continue
                etype = evt.get('type')
                if etype == 'message_stop':
                    break
                if etype == 'message_start':
                    if on_progress:
                        on_progress(_('  Model accepted request, '
                                      'processing input...'))
                elif etype == 'content_block_start':
                    if on_progress:
                        on_progress(_('  Output started, streaming JSON '
                                      'response...'))
                    output_started = True
                elif etype == 'content_block_delta':
                    # With Structured Outputs (output_config.format),
                    # the response is a regular text content block
                    # whose text is the JSON. Deltas are text_delta
                    # events — same code path as plain text streaming.
                    delta = evt.get('delta') or {}
                    dtype = delta.get('type')
                    s = ''
                    if dtype == 'text_delta':
                        s = str(delta.get('text') or '')
                        if s:
                            chunks.append(s)
                    elif dtype == 'input_json_delta':
                        # Shouldn't appear with output_config.format,
                        # but accept it defensively (legacy path).
                        s = str(delta.get('partial_json') or '')
                        if s:
                            chunks.append(s)
                    if s:
                        chars_received += len(s)
                        if (on_progress
                                and chars_received >= next_progress_at):
                            on_progress(_('  ...{} characters received')
                                        .format(chars_received))
                            next_progress_at = chars_received + progress_step
                elif etype == 'ping':
                    # Each ping confirms the connection is alive while
                    # the model is still processing input. Log every
                    # ping (they're typically every 30-60s) so the user
                    # can see activity well before output begins.
                    ping_count += 1
                    if not output_started and on_progress:
                        on_progress(_('  ...keepalive ping #{} received')
                                    .format(ping_count))
                elif etype == 'message_delta':
                    # Carries stop_reason and usage info (including
                    # prompt cache stats).
                    delta = evt.get('delta') or {}
                    sr = delta.get('stop_reason')
                    if sr:
                        final_stop_reason = sr
                    usage = evt.get('usage') or {}
                    cache_read = usage.get('cache_read_input_tokens')
                    cache_write = usage.get('cache_creation_input_tokens')
                    if on_progress and (cache_read or cache_write):
                        on_progress(_(
                            '  Cache: {} read tokens, {} write tokens '
                            '(read = 10% input cost; write = 125% '
                            'input cost, billed once)'
                        ).format(cache_read or 0, cache_write or 0))
                elif etype == 'error':
                    raise Exception(
                        _('Consistency review error: {}').format(
                            evt.get('error', {}).get(
                                'message', 'unknown')))
        finally:
            # Signal the reader to stop; with the 5s socket timeout, it
            # will wake from any blocked readline within 5 seconds and
            # check the flag.
            reader_done.set()
            # Brief join attempt so the reader's response.close() doesn't
            # race with ours. Don't block forever — the reader is daemon.
            try:
                reader_thread.join(timeout=0.5)
            except Exception:
                pass
            # Closing the response only AFTER setting reader_done helps
            # avoid the close()-blocked-by-readline scenario from
            # urllib3#2868. If reader is still in readline at this point,
            # close() may still hang on some platforms — but we've set
            # the daemon flag so worst case it lingers until process exit.
            try:
                response.close()
            except Exception:
                pass

        raw_text = ''.join(chunks).strip()
        if on_progress:
            on_progress(_('  ...complete: {} characters total '
                          '(stop_reason: {})')
                        .format(chars_received,
                                final_stop_reason or 'unknown'))

        # Stop_reason handling for Structured Outputs:
        # - 'end_turn': normal completion, JSON should parse
        # - 'max_tokens': output truncated mid-JSON; can't recover by
        #   parsing — must retry with higher max_tokens or smaller input
        # - 'refusal': model declined; output is refusal text, not JSON.
        #   Fall through to existing refusal-detection retry path.
        # - 'stop_sequence', 'pause_turn', 'tool_use': unexpected here
        if final_stop_reason == 'max_tokens':
            raise Exception(_(
                'Consistency review output was truncated (hit max_tokens '
                'limit at {} characters). The JSON is incomplete and '
                'cannot be parsed. Try splitting the book into smaller '
                'sections or running consistency pass on fewer paragraphs '
                'at a time.'
            ).format(chars_received))

        # Try direct JSON parse; fall back to extracting an object from
        # surrounding text (some models add commentary despite instructions).
        parsed = None
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            import re as _re
            match = _re.search(r'\{.*\}', raw_text, _re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = None
        if not isinstance(parsed, dict):
            # Parse failed — return raw text so caller can decide
            # whether to retry (refusal-shaped) or surface to the
            # user.
            return {
                'glossary': [],
                'corrections': [],
                'parsed': None,
                'raw_response': raw_text,
                'final_stop_reason': final_stop_reason,
            }

        # If the caller passed a non-default schema (e.g. brief
        # building), skip the consistency-specific glossary parsing
        # and surface the raw parsed dict directly.
        if schema is not self._CONSISTENCY_OUTPUT_SCHEMA:
            return {
                'glossary': [],
                'corrections': [],
                'parsed': parsed,
                'raw_response': '',
                'final_stop_reason': final_stop_reason,
            }

        # Validate glossary entries. Each entry has a canonical form
        # and a list of variants — verbatim strings from the source
        # text that should be normalized to the canonical via
        # deterministic word-boundary substitution at the host.
        glossary_raw = parsed.get('glossary') or []
        glossary = []
        if isinstance(glossary_raw, list):
            for g in glossary_raw:
                if not isinstance(g, dict):
                    continue
                canonical = (g.get('canonical') or '').strip()
                if not canonical:
                    continue
                variants_raw = g.get('variants') or []
                variants = []
                if isinstance(variants_raw, list):
                    seen = set()
                    for v in variants_raw:
                        if not isinstance(v, str):
                            continue
                        v = v.strip()
                        # Skip empties, the canonical itself, and dups.
                        if not v or v == canonical or v in seen:
                            continue
                        seen.add(v)
                        variants.append(v)
                glossary.append({
                    'canonical': canonical,
                    'variants': variants,
                    'type': (g.get('type') or '').strip(),
                    'gender': (g.get('gender') or '').strip().lower(),
                    'notes': (g.get('notes') or '').strip(),
                })

        return {
            'glossary': glossary,
            # corrections is no longer produced — kept as empty list
            # for backwards compatibility with consumers expecting
            # the old return shape.
            'corrections': [],
            'parsed': parsed,
            'raw_response': raw_text if not glossary else '',
            'final_stop_reason': final_stop_reason,
        }

    def get_models(self):
        model_endpoint = urljoin(self.endpoint, 'models')
        response = request(model_endpoint, headers=self.get_headers())
        return [i['id'] for i in json.loads(response)['data']]

    def get_headers(self):
        headers = {
            'Content-Type': 'application/json',
            'anthropic-version': self.api_version,
            'x-api-key': self.api_key,
            'User-Agent': 'Ebook-Translator/%s' % EbookTranslator.__version__,
        }

        # Enable beta features based on user configuration
        # More info: https://platform.claude.com/docs/en/about-claude/models/overview
        beta_features = []

        if self.model is not None:
            # For Claude Sonnet 3.7 - enable 128K output tokens
            # (requires user to enable this option)
            if self.enable_extended_output and self.model.startswith('claude-3-7-sonnet-'):
                beta_features.append('output-128k-2025-02-19')
            # For Claude 4.6 models - 1M context is native, no extra cost.
            # The beta header is kept as a no-op safeguard.
            elif self.enable_extended_context and (
                    self.model.startswith('claude-sonnet-4-6') or
                    self.model.startswith('claude-opus-4-6')):
                beta_features.append('context-1m-2025-08-07')

        # Prompt caching can be used with any Claude model
        # More info: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
        if self.enable_prompt_caching:
            beta_features.append('prompt-caching-2024-07-31')
            # Extended-cache-ttl pairs with cache_control.ttl='1h' on
            # the cached blocks. Without this header the API either
            # rejects the ttl field or silently falls back to the
            # 5-minute default.
            beta_features.append('extended-cache-ttl-2025-04-11')

        if beta_features:
            headers['anthropic-beta'] = ','.join(beta_features)

        return headers

    def get_body(self, text):
        # Calculate max_tokens based on model and input length
        # Estimate output will be similar length to input
        estimated_output_tokens = len(text) // 3  # Conservative: ~3 chars per token

        # Determine model's max output capability based on official docs
        # Source: https://platform.claude.com/docs/en/about-claude/models/overview
        if not self.model:
            model_max_output = 4_096
        elif self.model.startswith('claude-3-7-sonnet-') and self.enable_extended_output:
            # Claude 3.7 Sonnet with extended output beta flag
            model_max_output = 128_000
        elif self.model.startswith('claude-3-7-sonnet-'):
            # Claude 3.7 Sonnet without beta flag
            model_max_output = 64_000
        elif self.model.startswith('claude-sonnet-4-') or \
             self.model.startswith('claude-haiku-4-') or \
             self.model.startswith('claude-opus-4-5') or \
             self.model.startswith('claude-opus-4-1'):
            # Claude 4.5 (all variants) and Claude 4.1 Opus: 64K
            # Also Claude Sonnet 4.0, Haiku 4.5
            model_max_output = 64_000
        elif self.model.startswith('claude-opus-4-0'):
            # Claude Opus 4.0: 32K
            model_max_output = 32_000
        elif self.model.startswith('claude-3-haiku-'):
            # Claude Haiku 3.x: 4K
            model_max_output = 4_000
        elif self.model.startswith('claude-'):
            # Other Claude models: conservative 32K default
            model_max_output = 32_000
        else:
            # Non-Claude models or unknown: very conservative
            model_max_output = 4_096

        # Use estimated output or model max, whichever is smaller
        # Add 10% buffer for safety
        max_tokens = min(int(estimated_output_tokens * 1.1), model_max_output)
        # Minimum 4096 tokens
        max_tokens = max(max_tokens, 4_096)

        # Build system message with optional caching.
        # Three possible cached system shapes:
        #   1. brief + full_book_context (richest)
        #   2. brief only (compact, used when full_book_context is
        #      disabled or unavailable)
        #   3. full_book_context only (legacy pre-Phase-0 path)
        #   4. neither (un-cached fallback)
        has_brief = bool(getattr(self, 'translation_brief', None))
        has_book = bool(self.full_book_context)
        if self.enable_prompt_caching and (has_brief or has_book):
            # The prompt includes explicit instructions to prevent Claude
            # from refusing to translate content it identifies as
            # copyrighted. These clarify that this is a personal
            # translation tool operating on the user's own ebook
            # library.
            #
            # CRITICAL: when the cached system prompt contains the full
            # book context, the model can see what comes AFTER the
            # current paragraph. Without strong boundary discipline, it
            # tends to (a) translate fragments of subsequent paragraphs
            # ("leakage") or (b) hallucinate plausible continuations
            # ("completion-hallucination"). The mitigation block below
            # imposes hard anti-extension rules.
            mitigation = (
                ' The reference material below is provided to help you '
                'translate the user\'s personal copy of an ebook '
                'consistently. It is REFERENCE — do not translate it. '
                '\n'
                '\n'
                'BOUNDARY DISCIPLINE (CRITICAL). Each user message '
                'contains a single passage delimited by '
                '<<<BEGIN_SOURCE>>> and <<<END_SOURCE>>> markers. '
                'You MUST translate ONLY the text between those '
                'markers. The source\'s ending — wherever it lands, '
                'even mid-thought, even mid-sentence — is also the '
                'translation\'s ending.\n'
                '\n'
                'ABSOLUTE RULES:\n'
                '  • Do NOT add any content beyond the source. No '
                'continuation, no completion, no anticipation.\n'
                '  • Do NOT translate fragments of subsequent '
                'paragraphs visible in the cached reference. The '
                'cached book context is for terminology and gender '
                'consistency ONLY.\n'
                '  • Do NOT infer "what should come next" and append '
                'it. If the source ends on a question, your '
                'translation ends on that question — not on its '
                'answer.\n'
                '  • Do NOT add explanations, narrator interjections, '
                'or interpretive content not present in the source.\n'
                '\n'
                'SELF-VERIFICATION (perform mentally before '
                'finalizing). Compare:\n'
                '  1. Does your translation\'s last sentence '
                'correspond to the source\'s last sentence (the one '
                'just before <<<END_SOURCE>>>)?\n'
                '  2. Does your translation have approximately the '
                'same number of sentences as the source?\n'
                'If your translation continues past the source\'s '
                'ending, that is an error. Do not include any '
                'continuation in your output.')
            cached_prompt = self._get_prompt() + mitigation
            system_content = [
                {
                    "type": "text",
                    "text": cached_prompt,
                },
            ]
            # ── Translation Brief (Phase 1a) ──────────────────────
            # The brief is small (~5-15K tokens) and high-signal:
            # canonical names, character profiles with grammatical
            # gender, recurring terminology, style decisions. Inject
            # before the full-book context so it's the first
            # reference the model encounters after the prompt.
            if has_brief:
                try:
                    brief_json = json.dumps(
                        self.translation_brief,
                        ensure_ascii=False, indent=2)
                except Exception:
                    brief_json = ''
                if brief_json:
                    system_content.append({
                        "type": "text",
                        "text": (
                            "Translation Brief — REFERENCE document "
                            "with canonical {tlang} forms for "
                            "character names, character profiles "
                            "(including grammatical gender for "
                            "verb/adjective agreement), recurring "
                            "terminology, and style decisions. Apply "
                            "these consistently when translating the "
                            "segment in the user message. Do NOT "
                            "translate the brief itself — it is "
                            "metadata in the assistant's working "
                            "language.\n\n" + brief_json
                        ).format(tlang=self.target_lang),
                    })
            # ── Full book context (legacy / optional) ─────────────
            if has_book:
                system_content.append({
                    "type": "text",
                    "text": (
                        "Full book context for reference:\n\n"
                        + self.full_book_context),
                })
            # Cache_control on the LAST system block caches every
            # block up to and including it for 1 hour.
            system_content[-1]['cache_control'] = {
                'type': 'ephemeral', 'ttl': '1h'}
            # Use explicit delimiters around the source text. This
            # gives the model unambiguous boundary markers that are
            # easier to respect than prose ("the following section")
            # which the model can interpret loosely.
            user_message = (
                "Translate the source passage between the markers "
                "below. Output ONLY the translated text — no "
                "commentary, no continuation past <<<END_SOURCE>>>.\n"
                "\n"
                "<<<BEGIN_SOURCE>>>\n"
                + text +
                "\n<<<END_SOURCE>>>")
        else:
            # Standard system message without caching
            system_content = self._get_prompt()
            user_message = text

        body = {
            'stream': self.stream,
            'max_tokens': max_tokens,
            'model': self.model,
            'top_k': self.top_k,
            'system': system_content,
            'messages': [{'role': 'user', 'content': user_message}]
        }
        sampling_value = getattr(self, self.sampling)
        body.update({self.sampling: sampling_value})

        return json.dumps(body)

    def get_result(self, response: Response | str) -> str:
        if self.stream:
            return self._parse_stream(response)

        response_json = json.loads(response)
        response_content_text: str = response_json['content'][0]['text']
        return response_content_text

    def _parse_stream(self, data: Response) -> Generator:
        while True:
            try:
                line = data.readline().decode('utf-8').strip()
            except IncompleteRead:
                continue
            except Exception as e:
                raise Exception(
                    _('Can not parse returned response. Raw data: {}')
                    .format(str(e)))

            if line.startswith('data:'):
                chunk: dict = json.loads(line.split('data: ')[1])
                event_type: str = chunk['type']

                if event_type not in self.valid_event_types:
                    raise Exception(
                        _('Invalid event type received: {}')
                        .format(event_type))

                if event_type == 'message_stop':
                    break
                elif event_type == 'message_delta':
                    # Claude 4+ classifier-level refusals appear here as
                    # stop_reason="refusal" with no preceding text. Yield a
                    # synthetic refusal marker so the existing refusal
                    # detection (pattern + LLM) picks it up uniformly.
                    delta = chunk.get('delta') or {}
                    if delta.get('stop_reason') == 'refusal':
                        yield ('I cannot translate this copyrighted material. '
                               'I would be happy to help with shorter '
                               'excerpts from copyrighted content.')
                elif event_type == 'content_block_delta':
                    delta = chunk.get('delta')
                    if delta is not None:
                        yield str(delta.get('text'))
                elif event_type == 'error':
                    raise Exception(
                        _('Error received: {}')
                        .format(chunk['error']['message']))


class ClaudeBatchTranslate(ClaudeTranslate):
    """Message Batches API for asynchronous bulk translation with 50% cost reduction.

    The message batches API allows sending batches of up to 100,000 messages.
    Batches are processed asynchronously with results returned when complete.
    Costs 50% less than standard API calls.

    Combined with prompt caching, offers up to 84% total cost savings.
    Trade-off: Processing takes up to 24 hours (most <1 hour) vs immediate results.

    More info: https://docs.anthropic.com/en/docs/build-with-claude/message-batches
    """

    name = 'Claude Batch'
    alias = 'Claude Batch (Anthropic)'

    # Batch-specific settings
    stream = False  # Batches don't support streaming
    batch_poll_interval = 60.0  # Poll every 60 seconds

    def __init__(self):
        super().__init__()
        # Batch API uses different endpoints
        self.batch_endpoint = 'https://api.anthropic.com/v1/messages/batches'
        self.batch_poll_interval = self.config.get('batch_poll_interval', self.batch_poll_interval)

    def create_batch_request(self, custom_id, text):
        """Create a single request object for batch API."""
        # Parse the body that get_body() creates
        body_json = json.loads(self.get_body(text))
        # Remove stream parameter (not supported in batches)
        body_json.pop('stream', None)

        return {
            'custom_id': custom_id,
            'params': body_json
        }

    def create_batch(self, requests):
        """Submit a batch of translation requests."""
        batch_body = json.dumps({'requests': requests})
        response = request(
            self.batch_endpoint,
            data=batch_body,
            headers=self.get_headers(),
            method='POST',
            timeout=int(self.request_timeout),
            proxy_uri=self.proxy_uri if self.proxy_type == 'http' else None
        )
        batch = json.loads(response)
        return batch['id']

    def poll_batch(self, batch_id):
        """Poll batch status until completion."""
        while True:
            response = request(
                f'{self.batch_endpoint}/{batch_id}',
                headers=self.get_headers(),
                timeout=int(self.request_timeout),
                proxy_uri=self.proxy_uri if self.proxy_type == 'http' else None
            )
            batch = json.loads(response)

            if batch['processing_status'] == 'ended':
                return batch

            # Log progress
            counts = batch['request_counts']
            log.info(f"Batch {batch_id}: {counts['succeeded']}/{counts['processing']} completed")

            time.sleep(self.batch_poll_interval)

    def get_batch_results(self, results_url):
        """Retrieve and parse batch results."""
        response = request(
            results_url,
            headers=self.get_headers(),
            timeout=int(self.request_timeout),
            proxy_uri=self.proxy_uri if self.proxy_type == 'http' else None
        )

        # Results are in JSONL format (one JSON object per line)
        results = {}
        for line in response.decode('utf-8').strip().split('\n'):
            if line:
                result = json.loads(line)
                custom_id = result['custom_id']
                if result['result']['type'] == 'succeeded':
                    message = result['result']['message']
                    translation = message['content'][0]['text']
                    results[custom_id] = translation
                else:
                    # Handle errors
                    error_type = result['result'].get('error', {}).get('type', 'unknown')
                    raise Exception(f"Batch request {custom_id} failed: {error_type}")

        return results

    def translate(self, content):
        """Override translate to use batch API."""
        # For single paragraph translation, create a single-item batch
        batch_request = self.create_batch_request('translation-0', content)
        batch_id = self.create_batch([batch_request])

        log.info(f'Created batch {batch_id}, polling for completion...')
        batch = self.poll_batch(batch_id)

        if batch['results_url']:
            results = self.get_batch_results(batch['results_url'])
            return results.get('translation-0', '')
        else:
            raise Exception('Batch completed but no results URL provided')

