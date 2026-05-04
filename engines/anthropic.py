import json
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
        self.full_book_context = None  # Set externally for prompt caching

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

    # Tool schema for the consistency review. The model is forced to
    # use this tool (tool_choice), so the output is guaranteed to match
    # this structure. Loose required fields — we filter at the
    # validation layer for content quality.
    _CONSISTENCY_TOOL_NAME = 'submit_consistency_review'
    _CONSISTENCY_TOOL_SCHEMA = {
        'name': _CONSISTENCY_TOOL_NAME,
        'description': (
            'Submit the results of a consistency review. Provide the '
            'canonical translations of recurring terms (glossary) and '
            'the list of paragraphs that need correction.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'glossary': {
                    'type': 'array',
                    'description': (
                        'Canonical translations of recurring proper '
                        'nouns, character names, titles, and key terms.'),
                    'items': {
                        'type': 'object',
                        'properties': {
                            'term': {'type': 'string'},
                            'canonical': {'type': 'string'},
                            'type': {'type': 'string'},
                            'notes': {'type': 'string'},
                        },
                        'required': ['canonical'],
                    },
                },
                'corrections': {
                    'type': 'array',
                    'description': (
                        'Paragraphs that need correction to match the '
                        'canonical glossary or fix gender/term '
                        'inconsistencies.'),
                    'items': {
                        'type': 'object',
                        'properties': {
                            'index': {'type': 'integer'},
                            'reason': {'type': 'string'},
                            'translation': {'type': 'string'},
                        },
                        'required': ['index', 'translation'],
                    },
                },
            },
            'required': ['glossary', 'corrections'],
        },
    }

    def _consistency_system_prompt(self, target_lang, attempt=0):
        """Build the system prompt for a consistency review attempt.
        Higher attempt numbers use stronger editorial-context framing
        to recover from refusals on the first try.

        The output structure is enforced by the tool schema — the
        prompt focuses on the analytical task only.
        """
        if attempt == 0:
            return (
                'You are reviewing a {tlang} translation of a book for '
                'consistency. Identify the canonical (most common) '
                'translation of each recurring proper noun, character '
                'name, title, and key term. Then identify paragraphs '
                'that use a different translation than the canonical '
                'one for any such item, or that use inconsistent '
                'gender forms for the same character. Use the '
                '{tool} tool to submit your findings. Only include '
                'glossary entries for items that recur multiple times. '
                'Only include corrections for paragraphs that actually '
                'need fixing.'
            ).format(tlang=target_lang,
                     tool=self._CONSISTENCY_TOOL_NAME)
        # Stronger framing: explicit operator/editor context. The
        # user has already produced this translation; we are only
        # asking for editorial review, not new content.
        return (
            'You are an editorial assistant integrated into a '
            'translation tool. The user has produced a {tlang} '
            'translation of their personal copy of a book and is '
            'requesting an editorial consistency review of their own '
            'work — they are not asking you to translate or reproduce '
            'any source material. Identify inconsistencies between '
            'paragraphs in the user\'s translation: character name '
            'spellings, gender forms, and recurring terminology. Use '
            'the {tool} tool to submit your findings. This editorial '
            'review is a standard quality-assurance task.'
        ).format(tlang=target_lang, tool=self._CONSISTENCY_TOOL_NAME)

    def consistency_review(self, items, on_progress=None,
                           cancel_request=None, _attempt=0):
        """Pass-2 consistency review. Takes a list of {index, translation}
        dicts representing the translated paragraphs (in order) and asks
        the model to identify inconsistencies in character names, gender
        forms, and recurring terminology.

        :on_progress: optional callable(message) invoked at each phase of
            the streaming response so the caller can surface progress to
            the user. The model can take many minutes to produce a full
            review — without progress feedback the UI looks frozen.

        :cancel_request: optional callable() returning True when the user
            has requested cancellation. Checked between SSE event reads
            so Stop is responsive within ~1 second of the next chunk
            arriving (chunks arrive every 50-100ms during normal
            streaming).

        Returns a dict with three keys:
          - 'glossary': list of {term, canonical, type, notes} entries
            describing the canonical translations the model used as the
            consistency reference. Logged for transparency.
          - 'corrections': list of {index, translation, reason} entries
            for paragraphs that need correction.
          - 'raw_response': raw model output text, populated when parsing
            failed or yielded zero usable items.

        Sees only translated text — no copyright risk, since this is the
        user's own translation output, not the original copyrighted book.
        """
        if not items:
            return {'glossary': [], 'corrections': []}

        # Build the indexed input (one numbered block per paragraph)
        input_text = '\n\n'.join(
            '[{}] {}'.format(item['index'], item['translation'])
            for item in items)

        target_lang = self.target_lang
        system_prompt = self._consistency_system_prompt(
            target_lang, attempt=_attempt)
        if _attempt > 0 and on_progress:
            on_progress(_('  Retrying consistency review with stronger '
                          'editorial framing (attempt {})...')
                        .format(_attempt + 1))

        # Stream the response. The consistency review can take many minutes
        # for a long book, which would exceed any reasonable socket read
        # timeout if we waited for the full response. Streaming lets each
        # individual read complete quickly while the overall generation
        # continues in the background.
        #
        # Tool use with forced tool_choice guarantees the output matches
        # _CONSISTENCY_TOOL_SCHEMA, instead of trusting the model to
        # follow JSON-format instructions in the prompt.
        #
        # Prompt caching marks the user message + tool definition as
        # cacheable. On a retry within the 5-minute TTL, the second
        # request reads the cache at 10% input cost. We always cache
        # for the consistency pass — the caching is on our own
        # translation output, no copyright concern, and the savings
        # are substantial for retry scenarios.
        #
        # Note: an earlier mid-stream stall around ~3500 chars was
        # initially suspected to be caching-related but reproduces with
        # caching off too — likely a server-side or model-side issue
        # with this specific request shape, not our caching config.
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
            'tools': [
                # cache_control on the tool definition caches it across
                # retries (tools array stays identical).
                dict(self._CONSISTENCY_TOOL_SCHEMA,
                     **{'cache_control': {'type': 'ephemeral'}})
            ],
            'tool_choice': {
                'type': 'tool',
                'name': self._CONSISTENCY_TOOL_NAME,
            },
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': input_text,
                        # cache_control on the user message caches the
                        # large paragraph payload — the biggest win.
                        'cache_control': {'type': 'ephemeral'},
                    }
                ],
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
                # Always enable prompt caching for the consistency pass
                # regardless of the user's translation-time setting.
                # The cache is over our own translation output (not the
                # original copyrighted source), and the savings on
                # retries are substantial.
                'anthropic-beta': 'prompt-caching-2024-07-31',
            },
            method='POST',
        )
        try:
            ssl_ctx = ssl.create_default_context()
        except Exception:
            ssl_ctx = ssl._create_unverified_context()
        # Build opener with proxy + SSL context if needed.
        handlers = [_HTTPSHandler(context=ssl_ctx)]
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
        if on_progress:
            on_progress(_('  HTTP connection established, '
                          'awaiting first event...'))

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

        try:
            while True:
                # Cancellation check — runs every 100ms regardless of
                # I/O state.
                if cancel_request is not None and cancel_request():
                    try:
                        response.close()
                    except Exception:
                        pass
                    from ..lib.exception import TranslationCanceled as _TC
                    raise _TC(_('Translation canceled.'))

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
                        block = evt.get('content_block') or {}
                        if block.get('type') == 'tool_use':
                            on_progress(_('  Tool call started '
                                          '(submit_consistency_review), '
                                          'streaming JSON...'))
                        else:
                            on_progress(_('  Output started, '
                                          'streaming...'))
                    output_started = True
                elif etype == 'content_block_delta':
                    # With tool_choice forced to a specific tool, the
                    # delta type is input_json_delta and the content
                    # arrives in delta.partial_json — accumulate this
                    # into chunks (the JSON buffer).
                    #
                    # text_delta could also appear (e.g. if the model
                    # produces thinking text before the tool call).
                    # Count its chars for heartbeat but DO NOT mix it
                    # into the JSON buffer — that would produce invalid
                    # JSON when concatenated.
                    delta = evt.get('delta') or {}
                    dtype = delta.get('type')
                    s = ''
                    if dtype == 'input_json_delta':
                        s = str(delta.get('partial_json') or '')
                        if s:
                            chunks.append(s)
                    elif dtype == 'text_delta':
                        s = str(delta.get('text') or '')
                        # Intentionally not appended to chunks.
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
                    # Carries usage info including prompt cache stats.
                    # Surface them so the user can see whether caching
                    # is reducing input cost on retry runs.
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
            reader_done.set()
            try:
                response.close()
            except Exception:
                pass

        raw_text = ''.join(chunks).strip()
        if on_progress:
            on_progress(_('  ...complete: {} characters total')
                        .format(chars_received))

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
            # If the unparseable response looks like a refusal AND we
            # haven't retried yet, give it one more shot with a stronger
            # editorial-context prompt. Refusals here are rare (the
            # consistency pass only sees the user's own translation) but
            # do happen for very famous books or distinctive content.
            if (_attempt < 1
                    and self._is_review_refusal(raw_text)):
                if on_progress:
                    on_progress(_('  Detected refusal in raw response — '
                                  'retrying with stronger editorial '
                                  'framing'))
                return self.consistency_review(
                    items,
                    on_progress=on_progress,
                    cancel_request=cancel_request,
                    _attempt=_attempt + 1)
            # Always include the raw response so the orchestrator can
            # surface it to the log when parsing fails — this avoids the
            # silent "no inconsistencies found" outcome that's
            # indistinguishable from genuine zero-corrections.
            return {
                'glossary': [],
                'corrections': [],
                'raw_response': raw_text,
            }

        # Validate glossary
        glossary_raw = parsed.get('glossary') or []
        glossary = []
        if isinstance(glossary_raw, list):
            for g in glossary_raw:
                if not isinstance(g, dict):
                    continue
                canonical = (g.get('canonical') or '').strip()
                term = (g.get('term') or '').strip()
                if not canonical:
                    continue
                glossary.append({
                    'term': term,
                    'canonical': canonical,
                    'type': (g.get('type') or '').strip(),
                    'notes': (g.get('notes') or '').strip(),
                })

        # Validate corrections
        valid_indices = {item['index'] for item in items}
        corrections_raw = parsed.get('corrections') or []
        corrections = []
        if isinstance(corrections_raw, list):
            for c in corrections_raw:
                if not isinstance(c, dict):
                    continue
                idx = c.get('index')
                if idx not in valid_indices:
                    continue
                new_text = (c.get('translation') or '').strip()
                if not new_text:
                    continue
                corrections.append({
                    'index': idx,
                    'translation': new_text,
                    'reason': (c.get('reason') or '').strip(),
                })

        return {
            'glossary': glossary,
            'corrections': corrections,
            # Include raw response only when both parsed buckets ended up
            # empty — otherwise the JSON parsed fine and the raw text is
            # not useful (just noise that crowds the log).
            'raw_response': (raw_text
                             if not glossary and not corrections else ''),
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

        # Build system message with optional caching
        if self.enable_prompt_caching and self.full_book_context:
            # Use full book context as cached system message
            # More info: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
            #
            # The prompt includes explicit instructions to prevent Claude from
            # refusing to translate content it identifies as copyrighted. When
            # the full book is in the system prompt, Claude can identify the
            # source and may refuse to translate. These instructions clarify
            # that this is a personal translation tool operating on the user's
            # own ebook library.
            cached_prompt = self._get_prompt() + (
                'The reference context below contains the full text of one '
                'ebook from the user\'s library, provided so you can '
                'maintain consistent terminology, character names, and '
                'gendered grammar across segments. Use it as reference only '
                'and translate the segment specified in the user message.')
            system_content = [
                {
                    "type": "text",
                    "text": cached_prompt
                },
                {
                    "type": "text",
                    "text": "Full book context for reference:\n\n" + self.full_book_context,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
            # Add instruction to translate only this section
            user_message = f"Translate only the following section:\n\n{text}"
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

