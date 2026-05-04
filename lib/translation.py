import re
import time
import json
from types import GeneratorType

from calibre.utils.localization import _  # type: ignore

from ..engines import builtin_engines
from ..engines import GoogleFreeTranslateNew
from ..engines.base import Base
from ..engines.custom import CustomTranslate

from .utils import log, sep, trim, dummy, traceback_error
from .config import get_config
from .exception import (
    TranslationFailed, TranslationCanceled, RefusalExhausted)
from .handler import Handler


load_translations()  # type: ignore


class Glossary:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.glossary = []

    def load_from_file(self, path):
        content = None
        try:
            with open(path, 'r', newline=None) as f:
                content = f.read().strip()
        except Exception:
            pass
        if not content:
            return
        groups = re.split(r'\n{2,}', content.strip(u'\ufeff'))
        for group in filter(trim, groups):
            group = group.split('\n')
            self.glossary.append(
                (group[0], group[0] if len(group) < 2 else group[1]))

    def replace(self, content):
        for wid, words in enumerate(self.glossary):
            replacement = self.placeholder[0].format(format(wid, '06'))
            content = content.replace(words[0], replacement)
        return content

    def restore(self, content):
        for wid, words in enumerate(self.glossary):
            pattern = self.placeholder[1].format(format(wid, '06'))
            # Eliminate the impact of backslashes on substitution.
            content = re.sub(pattern, lambda _: words[1], content)
        return content


class ProgressBar:
    total = 0
    length = 0.0
    step = 0

    _count = 0

    def load(self, total):
        self.total = total
        self.step = 1.0 / total

    @property
    def count(self):
        self._count += 1
        self.length += self.step
        return self._count


class Translation:
    def __init__(self, translator, glossary):
        self.translator = translator
        self.glossary = glossary

        self.fresh = False
        self.batch = False
        self.progress = dummy
        self.log = dummy
        self.streaming = dummy
        self.callback = dummy
        self.cancel_request = dummy

        self.total = 0
        self.progress_bar = ProgressBar()
        self.abort_count = 0

    def set_fresh(self, fresh):
        self.fresh = fresh

    def set_batch(self, batch):
        self.batch = batch

    def set_progress(self, progress):
        self.progress = progress

    def set_logging(self, log):
        self.log = log

    def set_streaming(self, streaming):
        self.streaming = streaming

    def set_callback(self, callback):
        self.callback = callback

    def set_cancel_request(self, cancel_request):
        self.cancel_request = cancel_request

    def need_stop(self):
        # Cancel the request if there are more than max continuous errors.
        return self.translator.max_error_count > 0 and \
            self.abort_count >= self.translator.max_error_count

    def translate_text(self, row, text, retry=0, interval=0):
        """Translation engine service error code documentation:
        * https://cloud.google.com/apis/design/errors
        * https://www.deepl.com/docs-api/api-access/error-handling/
        * https://platform.openai.com/docs/guides/error-codes/api-errors
        * https://ai.youdao.com/DOCSIRMA/html/trans/api/wbfy/index.html
        * https://api.fanyi.baidu.com/doc/21
        """
        if self.cancel_request():
            raise TranslationCanceled(_('Translation canceled.'))
        try:
            translation = self.translator.translate(text)
            self.abort_count = 0
            return translation
        except Exception as e:
            if self.cancel_request() or self.need_stop():
                raise TranslationCanceled(_('Translation canceled.'))
            self.abort_count += 1
            message = _('Failed to retrieve data from translate engine API.')
            if retry >= self.translator.request_attempt:
                raise TranslationFailed('{}\n{}'.format(message, str(e)))
            retry += 1
            interval += 5
            # Logging any errors that occur during translation.
            logged_text = text[:200] + '...' if len(text) > 200 else text
            error_messages = [
                sep(), _('Original: {}').format(logged_text), sep('┈'),
                _('Status: Failed {} times / Sleeping for {} seconds')
                .format(retry, interval), sep('┈'), _('Error: {}')
                .format(traceback_error())]
            if row >= 0:
                error_messages.insert(1, _('Row: {}').format(row))
            self.log('\n'.join(error_messages), True)
            if self.translator.match_error(str(e)):
                raise TranslationCanceled(_('Translation canceled.'))
            time.sleep(interval)
            return self.translate_text(row, text, retry, interval)

    # Minimum chunk size (characters) below which we won't split further.
    MIN_SPLIT_CHUNK_SIZE = 200

    def _consume_streaming(self, generator):
        """Consume a streaming generator and return the full text. Honors
        cancellation requests and emits chars to the streaming UI when
        translating a single paragraph."""
        if self.total == 1:
            temp = ''
            clear = True
            for char in generator:
                if self.cancel_request():
                    raise TranslationCanceled(_('Translation canceled.'))
                if clear:
                    self.streaming('')
                    clear = False
                self.streaming(char)
                time.sleep(0.05)
                temp += char
            return temp
        temp_chars = []
        for char in generator:
            if self.cancel_request():
                raise TranslationCanceled(_('Translation canceled.'))
            temp_chars.append(char)
        return ''.join(temp_chars)

    def _translate_chunk_with_refusal_retries(self, row, text, label=''):
        """Translate one chunk with refusal-detection retries. Returns the
        glossary-restored translation. Raises RefusalExhausted if all
        retries fail due to refusals; other exceptions propagate."""
        refusal_retries = getattr(
            self.translator, 'refusal_max_retries', 0)
        translation = ''
        for refusal_attempt in range(refusal_retries + 1):
            self.streaming('')
            self.streaming(_('Translating...'))
            translation = self.translate_text(row, text)
            if isinstance(translation, GeneratorType):
                translation = self._consume_streaming(translation)
            translation = self.glossary.restore(translation)

            if not (hasattr(self.translator, 'is_translation_refusal')
                    and self.translator.is_translation_refusal(translation)):
                return translation

            logged = translation[:200]
            if len(translation) > 200:
                logged += '...'
            row_info = _('Row: {}').format(row) if row >= 0 else ''
            chunk_info = '[{}] '.format(label) if label else ''
            if refusal_attempt < refusal_retries:
                self.log('\n'.join(filter(None, [
                    sep(),
                    row_info,
                    _('{}Translation refusal detected, retrying ({}/{})')
                    .format(chunk_info, refusal_attempt + 1, refusal_retries),
                    sep('┈'),
                    _('Response: {}').format(logged),
                ])), True)
                time.sleep(2)
                continue
            raise RefusalExhausted(
                _('{}Translation refused after {} retries. {} Response: {}')
                .format(
                    chunk_info,
                    refusal_retries,
                    row_info + '.' if row_info else '',
                    logged))
        return translation

    def _split_text_for_retry(self, text):
        """Split text in half at the boundary closest to the midpoint that
        best preserves paragraph structure. Tries the merge separator
        ('\\n\\n') first, then sentence boundaries. Returns (left, right,
        joiner) where joiner is the original separator that should be used
        to recombine the translated halves — preserving alignment with the
        source paragraph count. Both halves must be at least
        MIN_SPLIT_CHUNK_SIZE characters. Raises ValueError if no acceptable
        boundary is found."""
        if len(text) < self.MIN_SPLIT_CHUNK_SIZE * 2:
            raise ValueError(_('Text too short to split'))

        midpoint = len(text) // 2
        min_size = self.MIN_SPLIT_CHUNK_SIZE

        # (regex matching the separator, joiner used when recombining)
        # Paragraph break is preferred — splitting there preserves the
        # original paragraph count, so do_aligment() still passes.
        # Sentence boundary is the fallback for chunks that consist of a
        # single long paragraph; we rejoin with a single space so we do
        # NOT introduce a new paragraph break that would fail alignment.
        candidates = [
            (r'\n\n+', '\n\n'),
            (r'(?<=[.!?])\s+', ' '),
        ]
        for pattern, joiner in candidates:
            best = None
            best_distance = None
            for m in re.finditer(pattern, text):
                left = text[:m.start()].rstrip()
                right = text[m.end():].lstrip()
                if len(left) < min_size or len(right) < min_size:
                    continue
                distance = abs(m.start() - midpoint)
                if best_distance is None or distance < best_distance:
                    best = (left, right, joiner)
                    best_distance = distance
            if best is not None:
                return best
        raise ValueError(_('No acceptable split boundary found'))

    def _translate_with_split(self, row, text, row_info):
        """Single-split fallback. Splits text at \\n\\n closest to midpoint
        (sentence boundary fallback) and translates each half with the
        same retry logic. Returns the joined translation. Raises
        RefusalExhausted if either half still refuses, or ValueError if
        the text can't be split."""
        xa, xb, joiner = self._split_text_for_retry(text)
        self.log('\n'.join(filter(None, [
            sep(),
            row_info,
            _('Splitting chunk and retrying each half '
              '({} + {} chars)').format(len(xa), len(xb)),
        ])), True)
        ta = self._translate_chunk_with_refusal_retries(
            row, xa, label=_('first half'))
        tb = self._translate_chunk_with_refusal_retries(
            row, xb, label=_('second half'))
        return ta.rstrip() + joiner + tb.lstrip()

    def _translate_without_cached_context(self, row, text, row_info):
        """Last-resort fallback: retry the full chunk with the cached
        book context temporarily disabled. The bare paragraph (no
        reference text in the system prompt) is much less identifiable
        as part of a specific copyrighted work."""
        if not hasattr(self.translator, 'full_book_context'):
            return self._translate_chunk_with_refusal_retries(
                row, text, label=_('no context'))
        saved = self.translator.full_book_context
        self.translator.full_book_context = None
        try:
            self.log('\n'.join(filter(None, [
                sep(),
                row_info,
                _('Final fallback: retrying without cached book context'),
            ])), True)
            return self._translate_chunk_with_refusal_retries(
                row, text, label=_('no context'))
        finally:
            self.translator.full_book_context = saved

    def translate_paragraph(self, paragraph):
        if self.cancel_request():
            raise TranslationCanceled(_('Translation canceled.'))
        if paragraph.translation and not self.fresh:
            paragraph.is_cache = True
            return

        text = self.glossary.replace(paragraph.original)
        row_info = _('Row: {}').format(paragraph.row) \
            if paragraph.row >= 0 else ''

        # Three-tier fallback chain:
        #   1. Full chunk with cached context + retries
        #   2. Split chunk into halves, each with retries (atomic)
        #   3. Full chunk WITHOUT cached context (bare paragraph)
        try:
            translation = self._translate_chunk_with_refusal_retries(
                paragraph.row, text)
        except RefusalExhausted as full_failure:
            try:
                translation = self._translate_with_split(
                    paragraph.row, text, row_info)
            except ValueError as split_err:
                self.log('\n'.join(filter(None, [
                    sep(),
                    row_info,
                    _('Cannot split chunk: {}').format(str(split_err)),
                ])), True)
                # Skip directly to no-context fallback
                try:
                    translation = self._translate_without_cached_context(
                        paragraph.row, text, row_info)
                except RefusalExhausted:
                    raise full_failure
            except RefusalExhausted:
                # Split halves still refused — try without cached context
                try:
                    translation = self._translate_without_cached_context(
                        paragraph.row, text, row_info)
                except RefusalExhausted:
                    raise full_failure

        paragraph.translation = translation.strip()
        # Apply aligment checking and processing.
        if self.translator.merge_enabled:
            paragraph.do_aligment(self.translator.separator)
        paragraph.engine_name = self.translator.name
        paragraph.target_lang = self.translator.get_target_lang()
        paragraph.is_cache = False

    def process_translation(self, paragraph):
        self.progress(
            self.progress_bar.length, _('Translating: {}/{}').format(
                self.progress_bar.count, self.progress_bar.total))

        self.streaming(paragraph)
        self.callback(paragraph)

        row = paragraph.row
        original = paragraph.original.strip()
        if paragraph.error is None:
            # Only log verbose content if log_content setting is enabled
            from .config import get_config
            if get_config().get('log_content', True):
                self.log(sep())
                if row >= 0:
                    self.log(_('Row: {}').format(row))
                self.log(_('Original: {}').format(original))
                self.log(sep('┈'))
                message = _('Translation: {}')
                if paragraph.is_cache:
                    message = _('Translation (Cached): {}')
                self.log(message.format(paragraph.translation.strip()))

    # Patterns that mark a paragraph as front-matter / identifying content.
    # Paragraphs matching any of these are excluded from the cached book
    # context (but still translated) to reduce the chance the model can
    # identify the source as a specific copyrighted work and refuse.
    _IDENTIFYING_PATTERNS = [
        re.compile(r'©'),
        re.compile(r'\bcopyright\b', re.IGNORECASE),
        re.compile(r'\ball\s+rights\s+reserved\b', re.IGNORECASE),
        re.compile(r'\bISBN\b'),
        re.compile(r'\bfirst\s+(edition|published|printing)\b', re.IGNORECASE),
        re.compile(r'\bpublished\s+by\b', re.IGNORECASE),
        re.compile(r'library\s+of\s+congress', re.IGNORECASE),
        re.compile(r'cataloging[- ]in[- ]publication', re.IGNORECASE),
    ]

    def _strip_identifying_content(self, paragraphs):
        """Filter out paragraphs containing explicit identifying markers
        (copyright notices, ISBN, publisher info) from the cached context.
        Returns the filtered list and the count of stripped paragraphs.
        Paragraphs are still translated — only excluded from the reference
        context Claude sees."""
        kept = []
        stripped = 0
        for p in paragraphs:
            text = p.original or ''
            if any(pat.search(text) for pat in self._IDENTIFYING_PATTERNS):
                stripped += 1
                continue
            kept.append(p)
        return kept, stripped

    def handle(self, paragraphs=[]):
        start_time = time.time()
        char_count = 0
        for paragraph in paragraphs:
            self.total += 1
            char_count += len(paragraph.original)

        # Set up prompt caching if enabled
        # Collect full book context for caching (Claude only)
        if hasattr(self.translator, 'enable_prompt_caching') and self.translator.enable_prompt_caching:
            cached_paragraphs, stripped = self._strip_identifying_content(
                paragraphs)
            full_context = '\n\n'.join(
                [p.original for p in cached_paragraphs])
            self.translator.full_book_context = full_context
            self.log(sep())
            self.log(_('Prompt caching enabled - using full book context'))
            self.log(_('Context size: {} characters').format(len(full_context)))
            if stripped > 0:
                self.log(_(
                    'Stripped {} identifying paragraph(s) (copyright notice, '
                    'ISBN, etc.) from cached context to reduce refusal risk'
                ).format(stripped))

        self.log(sep())
        self.log(_('Start to translate ebook content'))
        self.log(sep('┈'))
        self.log(_('Item count: {}').format(self.total))
        self.log(_('Character count: {}').format(char_count))

        if self.total < 1:
            raise Exception(_('There is no content need to translate.'))
        self.progress_bar.load(self.total)

        handler = Handler(
            paragraphs, self.translator.concurrency_limit,
            self.translate_paragraph, self.process_translation,
            self.translator.request_interval)
        handler.handle()

        # Pass-2 consistency review (Claude only, opt-in). Runs after the
        # main translation completes. Reviews the translated text for
        # inconsistencies (character names, gender, terminology) and
        # applies corrections.
        if not (self.batch and self.need_stop()):
            self.consistency_pass(paragraphs)

        self.log(sep())
        if self.batch and self.need_stop():
            raise Exception(_('Translation failed.'))
        consuming = round((time.time() - start_time) / 60, 2)
        self.log(_('Time consuming: {} minutes').format(consuming))
        self.log(_('Translation completed.'))
        self.progress(1, _('Translation completed.'))

    def consistency_pass(self, paragraphs):
        """Run a Pass-2 consistency review over the translated paragraphs
        if the engine supports it and the user has enabled the feature.
        Identifies inconsistencies in character names, gender forms, and
        terminology, then applies corrections to the paragraphs and
        triggers the cache update via the existing callback."""
        if not getattr(self.translator, 'enable_consistency_pass', False):
            return
        if not hasattr(self.translator, 'consistency_review'):
            return
        if self.cancel_request():
            return

        # Stable ordering: use the row attribute (or insertion order) so
        # the index we send to the model lines up with our local lookup.
        items = []
        index_to_paragraph = {}
        for i, p in enumerate(paragraphs):
            if p.translation and p.error is None:
                items.append({'index': i, 'translation': p.translation})
                index_to_paragraph[i] = p

        if not items:
            return

        self.log(sep())
        self.log(_('Running consistency pass on {} paragraphs...')
                 .format(len(items)))

        try:
            corrections = self.translator.consistency_review(items)
        except Exception as e:
            self.log(_('Consistency pass failed: {}').format(str(e)), True)
            return

        if not corrections:
            self.log(_('Consistency pass: no inconsistencies found.'))
            return

        applied = 0
        for c in corrections:
            if self.cancel_request():
                break
            paragraph = index_to_paragraph.get(c['index'])
            if paragraph is None:
                continue
            old_translation = paragraph.translation
            paragraph.translation = c['translation']
            if self.translator.merge_enabled:
                paragraph.do_aligment(self.translator.separator)
            applied += 1
            row_label = (paragraph.row if paragraph.row >= 0
                         else c['index'])
            self.log(sep('┈'))
            self.log(_('Corrected row {}: {}')
                     .format(row_label, c.get('reason', '')))
            if get_config().get('log_content', True):
                before = old_translation[:200] + (
                    '...' if len(old_translation) > 200 else '')
                after = paragraph.translation[:200] + (
                    '...' if len(paragraph.translation) > 200 else '')
                self.log(_('Before: {}').format(before))
                self.log(_('After:  {}').format(after))
            # Trigger cache update + UI refresh through the existing
            # callback (advanced.py's translation_callback handles this).
            self.callback(paragraph)

        self.log(sep('┈'))
        self.log(_('Consistency pass: applied {} correction(s).')
                 .format(applied))


def get_engine_class(engine_name=None):
    config = get_config()
    engine_name = engine_name or config.get('translate_engine')
    engines: dict[str, type[Base]] = {
        engine.name: engine for engine in builtin_engines
        if engine.name is not None}
    custom_engines = config.get('custom_engines') or {}
    if engine_name in engines:
        engine_class = engines[engine_name]
    elif engine_name in custom_engines:
        engine_class = CustomTranslate
        engine_data = json.loads(custom_engines[engine_name])
        engine_class.set_engine_data(engine_data)
    else:
        engine_class = GoogleFreeTranslateNew
    engine_preferences = config.get('engine_preferences') or {}
    engine_class.set_config(engine_preferences.get(engine_class.name) or {})
    return engine_class


def get_translator(engine_class=None):
    config = get_config()
    engine_class = engine_class or get_engine_class()
    translator = engine_class()
    translator.set_search_paths(config.get('search_paths'))
    if config.get('proxy_enabled'):
        proxy_type: str | None = config.get('proxy_type')
        proxy_setting: dict[str, list] | None = config.get('proxy_setting')
        if proxy_type is not None and proxy_setting is not None:
            # Compatible with old proxy settings stored as a list.
            if isinstance(proxy_setting, list):
                proxy_setting = {'http': proxy_setting}
            host, port = proxy_setting.get(proxy_type) or ['', '']
            translator.set_proxy(proxy_type, host, port)
    translator.set_merge_enabled(config.get('merge_enabled'))
    return translator


def get_translation(translator, log=None):
    config = get_config()
    glossary = Glossary(translator.placeholder)
    if config.get('glossary_enabled'):
        glossary.load_from_file(config.get('glossary_path'))
    translation = Translation(translator, glossary)
    if get_config().get('log_translation'):
        translation.set_logging(log)
    return translation
