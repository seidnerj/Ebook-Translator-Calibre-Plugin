import time
from types import MethodType

from qt.core import (  # type: ignore
    Qt, QObject, QDialog, QGroupBox, QWidget, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QPushButton, QSplitter, QLabel, QThread, QLineEdit,
    QGridLayout, QProgressBar, pyqtSignal, pyqtSlot, QPixmap, QEvent,
    QStackedWidget, QSpacerItem, QTabWidget, QCheckBox,
    QComboBox, QSizePolicy, QTextCursor)
from calibre.constants import __version__  # type: ignore
from calibre.gui2 import I  # type: ignore
from calibre.utils.localization import _  # type: ignore

from . import EbookTranslator
from .lib.utils import traceback_error
from .lib.config import get_config
from .lib.encodings import encoding_list
from .lib.cache import Paragraph, get_cache
from .lib.translation import get_engine_class, get_translator, get_translation
from .lib.element import get_element_handler
from .lib.conversion import extract_item, extra_formats
from .engines.openai import ChatgptTranslate, ChatgptBatchTranslate
from .engines.anthropic import ClaudeTranslate
from .engines.custom import CustomTranslate
from .components import (
    EngineList, Footer, SourceLang, TargetLang, InputFormat, OutputFormat,
    AlertMessage, AdvancedTranslationTable, StatusColor, TranslationStatus,
    set_shortcut, ChatgptBatchTranslationManager)
from .components.editor import CodeEditor


load_translations()  # type: ignore


class EditorWorker(QObject):
    start = pyqtSignal((str,), (str, object))
    show = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        QObject.__init__(self)
        self.start[str].connect(self.show_message)
        self.start[str, object].connect(self.show_message)

    @pyqtSlot(str)
    @pyqtSlot(str, object)
    def show_message(self, message, callback=None):
        time.sleep(0.01)
        self.show.emit(message)
        time.sleep(1)
        self.show.emit('')
        if callback is not None:
            callback()
        self.finished.emit()


class PreparationWorker(QObject):
    start = pyqtSignal()
    progress = pyqtSignal(int)
    progress_message = pyqtSignal(str)
    progress_detail = pyqtSignal(str)
    close = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, engine_class, ebook):
        QObject.__init__(self)
        self.current_engine = engine_class
        self.ebook = ebook

        self.on_working = False
        self.canceled = False

        self.start.connect(self.prepare_ebook_data)

    def clean_cache(self, cache):
        if cache.is_fresh():
            cache.destroy()
        self.on_working = False
        self.close.emit(1)

    def set_canceled(self, canceled):
        self.canceled = canceled

    # def cancel(self):
    #     return self.thread().isInterruptionRequested()

    @pyqtSlot()
    def prepare_ebook_data(self):
        self.on_working = True
        input_path = self.ebook.get_input_path()
        element_handler = get_element_handler(
            self.current_engine.placeholder, self.current_engine.separator,
            self.ebook.target_direction)
        from .lib.utils import get_cache_id
        merge_length = element_handler.get_merge_length()
        cache_id = get_cache_id(input_path, self.current_engine.name, self.ebook.target_lang,
                                merge_length, self.ebook.encoding)
        cache = get_cache(cache_id)

        if cache.is_fresh() or not cache.is_persistence():
            self.progress_detail.emit(
                'Start processing the ebook: %s' % self.ebook.title)
            cache.set_info('title', self.ebook.title)
            cache.set_info('engine_name', self.current_engine.name)
            cache.set_info('target_lang', self.ebook.target_lang)
            cache.set_info('merge_length', merge_length)
            cache.set_info('plugin_version', EbookTranslator.__version__)
            cache.set_info('calibre_version', __version__)
            # --------------------------
            a = time.time()
            # --------------------------
            self.progress_message.emit(_('Extracting ebook content...'))
            try:
                elements = extract_item(
                    input_path, self.ebook.input_format, self.ebook.encoding,
                    self.progress_detail.emit)
            except Exception:
                self.progress_message.emit(
                    _('Failed to extract ebook content'))
                self.progress_detail.emit('\n' + traceback_error())
                self.progress.emit(100)
                self.clean_cache(cache)
                return
            if self.canceled:
                self.clean_cache(cache)
                return
            self.progress.emit(30)
            b = time.time()
            self.progress_detail.emit('extracting timing: %s' % (b - a))
            if self.canceled:
                self.clean_cache(cache)
                return
            # --------------------------
            self.progress_message.emit(_('Filtering ebook content...'))
            original_group = element_handler.prepare_original(elements)
            self.progress.emit(80)
            c = time.time()
            self.progress_detail.emit('filtering timing: %s' % (c - b))
            if self.canceled:
                self.clean_cache(cache)
                return
            # --------------------------
            self.progress_message.emit(_('Preparing user interface...'))
            cache.save(original_group)
            self.progress.emit(100)
            d = time.time()
            self.progress_detail.emit('cache timing: %s' % (d - c))
            if self.canceled:
                self.clean_cache(cache)
                return
        else:
            self.progress_detail.emit(
                'Loading data from cache and preparing user interface...')
            time.sleep(0.1)

        self.finished.emit(cache_id)
        self.on_working = False


class TranslationWorker(QObject):
    start = pyqtSignal()
    close = pyqtSignal(int)
    finished = pyqtSignal()
    translate = pyqtSignal(list, bool)
    consistency = pyqtSignal(list)
    consistency_completed = pyqtSignal()
    agreement = pyqtSignal(list)
    agreement_completed = pyqtSignal(int)
    logging = pyqtSignal(str, bool)
    # error = pyqtSignal(str, str, str)
    streaming = pyqtSignal(object)
    callback = pyqtSignal(object)

    def __init__(self, engine_class, ebook):
        QObject.__init__(self)
        self.source_lang = ebook.source_lang
        self.target_lang = ebook.target_lang
        self.current_engine = engine_class

        self.on_working = False
        self.canceled = False
        self.need_close = False
        # The Translation Brief (a dict) is held here in-memory after
        # build, and also persisted to cache via `translation_brief`
        # info key. Drafting reads it from whichever source is
        # populated and pins it on the translator via
        # `translator.translation_brief = ...`.
        self.brief = None
        # Cache reference, set externally by the dialog after the
        # cache is constructed. Used by the auto-trigger to fetch
        # all paragraphs for brief building, and to persist a
        # newly-built brief.
        self.cache = None
        self.translate.connect(self.translate_paragraphs)
        self.consistency.connect(self.run_consistency_pass)
        self.agreement.connect(self.run_agreement_pass)
        # self.finished.connect(lambda: self.set_canceled(False))

    def set_source_lang(self, lang):
        self.source_lang = lang

    def set_target_lang(self, lang):
        self.target_lang = lang

    def set_engine_class(self, engine_class):
        self.current_engine = engine_class

    def set_canceled(self, canceled):
        self.canceled = canceled

    def cancel_request(self):
        return self.canceled

    def set_need_close(self, need_close):
        self.need_close = need_close

    def _auto_build_brief(self, translator, log):
        """Auto-build a Translation Brief from all source paragraphs
        in the cache, before per-paragraph translation begins. This
        runs the full three-turn pipeline (build → language review →
        logic review) — same flow as the manual 'Build Brief' button,
        but with quieter logging suitable for an in-line preparation
        step.

        Returns the brief dict on success, or None if the build
        failed for any reason. The caller persists the brief to
        cache via `cache.set_info('translation_brief', ...)`.
        """
        from .lib.translation import Translation as _Translation
        all_paragraphs = self.cache.all_paragraphs()
        if not all_paragraphs:
            return None
        log('═' * 50)
        log(_('Auto-building Translation Brief before drafting...'))
        log(_('(First translation in this book — building a '
              'reference document with canonical names, character '
              'profiles with gender, and recurring terminology so '
              'subsequent paragraph translations stay consistent. '
              'This takes ~2-3 minutes once per book; afterwards the '
              'brief is cached and reused on every subsequent '
              'translate.)'))
        patterns = _Translation._IDENTIFYING_PATTERNS
        items = []
        stripped = 0
        for i, p in enumerate(all_paragraphs):
            text = (p.original or '').strip()
            if not text:
                continue
            if any(pat.search(text) for pat in patterns):
                stripped += 1
                continue
            items.append({'index': i, 'text': text})
        if not items:
            log(_('No source paragraphs available for brief '
                  'building.'), True)
            return None
        log(_('Source: {} eligible blocks (stripped {} '
              'identifying paragraphs).').format(
                len(items), stripped))
        try:
            brief = translator.build_translation_brief(
                items,
                on_progress=log,
                cancel_request=self.cancel_request)
        except Exception as e:
            log(_('Brief build failed: {}').format(str(e)), True)
            log(traceback_error(), True)
            return None
        if brief is None:
            log(_('Brief build returned no usable result; '
                  'proceeding without brief.'), True)
            return None
        # Quieter summary than the manual-button path.
        chars = brief.get('characters') or []
        terms = brief.get('terminology') or []
        log(_('Brief built: {} character(s), {} term(s).').format(
            len(chars), len(terms)))
        log('═' * 50)
        return brief

    def _auto_agreement_pass(self, translator, log):
        """Run the Agreement Pass over every paragraph in the cache
        immediately after drafting completes. Quieter logging than
        the manual-button path; fixes are applied to cache and the
        UI is refreshed via callback as each paragraph mutates.
        """
        paragraphs = self.cache.all_paragraphs() or []
        items = []
        for i, p in enumerate(paragraphs):
            t = (getattr(p, 'translation', None) or '').strip()
            if t:
                items.append({'index': i,
                              'translation': p.translation})
        if not items:
            return
        log('═' * 50)
        log(_('Running post-translation Agreement Pass...'))
        result = translator.agreement_review(
            items, self.brief, on_progress=log,
            cancel_request=self.cancel_request)
        fixes = result.get('fixes') or []
        unfixable = result.get('unfixable') or []
        considered = result.get('considered', 0)

        applied = 0
        char_index = {}
        for c in (self.brief.get('characters') or []):
            if isinstance(c, dict) and c.get('id'):
                char_index[c['id']] = c
        for f in fixes:
            idx = f.get('block_index')
            old_str = f.get('old_str') or ''
            new_str = f.get('new_str') or ''
            if (not isinstance(idx, int)
                    or idx < 0 or idx >= len(paragraphs)
                    or not old_str or old_str == new_str):
                continue
            p = paragraphs[idx]
            t = getattr(p, 'translation', None) or ''
            if t.count(old_str) != 1:
                continue
            p.translation = t.replace(old_str, new_str, 1)
            applied += 1
            try:
                self.cache.update_paragraph(p)
            except Exception:
                pass
            try:
                self.callback.emit(p)
            except Exception:
                pass
            cid = f.get('character_id') or ''
            cname = (char_index.get(cid, {}).get('canonical_name')
                     if cid else '') or cid or '—'
            kind = f.get('kind') or 'other'
            log('  ✓ block_{} [{}] {} — {!r} → {!r}'.format(
                idx, kind, cname, old_str, new_str))
        if unfixable:
            log(_('Rejected (uniqueness failed): {}').format(
                len(unfixable)))
        log(_('Agreement Pass: reviewed {} character-mentioning '
              'paragraph(s); {} fix(es) applied.').format(
            considered, applied))
        try:
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d %H:%M')
            self.cache.set_info('last_agreement_pass', ts)
        except Exception:
            pass
        try:
            self.agreement_completed.emit(applied)
        except Exception:
            pass

    @pyqtSlot(list, bool)
    def translate_paragraphs(self, paragraphs=[], fresh=False):
        """:fresh: retranslate all paragraphs."""
        self.on_working = True
        self.start.emit()
        translator = get_translator(self.current_engine)
        translator.set_source_lang(self.source_lang)
        translator.set_target_lang(self.target_lang)

        import json as _json
        log = lambda text, error=False: self.logging.emit(text, error)

        # ── Auto-trigger Translation Brief build ──────────────────
        # If no brief exists yet AND the engine supports brief
        # building AND the user hasn't opted out via setting AND we
        # have access to cached source paragraphs, build a brief
        # before drafting starts. The brief is the difference
        # between "translate each paragraph in isolation" and
        # "translate with full canonical-name and gender awareness."
        if (self.brief is None
                and hasattr(translator, 'build_translation_brief')
                and getattr(translator, 'enable_translation_brief',
                            False)
                and self.cache is not None):
            try:
                brief = self._auto_build_brief(translator, log)
                if brief is not None:
                    self.brief = brief
                    try:
                        self.cache.set_info(
                            'translation_brief',
                            _json.dumps(brief, ensure_ascii=False))
                    except Exception:
                        pass
            except Exception as e:
                log(_('Auto-brief build failed: {} — proceeding '
                      'without brief.').format(str(e)), True)

        # Pin the brief on the translator so engines that support
        # brief-aware drafting can inject it into their per-paragraph
        # system prompt. Strip review-pipeline metadata (the
        # _review_*_changes keys) before injecting — those are build-
        # time artifacts, not part of the brief's reference content.
        if self.brief is not None:
            brief_for_drafting = {
                k: v for k, v in self.brief.items()
                if not (isinstance(k, str) and k.startswith('_review_'))}
            translator.translation_brief = brief_for_drafting

        translation = get_translation(translator)
        translation.set_fresh(fresh)
        translation.set_logging(
            lambda text, error=False: self.logging.emit(text, error))
        translation.set_streaming(self.streaming.emit)
        translation.set_callback(self.callback.emit)
        translation.set_cancel_request(self.cancel_request)
        translation.handle(paragraphs)

        # ── Auto-trigger Agreement Pass ────────────────────────────
        # Once drafting is complete, run a post-translation
        # Agreement Pass to fix residual gender/number/pronoun drift
        # against the brief's canonical morphology. Gated on the
        # same setting that gates the manual button — disabling the
        # setting suppresses both the button and this auto-run.
        # Cancellation, missing brief, or unsupported engine all
        # silently skip (logged, non-fatal).
        if (not self.cancel_request()
                and self.brief is not None
                and self.cache is not None
                and getattr(translator, 'supports_agreement_review',
                            False)
                and getattr(translator, 'enable_agreement_pass',
                            False)
                and hasattr(translator, 'agreement_review')):
            try:
                self._auto_agreement_pass(translator, log)
            except Exception as e:
                log(_('Auto Agreement Pass failed: {}').format(
                    str(e)), True)

        self.on_working = False
        self.finished.emit()
        if self.need_close:
            time.sleep(0.5)
            self.close.emit(0)

    @pyqtSlot(list)
    def run_consistency_pass(self, paragraphs=[]):
        """Phase 0 (validation spike): the 'Consistency Pass' button
        is temporarily repurposed to trigger Translation Brief
        construction. The brief is logged for inspection — no
        persistence to cache yet, no apply phase. Once Phase 0
        validates that brief construction works on real copyrighted
        material, this slot will be replaced by separate prep /
        terminology / agreement slots in Phase 1.
        """
        import json as _json
        from .lib.translation import Translation as _Translation
        self.on_working = True
        self.start.emit()
        translator = get_translator(self.current_engine)
        translator.set_source_lang(self.source_lang)
        translator.set_target_lang(self.target_lang)
        log = lambda text, error=False: self.logging.emit(text, error)

        if not hasattr(translator, 'build_translation_brief'):
            log(_('Brief building is not supported by this engine '
                  '(currently Claude only).'), True)
            self.on_working = False
            self.finished.emit()
            return

        try:
            # Build items from SOURCE text (not translation). The
            # brief is a pre-translation reference document.
            patterns = _Translation._IDENTIFYING_PATTERNS
            items = []
            stripped = 0
            for i, p in enumerate(paragraphs):
                text = (p.original or '').strip()
                if not text:
                    continue
                if any(pat.search(text) for pat in patterns):
                    stripped += 1
                    continue
                items.append({'index': i, 'text': text})

            log('═' * 50)
            log(_('Building Translation Brief...'))
            log(_('Source language: {} → Target language: {}')
                .format(self.source_lang, self.target_lang))
            log(_('Eligible source blocks: {} (stripped {} '
                  'identifying paragraphs)')
                .format(len(items), stripped))

            if not items:
                log(_('No eligible source paragraphs to analyze.'),
                    True)
                return

            brief = translator.build_translation_brief(
                items, on_progress=log,
                cancel_request=self.cancel_request)

            log('═' * 50)
            if brief is None:
                log(_('Brief build returned no usable result. '
                      'See raw response above for diagnosis.'), True)
                return

            # Surface the review change lists (if any) before the
            # final brief dump so the user can see what each critic
            # caught and what was applied.
            language_changes = brief.pop(
                '_review_language_changes', None) \
                if isinstance(brief, dict) else None
            logic_changes = brief.pop(
                '_review_logic_changes', None) \
                if isinstance(brief, dict) else None
            # Backwards compat with earlier single-review key.
            legacy_changes = brief.pop('_review_change_list', None) \
                if isinstance(brief, dict) else None
            if legacy_changes is not None and language_changes is None:
                language_changes = legacy_changes

            log(_('Refined brief (post-review). Dumping JSON to log:'))
            log(_json.dumps(brief, ensure_ascii=False, indent=2))

            def _dump_changes(label, changes):
                if changes is None:
                    return
                log('─' * 50)
                if changes:
                    log(_('{} review found {} issue(s):').format(
                        label, len(changes)))
                    for issue in changes:
                        cat = issue.get('category', '')
                        path = issue.get('field_path', '')
                        cur = issue.get('current', '')
                        sug = issue.get('suggested', '')
                        reason = issue.get('reason', '')
                        log('  [{}] {}: {!r} → {!r} — {}'.format(
                            cat, path, cur, sug, reason))
                else:
                    log(_('{} review found no issues.').format(label))

            _dump_changes(_('Language'), language_changes)
            _dump_changes(_('Logic'), logic_changes)

            # Surface a quick-scan summary alongside the JSON.
            log('─' * 50)
            summary = brief.get('source_summary') or {}
            themes = summary.get('themes') or []
            central = summary.get('central_conflict', '')
            if themes:
                log(_('Themes: {}').format(', '.join(themes)))
            if central:
                log(_('Central conflict: {}').format(central))

            chars = brief.get('characters') or []
            terms = brief.get('terminology') or []
            char_index = {c.get('id', ''): c for c in chars if c.get('id')}
            log(_('Summary: {} character(s), {} term(s).')
                .format(len(chars), len(terms)))
            for c in chars:
                cid = c.get('id', '')
                name = c.get('canonical_name', '')
                src = c.get('source_name', '')
                gender = c.get('gender', '')
                role = c.get('role', '')
                mentions = c.get('mention_count')
                first = c.get('first_occurrence_index')
                meta_bits = []
                if mentions is not None:
                    meta_bits.append('×{}'.format(mentions))
                if first is not None:
                    meta_bits.append('first@block_{}'.format(first))
                meta = ' [' + ', '.join(meta_bits) + ']' if meta_bits else ''
                log('  • [{}] {} ({} ← {}) — {}{}'.format(
                    cid, name, gender, src, role, meta))
                # Show relationships using canonical names where ids
                # resolve, ids otherwise.
                for rel in (c.get('relationships') or []):
                    to_id = rel.get('to_id', '')
                    rtype = rel.get('type', '')
                    target = char_index.get(to_id)
                    target_label = (target.get('canonical_name', to_id)
                                    if target else to_id)
                    log('       ↳ {} → {}'.format(rtype, target_label))
            for t in terms:
                tid = t.get('id', '')
                canon = t.get('canonical', '')
                src = t.get('source_form', '')
                ttype = t.get('type', '')
                dnt = t.get('do_not_translate', False)
                mentions = t.get('mention_count')
                first = t.get('first_occurrence_index')
                meta_bits = []
                if dnt:
                    meta_bits.append('DNT')
                if mentions is not None:
                    meta_bits.append('×{}'.format(mentions))
                if first is not None:
                    meta_bits.append('first@block_{}'.format(first))
                meta = ' [' + ', '.join(meta_bits) + ']' if meta_bits else ''
                log('  · [{}] {} ({}) ← {}{}'.format(
                    tid, canon, ttype, src, meta))

            # Phase 1a spike: hold the brief in-memory on the worker
            # so subsequent translate_paragraphs calls in the same
            # session can pin it on the translator. The dialog's
            # consistency_completed handler will also persist it to
            # cache so it survives plugin reloads.
            self.brief = brief
            self.consistency_completed.emit()
        except Exception as e:
            log(_('Brief build failed: {}').format(str(e)), True)
            log(traceback_error(), True)
        finally:
            self.on_working = False
            self.finished.emit()

    @pyqtSlot(list)
    def run_agreement_pass(self, paragraphs=[]):
        """Run an Agreement Pass over the translated paragraphs:
        scan for residual gender/number/pronoun drift against the
        canonical character morphology in the brief, request fixes
        from the engine, validate single-occurrence uniqueness, and
        apply the validated fixes back to the cache.

        Requires a brief to be present (the canonical morphology
        source). Skips silently with a log message if absent.
        """
        import json as _json
        self.on_working = True
        self.start.emit()
        translator = get_translator(self.current_engine)
        translator.set_source_lang(self.source_lang)
        translator.set_target_lang(self.target_lang)
        log = lambda text, error=False: self.logging.emit(text, error)

        if not getattr(translator, 'supports_agreement_review', False) \
                or not hasattr(translator, 'agreement_review'):
            log(_('Agreement Pass is not supported by this engine '
                  '(currently Claude only).'), True)
            self.on_working = False
            self.finished.emit()
            return

        # Rehydrate brief: prefer the in-memory copy, fall back to
        # the persisted one in cache.
        brief = self.brief
        if brief is None and self.cache is not None:
            try:
                brief_json = self.cache.get_info('translation_brief')
                if brief_json:
                    brief = _json.loads(brief_json)
            except Exception:
                brief = None
        if not brief:
            log(_('Agreement Pass requires a Translation Brief. '
                  'Click "Build Brief" first (or run a translation '
                  'with auto-brief enabled).'), True)
            self.on_working = False
            self.finished.emit()
            return

        try:
            log('═' * 50)
            log(_('Running Agreement Pass...'))
            log(_('Target language: {}').format(self.target_lang))

            # Items use the paragraphs' indices in the supplied list
            # — these match the block_N indices the engine emits in
            # its fixes. Caller passes all_paragraphs() so indices
            # are stable across the book.
            items = []
            for i, p in enumerate(paragraphs):
                t = (getattr(p, 'translation', None) or '').strip()
                if not t:
                    continue
                items.append({
                    'index': i,
                    'translation': p.translation,
                })
            log(_('Translated paragraphs available: {}/{}.').format(
                len(items), len(paragraphs)))
            if not items:
                log(_('No translated paragraphs to review.'), True)
                return

            result = translator.agreement_review(
                items, brief, on_progress=log,
                cancel_request=self.cancel_request)
            fixes = result.get('fixes') or []
            unfixable = result.get('unfixable') or []
            considered = result.get('considered', 0)

            log('─' * 50)
            log(_('Agreement Pass: reviewed {} paragraph(s) that '
                  'mention named characters; {} fix(es) validated, '
                  '{} rejected by uniqueness check.').format(
                considered, len(fixes), len(unfixable)))

            applied = 0
            char_index = {}
            for c in (brief.get('characters') or []):
                if isinstance(c, dict) and c.get('id'):
                    char_index[c['id']] = c
            for f in fixes:
                idx = f.get('block_index')
                old_str = f.get('old_str') or ''
                new_str = f.get('new_str') or ''
                if (not isinstance(idx, int)
                        or idx < 0 or idx >= len(paragraphs)
                        or not old_str or old_str == new_str):
                    continue
                p = paragraphs[idx]
                t = getattr(p, 'translation', None) or ''
                # Re-verify count == 1 at apply time (engine may
                # have validated against the same text, but be
                # defensive).
                if t.count(old_str) != 1:
                    continue
                p.translation = t.replace(old_str, new_str, 1)
                applied += 1
                if self.cache is not None:
                    try:
                        self.cache.update_paragraph(p)
                    except Exception:
                        pass
                # Surface to the table so the user sees the change.
                try:
                    self.callback.emit(p)
                except Exception:
                    pass
                cid = f.get('character_id') or ''
                cname = (char_index.get(cid, {}).get('canonical_name')
                         if cid else '') or cid or '—'
                kind = f.get('kind') or 'other'
                log('  ✓ block_{} [{}] {} — {!r} → {!r} ({})'.format(
                    idx, kind, cname, old_str, new_str,
                    f.get('reason') or ''))

            if unfixable:
                log('─' * 50)
                log(_('Rejected (uniqueness failed) — surfaced for '
                      'inspection:'))
                for u in unfixable:
                    log('  ✗ block_{}: {!r} → {!r} — {}'.format(
                        u.get('block_index', '?'),
                        u.get('old_str', ''),
                        u.get('new_str', ''),
                        u.get('reason', '')))

            log('═' * 50)
            log(_('Agreement Pass complete: {} fix(es) applied.')
                .format(applied))
            self.agreement_completed.emit(applied)
        except Exception as e:
            log(_('Agreement Pass failed: {}').format(str(e)), True)
            log(traceback_error(), True)
        finally:
            self.on_working = False
            self.finished.emit()


class CreateTranslationProject(QDialog):
    start_translation = pyqtSignal(object)

    def __init__(self, parent, ebook):
        QDialog.__init__(self, parent)
        self.ebook = ebook

        layout = QVBoxLayout(self)
        self.choose_format = self.layout_format()

        self.start_button = QPushButton(_('&Start'))
        # self.start_button.setStyleSheet(
        #     'padding:0;height:48;font-size:20px;color:royalblue;'
        #     'text-transform:uppercase;')
        self.start_button.clicked.connect(self.show_advanced)

        layout.addWidget(self.choose_format)
        layout.addWidget(self.start_button)

    def layout_format(self):
        engine_class = get_engine_class()
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        input_group = QGroupBox(_('Input Format'))
        input_layout = QGridLayout(input_group)
        input_format = InputFormat(self.ebook.files.keys())
        # input_format.setFixedWidth(150)
        input_layout.addWidget(input_format)
        layout.addWidget(input_group, 0, 0, 1, 3)

        output_group = QGroupBox(_('Output Format'))
        output_layout = QGridLayout(output_group)
        output_format = OutputFormat()
        # output_format.setFixedWidth(150)
        output_layout.addWidget(output_format)
        layout.addWidget(output_group, 0, 3, 1, 3)

        source_group = QGroupBox(_('Source Language'))
        source_layout = QVBoxLayout(source_group)
        source_lang = SourceLang()
        source_lang.setFixedWidth(150)
        source_layout.addWidget(source_lang)
        layout.addWidget(source_group, 1, 0, 1, 2)

        target_group = QGroupBox(_('Target Language'))
        target_layout = QVBoxLayout(target_group)
        target_lang = TargetLang()
        target_lang.setFixedWidth(150)
        target_layout.addWidget(target_lang)
        layout.addWidget(target_group, 1, 2, 1, 2)

        source_lang.refresh.emit(
            engine_class.lang_codes.get('source'),
            engine_class.config.get('source_lang'),
            not issubclass(engine_class, CustomTranslate))

        target_lang.refresh.emit(
            engine_class.lang_codes.get('target'),
            engine_class.config.get('target_lang'))

        def change_input_format(_format):
            self.ebook.set_input_format(_format)
        change_input_format(input_format.currentText())
        input_format.currentTextChanged.connect(change_input_format)

        def change_output_format(_format):
            self.ebook.set_output_format(_format)
        if self.ebook.is_extra_format():
            output_format.lock_format(self.ebook.input_format)
            change_output_format(self.ebook.input_format)
        else:
            change_output_format(output_format.currentText())
            output_format.currentTextChanged.connect(change_output_format)

        def change_source_lang(lang):
            self.ebook.set_source_lang(lang)
        change_source_lang(source_lang.currentText())
        source_lang.currentTextChanged.connect(change_source_lang)

        def change_target_lang(lang):
            self.ebook.set_target_lang(lang)
            self.ebook.set_lang_code(
                engine_class.get_iso639_target_code(lang))
        change_target_lang(target_lang.currentText())
        target_lang.currentTextChanged.connect(change_target_lang)

        if self.ebook.input_format in extra_formats.keys():
            encoding_group = QGroupBox(_('Encoding'))
            encoding_layout = QVBoxLayout(encoding_group)
            encoding_select = QComboBox()
            encoding_select.setFixedWidth(150)
            encoding_select.addItems(encoding_list)
            encoding_layout.addWidget(encoding_select)
            layout.addWidget(encoding_group, 1, 4, 1, 2)

            def change_encoding(encoding):
                self.ebook.set_encoding(encoding)
            encoding_select.currentTextChanged.connect(change_encoding)
        else:
            direction_group = QGroupBox(_('Target Directionality'))
            direction_layout = QVBoxLayout(direction_group)
            direction_list = QComboBox()
            direction_list.setFixedWidth(150)
            direction_list.addItem(_('Auto'), 'auto')
            direction_list.addItem(_('Left to Right'), 'ltr')
            direction_list.addItem(_('Right to Left'), 'rtl')
            direction_layout.addWidget(direction_list)
            layout.addWidget(direction_group, 1, 4, 1, 2)

            def change_direction(_index):
                _direction = direction_list.itemData(_index)
                self.ebook.set_target_direction(_direction)
            direction_list.currentIndexChanged.connect(change_direction)

            engine_target_lange_codes = engine_class.lang_codes.get('target')
            if engine_target_lange_codes is not None and \
                    self.ebook.target_lang in engine_target_lange_codes:
                target_lang_code = engine_target_lange_codes[
                    self.ebook.target_lang]
                direction = engine_class.get_lang_directionality(
                    target_lang_code)
                index = direction_list.findData(direction)
                direction_list.setCurrentIndex(index)

        return widget

    @pyqtSlot()
    def show_advanced(self):
        self.done(0)
        self.start_translation.emit(self.ebook)


class AdvancedTranslation(QDialog):
    paragraph_sig = pyqtSignal(object)
    ebook_title = pyqtSignal()
    progress_bar = pyqtSignal()
    batch_translation = pyqtSignal()

    def __init__(self, plugin, parent, worker, ebook):
        QDialog.__init__(self, parent)

        self.ui_settings = plugin.ui_settings
        self.api = parent.current_db.new_api
        self.worker = worker
        self.ebook = ebook

        self.config = get_config()
        self.alert = AlertMessage(self)
        self.footer = Footer()
        # self.error = JobError(self)
        self.current_engine = get_engine_class()
        self.cache = None
        self.merge_enabled = False

        self.progress_step = 0
        self.translate_all = False

        # Create threads per-instance to avoid accumulating signal
        # connections across dialog instances.
        self.preparation_thread = QThread()
        self.trans_thread = QThread()
        self.editor_thread = QThread()

        self.editor_worker = EditorWorker()
        self.editor_worker.moveToThread(self.editor_thread)
        self.editor_thread.start()

        self.trans_worker = TranslationWorker(self.current_engine, self.ebook)
        self.trans_worker.close.connect(self.done)
        self.trans_worker.moveToThread(self.trans_thread)
        self.trans_thread.start()

        self.preparation_worker = PreparationWorker(
            self.current_engine, self.ebook)
        self.preparation_worker.close.connect(self.done)
        self.preparation_worker.moveToThread(self.preparation_thread)
        self.preparation_thread.start()

        layout = QVBoxLayout(self)

        self.waiting = self.layout_progress()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.waiting)
        layout.addWidget(self.stack)
        layout.addWidget(self.footer)

        def working_status():
            self.logging_text.clear()
            self.errors_text.clear()
        self.trans_worker.start.connect(working_status)

        self.trans_worker.logging.connect(
            lambda text, error: self.errors_text.appendPlainText(text)
            if error else self.logging_text.appendPlainText(text))

        def working_finished():
            if self.translate_all and not self.trans_worker.cancel_request():
                failures = len(self.table.get_selected_paragraphs(True, True))
                if failures > 0:
                    message = _(
                        'Failed to translate {} paragraph(s), '
                        'Would you like to retry?')
                    if self.alert.ask(message.format(failures)) == 'yes':
                        self.translate_all_paragraphs()
                        return
                else:
                    self.alert.pop(_('Translation completed.'))
            self.trans_worker.set_canceled(False)
            self.translate_all = False
        self.trans_worker.finished.connect(working_finished)

        # self.trans_worker.error.connect(
        #     lambda title, reason, detail: self.error.show_error(
        #         title, _('Failed') + ': ' + reason, det_msg=detail))

        def prepare_table_layout(cache_id):
            self.cache = get_cache(cache_id)
            # Hand the cache to the worker so the auto-brief build
            # path can fetch all source paragraphs and persist a
            # newly-built brief without round-tripping through the
            # dialog.
            self.trans_worker.cache = self.cache
            merge_length = self.cache.get_info('merge_length') or 0
            self.merge_enabled = int(merge_length) > 0
            paragraphs = self.cache.all_paragraphs()
            if len(paragraphs) < 1:
                self.alert.pop(
                    _('There is no content that needs to be translated.'),
                    'warning')
                self.done(0)
                return
            # Phase 1a spike: rehydrate the brief from cache (if a
            # previous session built one) so the worker has it
            # available for translation calls without requiring the
            # user to rebuild.
            try:
                import json as _json_load
                brief_json = self.cache.get_info('translation_brief')
                if brief_json:
                    self.trans_worker.brief = _json_load.loads(brief_json)
            except Exception:
                # Corrupt JSON or missing key — proceed without brief.
                pass
            self.table = AdvancedTranslationTable(self, paragraphs)
            self.panel = self.layout_panel()
            self.stack.addWidget(self.panel)
            self.stack.setCurrentWidget(self.panel)
            self.table.setFocus(Qt.OtherFocusReason)
        self.preparation_worker.finished.connect(prepare_table_layout)
        self.preparation_worker.start.emit()

    def layout_progress(self):
        widget = QWidget()
        layout = QGridLayout(widget)

        try:
            cover_image = self.api.cover(self.ebook.id, as_pixmap=True)
        except Exception:
            cover_image = QPixmap(self.api.cover(self.ebook.id, as_image=True))
        if cover_image.isNull():
            cover_image = QPixmap(I('default_cover.png'))
        cover_image = cover_image.scaledToHeight(
            480, Qt.TransformationMode.SmoothTransformation)

        cover = QLabel()
        cover.setAlignment(Qt.AlignCenter)
        cover.setPixmap(cover_image)

        title = QLabel()
        title.setMaximumWidth(cover_image.width())
        title.setText(title.fontMetrics().elidedText(
            self.ebook.title, Qt.ElideRight, title.width()))
        title.setToolTip(self.ebook.title)

        progress_bar = QProgressBar()
        progress_bar.setFormat('')
        progress_bar.setValue(0)
        # progress_bar.setFixedWidth(300)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)

        def show_progress(value):
            if progress_bar.maximum() == 0:
                progress_bar.setMaximum(100)
            progress_bar.setValue(value)
        self.preparation_worker.progress.connect(show_progress)

        label = QLabel(_('Loading ebook data, please wait...'))
        label.setAlignment(Qt.AlignCenter)
        self.preparation_worker.progress_message.connect(label.setText)

        detail = QPlainTextEdit()
        detail.setReadOnly(True)
        self.preparation_worker.progress_detail.connect(detail.appendPlainText)

        layout.addWidget(cover, 0, 0)
        layout.addWidget(title, 1, 0)
        layout.addItem(QSpacerItem(0, 20), 2, 0, 1, 3)
        layout.addWidget(progress_bar, 3, 0)
        layout.addWidget(label, 4, 0)
        layout.addItem(QSpacerItem(0, 0), 5, 0, 1, 3)
        layout.addItem(QSpacerItem(10, 0), 0, 1, 6, 1)
        layout.addWidget(detail, 0, 2, 6, 1)
        # layout.setRowStretch(0, 1)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(2, 1)
        # layout.setColumnStretch(2, 1)

        return widget

    def layout_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        review_index = tabs.addTab(self.layout_review(), _('Review'))
        log_index = tabs.addTab(self.layout_log(), _('Log'))
        errors_index = tabs.addTab(self.layout_errors(), _('Errors'))
        tabs.setStyleSheet('QTabBar::tab {min-width:120px;}')

        self.trans_worker.start.connect(
            lambda: (self.translate_all or self.table.selected_count() > 1)
            and tabs.setCurrentIndex(log_index))
        self.trans_worker.finished.connect(
            lambda: tabs.setCurrentIndex(
                errors_index if self.errors_text.toPlainText()
                and len(self.table.get_selected_paragraphs(True, True)) > 0
                else review_index))
        splitter = QSplitter()
        splitter.addWidget(self.layout_table())
        splitter.addWidget(tabs)
        splitter.setSizes([int(splitter.width() / 2)] * 2)

        layout.addWidget(self.layout_control())
        layout.addWidget(splitter, 1)

        return widget

    def layout_filter(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        categories = QComboBox()
        categories.addItem(_('All'), 'all')
        if self.merge_enabled:
            categories.addItem(_('Non-aligned'), 'non_aligned')
        categories.addItem(_('Translated'), 'translated')
        categories.addItem(_('Untranslated'), 'untranslated')

        content_types = QComboBox()
        content_types.addItem(_('Original Text'), 'original_text')
        content_types.addItem(_('Original Code'), 'original_code')
        content_types.addItem(_('Translation Text'), 'translation_text')

        search_input = QLineEdit()
        search_input.setPlaceholderText(_('keyword for filtering'))
        set_shortcut(
            search_input, 'search', search_input.setFocus,
            search_input.placeholderText())

        reset_button = QPushButton(_('Reset'))
        reset_button.setVisible(False)

        def filter_table_items(index):
            self.table.show_all_rows()
            category = categories.itemData(index)
            if category == 'non_aligned':
                self.table.hide_by_paragraphs(self.table.aligned_paragraphs())
            elif category == 'translated':
                self.table.hide_by_paragraphs(
                    self.table.untranslated_paragraphs())
            elif category == 'untranslated':
                self.table.hide_by_paragraphs(
                    self.table.translated_paragraphs())

        def filter_by_category(index):
            reset_button.setVisible(index != 0)
            filter_table_items(index)
            self.table.show_by_text(
                search_input.text(), content_types.currentData())
        categories.currentIndexChanged.connect(filter_by_category)

        def filter_by_content_type(index):
            reset_button.setVisible(index != 0)
            filter_table_items(categories.currentIndex())
            self.table.show_by_text(
                search_input.text(), content_types.itemData(index))
        content_types.currentIndexChanged.connect(filter_by_content_type)

        def filter_by_keyword(text):
            reset_button.setVisible(text != '')
            filter_table_items(categories.currentIndex())
            self.table.show_by_text(text, content_types.currentData())
        search_input.textChanged.connect(filter_by_keyword)

        def reset_filter_criteria():
            categories.setCurrentIndex(0)
            content_types.setCurrentIndex(0)
            search_input.clear()
            reset_button.setVisible(False)
        reset_button.clicked.connect(reset_filter_criteria)

        # def reset_filter():
        #     filter_table_items(categories.currentIndex())
        #     self.table.show_by_text(search_input.text())
        # self.editor_worker.finished.connect(reset_filter)
        # self.trans_worker.finished.connect(reset_filter)

        layout.addWidget(categories)
        layout.addWidget(content_types)
        layout.addWidget(search_input)
        layout.addWidget(reset_button)

        return widget

    def layout_table(self):
        widget = QWidget()
        widget.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        progress_bar = QProgressBar()
        progress_bar.setMaximum(100000000)
        progress_bar.setVisible(False)

        def write_progress():
            value = progress_bar.value() + self.progress_step
            if value > progress_bar.maximum():
                value = progress_bar.maximum()
            progress_bar.setValue(value)
        self.progress_bar.connect(write_progress)

        paragraph_count = QLabel()
        non_aligned_paragraph_count = QLabel()
        non_aligned_paragraph_count.setVisible(False)

        counter = QWidget()
        counter_layout = QHBoxLayout(counter)
        counter_layout.setContentsMargins(0, 0, 0, 0)
        counter_layout.setSpacing(0)
        counter_layout.addWidget(paragraph_count)
        counter_layout.addWidget(non_aligned_paragraph_count)
        counter_layout.addStretch(1)
        self.footer.layout().insertWidget(0, counter)

        def get_paragraph_count(select_all=True):
            item_count = char_count = 0
            paragraphs = self.table.get_selected_paragraphs(
                select_all=select_all)
            for paragraph in paragraphs:
                item_count += 1
                char_count += len(paragraph.original)
            return (item_count, char_count)
        all_item_count, all_char_count = get_paragraph_count(True)

        def item_selection_changed():
            item_count, char_count = get_paragraph_count(False)
            total = '%s/%s' % (item_count, all_item_count)
            parts = '%s/%s' % (char_count, all_char_count)
            paragraph_count.setText(
                _('Total items: {}').format(total) + ' · '
                + _('Character count: {}').format(parts))
        item_selection_changed()
        self.table.itemSelectionChanged.connect(item_selection_changed)

        if self.merge_enabled:
            non_aligned_paragraph_count.setVisible(True)

            def show_none_aligned_count():
                non_aligned_paragraph_count.setText(
                    ' · ' + _('Non-aligned items: {}')
                    .format(self.table.non_aligned_count))
            show_none_aligned_count()
            self.table.row.connect(show_none_aligned_count)

        filter_widget = self.layout_filter()

        layout.addWidget(filter_widget)
        layout.addWidget(self.table, 1)
        layout.addWidget(progress_bar)
        layout.addWidget(self.layout_table_control())

        def working_start():
            if self.translate_all or self.table.selected_count() > 1:
                filter_widget.setVisible(False)
                progress_bar.setValue(0)
                progress_bar.setVisible(True)
                counter.setVisible(False)
        self.trans_worker.start.connect(working_start)

        def working_end():
            filter_widget.setVisible(True)
            progress_bar.setVisible(False)
            counter.setVisible(True)
        self.trans_worker.finished.connect(working_end)

        return widget

    def layout_table_control(self):
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)

        delete_button = QPushButton(_('Delete'))
        delete_button.setToolTip(delete_button.text() + ' [Del]')
        batch_translation = QPushButton(
            ' %s (%s)' % (_('Batch Translation'), _('Beta')))
        # Manual brief-build button (also auto-triggered on first
        # translate when the brief is missing and the engine
        # supports it). Kept as a manual rebuild path for when the
        # user wants to regenerate the brief explicitly — e.g. after
        # discovering issues in the brief or after editing source
        # paragraphs.
        consistency_pass_button = QPushButton(
            '  %s  ' % _('Build Brief'))
        # Agreement Pass: post-translation revision pass that fixes
        # gender/number/pronoun drift in the translated text against
        # the brief. Visible only for Claude; disabled until both a
        # brief exists and at least one translation is present.
        agreement_pass_button = QPushButton(
            '  %s  ' % _('Run Agreement Pass'))
        translate_all = QPushButton('  %s  ' % _('Translate All'))
        translate_untranslated = QPushButton(
            '  %s  ' % _('Translate Untranslated'))
        translate_selected = QPushButton('  %s  ' % _('Translate Selected'))

        delete_button.clicked.connect(self.table.delete_selected_rows)
        consistency_pass_button.clicked.connect(
            self.run_consistency_pass)
        agreement_pass_button.clicked.connect(
            self.run_agreement_pass)
        translate_all.clicked.connect(self.translate_all_paragraphs)
        translate_untranslated.clicked.connect(
            self.translate_untranslated_paragraphs)
        translate_selected.clicked.connect(self.translate_selected_paragraph)

        # Track the button so update_consistency_pass_button() can read its
        # state.
        self.consistency_pass_button = consistency_pass_button
        self.agreement_pass_button = agreement_pass_button

        action_layout.addWidget(delete_button)
        action_layout.addStretch(1)
        action_layout.addWidget(batch_translation)
        action_layout.addWidget(consistency_pass_button)
        action_layout.addWidget(agreement_pass_button)
        action_layout.addWidget(translate_all)
        action_layout.addWidget(translate_untranslated)
        action_layout.addWidget(translate_selected)

        stop_widget = QWidget()
        stop_layout = QHBoxLayout(stop_widget)
        stop_layout.setContentsMargins(0, 0, 0, 0)
        # stop_layout.addStretch(1)
        stop_button = QPushButton(_('Stop'))
        stop_layout.addWidget(stop_button)

        delete_button.setDisabled(True)
        translate_selected.setDisabled(True)

        # Consistency pass: the button is a manual trigger and is shown
        # for Claude regardless of the "auto-run after translation"
        # setting. The setting only controls whether the pass runs
        # automatically at the end of translate_paragraphs; users can
        # always invoke it manually via this button.
        consistency_pass_button.setVisible(False)
        agreement_pass_button.setVisible(False)

        def _last_consistency_pass_label():
            """Return a human-readable 'Last run: ...' string from cache,
            or 'never' if no record exists."""
            ts = None
            if self.cache is not None:
                ts = self.cache.get_info('last_consistency_pass')
            if not ts:
                return _('never')
            return ts

        def update_consistency_pass_button():
            from .engines.anthropic import ClaudeTranslate
            engine_class = self.current_engine
            visible = (
                engine_class is not None
                and issubclass(engine_class, ClaudeTranslate)
            )
            consistency_pass_button.setVisible(visible)
            if not visible:
                return
            # Enabled whenever source text exists — brief is built
            # from source, translation isn't required.
            total_rows = self.table.rowCount()
            has_source = total_rows > 0
            consistency_pass_button.setEnabled(has_source)
            if has_source:
                consistency_pass_button.setToolTip(_(
                    'Build a Translation Brief — a structured '
                    'reference document with canonical names, '
                    'character profiles (with grammatical gender '
                    'for verb/adjective agreement), recurring '
                    'terminology, and style decisions. Generated '
                    'once from the source text; cached and reused '
                    'on every subsequent translation. Translation '
                    'auto-builds this on first run if missing; '
                    'click here to rebuild manually.\n\nLast run: {}'
                ).format(_last_consistency_pass_label()))
            else:
                consistency_pass_button.setToolTip(_(
                    'Disabled until source paragraphs are loaded.'))

        # Save references the callback chain can reach.
        self.update_consistency_pass_button = update_consistency_pass_button

        def _last_agreement_pass_label():
            ts = None
            if self.cache is not None:
                ts = self.cache.get_info('last_agreement_pass')
            if not ts:
                return _('never')
            return ts

        def _has_brief_in_cache():
            if self.cache is None:
                return False
            try:
                return bool(self.cache.get_info('translation_brief'))
            except Exception:
                return False

        def _has_any_translation():
            if self.cache is None:
                return False
            try:
                paras = self.cache.all_paragraphs() or []
            except Exception:
                return False
            for p in paras:
                if (getattr(p, 'translation', None) or '').strip():
                    return True
            return False

        def update_agreement_pass_button():
            from .engines.anthropic import ClaudeTranslate
            engine_class = self.current_engine
            opted_in = bool(
                engine_class is not None
                and engine_class.config.get(
                    'enable_agreement_pass',
                    getattr(engine_class,
                            'enable_agreement_pass', False)))
            visible = (
                engine_class is not None
                and issubclass(engine_class, ClaudeTranslate)
                and getattr(engine_class,
                            'supports_agreement_review', False)
                and opted_in
            )
            agreement_pass_button.setVisible(visible)
            if not visible:
                return
            has_brief = _has_brief_in_cache()
            has_translation = _has_any_translation()
            agreement_pass_button.setEnabled(
                has_brief and has_translation)
            if not has_brief:
                agreement_pass_button.setToolTip(_(
                    'Build a Translation Brief first — the '
                    'Agreement Pass uses canonical character '
                    'morphology from the brief to detect drift.'))
            elif not has_translation:
                agreement_pass_button.setToolTip(_(
                    'Translate at least one paragraph first — the '
                    'Agreement Pass reviews translated text.'))
            else:
                agreement_pass_button.setToolTip(_(
                    'Scan translated paragraphs for residual '
                    'gender / number / pronoun agreement drift '
                    'against the canonical character morphology '
                    'in the brief, and apply single-occurrence '
                    'fixes. Sees only translated text — no '
                    'copyrighted source material is sent.\n\n'
                    'Last run: {}'
                ).format(_last_agreement_pass_label()))

        self.update_agreement_pass_button = update_agreement_pass_button

        def record_consistency_pass_time():
            """Persist the run timestamp in cache info so it survives
            plugin reloads. Phase 1a spike: also persist the brief
            itself so subsequent sessions can use it for drafting."""
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d %H:%M')
            if self.cache is not None:
                self.cache.set_info('last_consistency_pass', ts)
                # Persist the brief so future translate runs (even
                # after plugin reload) pick it up automatically.
                if getattr(self.trans_worker, 'brief', None) is not None:
                    try:
                        import json as _json_persist
                        self.cache.set_info(
                            'translation_brief',
                            _json_persist.dumps(
                                self.trans_worker.brief,
                                ensure_ascii=False))
                    except Exception:
                        pass
            update_consistency_pass_button()
            # A fresh brief makes Agreement Pass eligible if a
            # translation already exists.
            update_agreement_pass_button()
        self.trans_worker.consistency_completed.connect(
            record_consistency_pass_time)

        def record_agreement_pass_time(applied):
            from datetime import datetime
            ts = datetime.now().strftime('%Y-%m-%d %H:%M')
            if self.cache is not None:
                try:
                    self.cache.set_info('last_agreement_pass', ts)
                except Exception:
                    pass
            update_agreement_pass_button()
        self.trans_worker.agreement_completed.connect(
            record_agreement_pass_time)

        self.batch_translation.connect(
            lambda: batch_translation.setVisible(
                self.current_engine == ChatgptTranslate))
        self.batch_translation.emit()
        update_consistency_pass_button()
        update_agreement_pass_button()

        def start_batch_translation():
            translator = get_translator(self.current_engine)
            translator.set_source_lang(self.ebook.source_lang)
            translator.set_target_lang(self.ebook.target_lang)
            batch_translator = ChatgptBatchTranslate(translator)
            batch = ChatgptBatchTranslationManager(
                batch_translator, self.cache, self.table, self)
            batch.exec_()
        batch_translation.clicked.connect(start_batch_translation)

        def item_selection_changed():
            disabled = self.table.selected_count() < 1
            delete_button.setDisabled(disabled)
            translate_selected.setDisabled(disabled)
        item_selection_changed()
        self.table.itemSelectionChanged.connect(item_selection_changed)

        def stop_translation():
            action = self.alert.ask(
                _('Are you sure you want to stop the translation progress?'))
            if action != 'yes':
                return
            stop_button.setDisabled(True)
            stop_button.setText(_('Stopping...'))
            self.trans_worker.set_canceled(True)
        stop_button.clicked.connect(stop_translation)

        def terminate_finished():
            stop_button.setDisabled(False)
            stop_button.setText(_('Stop'))
            self.paragraph_sig.emit(self.table.current_paragraph())
        self.trans_worker.finished.connect(terminate_finished)

        stack = QStackedWidget()
        stack.addWidget(action_widget)
        stack.addWidget(stop_widget)

        def working_start():
            stack.setCurrentWidget(stop_widget)
            action_widget.setDisabled(True)
        self.trans_worker.start.connect(working_start)

        def working_finished():
            stack.setCurrentWidget(action_widget)
            action_widget.setDisabled(False)
            # The action_widget re-enable also re-enables the consistency
            # pass button unconditionally; reapply the "all translated"
            # gate so it only stays enabled when appropriate.
            if hasattr(self, 'update_consistency_pass_button'):
                self.update_consistency_pass_button()
            if hasattr(self, 'update_agreement_pass_button'):
                self.update_agreement_pass_button()
        self.trans_worker.finished.connect(working_finished)

        return stack

    def layout_control(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        cache_group = QGroupBox(_('Cache Status'))
        cache_layout = QVBoxLayout(cache_group)
        cache_status = QLabel(
            _('Enabled') if self.cache and self.cache.is_persistence()
            else _('Disabled'))
        cache_status.setAlignment(Qt.AlignCenter)
        cache_status.setStyleSheet(
            'border-radius:2px;color:white;background-color:%s;'
            % ('green' if self.cache and self.cache.is_persistence()
               else 'grey'))
        cache_layout.addWidget(cache_status)

        engine_group = QGroupBox(_('Translation Engine'))
        engine_layout = QVBoxLayout(engine_group)
        engine_list = EngineList(self.current_engine.name)
        engine_list.setMaximumWidth(150)
        engine_layout.addWidget(engine_list)

        source_group = QGroupBox(_('Source Language'))
        source_layout = QVBoxLayout(source_group)
        source_lang = SourceLang()
        source_lang.setMaximumWidth(150)
        source_layout.addWidget(source_lang)

        target_group = QGroupBox(_('Target Language'))
        target_layout = QVBoxLayout(target_group)
        target_lang = TargetLang()
        target_lang.setMaximumWidth(150)
        target_layout.addWidget(target_lang)

        title_group = QGroupBox(_('Custom Ebook Title'))
        title_layout = QHBoxLayout(title_group)
        custom_title = QCheckBox()
        ebook_title = QLineEdit()
        ebook_title.setToolTip(
            _('By default, title metadata will be translated.'))
        ebook_title.setText(self.ebook.title)
        ebook_title.setCursorPosition(0)
        ebook_title.setDisabled(True)
        title_layout.addWidget(custom_title)
        title_layout.addWidget(ebook_title)

        def enable_custom_title(checked):
            ebook_title.setDisabled(not checked)
            self.ebook.set_custom_title(
                ebook_title.text() if checked else None)
            if checked:
                ebook_title.setFocus(Qt.MouseFocusReason)
        custom_title.stateChanged.connect(enable_custom_title)

        def change_ebook_title():
            if ebook_title.text() == '':
                ebook_title.undo()
            self.ebook.set_custom_title(ebook_title.text())
        ebook_title.editingFinished.connect(change_ebook_title)

        # if self.config.get('to_library'):
        #     ebook_title.setDisabled(True)
        #     ebook_title.setToolTip(_(
        #         "The ebook's filename is automatically managed by Calibre "
        #         'according to metadata since the output path is set to '
        #         'Calibre Library.'))
        # ebook_title.textChanged.connect(self.ebook.set_custom_title)

        output_group = QGroupBox(_('Output Ebook'))
        output_layout = QHBoxLayout(output_group)
        output_button = QPushButton(_('Output'))
        output_format = OutputFormat()
        output_layout.addWidget(output_format)
        output_layout.addWidget(output_button)

        layout.addWidget(cache_group)
        layout.addWidget(engine_group)
        layout.addWidget(source_group)
        layout.addWidget(target_group)
        layout.addWidget(title_group, 1)
        layout.addWidget(output_group)

        source_lang.currentTextChanged.connect(
            self.trans_worker.set_source_lang)
        target_lang.currentTextChanged.connect(
            self.trans_worker.set_target_lang)

        def refresh_languages():
            source_lang.refresh.emit(
                self.current_engine.lang_codes.get('source'),
                self.ebook.source_lang,
                not isinstance(self.current_engine, CustomTranslate))
            target_lang.refresh.emit(
                self.current_engine.lang_codes.get('target'),
                self.ebook.target_lang)
        refresh_languages()
        self.ebook.set_source_lang(source_lang.currentText())

        def choose_engine(index):
            engine_name = engine_list.itemData(index)
            self.current_engine = get_engine_class(engine_name)
            self.trans_worker.set_engine_class(self.current_engine)
            self.batch_translation.emit()
            refresh_languages()
        engine_list.currentIndexChanged.connect(choose_engine)

        output_format.setCurrentText(self.ebook.output_format)

        def change_output_format(_format):
            self.ebook.set_output_format(_format)
        if self.ebook.is_extra_format():
            output_format.lock_format(self.ebook.input_format)
            change_output_format(self.ebook.input_format)
        else:
            change_output_format(output_format.currentText())
            output_format.currentTextChanged.connect(change_output_format)

        def output_ebook():
            if len(self.table.findItems(_('Translated'), Qt.MatchExactly)) < 1:
                self.alert.pop(_('The ebook has not been translated yet.'))
                return
            if self.table.non_aligned_count > 0:
                message = _(
                    'The number of lines in some translation units differs '
                    'between the original text and the translated text. Are '
                    'you sure you want to output without checking alignment?')
                if self.alert.ask(message) != 'yes':
                    return
            self.worker.translate_ebook(self.ebook, cache_only=True)
            self.done(1)
        output_button.clicked.connect(output_ebook)

        def working_start():
            if self.translate_all:
                widget.setVisible(False)
            widget.setDisabled(True)
        self.trans_worker.start.connect(working_start)

        def working_finished():
            widget.setVisible(True)
            widget.setDisabled(False)
        self.trans_worker.finished.connect(working_finished)

        return widget

    def layout_review(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        orientation = self.ui_settings.value(
            'review/layout_orientation', 'vertical')
        self.review_splitter = QSplitter(
            Qt.Horizontal if orientation == 'horizontal' else Qt.Vertical)
        self.review_splitter.setContentsMargins(0, 0, 0, 0)
        raw_text = CodeEditor()
        raw_text.setReadOnly(True)
        original_text = CodeEditor()
        original_text.setReadOnly(True)
        translation_text = CodeEditor()
        if self.ebook.target_direction == 'rtl':
            translation_text.setLayoutDirection(Qt.RightToLeft)
            document = translation_text.document()
            option = document.defaultTextOption()
            option.setAlignment(Qt.AlignRight)
            document.setDefaultTextOption(option)
        translation_text.setPlaceholderText(_('No translation yet'))
        self.review_splitter.addWidget(raw_text)
        self.review_splitter.addWidget(original_text)
        self.review_splitter.addWidget(translation_text)
        _size = [0] + [int(self.review_splitter.width() / 2)] * 2
        if self.review_splitter.orientation() == Qt.Vertical:
            _size = [0] + [int(self.review_splitter.height() / 2)] * 2
        self.review_splitter.setSizes(_size)

        def synchronizeScrollbars(editors):
            """Sync scrollbars between editors using simple pixel-based approach."""
            for editor in editors:
                for other_editor in editors:
                    if editor != other_editor:
                        editor.verticalScrollBar().valueChanged.connect(
                            other_editor.verticalScrollBar().setValue)
        synchronizeScrollbars((raw_text, original_text, translation_text))

        translation_text.cursorPositionChanged.connect(
            translation_text.ensureCursorVisible)

        def refresh_translation(paragraph):
            # TODO: check - why/how can "paragraph" be None and what should we do in such case?
            if paragraph is not None:
                raw_text.setPlainText(paragraph.raw.strip())
                original_text.setPlainText(paragraph.original.strip())
                translation_text.setPlainText(paragraph.translation)

        self.paragraph_sig.connect(refresh_translation)

        self.trans_worker.start.connect(
            lambda: translation_text.setReadOnly(True))
        self.trans_worker.finished.connect(
            lambda: translation_text.setReadOnly(False))

        # default_flag = translation_text.textInteractionFlags()

        # def disable_translation_text():
        #     if self.trans_worker.on_working:
        #         translation_text.setTextInteractionFlags(Qt.TextEditable)
        #         end = getattr(QTextCursor.MoveOperation, 'End', None) \
        #             or QTextCursor.End
        #         translation_text.moveCursor(end)
        #     else:
        #         translation_text.setTextInteractionFlags(default_flag)
        # translation_text.cursorPositionChanged.connect(
        #     disable_translation_text)

        def auto_open_close_splitter():
            if self.review_splitter.sizes()[0] > 0:
                sizes = [0] + [int(self.review_splitter.height() / 2)] * 2
            else:
                sizes = [int(self.review_splitter.height() / 3)] * 3
            self.review_splitter.setSizes(sizes)

        self.install_widget_event(
            self.review_splitter,
            self.review_splitter.handle(1),
            QEvent.MouseButtonDblClick,
            auto_open_close_splitter)

        self.table.itemDoubleClicked.connect(
            lambda item: auto_open_close_splitter())

        control = QWidget()
        control_layout = QHBoxLayout(control)
        control_layout.setContentsMargins(0, 0, 0, 0)

        self.trans_worker.start.connect(
            lambda: control.setVisible(False))
        self.trans_worker.finished.connect(
            lambda: control.setVisible(True))

        save_status = QLabel()
        save_button = QPushButton(_('&Save'))
        save_button.setDisabled(True)

        # Word Wrap toggle button
        word_wrap_button = QPushButton(_("Word Wrap"))
        word_wrap_button.setCheckable(True)

        word_wrap_enabled = self.ui_settings.value(
            'review/word_wrap', True, type=bool)  # Default to enabled
        word_wrap_button.setChecked(word_wrap_enabled)

        # Set initial word wrap state for all editors
        # Try different approaches for different Qt versions
        try:
            # Method 1: Modern Qt with LineWrapMode enum
            wrap_enabled = QPlainTextEdit.LineWrapMode.WidgetWidth
            wrap_disabled = QPlainTextEdit.LineWrapMode.NoWrap
        except AttributeError:
            try:
                # Method 2: Older Qt versions
                wrap_enabled = QPlainTextEdit.WidgetWidth
                wrap_disabled = QPlainTextEdit.NoWrap
            except AttributeError:
                # Method 3: Direct integer values as fallback
                wrap_enabled = 1  # WidgetWidth
                wrap_disabled = 0  # NoWrap

        raw_text.setLineWrapMode(wrap_enabled)
        original_text.setLineWrapMode(wrap_enabled)
        translation_text.setLineWrapMode(wrap_enabled)

        def toggle_word_wrap(checked):
            self.ui_settings.setValue('review/word_wrap', checked)
            wrap_mode = wrap_enabled if checked else wrap_disabled
            raw_text.setLineWrapMode(wrap_mode)
            original_text.setLineWrapMode(wrap_mode)
            translation_text.setLineWrapMode(wrap_mode)

        word_wrap_button.clicked.connect(toggle_word_wrap)

        layout_button = QPushButton(_("Horizontal Split"))
        layout_button.setCheckable(True)
        is_horizontal = self.ui_settings.value(
            'review/layout_orientation', 'vertical') == 'horizontal'
        layout_button.setChecked(is_horizontal)
        layout_button.toggled.connect(self.toggle_review_layout)

        status_indicator = TranslationStatus()

        control_layout.addWidget(status_indicator)
        control_layout.addWidget(word_wrap_button)
        control_layout.addWidget(layout_button)
        control_layout.addStretch(1)
        control_layout.addWidget(save_status)
        control_layout.addWidget(save_button)

        layout.addWidget(self.review_splitter, 1)
        layout.addWidget(control)

        def update_translation_status(row):
            paragraph = self.table.paragraph(row)
            if paragraph is None:
                return
            if not paragraph.translation:
                if paragraph.error is not None:
                    status_indicator.set_color(
                        StatusColor('red'), paragraph.error)
                else:
                    status_indicator.set_color(StatusColor('gray'))
            elif not paragraph.aligned and self.merge_enabled:
                status_indicator.set_color(
                    StatusColor('yellow'), )
            else:
                status_indicator.set_color(StatusColor('green'))
        self.table.row.connect(update_translation_status)

        def change_selected_item():
            if self.trans_worker.on_working:
                return
            paragraph = self.table.current_paragraph()
            if paragraph is None:
                return
            self.paragraph_sig.emit(paragraph)
            self.table.row.emit(paragraph.row)
        self.table.setCurrentItem(self.table.item(0, 0))
        change_selected_item()
        self.table.itemSelectionChanged.connect(change_selected_item)

        def translation_callback(paragraph):
            self.table.row.emit(paragraph.row)
            self.paragraph_sig.emit(paragraph)
            if self.cache is not None:
                self.cache.update_paragraph(paragraph)
            self.progress_bar.emit()
            # Refresh consistency-pass button state — translation status
            # of one paragraph just changed, which may flip the
            # "all translated" predicate.
            if hasattr(self, 'update_consistency_pass_button'):
                self.update_consistency_pass_button()
            if hasattr(self, 'update_agreement_pass_button'):
                self.update_agreement_pass_button()

        self.trans_worker.callback.connect(translation_callback)

        def streaming_translation(data):
            if data == '':
                translation_text.clear()
            elif isinstance(data, Paragraph):
                self.table.setCurrentItem(self.table.item(data.row, 0))
            else:
                # Check if user is at bottom (watching stream)
                scrollbar = translation_text.verticalScrollBar()
                was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10

                # Disable updates to prevent auto-scroll/flicker
                translation_text.setUpdatesEnabled(False)
                saved_position = scrollbar.value()

                # Append to document using cursor at end
                doc = translation_text.document()
                cursor = QTextCursor(doc)
                end_position = getattr(QTextCursor.MoveOperation, 'End', None) or QTextCursor.End
                cursor.movePosition(end_position)
                cursor.insertText(data)

                # Restore scroll position before re-enabling updates
                if not was_at_bottom:
                    scrollbar.setValue(saved_position)

                # Re-enable updates - this triggers repaint
                translation_text.setUpdatesEnabled(True)

                # Scroll to bottom only if user was watching
                if was_at_bottom:
                    scrollbar.setValue(scrollbar.maximum())
        self.trans_worker.streaming.connect(streaming_translation)

        def modify_translation():
            if self.trans_worker.on_working and \
                    self.table.selected_count() > 1:
                return

            paragraph = self.table.current_paragraph()

            # TODO: check - why/how can "paragraph" be None and what should we
            # do in such case?
            if paragraph is not None:
                translation = translation_text.toPlainText()
                save_button.setDisabled(
                    translation == (paragraph.translation or ''))

        translation_text.textChanged.connect(modify_translation)

        self.editor_worker.show.connect(save_status.setText)

        def save_translation():
            paragraph = self.table.current_paragraph()

            # TODO: check - why/how can "paragraph" be None and what should we
            # do in such case?
            if paragraph is not None:
                save_button.setDisabled(True)
                translation = translation_text.toPlainText()
                paragraph.translation = translation
                paragraph.engine_name = self.current_engine.name
                paragraph.target_lang = self.ebook.target_lang
                self.table.row.emit(paragraph.row)
                if self.cache is not None:
                    self.cache.update_paragraph(paragraph)
                translation_text.setFocus(Qt.OtherFocusReason)
                self.editor_worker.start[str].emit(
                    _('Your changes have been saved.'))

        save_button.clicked.connect(save_translation)
        set_shortcut(save_button, 'save', save_translation, save_button.text())

        return widget

    def toggle_review_layout(self, checked):
        if checked:
            self.review_splitter.setOrientation(Qt.Horizontal)
            self.ui_settings.setValue('review/layout_orientation', 'horizontal')
        else:
            self.review_splitter.setOrientation(Qt.Vertical)
            self.ui_settings.setValue('review/layout_orientation', 'vertical')

    def layout_log(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.logging_text = QPlainTextEdit()
        self.logging_text.setPlaceholderText(_('Translation log'))
        self.logging_text.setReadOnly(True)
        layout.addWidget(self.logging_text)

        return widget

    def layout_errors(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.errors_text = QPlainTextEdit()
        self.errors_text.setPlaceholderText(_('Error log'))
        self.errors_text.setReadOnly(True)
        layout.addWidget(self.errors_text)

        return widget

    def get_progress_step(self, total):
        return int(round(100.0 / (total or 1), 100) * 1000000)

    def check_max_tokens_capacity(self):
        """Check if merge length might exceed model's max output tokens."""
        if not issubclass(self.current_engine, ClaudeTranslate):
            return True  # Only check for Claude models

        config = get_config()
        merge_length = config.get('merge_length', 1800)
        merge_enabled = config.get('merge_enabled', False)

        if not merge_enabled:
            return True  # No merge, chunks will be small

        # Get model and settings from engine
        engine_config = self.current_engine.config
        model = engine_config.get('model', self.current_engine.model)
        enable_extended_output = engine_config.get('enable_extended_output', False)

        # Estimate output tokens (conservative: 3 chars per token, +10% buffer)
        estimated_output_tokens = int((merge_length / 3) * 1.1)

        # Determine model's max output (same logic as in anthropic.py get_body)
        if not model:
            model_max_output = 4_096
        elif model.startswith('claude-3-7-sonnet-') and enable_extended_output:
            model_max_output = 128_000
        elif model.startswith('claude-3-7-sonnet-'):
            model_max_output = 64_000
        elif model.startswith('claude-sonnet-4-') or model.startswith('claude-haiku-4-') or \
             model.startswith('claude-opus-4-5') or model.startswith('claude-opus-4-1'):
            model_max_output = 64_000
        elif model.startswith('claude-opus-4-0'):
            model_max_output = 32_000
        elif model.startswith('claude-3-haiku-'):
            model_max_output = 4_000
        else:
            model_max_output = 32_000

        # Check if estimated output exceeds model capacity
        if estimated_output_tokens > model_max_output:
            message = _(
                'Warning: Merge length ({:,} chars ≈ {:,} tokens) may exceed model max output ({:,} tokens).\n\n'
                'This could result in incomplete translations. Consider:\n'
                '• Reducing merge length\n'
                '• Enabling "Extended Output" (Claude 3.7 Sonnet: 128K tokens)\n\n'
                'Continue anyway?'
            ).format(merge_length, estimated_output_tokens, model_max_output)
            return self.alert.ask(message) == 'yes'

        return True

    def translate_all_paragraphs(self):
        """Translate the untranslated paragraphs when at least one is selected.
        Otherwise, retranslate all paragraphs regardless of prior translation.
        """
        paragraphs = self.table.get_selected_paragraphs(True, True)
        is_fresh = len(paragraphs) < 1
        if is_fresh:
            paragraphs = self.table.get_selected_paragraphs(False, True)
        self.progress_step = self.get_progress_step(len(paragraphs))

        # Check max_tokens capacity before starting
        if not self.check_max_tokens_capacity():
            return

        if not self.translate_all:
            message = _(
                'Are you sure you want to translate all {:n} paragraphs?')
            if self.alert.ask(message.format(len(paragraphs))) != 'yes':
                return
        self.translate_all = True
        self.trans_worker.translate.emit(paragraphs, is_fresh)

    def translate_untranslated_paragraphs(self):
        """Translate only paragraphs that don't have a translation yet."""
        paragraphs = self.table.get_selected_paragraphs(True, True)
        if len(paragraphs) < 1:
            self.alert.pop(
                _('All paragraphs have already been translated.'), 'info')
            return
        self.progress_step = self.get_progress_step(len(paragraphs))
        if not self.check_max_tokens_capacity():
            return
        self.translate_all = True
        self.trans_worker.translate.emit(paragraphs, False)

    def run_consistency_pass(self):
        """Phase 0 spike: build a Translation Brief from source text
        and dump it to the log. Translations are not required (and
        not consulted) — the brief is a pre-translation reference
        document built from the source.
        """
        paragraphs = self.table.get_selected_paragraphs(False, True)
        if len(paragraphs) < 1:
            return
        self.progress_step = self.get_progress_step(len(paragraphs))
        self.trans_worker.consistency.emit(paragraphs)

    def run_agreement_pass(self):
        """Run an Agreement Pass over every paragraph in the book
        (selection is ignored — agreement is a global revision pass
        because canonical character morphology spans the whole work).
        Worker filters out paragraphs that lack a translation.
        """
        if self.cache is None:
            return
        paragraphs = self.cache.all_paragraphs() or []
        if len(paragraphs) < 1:
            return
        self.progress_step = self.get_progress_step(len(paragraphs))
        self.trans_worker.agreement.emit(paragraphs)

    def translate_selected_paragraph(self):
        paragraphs = self.table.get_selected_paragraphs()
        # Consider selecting all paragraphs as translating all.
        if len(paragraphs) == self.table.rowCount():
            self.translate_all_paragraphs()
        else:
            # Check max_tokens capacity before starting
            if not self.check_max_tokens_capacity():
                return
            self.progress_step = self.get_progress_step(len(paragraphs))
            self.trans_worker.translate.emit(paragraphs, True)

    def install_widget_event(
            self, source, target, action, callback, stop=False):
        def eventFilter(self, object, event):
            if event.type() == action:
                callback()
            return stop
        source.eventFilter = MethodType(eventFilter, source)
        target.installEventFilter(source)

    def terminate_preparework(self):
        if self.preparation_worker.on_working:
            if self.preparation_worker.canceled:
                return False
            action = self.alert.ask(
                _('Are you sure you want to cancel the preparation progress?'))
            if action != 'yes':
                return False
            self.preparation_worker.set_canceled(True)
            self.preparation_worker.progress_message.emit('Canceling...')
            return False
        return True

    def terminate_translation(self):
        if self.trans_worker.on_working:
            action = self.alert.ask(
                _('Are you sure you want to cancel the translation progress?'))
            if action != 'yes':
                return False
            self.trans_worker.set_need_close(True)
            self.trans_worker.set_canceled(True)
            return False
        return True

    def done(self, result):
        if not self.terminate_preparework():
            return
        if not self.terminate_translation():
            return

        # Close cache first to prevent corruption if cleanup crashes.
        if self.cache is not None:
            if self.cache.is_persistence():
                self.cache.close()
            elif result == 0:
                self.cache.destroy()

        # Disconnect all worker signals before quitting threads to prevent
        # pending cross-thread queued events from being delivered to stale
        # Python slots after the workers' C++ objects are destroyed.
        for worker in (self.trans_worker, self.preparation_worker,
                       self.editor_worker):
            try:
                worker.disconnect()
            except TypeError:
                pass

        # self.preparation_thread.requestInterruption()
        self.preparation_thread.quit()
        self.preparation_thread.wait()
        self.trans_thread.quit()
        self.trans_thread.wait()
        self.editor_thread.quit()
        self.editor_thread.wait()

        QDialog.done(self, result)
