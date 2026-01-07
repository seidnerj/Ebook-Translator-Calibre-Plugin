import os
import os.path
from types import MethodType
from typing import Callable
from tempfile import gettempdir

from calibre import sanitize_file_name  # type: ignore
from calibre.gui2 import Dispatcher  # type: ignore
from calibre.constants import DEBUG, __version__  # type: ignore
from calibre.utils.localization import _  # type: ignore
from calibre.utils.logging import Stream  # type: ignore
from calibre.ebooks.conversion.plumber import (  # type: ignore
    Plumber, CompositeProgressReporter)
from calibre.ptempfile import PersistentTemporaryFile  # type: ignore
from calibre.ebooks.metadata.meta import (  # type: ignore
    get_metadata, set_metadata)

from .. import EbookTranslator

from .config import get_config
from .utils import log, sep, open_path, open_file
from .cache import get_cache
from .element import (
    get_element_handler, get_srt_elements, get_toc_elements, get_page_elements,
    get_metadata_elements, get_pgn_elements)
from .translation import get_translator, get_translation
from .exception import ConversionAbort


load_translations()  # type: ignore


class PrepareStream:
    mode = 'r'

    def __init__(self, callback):
        self.callback = callback
        self.temp = ''

    def write(self, text):
        self.temp += text
        if text == '\n':
            self.callback(self.temp.strip('\n'))
            self.temp = ''

    def flush(self):
        pass


def convert_book(
        input_path, output_path, translation, element_handler, cache,
        debug_info, encoding, notification) -> None:
    """Process ebooks that Calibre supported."""
    plumber = Plumber(
        input_path, output_path, log=log, report_progress=notification)
    _convert = plumber.output_plugin.convert
    elements = []

    def convert(self, oeb, output_path, input_plugin, opts, log):
        backup_progress = self.report_progress.global_min
        self.report_progress = CompositeProgressReporter(0, 1, notification)
        log.info('Translating ebook content... (this will take a while)')
        log.info(debug_info)

        # Log all plugin settings if log_translation is enabled
        config = get_config()
        if config.get('log_translation', True):
            log.info(sep())
            log.info('Plugin Settings:')
            log.info(sep('┈'))
            log.info('  Translation Position: %s' % config.get('translation_position'))
            log.info('  Merge Enabled: %s' % config.get('merge_enabled'))
            if config.get('merge_enabled'):
                log.info('  Merge Length: %s' % config.get('merge_length'))
            log.info('  Cache Enabled: %s' % config.get('cache_enabled'))
            log.info('  Translate Missing Metadata: %s' % config.get('translate_missing_metadata', False))
            log.info('  Log Translation: %s' % config.get('log_translation', True))
            log.info('  Log Content: %s' % config.get('log_content', True))
            log.info('  Original Color: %s' % config.get('original_color'))
            log.info('  Translation Color: %s' % config.get('translation_color'))
            log.info('  Filter Scope: %s' % config.get('filter_scope'))
            log.info('  Rule Mode: %s' % config.get('rule_mode'))
            log.info('  Glossary Enabled: %s' % config.get('glossary_enabled'))

            ebook_metadata = config.get('ebook_metadata', {})
            if ebook_metadata:
                log.info('  Metadata Translation Settings:')
                for key, value in ebook_metadata.items():
                    log.info('    %s: %s' % (key, value))
            log.info(sep())

        translation.set_progress(self.report_progress)

        config = get_config()
        ebook_metadata_config = config.get('ebook_metadata') or {}

        # Handle metadata separately from content (not merged, individual cache entries)
        # Always extract and store metadata, even if translation is disabled (for UI display)
        from .utils import uid
        metadata_elements = get_metadata_elements(oeb.metadata)
        metadata_element_map = {}  # Map original text to element for applying translations

        metadata_originals = []
        metadata_field_names = {}  # Map item_id to field name for logging
        # Store metadata originals directly in cache (bypass element_handler)
        # Only store ENABLED metadata fields (ignored ones don't appear in UI)
        for eid, metadata_elem in enumerate(metadata_elements):
            if metadata_elem.ignored:
                continue  # Skip disabled metadata fields

            original_value = metadata_elem.get_content()
            # Use negative IDs to avoid conflicts with content
            item_id = -(eid + 1)
            md5 = uid('metadata_%s%s' % (eid, original_value))

            # Determine field name from OEB metadata
            field_name = 'unknown'
            for key in oeb.metadata.iterkeys():
                items = getattr(oeb.metadata, key)
                for item in items:
                    if item == metadata_elem.element:
                        field_name = key
                        break
                if field_name != 'unknown':
                    break

            # Store as 7-tuple with page='content.opf'
            # Only enabled metadata appears in UI
            metadata_originals.append((item_id, md5, original_value, original_value, False, None, 'content.opf'))
            metadata_field_names[item_id] = field_name
            metadata_element_map[original_value] = metadata_elem

        # Handle TOC separately - merge all TOC entries into one paragraph
        toc_elements = get_toc_elements(oeb.toc.nodes, [])
        toc_originals = []
        toc_element_list = []  # Keep references for applying translations

        if toc_elements:
            # Merge all TOC entries with separator
            toc_texts = []
            for toc_elem in toc_elements:
                if not toc_elem.ignored:
                    toc_texts.append(toc_elem.get_content())
                    toc_element_list.append(toc_elem)

            if toc_texts:
                merged_toc = '\n\n'.join(toc_texts)
                toc_id = 'toc_merged'
                md5 = uid('toc_%s' % merged_toc)
                # Store merged TOC as single entry with page='toc.ncx'
                toc_originals.append((toc_id, md5, merged_toc, merged_toc, False, None, 'toc.ncx'))

        # Extract content/page elements only (no TOC, no metadata)
        elements.extend(list(get_page_elements(oeb.manifest.items)))
        content_originals = element_handler.prepare_original(elements)

        # Synchronize cache with current extraction
        if config.get('log_translation', True):
            log.info('[CACHE] Preparing to save: %d metadata, %d TOC, %d content' %
                    (len(metadata_originals), len(toc_originals), len(content_originals)))

        if cache.is_fresh():
            # Fresh cache - just save everything
            if config.get('log_translation', True):
                log.info('[CACHE] Fresh cache - saving all entries')

            cache.save(content_originals)

            if config.get('log_translation', True):
                log.info('[CACHE] Content saved, adding %d metadata entries' % len(metadata_originals))

            for metadata_item in metadata_originals:
                cache.add(*metadata_item)

            if config.get('log_translation', True):
                log.info('[CACHE] Metadata added, adding %d TOC entries' % len(toc_originals))

            for toc_item in toc_originals:
                cache.add(*toc_item)

            cache.connection.commit()

            if config.get('log_translation', True):
                log.info('[CACHE] All entries saved and committed')
        else:
            # Existing cache - synchronize structure
            # 1. Remove obsolete entries (content paragraphs that no longer exist)
            # 2. Add missing entries (new metadata, TOC, or content)

            # Get current cache IDs
            existing_ids = set([item[0] for item in cache.cursor.execute('SELECT id FROM cache').fetchall()])

            # Get new IDs from current extraction
            new_content_ids = set([item[0] for item in content_originals])
            new_metadata_ids = set([item[0] for item in metadata_originals])
            new_toc_ids = set([item[0] for item in toc_originals])
            new_all_ids = new_content_ids | new_metadata_ids | new_toc_ids

            # Find IDs to remove (in cache but not in new extraction)
            # Only remove if current extraction is complete (has content)
            if len(content_originals) > 0:
                ids_to_remove = existing_ids - new_all_ids
                if ids_to_remove:
                    placeholders = ', '.join(['?'] * len(ids_to_remove))
                    cache.cursor.execute(
                        'DELETE FROM cache WHERE id IN (%s)' % placeholders,
                        tuple(ids_to_remove))
                    if config.get('log_translation', True):
                        log.info('[CACHE SYNC] Removed %d obsolete entries' % len(ids_to_remove))

            # Add missing metadata entries
            metadata_added = 0
            for metadata_item in metadata_originals:
                item_id = metadata_item[0]
                if item_id not in existing_ids:
                    cache.add(*metadata_item)
                    metadata_added += 1
                    field_name = metadata_field_names.get(item_id, 'unknown')
                    if config.get('log_translation', True):
                        log.info('[CACHE SYNC] Added metadata [%s]: "%s"' %
                                (field_name, metadata_item[3][:50]))

            # Add missing TOC entry
            toc_added = 0
            for toc_item in toc_originals:
                item_id = toc_item[0]
                if item_id not in existing_ids:
                    cache.add(*toc_item)
                    toc_added += 1
                    if config.get('log_translation', True):
                        log.info('[CACHE SYNC] Added merged TOC entry')

            if metadata_added > 0 or toc_added > 0:
                cache.connection.commit()
                if config.get('log_translation', True):
                    if metadata_added > 0:
                        log.info('[CACHE SYNC] Added %d metadata entries to existing cache' % metadata_added)
                    if toc_added > 0:
                        log.info('[CACHE SYNC] Added TOC entry to existing cache')

        # Load all paragraphs from cache
        all_paragraphs = cache.all_paragraphs()

        # Separate metadata, TOC, and content
        metadata_paragraphs = [p for p in all_paragraphs if p.page == 'content.opf']
        toc_paragraphs = [p for p in all_paragraphs if p.page == 'toc.ncx']
        content_paragraphs = [p for p in all_paragraphs if p.page not in ('content.opf', 'toc.ncx')]

        # Translate metadata fields separately
        if metadata_paragraphs:
            log.info(sep())
            log.info(_('Translating metadata fields...'))
            log.info(sep('┈'))

            # Disable streaming for metadata
            original_stream = translation.translator.stream
            translation.translator.stream = False

            # Track which fields exist to avoid auto-populating sort fields
            existing_fields = set()
            for meta_para in metadata_paragraphs:
                # Determine field name from original value
                for key in oeb.metadata.iterkeys():
                    items = getattr(oeb.metadata, key)
                    for item in items:
                        if item.content == meta_para.original:
                            existing_fields.add(key)
                            break

            # Store translated title/creator for auto-populating sort fields
            translated_title = None
            translated_creators = []

            for meta_para in metadata_paragraphs:
                # Determine which field this paragraph belongs to
                field_name = None
                for key in oeb.metadata.iterkeys():
                    items = getattr(oeb.metadata, key)
                    for item in items:
                        if item.content == meta_para.original:
                            field_name = key
                            break
                    if field_name:
                        break

                # Skip if already translated and not fresh
                if meta_para.translation and not translation.fresh:
                    if config.get('log_translation', True):
                        log.info('✓ Metadata cached: "%s" → "%s"' %
                                (meta_para.original[:50], meta_para.translation[:50]))
                    # Apply cached translation to element
                    if meta_para.original in metadata_element_map:
                        metadata_element_map[meta_para.original].element.content = meta_para.translation

                    # Track for auto-populating sort fields
                    if field_name == 'title':
                        translated_title = meta_para.translation
                    elif field_name == 'creator':
                        translated_creators.append(meta_para.translation)
                    continue

                # Translate this metadata field
                if config.get('log_translation', True):
                    log.info('○ Translating metadata [%s]: "%s"' % (field_name or 'unknown', meta_para.original[:50]))

                try:
                    result = translation.translator.translate(meta_para.original)
                    if hasattr(result, '__iter__') and not isinstance(result, str):
                        result = ''.join(result)
                    if result and result.strip():
                        meta_para.translation = result.strip()
                        meta_para.engine_name = translation.translator.name
                        meta_para.target_lang = translation.translator.get_target_lang()
                        # Update cache
                        cache.update_paragraph(meta_para)
                        # Apply to OEB element
                        if meta_para.original in metadata_element_map:
                            metadata_element_map[meta_para.original].element.content = meta_para.translation

                        # Track for auto-populating sort fields
                        if field_name == 'title':
                            translated_title = meta_para.translation
                        elif field_name == 'creator':
                            translated_creators.append(meta_para.translation)

                        if config.get('log_translation', True):
                            log.info('  → "%s"' % meta_para.translation[:50])
                except Exception as e:
                    log.error('Failed to translate metadata: %s' % e)

            # Auto-populate title_sort if title was translated and title_sort doesn't exist
            if translated_title and 'title_sort' not in existing_fields:
                oeb.metadata.add('title_sort', translated_title)
                if config.get('log_translation', True):
                    log.info('✓ Auto-populated title_sort: "%s"' % translated_title[:50])

            # Auto-populate author_sort if creator was translated and author_sort doesn't exist
            if translated_creators and 'author_sort' not in existing_fields:
                # Use first creator for author_sort
                oeb.metadata.add('author_sort', translated_creators[0])
                if config.get('log_translation', True):
                    log.info('✓ Auto-populated author_sort: "%s"' % translated_creators[0][:50])

            # Restore streaming
            translation.translator.stream = original_stream
            log.info(sep())

        # Translate TOC (merged paragraph)
        if toc_paragraphs:
            log.info(sep())
            log.info(_('Translating TOC...'))
            log.info(sep('┈'))

            for toc_para in toc_paragraphs:
                # Skip if already translated and not fresh
                if toc_para.translation and not translation.fresh:
                    if config.get('log_translation', True):
                        log.info('✓ TOC cached')
                else:
                    # Translate merged TOC
                    if config.get('log_translation', True):
                        log.info('○ Translating merged TOC')

                    try:
                        result = translation.translator.translate(toc_para.original)
                        if hasattr(result, '__iter__') and not isinstance(result, str):
                            result = ''.join(result)
                        if result and result.strip():
                            toc_para.translation = result.strip()
                            toc_para.engine_name = translation.translator.name
                            toc_para.target_lang = translation.translator.get_target_lang()
                            cache.update_paragraph(toc_para)
                            if config.get('log_translation', True):
                                log.info('  → TOC translated')
                    except Exception as e:
                        log.error('Failed to translate TOC: %s' % e)

                # Split translated TOC back to individual elements
                if toc_para.translation and toc_element_list:
                    translated_parts = toc_para.translation.split('\n\n')
                    for i, toc_elem in enumerate(toc_element_list):
                        if i < len(translated_parts):
                            toc_elem.element.title = translated_parts[i].strip()

            log.info(sep())

        # Translate content using normal paragraph system
        translation.handle(content_paragraphs)

        # Apply content translations to elements
        element_handler.add_translations(content_paragraphs)

        # RTL/LTR metadata will be added in translate_done() phase
        # (after EPUB file is written, so we can directly modify the OPF)

        log.info(sep())
        log.info(_('Start to convert ebook format...'))
        log.info(sep())

        self.report_progress = CompositeProgressReporter(
            backup_progress, 1, notification)
        self.report_progress(0., _('Outputting ebook file...'))
        _convert(oeb, output_path, input_plugin, opts, log)

    plumber.output_plugin.convert = MethodType(convert, plumber.output_plugin)
    plumber.run()


def convert_srt(
        input_path, output_path, translation, element_handler, cache,
        debug_info, encoding, notification) -> None:
    log.info('Translating subtitles content... (this will take a while)')
    log.info(debug_info)

    elements = get_srt_elements(input_path, encoding)
    original_group = element_handler.prepare_original(elements)
    cache.save(original_group)

    paragraphs = cache.all_paragraphs()
    translation.set_progress(notification)
    translation.handle(paragraphs)
    element_handler.add_translations(paragraphs)

    log.info(sep())
    log.info(_('Starting to output subtitles file...'))
    log.info(sep())

    with open(output_path, 'w') as file:
        file.write('\n\n'.join([e.get_translation() for e in elements]))

    log.info(_('The translation of the subtitles file was completed.'))


def convert_pgn(
        input_path, output_path, translation, element_handler, cache,
        debug_info, encoding, notification) -> None:
    log.info('Translating PGN content... (this may be take a while)')
    log.info(debug_info)

    elements = get_pgn_elements(input_path, encoding)
    original_group = element_handler.prepare_original(elements)
    cache.save(original_group)

    paragraphs = cache.all_paragraphs()
    translation.set_progress(notification)
    translation.handle(paragraphs)
    element_handler.add_translations(paragraphs)

    log.info(sep())
    log.info(_('Starting to output PGN file...'))
    log.info(sep())

    pgn_content = open_file(input_path, encoding)
    for element in elements:
        pgn_content = pgn_content.replace(
            element.get_raw(), element.get_translation(), 1)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(pgn_content)

    log.info(_('The translation of the PGN file was completed.'))


extra_formats: dict[str, dict[str, Callable]] = {
    'srt': {
        'extractor': get_srt_elements,
        'convertor': convert_srt,
    },
    'pgn': {
        'extractor': get_pgn_elements,
        'convertor': convert_pgn,
    }
}


def extract_item(input_path, input_format, encoding, callback=None):
    if callback is not None:
        log.outputs = [Stream(PrepareStream(callback))]
    handler = extra_formats.get(input_format)
    extractor = extract_book if handler is None else handler['extractor']
    return extractor(input_path, encoding)


def extract_book(input_path, encoding):
    elements = []
    output_path = os.path.join(gettempdir(), 'temp.epub')
    plumber = Plumber(input_path, output_path, log=log)

    def convert(self, oeb, output_path, input_plugin, opts, log):
        # for item in oeb.manifest.items:
        #     if item.media_type == 'text/css':
        #         for rule in item.data.cssRules:
        #             print('='*20)
        #             # CSSStyleRule or CSSPageRule
        #             print(type(rule))
        #             # CSSStyleDeclaration
        #             print(rule.style.keys())
        elements.extend(get_metadata_elements(oeb.metadata))
        elements.extend(get_toc_elements(oeb.toc.nodes, []))
        elements.extend(get_page_elements(oeb.manifest.items))
        raise ConversionAbort()
    plumber.output_plugin.convert = MethodType(convert, plumber.output_plugin)
    try:
        plumber.run()
    except ConversionAbort:
        return elements


def convert_item(
        ebook_title, input_path, output_path, source_lang, target_lang,
        cache_only, is_batch, format, encoding, direction, notification):
    """The following parameters need attention:
    :cache_only: Only use the translation which exists in the cache.
    :notification: It is automatically added by arbitrary_n.
    """
    translator = get_translator()
    translator.set_source_lang(source_lang)
    translator.set_target_lang(target_lang)

    element_handler = get_element_handler(
        translator.placeholder, translator.separator, direction)
    element_handler.set_translation_lang(
        translator.get_iso639_target_code(target_lang))
    element_handler.set_source_lang(
        translator.get_source_code(translator.source_lang))

    from .utils import get_cache_id
    merge_length = element_handler.get_merge_length()
    cache_id = get_cache_id(input_path, translator.name, target_lang, merge_length, encoding)
    cache = get_cache(cache_id)
    cache.set_cache_only(cache_only)
    cache.set_info('title', ebook_title)
    cache.set_info('engine_name', translator.name)
    cache.set_info('target_lang', target_lang)
    cache.set_info('merge_length', merge_length)
    cache.set_info('plugin_version', EbookTranslator.__version__)
    cache.set_info('calibre_version', __version__)
    cache.set_info('source_lang', source_lang)
    cache.set_info('target_direction', direction)

    translation = get_translation(
        translator, lambda text, error=False: log.info(text))
    translation.set_batch(is_batch)
    translation.set_callback(cache.update_paragraph)

    debug_info = '{0}\n| Diagnosis Information\n{0}'.format(sep())
    debug_info += '\n| Calibre Version: %s\n' % __version__
    debug_info += '| Plugin Version: %s\n' % EbookTranslator.__version__
    debug_info += '| Translation Engine: %s\n' % translator.name
    debug_info += '| Source Language: %s\n' % source_lang
    debug_info += '| Target Language: %s\n' % target_lang
    debug_info += '| Encoding: %s\n' % encoding
    debug_info += '| Cache Enabled: %s\n' % cache.is_persistence()
    debug_info += '| Merging Length: %s\n' % element_handler.merge_length
    debug_info += '| Concurrent requests: %s\n' % translator.concurrency_limit
    debug_info += '| Request Interval: %s\n' % translator.request_interval
    debug_info += '| Request Attempt: %s\n' % translator.request_attempt
    debug_info += '| Request Timeout: %s\n' % translator.request_timeout
    debug_info += '| Input Path: %s\n' % input_path
    debug_info += '| Output Path: %s' % output_path

    handler: dict[str, Callable] | None = extra_formats.get(format)
    convertor = convert_book if handler is None else handler['convertor']
    convertor(
        input_path, output_path, translation, element_handler, cache,
        debug_info, encoding, notification)

    cache.done()


class ConversionWorker:
    def __init__(self, gui, icon):
        self.gui = gui
        self.icon = icon
        self.config = get_config()
        self.db = gui.current_db
        self.api = self.db.new_api
        self.working_jobs = self.gui.bookfere_ebook_translator.jobs

    def translate_ebook(self, ebook, cache_only=False, is_batch=False):
        input_path = ebook.get_input_path()
        if not self.config.get('to_library'):
            filename = sanitize_file_name(ebook.title[:200])
            output_path = self.config.get('output_path')
            if output_path is None or not os.path.isdir(output_path):
                raise Exception(
                    _('Please set a valid output path.'))
            output_path = os.path.join(
                output_path, f'{filename}.{ebook.output_format}')
        else:
            output_path = PersistentTemporaryFile(
                suffix='.' + ebook.output_format).name
        job = self.gui.job_manager.run_job(
            Dispatcher(self.translate_done),
            'arbitrary_n',
            args=(
                'calibre_plugins.ebook_translator.lib.conversion',
                'convert_item',
                (ebook.title, input_path, output_path, ebook.source_lang,
                 ebook.target_lang, cache_only, is_batch, ebook.input_format,
                 ebook.encoding, ebook.target_direction)),
            description=(_('[{} > {}] Translating "{}"').format(
                ebook.source_lang, ebook.target_lang, ebook.title)))
        self.working_jobs[job] = (ebook, output_path, cache_only)

    def translate_done(self, job):
        ebook, output_path, cache_only = self.working_jobs.pop(job)
        log.info('translate_done called: cache_only=%s, format=%s' % (cache_only, ebook.output_format))

        if job.failed:
            if not DEBUG:
                self.gui.job_exception(
                    job, dialog_title=_('Translation job failed'))
            return

        # TODO: Try to use the calibre generated metadata file.
        ebook_metadata_config = self.config.get('ebook_metadata') or {}
        log.info('ebook_metadata_config: %s' % ebook_metadata_config)
        log.info('is_extra_format: %s' % ebook.is_extra_format())

        if not ebook.is_extra_format():
            with open(output_path, 'r+b') as file:
                metadata = get_metadata(file, ebook.output_format)
                log.info('Read metadata from file: title=%s, authors=%s, publisher=%s, series=%s' % (
                    metadata.title, metadata.authors, metadata.publisher, metadata.series))

                # Add RTL/LTR metadata to OPF if directions differ
                # Do this by directly modifying the EPUB's content.opf file
                from lxml import etree
                from ..engines.languages import lang_directionality

                translator = get_translator()
                source_lang_code = translator.get_source_code(ebook.source_lang)
                source_lang_base = source_lang_code.split('-')[0] if source_lang_code else None
                source_direction = lang_directionality.get(source_lang_code,
                                                          lang_directionality.get(source_lang_base, 'ltr'))
                target_direction = ebook.target_direction.lower() if ebook.target_direction else None

                if target_direction in ('rtl', 'ltr') and source_direction != target_direction:
                    try:
                        from calibre.ebooks.oeb.polish.container import get_container
                        from lxml import etree

                        writing_mode = 'horizontal-rl' if target_direction == 'rtl' else 'horizontal-lr'

                        # Use Calibre's polish Container for efficient EPUB modification
                        container = get_container(output_path, tweak_mode=True)
                        opf_root = container.opf

                        # Add primary-writing-mode meta
                        ns = {'opf': 'http://www.idpf.org/2007/opf'}
                        metadata_elem = opf_root.xpath('//opf:metadata', namespaces=ns)[0]
                        etree.SubElement(metadata_elem, '{http://www.idpf.org/2007/opf}meta',
                                        attrib={'name': 'primary-writing-mode', 'content': writing_mode})

                        # Set spine page-progression-direction
                        spine_elem = opf_root.xpath('//opf:spine', namespaces=ns)[0]
                        spine_elem.set('page-progression-direction', target_direction)

                        # Commit changes - Container efficiently updates only the OPF file in ZIP
                        container.dirty(container.opf_name)
                        container.commit()

                        log.info('Added RTL metadata: primary-writing-mode=%s, page-progression-direction=%s' %
                                (writing_mode, target_direction))
                    except Exception as e:
                        log.warn('Failed to add RTL metadata to EPUB: %s' % e)

                ebook_title = metadata.title
                if ebook.custom_title is not None:
                    ebook_title = ebook.custom_title
                if ebook_metadata_config.get('lang_mark'):
                    ebook_title = '%s [%s]' % (ebook_title, ebook.target_lang)
                metadata.title = ebook_title

                # Update title_sort to match (custom title or translated title with suffix)
                if ebook_metadata_config.get('translate_title', False):
                    metadata.title_sort = ebook_title

                if ebook_metadata_config.get('lang_code', True):
                    metadata.language = ebook.lang_code
                subjects = ebook_metadata_config.get('subjects')
                metadata.tags += (subjects or []) + [
                    'Translated by Ebook Translator: '
                    'https://translator.bookfere.com']
                # metadata.authors = ['bookfere.com']
                # metadata.book_producer = 'Ebook Translator'
                set_metadata(file, metadata, ebook.output_format)
        else:
            metadata = self.api.get_metadata(ebook.id)
            ebook_title = ebook.title
            if ebook.custom_title is not None:
                ebook_title = ebook.custom_title
            if ebook_metadata_config.get('lang_mark'):
                ebook_title = '%s [%s]' % (ebook_title, ebook.target_lang)
            metadata.title = ebook_title

        if self.config.get('to_library'):
            book_id = self.db.create_book_entry(metadata)
            self.api.add_format(
                book_id, ebook.output_format, output_path, run_hooks=False)
            self.gui.library_view.model().books_added(1)
            output_path = self.api.format_abspath(book_id, ebook.output_format)
        else:
            dirname = os.path.dirname(output_path)
            filename = sanitize_file_name(ebook_title[:200])
            new_output_path = os.path.join(
                dirname, '%s.%s' % (filename, ebook.output_format))
            os.rename(output_path, new_output_path)
            output_path = new_output_path

        self.gui.status_bar.show_message(
            job.description + ' ' + _('completed'), 5000)

        def callback(payload):
            if ebook.input_format in extra_formats.keys():
                open_path(output_path)
            else:
                kwargs = {'args': ['ebook-viewer', output_path]}
                payload('ebook-viewer', kwargs=kwargs)

        if self.config.get('show_notification', True):
            self.gui.proceed_question(
                callback,
                self.gui.job_manager.launch_gui_app,
                job.log_path,
                _('Ebook Translation Log'), _('Translation Completed'),
                _('The translation of "{}" was completed. Do you want to '
                  'open the book?').format(ebook_title),
                log_is_file=True, icon=self.icon, auto_hide_after=10)
