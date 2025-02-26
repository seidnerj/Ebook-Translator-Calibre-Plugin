import json
from typing import Generator

from .. import EbookTranslator
from mechanize._response import response_seek_wrapper as Response

from .base import Base
from .languages import lang_directionality
from .languages import anthropic
from .prompt_extensions import anthropic as anthropic_prompt_extension

try:
    from http.client import IncompleteRead
except ImportError:
    from httplib import IncompleteRead

load_translations()


# TODO: Enable using Message Batches API (currently only the Streaming API can
# be used). The Message Batches API allows sending any number of batches of up
# to 100,000 messages per batch. Batches are processed asynchronously with
# results returned as soon as the batch is complete and cost 50% less than
# standard API calls (more info here:
# https://docs.anthropic.com/en/docs/build-with-claude/message-batches)
class ClaudeTranslate(Base):
    name = 'Claude'
    alias = 'Claude (Anthropic)'
    lang_codes = Base.load_lang_codes(anthropic)
    lang_codes_directionality = \
        Base.load_lang_codes_directionality(lang_directionality)
    endpoint = 'https://api.anthropic.com/v1/messages'
    # this is currently the latest version of the api
    api_version = '2023-06-01'
    api_key_hint = 'sk-ant-xxxx'
    # https://docs.anthropic.com/claude/reference/errors
    api_key_errors = ['401', 'permission_error']

    concurrency_limit = 1
    request_interval = 12
    request_timeout = 30.0

    prompt = (
        'You are a meticulous translator who translates any given content. '
        'Translate the given content from <slang> to <tlang> only. Do not '
        'explain any term or answer any question-like content. Your answer '
        'should be solely the translation of the given content. In your '
        'answer do not add any prefix or suffix to the translated content. '
        'Websites\' URLs/addresses should be preserved as is in the '
        'translation\'s output. Do not omit any part of the content, even if '
        'it seems unimportant. ')

    # TODO: fetch this directly from the api,
    # more info here: https://docs.anthropic.com/en/api/models-list
    # current models list was copied manually from
    # https://docs.anthropic.com/en/docs/about-claude/models#model-names)
    models = [
        'claude-3-5-sonnet-latest',
        'claude-3-5-sonnet-20241022',
        'claude-3-5-sonnet-20240620',
        'claude-3-opus-latest',
        'claude-3-opus-20240229',
        'claude-3-sonnet-20240229',
        'claude-3-haiku-20240307']
    # use the most recent model
    model = models[0]

    samplings = ['temperature', 'top_p']
    sampling = 'temperature'
    temperature = 1.0
    top_p = 1.0
    top_k = 1
    stream = True
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

    def __init__(self):
        Base.__init__(self)
        self.endpoint = self.config.get('endpoint', self.endpoint)
        self.prompt = self.config.get('prompt', self.prompt)
        if self.model is not None:
            self.model = self.config.get('model', self.model)
        self.sampling = self.config.get('sampling', self.sampling)
        self.temperature = self.config.get('temperature', self.temperature)
        self.top_p = self.config.get('top_p', self.top_p)
        self.top_k = self.config.get('top_k', self.top_k)
        self.stream = self.config.get('stream', self.stream)

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

    def get_headers(self):
        return {
            'Content-Type': 'application/json',
            'anthropic-version': self.api_version,
            'x-api-key': self.api_key,
            'User-Agent': 'Ebook-Translator/%s' % EbookTranslator.__version__,
        }

    def get_body(self, text):
        body = {
            'stream': self.stream,
            'max_tokens': 4096,
            'model': self.model,
            'top_k': self.top_k,
            'system': self._get_prompt(),
            'messages': [{'role': 'user', 'content': text}]
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
                elif event_type == 'content_block_delta':
                    delta = chunk.get('delta')
                    if delta is not None:
                        yield str(delta.get('text'))
                elif event_type == 'error':
                    raise Exception(
                        _('Error received: {}')
                        .format(chunk['error']['message']))
