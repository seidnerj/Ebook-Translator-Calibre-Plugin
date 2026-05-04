class UnexpectedResult(Exception):
    pass


class ConversionFailed(Exception):
    pass


class ConversionAbort(Exception):
    pass


class TranslationFailed(Exception):
    pass


class RefusalExhausted(TranslationFailed):
    """Translation refused after exhausting refusal retries. Subclass of
    TranslationFailed so existing handlers still catch it; the distinct
    type lets the splitting logic identify a recoverable refusal failure
    versus other translation failures (network, parse errors, etc.)."""
    pass


class TranslationCanceled(Exception):
    pass


class BadApiKeyFormat(TranslationCanceled):
    pass


class NoAvailableApiKey(TranslationCanceled):
    pass


class UnsupportedModel(Exception):
    pass
