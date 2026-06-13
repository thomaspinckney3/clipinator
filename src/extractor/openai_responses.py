from __future__ import annotations

import sys
import time
from typing import Any

from extractor.transcribe import env_float, env_int

# Long Responses API generations (large image input + tens of thousands of
# output tokens + reasoning) can hold an idle HTTP connection for minutes while
# the model works, and that idle wait is where flaky links drop with
# "server disconnected without sending a response". Streaming keeps the
# connection active for the whole generation, which removes that failure mode
# without changing the request (the full contact sheet is still sent). A retry
# loop with exponential backoff covers any connection/timeout error that still
# slips through. Retries keep the *result* correct (every stage call is
# stateless and stored=False), but a disconnect late in generation re-runs the
# whole request, so output tokens already produced on the failed attempt are
# billed again — the retry budget is deliberately small to bound that cost.

DEFAULT_MAX_CONNECTION_RETRIES = 4
DEFAULT_RETRY_BASE_DELAY_SECONDS = 2.0
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0


def _connection_error_types() -> tuple[type[BaseException], ...]:
    import httpx
    from openai import APIConnectionError, APITimeoutError

    # Retry only genuinely transient transport failures. ConnectError/ReadError/
    # WriteError/ReadTimeout/RemoteProtocolError (the idle-disconnect case) are
    # transient; we exclude httpx.TransportError's client-side request-
    # construction members (LocalProtocolError, UnsupportedProtocol, ProxyError),
    # which would never succeed on retry. HTTP 4xx/5xx surface as
    # openai.APIStatusError (not caught here) and are correctly not retried.
    return (
        APIConnectionError,
        APITimeoutError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.WriteError,
        httpx.WriteTimeout,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
    )


def create_response_streaming(
    client: Any,
    *,
    label: str = "request",
    max_connection_retries: int | None = None,
    **create_kwargs: Any,
) -> Any:
    """Run a Responses API call in streaming mode with connection retry.

    Drop-in replacement for ``client.responses.create(**kwargs)``: it streams
    the response (keeping the connection warm during generation) and returns
    the final ``Response`` object, identical in type and shape to a
    non-streaming create(), so all downstream handling is unchanged.

    On a transient connection/timeout error the whole request is retried with
    exponential backoff (small budget — see module note on re-billing).
    ``OPENAI_STREAM_MAX_RETRIES`` and ``OPENAI_STREAM_RETRY_BASE_SECONDS``
    override the defaults; ``OPENAI_STREAM_PROGRESS_SECONDS`` sets the
    progress-log cadence.
    """
    retries = (
        max_connection_retries
        if max_connection_retries is not None
        else (env_int("OPENAI_STREAM_MAX_RETRIES", DEFAULT_MAX_CONNECTION_RETRIES) or 0)
    )
    base_delay = env_float("OPENAI_STREAM_RETRY_BASE_SECONDS", DEFAULT_RETRY_BASE_DELAY_SECONDS) or 0.0
    # Progress logging only — it prints as events arrive and so cannot detect an
    # idle stall (that is the httpx read-timeout's job). It is not a heartbeat.
    progress_interval = (
        env_float("OPENAI_STREAM_PROGRESS_SECONDS", DEFAULT_PROGRESS_INTERVAL_SECONDS) or 0.0
    )
    connection_errors = _connection_error_types()

    attempt = 0
    while True:
        try:
            with client.responses.stream(**create_kwargs) as stream:
                output_chars = 0
                last_progress = time.monotonic()
                for event in stream:
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str):
                        output_chars += len(delta)
                    if progress_interval > 0:
                        now = time.monotonic()
                        if now - last_progress >= progress_interval:
                            print(
                                f"  {label}: streaming, ~{output_chars} output chars so far",
                                flush=True,
                            )
                            last_progress = now
                return stream.get_final_response()
        except connection_errors as exc:
            attempt += 1
            if attempt > retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"  {label}: connection error ({type(exc).__name__}: {exc}); "
                f"retry {attempt}/{retries} in {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
