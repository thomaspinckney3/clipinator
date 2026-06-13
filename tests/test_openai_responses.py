import httpx
import pytest
from openai import APIConnectionError

from extractor import openai_responses
from extractor.openai_responses import _connection_error_types, create_response_streaming


class FakeStream:
    def __init__(self, events, final):
        self._events = events
        self._final = final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_response(self):
        return self._final


class FakeResponses:
    """Raises a configured exception on the first `fail_times` stream() calls,
    then yields a successful stream. Records how many times it was called."""
    def __init__(self, fail_times=0, exc=None, events=None, final="OK"):
        self.fail_times = fail_times
        self.exc = exc or httpx.RemoteProtocolError("server disconnected")
        self.events = events or []
        self.final = final
        self.calls = 0

    def stream(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return FakeStream(self.events, self.final)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses


class TestConnectionErrorTypes:
    def test_includes_transient(self):
        types = _connection_error_types()
        assert APIConnectionError in types
        assert httpx.RemoteProtocolError in types
        assert httpx.ReadTimeout in types
        assert httpx.ConnectError in types

    def test_excludes_client_side_construction_errors(self):
        types = _connection_error_types()
        # F9: these never succeed on retry and must not be caught
        assert not issubclass(httpx.LocalProtocolError, types)
        assert not issubclass(httpx.UnsupportedProtocol, types)


class TestRetry:
    def test_success_first_try(self):
        client = FakeClient(FakeResponses(fail_times=0, final="RESULT"))
        out = create_response_streaming(client, model="m", input=[])
        assert out == "RESULT"
        assert client.responses.calls == 1

    def test_retries_then_succeeds(self):
        client = FakeClient(FakeResponses(fail_times=2, final="RESULT"))
        out = create_response_streaming(client, max_connection_retries=3,
                                        model="m", input=[])
        assert out == "RESULT"
        assert client.responses.calls == 3  # 2 failures + 1 success

    def test_retries_exhausted_reraises(self):
        client = FakeClient(FakeResponses(fail_times=5))
        with pytest.raises(httpx.RemoteProtocolError):
            create_response_streaming(client, max_connection_retries=2,
                                      model="m", input=[])
        assert client.responses.calls == 3  # initial + 2 retries

    def test_non_retryable_propagates_immediately(self):
        client = FakeClient(FakeResponses(fail_times=1, exc=httpx.LocalProtocolError("bad")))
        with pytest.raises(httpx.LocalProtocolError):
            create_response_streaming(client, max_connection_retries=3,
                                      model="m", input=[])
        assert client.responses.calls == 1  # not retried

    def test_zero_retries(self, monkeypatch):
        monkeypatch.setenv("OPENAI_STREAM_MAX_RETRIES", "0")
        client = FakeClient(FakeResponses(fail_times=1))
        with pytest.raises(httpx.RemoteProtocolError):
            create_response_streaming(client, model="m", input=[])
        assert client.responses.calls == 1

    def test_retry_delay_is_not_real_time(self, monkeypatch):
        # keep the test fast and assert backoff is invoked
        sleeps = []
        monkeypatch.setattr(openai_responses.time, "sleep", lambda s: sleeps.append(s))
        monkeypatch.setenv("OPENAI_STREAM_RETRY_BASE_SECONDS", "1")
        client = FakeClient(FakeResponses(fail_times=2, final="R"))
        out = create_response_streaming(client, max_connection_retries=3, model="m", input=[])
        assert out == "R"
        assert sleeps == [1.0, 2.0]  # exponential backoff base*2**(n-1)

    def test_progress_logging_counts_deltas(self, monkeypatch, capsys):
        monkeypatch.setattr(openai_responses.time, "sleep", lambda s: None)
        monkeypatch.setenv("OPENAI_STREAM_PROGRESS_SECONDS", "0")  # disabled, just exercise loop

        class Ev:
            def __init__(self, d):
                self.delta = d
        client = FakeClient(FakeResponses(fail_times=0, events=[Ev("ab"), Ev("cde")], final="R"))
        out = create_response_streaming(client, model="m", input=[])
        assert out == "R"
