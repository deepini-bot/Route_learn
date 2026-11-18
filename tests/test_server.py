import json
import os
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from routelearn.gateway import Gateway
from routelearn.features import extract
from routelearn.policies import AlwaysArm, BanditPolicy
from routelearn.providers import (
    AnthropicProvider,
    Completion,
    LLMJudge,
    MockProvider,
)
from routelearn.server import create_app
from routelearn.storage import SQLiteStateStore


class AttemptProvider:
    def stream(self, arm_idx, prompt, meta=None, history=None):
        text = f"answer-{arm_idx}"
        yield text
        return Completion(text, 10, 2, 0.001, 0.01)


class SequenceJudge:
    cost = 0.0

    def __init__(self):
        self.scores = iter([0.1, 0.9])

    def score(self, prompt, completion, history=None):
        return next(self.scores)


class CapturingProvider:
    def __init__(self):
        self.history = None

    def complete(self, arm_idx, prompt, meta=None, history=None):
        self.history = history
        return Completion("captured", 10, 2, 0.001, 0.01)


class TestServer(unittest.TestCase):
    def _client(self, directory, provider=None, *, rate=0, spend=0, keys=(),
                allow_tier_override=False):
        store = SQLiteStateStore(os.path.join(directory, "routelearn.db"))
        policy = BanditPolicy()
        store.initialize_policy(policy)
        policy.store = store
        gateway = Gateway(policy, provider or MockProvider(), store=store)
        return TestClient(create_app(
            gateway,
            store,
            rate_limit_per_minute=rate,
            daily_spend_limit=spend,
            api_keys=keys,
            allow_tier_override=allow_tier_override,
        )), store, gateway

    def test_chat_forwards_history_and_explain_uses_prior_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = CapturingProvider()
            client, _, _ = self._client(directory, provider)
            history = [{"role": "user", "content": "Analyze the architecture carefully."}]
            response = client.post("/chat", json={"prompt": "ok, do it", "history": history})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(provider.history, history)

            explained = client.post(
                "/explain", json={"prompt": "ok, do it", "history": history}
            ).json()
            self.assertGreater(explained["features"]["n_reason"], 0)

    def test_stream_returns_json_sse_deltas_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            client, _, _ = self._client(directory)
            response = client.post("/chat/stream", json={"prompt": "hello"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
            events = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            self.assertTrue(any(event["type"] == "delta" for event in events))
            done = next(event for event in events if event["type"] == "done")
            answer = "".join(event["text"] for event in events if event["type"] == "delta")
            self.assertEqual(answer, f"[mock answer from {done['model']}]")

    def test_quotas_are_per_configured_key_and_feedback_is_key_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            client, _, _ = self._client(
                directory, rate=1, keys=("key-a", "key-b")
            )
            headers_a, headers_b = {"X-API-Key": "key-a"}, {"X-API-Key": "key-b"}
            first = client.post("/chat", json={"prompt": "hello"}, headers=headers_a)
            self.assertEqual(first.status_code, 200)
            limited = client.post("/chat", json={"prompt": "again"}, headers=headers_a)
            self.assertEqual(limited.status_code, 429)
            other_key = client.post("/chat", json={"prompt": "hello"}, headers=headers_b)
            self.assertEqual(other_key.status_code, 200)

            feedback = {"request_id": first.json()["request_id"], "score": 0.9}
            self.assertEqual(client.post("/feedback", json=feedback, headers=headers_b).status_code, 404)
            self.assertEqual(client.post("/feedback", json=feedback, headers=headers_a).status_code, 200)
            self.assertEqual(client.get("/stats").status_code, 401)

    def test_daily_cap_rejects_before_calling_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = CapturingProvider()
            client, _, _ = self._client(directory, provider, spend=0.001)
            response = client.post("/chat", json={"prompt": "hello"})
            self.assertEqual(response.status_code, 429)
            self.assertIsNone(provider.history)

    def test_config_persists_shared_lambda(self):
        with tempfile.TemporaryDirectory() as directory:
            client, store, gateway = self._client(directory)
            response = client.post("/config", json={"lam": 0.9})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["lam"], 0.9)
            store.refresh_policy(gateway.policy)
            self.assertEqual(gateway.policy.lam, 0.9)

    def test_openapi_has_request_examples_for_every_post_api(self):
        with tempfile.TemporaryDirectory() as directory:
            client, _, _ = self._client(directory)
            schema = client.get("/openapi.json").json()
            for path in ("/chat", "/chat/stream", "/explain", "/feedback", "/config"):
                request_body = schema["paths"][path]["post"]["requestBody"]
                content = request_body["content"]["application/json"]
                self.assertIn("schema", content)
                self.assertTrue(content["examples"])
            self.assertEqual(
                schema["paths"]["/chat"]["post"]["requestBody"]["content"]
                ["application/json"]["examples"]["test_medium"]["value"]["force_tier"],
                "medium",
            )
            self.assertNotIn("requestBody", schema["paths"]["/stats"]["get"])

    def test_explicit_tier_override_is_gated_and_selects_requested_model(self):
        with tempfile.TemporaryDirectory() as directory:
            disabled, _, disabled_gateway = self._client(directory)
            denied = disabled.post(
                "/chat", json={"prompt": "simple prompt", "force_tier": "medium"}
            )
            self.assertEqual(denied.status_code, 403)
            self.assertEqual(disabled_gateway.log, [])

        with tempfile.TemporaryDirectory() as directory:
            client, _, _ = self._client(directory, allow_tier_override=True)
            for requested_tier, model in (
                ("medium", "claude-sonnet-5-5"),
                ("large", "claude-opus-5-5"),
            ):
                response = client.post(
                    "/chat",
                    json={"prompt": "simple prompt", "force_tier": requested_tier},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["tier"], requested_tier)
                self.assertEqual(response.json()["model"], model)

            explained = client.post(
                "/explain",
                json={"prompt": "simple prompt", "force_tier": "large"},
            )
            self.assertEqual(explained.json()["selected_tier"], "large")
            streamed = client.post(
                "/chat/stream",
                json={"prompt": "simple prompt", "force_tier": "medium"},
            )
            events = [
                json.loads(line.removeprefix("data: "))
                for line in streamed.text.splitlines()
                if line.startswith("data: ")
            ]
            done = next(event for event in events if event["type"] == "done")
            self.assertEqual(done["tier"], "medium")

    def test_stream_hides_rejected_cascade_attempt(self):
        gateway = Gateway(
            AlwaysArm(0, "small"),
            AttemptProvider(),
            SequenceJudge(),
            cascade_tau=0.5,
        )
        events = list(gateway.stream("hard prompt"))
        deltas = [value for event, value in events if event == "delta"]
        result = next(value for event, value in events if event == "result")
        self.assertEqual(deltas, ["answer-1"])
        self.assertEqual(result.escalations, 1)

    def test_sqlite_migrates_existing_feedback_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "routelearn.db")
            db = sqlite3.connect(path)
            try:
                db.execute(
                    "CREATE TABLE pending_feedback("
                    "request_id TEXT PRIMARY KEY, arm INTEGER NOT NULL, features BLOB NOT NULL)"
                )
            finally:
                db.close()
            store = SQLiteStateStore(path)
            features = extract("a prompt")
            store.save_pending("request-id", 1, features, "tenant")
            restored = store.consume_pending("request-id", "tenant")
            self.assertEqual(restored[0], 1)
            self.assertEqual(restored[1].tolist(), features.tolist())

    def test_sqlite_policy_updates_are_shared_across_worker_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStateStore(os.path.join(directory, "routelearn.db"))
            first, second = BanditPolicy(), BanditPolicy()
            store.initialize_policy(first)
            store.initialize_policy(second)
            first.store = second.store = store
            features = extract("A hard reasoning prompt")

            first.update(0, features, 0.2)
            second.update(1, features, 0.9)
            first.refresh()
            self.assertEqual(first.model.counts.tolist(), [1, 1, 0])
            second.set_lam(0.9)
            first.refresh()
            self.assertEqual(first.lam, 0.9)


class TestAnthropicAdapters(unittest.TestCase):
    def test_completion_and_judge_use_api_and_conversation_history(self):
        calls = []

        class Messages:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(text="8.5")],
                    usage=SimpleNamespace(input_tokens=12, output_tokens=3),
                )

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.client = SimpleNamespace(messages=Messages())
        provider.max_tokens = 20
        history = [{"role": "user", "content": "Earlier request"}, {
            "role": "assistant", "content": "Earlier answer"
        }]
        completion = provider.complete(1, "Follow up", history=history)
        self.assertEqual(completion.text, "8.5")
        self.assertEqual(calls[-1]["messages"], history + [{"role": "user", "content": "Follow up"}])

        judge = LLMJudge.__new__(LLMJudge)
        judge.client = SimpleNamespace(messages=Messages())
        self.assertEqual(judge.score("Follow up", completion, history), 0.85)
        self.assertIn("Earlier request", calls[-1]["messages"][0]["content"])

    def test_anthropic_stream_yields_text_and_usage(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(text="hello world")],
            usage=SimpleNamespace(input_tokens=7, output_tokens=2),
        )

        class StreamContext:
            text_stream = iter(["hello", " world"])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get_final_message(self):
                return response

        class Messages:
            def stream(self, **kwargs):
                self.kwargs = kwargs
                return StreamContext()

        messages = Messages()
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider.client = SimpleNamespace(messages=messages)
        provider.max_tokens = 20
        stream = provider.stream(0, "hello")
        chunks = []
        while True:
            try:
                chunks.append(next(stream))
            except StopIteration as finished:
                completion = finished.value
                break
        self.assertEqual(chunks, ["hello", " world"])
        self.assertEqual(completion.text, "hello world")
        self.assertEqual(completion.tokens_out, 2)


if __name__ == "__main__":
    unittest.main()
