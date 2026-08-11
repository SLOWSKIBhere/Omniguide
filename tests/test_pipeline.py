import base64
import io
import os
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from PIL import Image

from agents.context import ContextBuilder
from agents.intent import IntentRouter
from agents.reasoning import ReasoningAgent
from agents.response import ResponseAgent
from models import IntentClassification, ScreenContext
from providers import (
    GeminiProvider,
    OpenAICompatibleProvider,
    ProviderCallError,
    ProviderResult,
    ProviderRouter,
    ProviderUnavailableError,
)


# requirements.txt installs google-genai, so tests must explicitly simulate
# the missing-dependency state instead of relying on the local environment.
@contextmanager
def _mock_missing_google_genai():
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                GeminiProvider,
                "dependency_ready",
                return_value=False,
                create=True,
            )
        )
        stack.enter_context(
            patch.object(
                GeminiProvider,
                "_sync_generate",
                side_effect=ProviderUnavailableError(
                    "google-genai is not installed"
                ),
            )
        )
        yield


class FakeProvider:
    def __init__(
        self,
        name,
        result=None,
        error=None,
        vision=True,
        text=True,
    ):
        self.name = name
        self.result = result
        self.error = error
        self.vision = vision
        self.text = text
        self.calls = 0

    def configured_for(self, *, vision):
        """Match the real provider configuration contract."""
        return self.vision if vision else self.text

    def available_for(self, *, vision):
        return self.configured_for(vision=vision)

    def describe(self):
        return {
            "name": self.name,
            "configured": True,
            "dependency_ready": True,
            "available": True,
        }

    async def generate_text(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return ProviderResult(
            text=self.result or "ok",
            tokens=7,
            provider=self.name,
            model="fake-model",
        )


class FailingVision:
    async def analyze(self, image):
        return (
            ScreenContext(),
            0,
            "vision down",
            {
                "stage": "vision",
                "status": "error",
                "error": "vision down",
            },
        )


class FailingOCR:
    async def extract(self, image):
        return (
            "",
            0,
            "ocr down",
            {
                "stage": "ocr",
                "status": "error",
                "error": "ocr down",
            },
        )


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_router_falls_through(self):
        first = FakeProvider(
            "first",
            error=ProviderCallError("quota"),
        )
        second = FakeProvider(
            "second",
            result="grounded answer",
        )
        router = ProviderRouter([first, second])

        result = await router.generate_text(
            system_prompt="s",
            user_prompt="u",
        )

        self.assertEqual(result.provider, "second")
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertTrue(result.attempts)

    async def test_provider_router_falls_through_on_invalid_json(self):
        first = FakeProvider("first", result="not JSON")
        second = FakeProvider(
            "second",
            result='{"answer": "grounded"}',
        )
        router = ProviderRouter([first, second])

        result = await router.generate_json(
            system_prompt="s",
            user_prompt="u",
        )

        self.assertEqual(result.provider, "second")
        self.assertEqual(result.data, {"answer": "grounded"})
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual(
            result.attempts,
            ["first: Provider returned non-JSON output"],
        )

    async def test_generate_text_reports_specific_dependency_error_not_generic_message(
        self,
    ):
        """A configured Gemini provider reports a missing SDK specifically."""
        os.environ["GEMINI_API_KEY"] = "fake-key-for-test"
        try:
            with _mock_missing_google_genai():
                gemini = GeminiProvider()
                self.assertTrue(gemini.is_configured())
                self.assertFalse(gemini.dependency_ready())

                fallback = FakeProvider(
                    "fallback",
                    result="grounded answer",
                )
                router = ProviderRouter([gemini, fallback])

                result = await router.generate_text(
                    system_prompt="s",
                    user_prompt="u",
                )

                self.assertEqual(result.provider, "fallback")
                self.assertEqual(fallback.calls, 1)
                self.assertEqual(len(result.attempts), 1)
                self.assertIn("gemini", result.attempts[0])
                self.assertIn(
                    "google-genai is not installed",
                    result.attempts[0],
                )
                self.assertNotIn(
                    "is not configured",
                    result.attempts[0],
                )
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

    async def test_generate_text_gemini_only_raises_specific_error_not_generic(
        self,
    ):
        """A Gemini-only configuration preserves the missing-SDK error."""
        os.environ["GEMINI_API_KEY"] = "fake-key-for-test"
        try:
            with _mock_missing_google_genai():
                gemini = GeminiProvider()
                router = ProviderRouter([gemini])

                with self.assertRaises(ProviderCallError) as ctx:
                    await router.generate_text(
                        system_prompt="s",
                        user_prompt="u",
                    )

                message = str(ctx.exception)
                self.assertIn(
                    "google-genai is not installed",
                    message,
                )
                self.assertNotIn(
                    "Set OPENAI_COMPAT_API_KEY or GEMINI_API_KEY",
                    message,
                )
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

    async def test_generate_json_reports_specific_dependency_error_not_generic_message(
        self,
    ):
        """The JSON path preserves the missing-SDK error during failover."""
        os.environ["GEMINI_API_KEY"] = "fake-key-for-test"
        try:
            with _mock_missing_google_genai():
                gemini = GeminiProvider()
                fallback = FakeProvider(
                    "fallback",
                    result='{"answer": "grounded"}',
                )
                router = ProviderRouter([gemini, fallback])

                result = await router.generate_json(
                    system_prompt="s",
                    user_prompt="u",
                )

                self.assertEqual(result.provider, "fallback")
                self.assertEqual(
                    result.data,
                    {"answer": "grounded"},
                )
                self.assertEqual(len(result.attempts), 1)
                self.assertIn(
                    "google-genai is not installed",
                    result.attempts[0],
                )
                self.assertNotIn(
                    "is not configured",
                    result.attempts[0],
                )
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

    async def test_context_is_not_grounded_when_evidence_agents_fail(
        self,
    ):
        builder = ContextBuilder(
            FailingVision(),
            FailingOCR(),
        )

        context, _, errors, chain, _ = await builder.build("abc")

        self.assertFalse(context.grounded)
        self.assertEqual(chain, [])
        self.assertEqual(len(errors), 2)

    async def test_reasoning_refuses_ungrounded_context(self):
        provider = FakeProvider(
            "unused",
            result="should never be called",
        )
        agent = ReasoningAgent(ProviderRouter([provider]))

        text, _, error, trace = await agent.reason(
            "fix it",
            ScreenContext(grounded=False),
            IntentClassification(),
        )

        self.assertEqual(text, "")
        self.assertIn("Grounding contract failed", error)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(trace["status"], "error")

    async def test_deterministic_intent_needs_no_model(self):
        os.environ["INTENT_ROUTER_MODE"] = "deterministic"
        provider = FakeProvider("unused", result="{}")
        router = IntentRouter(ProviderRouter([provider]))

        intent, tokens, error, trace = await router.classify(
            "Why is this error crashing?",
            ScreenContext(),
        )

        self.assertEqual(
            intent.intent_type.value,
            "debug_help",
        )
        self.assertEqual(tokens, 0)
        self.assertIsNone(error)
        self.assertEqual(provider.calls, 0)
        self.assertEqual(trace["provider"], "deterministic")

    async def test_response_never_labels_unverified_text_as_answer(
        self,
    ):
        result = ResponseAgent.build(
            run_id="r1",
            response_text="plausible but ungrounded model answer",
            context=ScreenContext(grounded=False),
            intent=IntentClassification(),
            latency_ms=1,
            tokens=0,
            errors=["vision failed"],
            agent_chain=[],
            traces=[
                {
                    "stage": "reasoning",
                    "status": "blocked",
                }
            ],
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn(
            "plausible but ungrounded",
            result["response"],
        )

    async def test_unverified_response_confidence_is_zero(self):
        result = ResponseAgent.build(
            run_id="r2",
            response_text="plausible but unverified model answer",
            context=ScreenContext(
                grounded=True,
                confidence=0.98,
            ),
            intent=IntentClassification(confidence=0.99),
            latency_ms=1,
            tokens=7,
            errors=["reasoning failed"],
            agent_chain=["vision", "intent"],
            traces=[
                {
                    "stage": "reasoning",
                    "status": "error",
                }
            ],
        )

        self.assertFalse(result["verified"])
        self.assertEqual(result["confidence"], 0.0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("OPENAI_COMPAT_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)

        from fastapi.testclient import TestClient
        import main

        cls._patcher = patch.object(
            GeminiProvider,
            "dependency_ready",
            return_value=False,
            create=True,
        )
        cls._patcher.start()
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()

    def test_health_reports_no_provider_without_keys(self):
        data = self.client.get("/health").json()

        self.assertFalse(data["model_available"])
        self.assertFalse(data["provider_ready"])
        self.assertFalse(data["gemini_key_configured"])
        self.assertEqual(
            data["execution_contract"],
            "screen_grounded_only",
        )

    def test_health_distinguishes_configured_from_dependency_ready(
        self,
    ):
        """Health distinguishes credentials from runtime readiness."""
        os.environ["GEMINI_API_KEY"] = "fake-key-for-test"
        try:
            import importlib
            import main
            from fastapi.testclient import TestClient

            with patch.object(
                GeminiProvider,
                "dependency_ready",
                return_value=False,
                create=True,
            ):
                importlib.reload(main)
                client = TestClient(main.app)
                data = client.get("/health").json()

                gemini_entry = next(
                    provider
                    for provider in data["providers"]
                    if provider["name"] == "gemini"
                )

                self.assertTrue(gemini_entry["configured"])
                self.assertFalse(
                    gemini_entry["dependency_ready"],
                )
                self.assertFalse(gemini_entry["available"])
                self.assertTrue(data["gemini_key_configured"])
        finally:
            os.environ.pop("GEMINI_API_KEY", None)

            import importlib
            import main

            importlib.reload(main)

    def test_ask_returns_503_instead_of_fake_answer(self):
        image = Image.new("RGB", (40, 40), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode()

        response = self.client.post(
            "/ask",
            json={
                "image": encoded,
                "query": "What is on my screen?",
            },
        )

        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertFalse(data["verified"])
        self.assertFalse(data["grounded"])
        self.assertNotIn("I can see", data["response"])
        self.assertEqual(data["error"], data["response"])


if __name__ == "__main__":
    unittest.main()