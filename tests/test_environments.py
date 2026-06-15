import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import json
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import app
from rlm.engine import RLMEngine
from rlm.models import MockChatModel

class TestEnvironments(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_engine_init_with_local_env(self):
        """Verify that RLMEngine can be initialized with environment parameters."""
        mock_model = MockChatModel()
        engine = RLMEngine(
            model=mock_model,
            max_depth=1,
            max_steps=5,
            verbose=False,
            environment="local"
        )
        self.assertEqual(engine.environment, "local")

    @patch("server.get_model")
    @patch("server.RLMEngine")
    def test_run_endpoint_with_environment(self, mock_engine_class, mock_get_model):
        """Verify that /api/run accepts environment form parameter and passes it."""
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run.return_value = "Mocked final answer"

        response = self.client.post(
            "/api/run",
            data={
                "query": "Test query",
                "provider": "mock",
                "model_name": "gpt-4o-mini",
                "max_depth": 2,
                "max_steps": 5,
                "context_text": "Sample document content.",
                "environment": "ipython"
            }
        )

        self.assertEqual(response.status_code, 200)
        # Verify engine was initialized with environment="ipython"
        mock_engine_class.assert_called_once()
        _, kwargs = mock_engine_class.call_args
        self.assertEqual(kwargs.get("environment"), "ipython")

    def test_get_environment_local(self):
        """Verify that get_environment correctly returns a LocalREPL instance."""
        from rlm.environments import get_environment
        from rlm.environments.local_repl import LocalREPL
        
        env_kwargs = {
            "context_payload": "Hello from tests!",
            "depth": 1,
        }
        env = get_environment("local", env_kwargs)
        self.assertIsInstance(env, LocalREPL)
        self.assertEqual(env.locals.get("context"), "Hello from tests!")
        env.cleanup()

    def test_local_repl_execute_code(self):
        """Verify that LocalREPL executes code, maintains state, and outputs stdout."""
        from rlm.environments.local_repl import LocalREPL
        
        env = LocalREPL(context_payload="test_context")
        res1 = env.execute_code("x = 10\nprint('x is set')")
        self.assertEqual(res1.stdout.strip(), "x is set")
        self.assertEqual(env.locals.get("x"), 10)
        
        res2 = env.execute_code("x += 5\nprint(f'x is {x}')")
        self.assertEqual(res2.stdout.strip(), "x is 15")
        self.assertEqual(env.locals.get("x"), 15)
        
        env.cleanup()

    def test_sandbox_delegation_to_environment(self):
        """Verify that Sandbox delegates code execution to a passed environment."""
        from rlm.sandbox import Sandbox
        from rlm.environments.local_repl import LocalREPL
        
        env = LocalREPL(context_payload="test_context")
        sandbox = Sandbox(
            context="test_context",
            llm_query_fn=lambda q, s: "llm",
            rlm_query_fn=lambda q, s: "rlm",
            environment=env
        )
        
        res = sandbox.run_code("y = 100\nprint(f'y is {y}')")
        self.assertTrue(res["success"])
        self.assertEqual(res["stdout"].strip(), "y is 100")
        self.assertEqual(sandbox.local_vars.get("y"), 100)
        self.assertEqual(env.locals.get("y"), 100)
        
        env.cleanup()

if __name__ == "__main__":
    unittest.main()
