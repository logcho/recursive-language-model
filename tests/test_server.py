import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import json
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import app

class TestServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("server.get_model")
    @patch("server.RLMEngine")
    def test_run_endpoint_streaming(self, mock_engine_class, mock_get_model):
        """Verify that /api/run accepts form data, triggers RLMEngine, and streams events."""
        # Mock engine instance and run method
        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        
        # When engine.run is called, simulate sending a callback event and then returning
        def mock_run(query, context):
            # The engine calls callback in real runs; we simulate that by calling the callback parameter passed in init
            init_args, init_kwargs = mock_engine_class.call_args
            callback = init_kwargs.get("callback")
            if callback:
                callback({"type": "test_event", "data": "hello"})
            return "Final mock response"
            
        mock_engine.run.side_effect = mock_run

        response = self.client.post(
            "/api/run",
            data={
                "query": "Test query",
                "provider": "mock",
                "model_name": "gpt-4o-mini",
                "max_depth": 3,
                "max_steps": 10,
                "context_text": "Sample document content here."
            }
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        
        # Decode streaming events
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                event_data = json.loads(line[6:])
                events.append(event_data)
                
        # We expect a custom test_event and then a complete event
        self.assertTrue(len(events) >= 2)
        self.assertEqual(events[0], {"type": "test_event", "data": "hello"})
        self.assertEqual(events[1], {"type": "complete", "final_answer": "Final mock response"})

if __name__ == "__main__":
    unittest.main()
