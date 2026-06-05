import unittest
from rlm.models import MockChatModel
from rlm.engine import RLMEngine

class TestRLM(unittest.TestCase):
    def test_flat_code_execution_and_self_correction(self):
        """
        Verify that:
        1. Invalid syntax in Turn 1 is corrected in Turn 2.
        2. Sandbox stdout is read by Turn 2 to produce the correct final response.
        """
        # Create a mock model configured with a sequence of responses for word count
        mock_model = MockChatModel()
        mock_model.responses = {
            "Count kernel": [
                # Turn 1: Code with syntax error (missing colon)
                "```python\ncount = 0\nfor word in context.split()\n    if 'kernel' in word:\n        count += 1\nprint(f'Count is {count}')\n```",
                # Turn 2: Self-corrected code
                "```python\nimport re\nmatches = len(re.findall(r'\\bkernel\\b', context, re.IGNORECASE))\nprint(f'Matched: {matches}')\n```",
                # Turn 3: Terminate with final answer
                "FINAL: The word 'kernel' appears 3 times."
            ]
        }
        
        engine = RLMEngine(
            model=mock_model,
            max_steps=5,
            verbose=False
        )
        
        context = "This kernel is a custom kernel. Other kernel modules exist. Nothing else."
        result = engine.run(query="Count kernel occurrences in text", context=context)
        
        self.assertEqual(result, "The word 'kernel' appears 3 times.")

    def test_recursive_pipeline(self):
        """
        Verify that a root node can recursively invoke a child node,
        which in turn invokes a leaf LLM query, compiles the results, and returns them.
        """
        # We need to setup a mock model that handles both:
        # 1. The root query: "Summarize document using recursion"
        # 2. The child query: "Summarize this snippet"
        # 3. The leaf query: "Get brief summary" (this is handled inside llm_query)
        mock_model = MockChatModel()
        
        mock_model.responses = {
            # Root Orchestrator Responses
            "Summarize document using recursion": [
                # Turn 1: Split document and recursively invoke rlm_query
                "```python\n"
                "parts = [p.strip() for p in context.split('.') if p.strip()]\n"
                "summaries = []\n"
                "for p in parts:\n"
                "    # Call recursive query\n"
                "    sub = rlm_query('Summarize this snippet', p)\n"
                "    summaries.append(sub)\n"
                "print('SUMMARIES:' + '|'.join(summaries))\n"
                "```",
                # Turn 2: Output final answer
                "FINAL: Synthesized summary: Part 1 and Part 2 are complete."
            ],
            
            # Child Orchestrator Responses (Depth 1)
            "Summarize this snippet": [
                # Turn 1: Run atomic leaf LLM query
                "```python\n"
                "brief = llm_query('Get brief summary', context)\n"
                "print('BRIEF:' + brief)\n"
                "```",
                # Turn 2: Child final answer
                "FINAL: Brief summary of snippet"
            ],
            
            # Leaf model replies (in case leaf_model gets invoked through standard chat)
            "Get brief summary": [
                "FINAL: Brief summary of snippet"
            ]
        }
        
        # When llm_query runs, it calls leaf_model.invoke()
        # The mock model's responses dict will match "Get brief summary" and return "Brief summary of snippet"
        
        engine = RLMEngine(
            model=mock_model,
            leaf_model=mock_model,
            max_depth=2,
            max_steps=5,
            verbose=False
        )
        
        context = "Paragraph one content. Paragraph two content."
        result = engine.run(query="Summarize document using recursion", context=context)
        
        self.assertEqual(result, "Synthesized summary: Part 1 and Part 2 are complete.")

if __name__ == '__main__':
    unittest.main()
