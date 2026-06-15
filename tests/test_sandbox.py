import unittest
from rlm.sandbox import Sandbox

class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.context = "This is the raw context of length 32."
        self.llm_calls = []
        self.rlm_calls = []
        
        def dummy_llm(q, s):
            self.llm_calls.append((q, s))
            return f"llm_result_for_{q}"
            
        def dummy_rlm(q, s):
            self.rlm_calls.append((q, s))
            return f"rlm_result_for_{q}"
            
        self.sandbox = Sandbox(
            context=self.context,
            llm_query_fn=dummy_llm,
            rlm_query_fn=dummy_rlm
        )

    def test_initial_state(self):
        """Verify the sandbox starts with context and standard libraries bound."""
        self.assertEqual(self.sandbox.local_vars["context"], self.context)
        self.assertTrue("re" in self.sandbox.local_vars)
        self.assertTrue("json" in self.sandbox.local_vars)

    def test_scope_persistence(self):
        """Verify variables defined in one run persist into the next run."""
        # Turn 1: define a variable
        res1 = self.sandbox.run_code("my_counter = 42\nprint('Counter set')")
        self.assertTrue(res1["success"])
        self.assertEqual(res1["stdout"].strip(), "Counter set")
        self.assertEqual(self.sandbox.local_vars.get("my_counter"), 42)
        
        # Turn 2: modify and print the variable
        res2 = self.sandbox.run_code("my_counter += 10\nprint(f'Counter is now {my_counter}')")
        self.assertTrue(res2["success"])
        self.assertEqual(res2["stdout"].strip(), "Counter is now 52")
        self.assertEqual(self.sandbox.local_vars.get("my_counter"), 52)

    def test_stdout_stderr_capture(self):
        """Verify printing and error outputs are captured separately."""
        res = self.sandbox.run_code("import sys\nprint('Standard Output')\nprint('Standard Error', file=sys.stderr)")
        self.assertTrue(res["success"])
        self.assertEqual(res["stdout"].strip(), "Standard Output")
        self.assertEqual(res["stderr"].strip(), "Standard Error")

    def test_exception_handling(self):
        """Verify errors are gracefully caught, returning failure status and traceback."""
        res = self.sandbox.run_code("x = 10\ny = 0\nz = x / y")
        self.assertFalse(res["success"])
        self.assertIn("ZeroDivisionError: division by zero", res["exception"])
        self.assertIsNotNone(res["traceback"])
        self.assertIn("line 3", res["traceback"])
        self.assertIn("ZeroDivisionError", res["traceback"])

    def test_variables_summary(self):
        """Verify that user-defined variables are properly listed and system ones are ignored."""
        # Initially empty custom variables
        self.assertEqual(self.sandbox.get_variables_summary(), "No custom variables defined.")
        
        # Run code setting list and string
        self.sandbox.run_code("names = ['alice', 'bob']\ntitle = 'The Adventures of Tom Sawyer'\nnumber = 99")
        summary = self.sandbox.get_variables_summary()
        
        self.assertIn("- `names` (list): list (size: 2)", summary)
        self.assertIn("- `title` (str): 'The Adventures of Tom Sawyer'", summary)
        self.assertIn("- `number` (int): 99", summary)
        self.assertNotIn("context", summary)
        self.assertNotIn("llm_query", summary)

    def test_injected_callbacks(self):
        """Verify code execution can call llm_query and rlm_query hooks."""
        code = """
summary = llm_query("Summarize first sentence", context[:10])
print(f"Summary: {summary}")
tree_data = rlm_query("Compile a structure", context)
print(f"Tree: {tree_data}")
"""
        res = self.sandbox.run_code(code)
        self.assertTrue(res["success"])
        self.assertEqual(len(self.llm_calls), 1)
        self.assertEqual(self.llm_calls[0], ("Summarize first sentence", "This is th"))
        self.assertEqual(len(self.rlm_calls), 1)
        self.assertEqual(self.rlm_calls[0], ("Compile a structure", self.context))
        
        self.assertIn("Summary: llm_result_for_Summarize first sentence", res["stdout"])
        self.assertIn("Tree: rlm_result_for_Compile a structure", res["stdout"])

    def test_navigation_tools(self):
        """Verify get_logical_chunks and search_context functions inside sandbox execution."""
        context = "Chapter 1. This is the first section.\n\nChapter 2. This is the second section."
        sandbox = Sandbox(
            context=context,
            llm_query_fn=lambda q, s: "",
            rlm_query_fn=lambda q, s: ""
        )
        
        res_chunks = sandbox.run_code("chunks = get_logical_chunks()")
        self.assertTrue(res_chunks["success"])
        chunks = sandbox.local_vars.get("chunks")
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], 0)
        self.assertEqual(chunks[0]["start"], 0)
        self.assertEqual(chunks[0]["end"], len(context))
        self.assertIn("Chapter 1", chunks[0]["preview"])
        
        res_search = sandbox.run_code("matches = search_context('Chapter')")
        self.assertTrue(res_search["success"])
        matches = sandbox.local_vars.get("matches")
        self.assertIsNotNone(matches)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0], (0, 7))
        self.assertEqual(matches[1], (context.find("Chapter 2"), context.find("Chapter 2") + 7))

    def test_swapped_arguments(self):
        """Verify that llm_query and rlm_query detect and auto-correct swapped parameters."""
        context = "SIRA stands for something important."
        llm_args = []
        rlm_args = []
        
        sandbox = Sandbox(
            context=context,
            llm_query_fn=lambda q, s: (llm_args.append((q, s)) or "llm_ok"),
            rlm_query_fn=lambda q, s: (rlm_args.append((q, s)) or "rlm_ok")
        )
        
        res_llm = sandbox.run_code("res = llm_query(context, 'What does SIRA stand for?')")
        self.assertTrue(res_llm["success"])
        self.assertEqual(sandbox.local_vars.get("res"), "llm_ok")
        self.assertEqual(len(llm_args), 1)
        self.assertEqual(llm_args[0], ("What does SIRA stand for?", context))
        self.assertIn("WARNING: Detected swapped arguments in llm_query", res_llm["stderr"])
        
        res_rlm = sandbox.run_code("res2 = rlm_query(context, 'Describe SIRA')")
        self.assertTrue(res_rlm["success"])
        self.assertEqual(sandbox.local_vars.get("res2"), "rlm_ok")
        self.assertEqual(len(rlm_args), 1)
        self.assertEqual(rlm_args[0], ("Describe SIRA", context))
        self.assertIn("WARNING: Detected swapped arguments in rlm_query", res_rlm["stderr"])

    def test_batched_calls_and_answer_dict(self):
        """Verify new RLM prompt sandbox capabilities (batched queries, SHOW_VARS, answer dictionary)."""
        llm_args = []
        sandbox = Sandbox(
            context="SIRA stands for something important.",
            llm_query_fn=lambda q, s: (llm_args.append((q, s)) or f"res_{q}"),
            rlm_query_fn=lambda q, s: f"rlm_{q}"
        )
        
        # Test SHOW_VARS
        res_show = sandbox.run_code("vars_str = SHOW_VARS()\nmy_custom_var = 100")
        self.assertTrue(res_show["success"])
        self.assertIn("No custom variables defined.", sandbox.local_vars.get("vars_str"))
        
        # Test answer dictionary
        res_ans = sandbox.run_code("answer['content'] = 'Final synthesis'\nanswer['ready'] = True")
        self.assertTrue(res_ans["success"])
        self.assertEqual(sandbox.local_vars["answer"]["content"], "Final synthesis")
        self.assertTrue(sandbox.local_vars["answer"]["ready"])
        
        # Test batched query wrapper
        res_batch = sandbox.run_code("results = llm_query_batched(['query1', 'query2'])")
        self.assertTrue(res_batch["success"])
        self.assertEqual(sandbox.local_vars.get("results"), ["res_query1", "res_query2"])

if __name__ == '__main__':
    unittest.main()
