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

if __name__ == '__main__':
    unittest.main()
