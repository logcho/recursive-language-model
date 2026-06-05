import sys
import io
import traceback
from typing import Any, Dict, Callable

class Sandbox:
    """
    Stateful Python REPL sandbox.
    Keeps a persistent namespace across multiple execution turns, isolates the
    context, and exposes LLM worker APIs (llm_query, rlm_query) in its environment.
    """
    def __init__(
        self,
        context: str,
        llm_query_fn: Callable[[str, str], str],
        rlm_query_fn: Callable[[str, str], str]
    ):
        self.context = context
        self.llm_query_fn = llm_query_fn
        self.rlm_query_fn = rlm_query_fn
        
        # Local state namespace that persists across runs
        self.local_vars: Dict[str, Any] = {
            "context": context,
            "llm_query": self._llm_query_wrapper,
            "rlm_query": self._rlm_query_wrapper,
        }
        
        # Pre-import safe libraries for convenience
        import re
        import json
        import math
        self.local_vars["re"] = re
        self.local_vars["json"] = json
        self.local_vars["math"] = math

    def _llm_query_wrapper(self, query: str, text_slice: str) -> str:
        """Wrapper to direct script-level calls to the engine-level leaf callback."""
        if not isinstance(query, str) or not isinstance(text_slice, str):
            raise TypeError("llm_query requires arguments (query: str, text_slice: str)")
        return self.llm_query_fn(query, text_slice)

    def _rlm_query_wrapper(self, query: str, text_slice: str) -> str:
        """Wrapper to direct script-level calls to the engine-level branch callback."""
        if not isinstance(query, str) or not isinstance(text_slice, str):
            raise TypeError("rlm_query requires arguments (query: str, text_slice: str)")
        return self.rlm_query_fn(query, text_slice)

    def run_code(self, code: str) -> Dict[str, Any]:
        """
        Executes a Python code string inside the persistent sandbox environment.
        Captures and redirects stdout and stderr, and maps exceptions.
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        # Redirect standard streams
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        
        success = True
        exception_str = None
        tb_str = None
        
        try:
            # We execute in both globals and locals pointing to self.local_vars
            # to preserve state changes (variables, imports, function declarations)
            exec(code, self.local_vars, self.local_vars)
        except Exception as e:
            success = False
            exception_str = f"{type(e).__name__}: {str(e)}"
            # Truncate traceback to avoid leaking internal framework stacks
            tb_lines = traceback.format_exception(*sys.exc_info())
            # Clean up framework lines from traceback to focus on the user script
            filtered_tb = []
            for line in tb_lines:
                if "exec(code" in line or "sandbox.py" in line:
                    continue
                filtered_tb.append(line)
            tb_str = "".join(filtered_tb)
        finally:
            # Restore standard streams
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            
        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()
        
        return {
            "success": success,
            "stdout": stdout_val,
            "stderr": stderr_val,
            "exception": exception_str,
            "traceback": tb_str,
        }

    def get_variables_summary(self) -> str:
        """
        Returns a formatted Markdown summary of non-system variables defined in the sandbox.
        """
        summary_lines = []
        ignored_names = {
            "__builtins__",
            "context",
            "llm_query",
            "rlm_query",
            "re",
            "json",
            "math"
        }
        
        for name, value in self.local_vars.items():
            if name in ignored_names or name.startswith("_") or callable(value):
                continue
            
            val_type = type(value).__name__
            if isinstance(value, str):
                snippet = f"'{value[:50]}...'" if len(value) > 50 else f"'{value}'"
            elif isinstance(value, (list, dict, set, tuple)):
                snippet = f"{val_type} (size: {len(value)})"
            else:
                snippet = str(value)
                
            summary_lines.append(f"- `{name}` ({val_type}): {snippet}")
            
        if not summary_lines:
            return "No custom variables defined."
        return "\n".join(summary_lines)
