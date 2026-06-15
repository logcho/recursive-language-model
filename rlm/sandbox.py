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
        rlm_query_fn: Callable[[str, str], str],
        environment: Any = None,
    ):
        self.context = context
        self.llm_query_fn = llm_query_fn
        self.rlm_query_fn = rlm_query_fn
        self.environment = environment
        
        # Local state namespace that persists across runs
        self.local_vars: Dict[str, Any] = {
            "context": context,
            "llm_query": self._llm_query_wrapper,
            "rlm_query": self._rlm_query_wrapper,
            "llm_query_batched": self._llm_query_batched_wrapper,
            "rlm_query_batched": self._rlm_query_batched_wrapper,
            "get_logical_chunks": self.get_logical_chunks,
            "search_context": self.search_context,
            "SHOW_VARS": self.show_vars,
            "answer": {"content": "", "ready": False},
        }
        
        # Pre-import safe libraries for convenience
        import re
        import json
        import math
        self.local_vars["re"] = re
        self.local_vars["json"] = json
        self.local_vars["math"] = math

    def _llm_query_wrapper(self, query: str, text_slice: Any = "", *args, **kwargs) -> str:
        """Wrapper to direct script-level calls to the engine-level leaf callback."""
        if not isinstance(query, str):
            raise TypeError("llm_query requires at least a string query parameter.")
            
        if not isinstance(text_slice, str):
            text_slice = ""
            
        # Check for swapped arguments (query and text_slice)
        if text_slice and (
            (query in self.context and text_slice not in self.context) or \
            (query in self.context and text_slice in self.context and len(query) > len(text_slice) * 5)
        ):
            query, text_slice = text_slice, query
            sys.stderr.write("WARNING: Detected swapped arguments in llm_query function call (query was document text, text_slice was query string). Automatically corrected order.\n")
            sys.stderr.flush()
            
        return self.llm_query_fn(query, text_slice)

    def _rlm_query_wrapper(self, query: str, text_slice: Any = "", *args, **kwargs) -> str:
        """Wrapper to direct script-level calls to the engine-level branch callback."""
        if not isinstance(query, str):
            raise TypeError("rlm_query requires at least a string query parameter.")
            
        if not isinstance(text_slice, str):
            text_slice = ""
            
        # Check for swapped arguments (query and text_slice)
        if text_slice and (
            (query in self.context and text_slice not in self.context) or \
            (query in self.context and text_slice in self.context and len(query) > len(text_slice) * 5)
        ):
            query, text_slice = text_slice, query
            sys.stderr.write("WARNING: Detected swapped arguments in rlm_query function call (query was document text, text_slice was query string). Automatically corrected order.\n")
            sys.stderr.flush()
            
        return self.rlm_query_fn(query, text_slice)

    def _llm_query_batched_wrapper(self, prompts: list, *args, **kwargs) -> list:
        """Wrapper to run multiple llm_query calls concurrently."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(lambda p: self._llm_query_wrapper(p, "", *args, **kwargs), prompts))
        return results

    def _rlm_query_batched_wrapper(self, prompts: list, *args, **kwargs) -> list:
        """Wrapper to run multiple rlm_query calls concurrently."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(executor.map(lambda p: self._rlm_query_wrapper(p, "", *args, **kwargs), prompts))
        return results

    def show_vars(self, *args, **kwargs) -> str:
        """Wrapper for SHOW_VARS()."""
        return self.get_variables_summary()

    def run_code(self, code: str) -> Dict[str, Any]:
        """
        Executes a Python code string inside the persistent sandbox environment.
        Captures and redirects stdout and stderr, and maps exceptions.
        """
        if self.environment:
            res = self.environment.execute_code(code)
            # Sync locals back into self.local_vars
            self.local_vars.clear()
            if hasattr(res, "locals") and res.locals:
                self.local_vars.update(res.locals)
            
            # Make sure we preserve the helpers so they don't disappear from variables snapshot/usage
            self.local_vars["context"] = self.context
            self.local_vars["llm_query"] = self._llm_query_wrapper
            self.local_vars["rlm_query"] = self._rlm_query_wrapper
            self.local_vars["llm_query_batched"] = self._llm_query_batched_wrapper
            self.local_vars["get_logical_chunks"] = self.get_logical_chunks
            self.local_vars["search_context"] = self.search_context
            self.local_vars["SHOW_VARS"] = self.show_vars
            
            # Make sure answer is populated
            if getattr(res, "final_answer", None) is not None:
                self.local_vars["answer"] = {"content": res.final_answer, "ready": True}
            
            # Sync environment.locals if it exists and is a dictionary (local REPL case)
            if hasattr(self.environment, "locals") and isinstance(self.environment.locals, dict):
                for k, v in self.environment.locals.items():
                    if k not in self.local_vars:
                        self.local_vars[k] = v
            
            return {
                "success": res.stderr == "",
                "stdout": res.stdout,
                "stderr": res.stderr,
                "exception": None if res.stderr == "" else res.stderr,
                "traceback": None,
            }

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
            "llm_query_batched",
            "rlm_query_batched",
            "get_logical_chunks",
            "search_context",
            "SHOW_VARS",
            "answer",
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

    def get_logical_chunks(self) -> list:
        """
        Splits context into logical chunks.
        Target chunk size: ~8000 characters, trying to split on paragraph boundaries (\n\n) or newlines (\n).
        """
        chunks = []
        text = self.context
        target_size = 8000
        
        start = 0
        total_len = len(text)
        chunk_id = 0
        
        if total_len == 0:
            return []
            
        while start < total_len:
            end = min(start + target_size, total_len)
            if end < total_len:
                # Find the last paragraph/newline break to split cleanly
                last_double_newline = text.rfind("\n\n", start, end)
                if last_double_newline > start + (target_size // 2):
                    end = last_double_newline + 2
                else:
                    last_newline = text.rfind("\n", start, end)
                    if last_newline > start + (target_size // 2):
                        end = last_newline + 1
                    else:
                        last_space = text.rfind(" ", start, end)
                        if last_space > start + (target_size // 2):
                            end = last_space + 1
            
            chunk_text = text[start:end]
            preview = chunk_text[:100].replace("\n", " ").strip() + ("..." if len(chunk_text) > 100 else "")
            chunks.append({
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "preview": preview
            })
            start = end
            chunk_id += 1
            
        return chunks

    def search_context(self, pattern: str) -> list:
        """
        Returns character boundaries (start_index, end_index) where pattern or regex matches.
        """
        import re
        matches = []
        try:
            # Try regex search first
            for m in re.finditer(pattern, self.context, re.IGNORECASE):
                matches.append((m.start(), m.end()))
        except re.error:
            # Fallback to literal search if pattern is not valid regex
            start = 0
            while True:
                idx = self.context.lower().find(pattern.lower(), start)
                if idx == -1:
                    break
                matches.append((idx, idx + len(pattern)))
                start = idx + max(1, len(pattern))
        return matches
