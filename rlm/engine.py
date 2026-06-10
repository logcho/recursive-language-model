from typing import Any, Dict, Optional, Callable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from rlm.sandbox import Sandbox
from rlm.graph import build_rlm_graph

class RLMEngine:
    """
    Recursive Language Model Engine.
    Sets up the stateful sandbox, configures the sub-routine callback tree (leaf/branch),
    compiles the LangGraph workflow, and runs queries recursively.
    """
    def __init__(
        self,
        model: BaseChatModel,
        leaf_model: Optional[BaseChatModel] = None,
        current_depth: int = 0,
        max_depth: int = 3,
        max_steps: int = 10,
        verbose: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.model = model
        self.leaf_model = leaf_model or model
        self.current_depth = current_depth
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.verbose = verbose
        self.callback = callback
        
        # Compile graph workflow once
        self.graph = build_rlm_graph().compile()

    def _log(self, message: str):
        """Prints a log message indented by the current recursion depth to the original terminal stdout."""
        if self.verbose:
            import sys
            indent = "  " * self.current_depth
            prefix = f"[Depth {self.current_depth}]"
            # Split messages to indent each line
            for line in message.split("\n"):
                sys.__stdout__.write(f"{indent}{prefix} {line}\n")
                sys.__stdout__.flush()

    def _llm_query_fn(self, query: str, text_slice: str) -> str:
        """Leaf Node callback (llm_query). Runs a single-turn LLM call on a text slice."""
        self._log(f"--- Leaf Node Call (llm_query) ---")
        self._log(f"Sub-Query: {query}")
        self._log(f"Text Slice Length: {len(text_slice)} chars")
        
        if self.callback:
            self.callback({
                "type": "leaf_start",
                "depth": self.current_depth,
                "query": query,
                "text_slice_len": len(text_slice)
            })
        
        messages = [
            SystemMessage(content=(
                "You are a fast, targeted helper leaf node in a Recursive Language Model pipeline. "
                "Answer the sub-query directly and accurately based on the provided text slice. "
                "Keep your response concise and focused only on answering the sub-query."
            )),
            HumanMessage(content=f"Sub-Query: {query}\n\nText Slice (Context):\n{text_slice}")
        ]
        
        response = self.leaf_model.invoke(messages)
        self._log(f"Leaf Node Response Summary: {response.content[:100]}...")
        
        # Extract token usage details
        from rlm.graph import extract_token_usage
        tokens = extract_token_usage(response)
        if tokens["total"] == 0:
            input_text = messages[0].content + messages[1].content
            output_text = response.content
            p = len(input_text) // 4
            c = len(output_text) // 4
            tokens = {
                "prompt": p,
                "completion": c,
                "total": p + c
            }
        
        if self.callback:
            self.callback({
                "type": "leaf_end",
                "depth": self.current_depth,
                "query": query,
                "response": response.content,
                "tokens": tokens
            })
            
        return response.content

    def _rlm_query_fn(self, query: str, text_slice: str) -> str:
        """Branch Node callback (rlm_query). Recursively launches a child sandbox loop."""
        self._log(f"--- Branch Node Call (rlm_query) ---")
        self._log(f"Sub-Query: {query}")
        self._log(f"Text Slice Length: {len(text_slice)} chars")
        
        if self.callback:
            self.callback({
                "type": "branch_start",
                "depth": self.current_depth,
                "query": query,
                "text_slice_len": len(text_slice)
            })
        
        if self.current_depth >= self.max_depth:
            self._log("Max depth limit reached! Falling back to flat leaf node (llm_query) instead of recursing.")
            res = self._llm_query_fn(query, text_slice)
            if self.callback:
                self.callback({
                    "type": "branch_end",
                    "depth": self.current_depth,
                    "query": query,
                    "response": res
                })
            return res
            
        # Spawn child engine with depth incremented
        child_engine = RLMEngine(
            model=self.model,
            leaf_model=self.leaf_model,
            current_depth=self.current_depth + 1,
            max_depth=self.max_depth,
            max_steps=self.max_steps,
            verbose=self.verbose,
            callback=self.callback
        )
        
        # Run child engine
        child_answer = child_engine.run(query, text_slice)
        self._log(f"Branch Node Recursive Answer Summary: {child_answer[:100]}...")
        
        if self.callback:
            self.callback({
                "type": "branch_end",
                "depth": self.current_depth,
                "query": query,
                "response": child_answer
            })
            
        return child_answer

    def run(self, query: str, context: str) -> str:
        """Runs the query over the isolated context using LangGraph."""
        self._log(f"Initializing Sandbox (Context Length: {len(context)} characters)")
        
        # Create persistent sandbox with callbacks
        sandbox = Sandbox(
            context=context,
            llm_query_fn=self._llm_query_fn,
            rlm_query_fn=self._rlm_query_fn
        )
        sandbox.callback = self.callback
        sandbox.current_depth = self.current_depth
        sandbox.max_depth = self.max_depth
        
        if self.callback:
            self.callback({
                "type": "engine_start",
                "depth": self.current_depth,
                "query": query,
                "context_len": len(context)
            })
        
        # Build initial state
        initial_state = {
            "query": query,
            "context_len": len(context),
            "sandbox": sandbox,
            "messages": [],
            "history_logs": "",
            "step_count": 0,
            "max_steps": self.max_steps,
            "final_answer": None,
            "status": "running"
        }
        
        self._log("Running neural orchestrator execution loop...")
        
        # Run state machine loop
        final_state = self.graph.invoke(
            initial_state,
            config={"configurable": {"model": self.model}}
        )
        
        # Handle termination results
        if final_state["final_answer"] is not None:
            self._log(f"Orchestration completed successfully.")
            if self.callback:
                self.callback({
                    "type": "engine_end",
                    "depth": self.current_depth,
                    "final_answer": final_state["final_answer"],
                    "success": True
                })
            return final_state["final_answer"]
        else:
            self._log(f"Orchestration exited without resolving (Step limit {self.max_steps} hit or error).")
            if self.callback:
                self.callback({
                    "type": "engine_end",
                    "depth": self.current_depth,
                    "final_answer": None,
                    "success": False,
                    "error": "Step limit exceeded"
                })
            # Fallback output
            return "Error: Could not resolve query. Step limit exceeded."
