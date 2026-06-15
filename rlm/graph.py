import re
from typing import Any, Dict, List, TypedDict, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from rlm.sandbox import Sandbox
# Prompts are imported below in orchestrator_node

def extract_token_usage(response) -> dict:
    """Extracts token usage details from a LangChain response message."""
    metadata = getattr(response, "response_metadata", {}) or {}
    
    # OpenAI format
    if "token_usage" in metadata:
        usage = metadata["token_usage"] or {}
        return {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0)
        }
    
    # Anthropic format
    if "usage" in metadata:
        usage = metadata["usage"] or {}
        p = usage.get("input_tokens", 0)
        c = usage.get("output_tokens", 0)
        return {
            "prompt": p,
            "completion": c,
            "total": p + c
        }
    
    # Google format
    if "usage_metadata" in metadata:
        usage = metadata["usage_metadata"] or {}
        p = usage.get("prompt_token_count", 0)
        c = usage.get("candidates_token_count", 0)
        return {
            "prompt": p,
            "completion": c,
            "total": p + c
        }
        
    return {"prompt": 0, "completion": 0, "total": 0}

class RLMState(TypedDict):
    query: str
    context_len: int
    sandbox: Sandbox
    messages: List[BaseMessage]
    history_logs: str
    step_count: int
    max_steps: int
    final_answer: Optional[str]
    status: str  # "running", "success", "error", "max_steps_reached"

def parse_code_block(text: str) -> Optional[str]:
    """Helper to extract the python/repl code block inside ```python ... ``` or ```repl ... ``` tags."""
    pattern = r"```(?:python|repl)\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_final_answer(text: str) -> Optional[str]:
    """Helper to extract final answer matching 'FINAL: <answer>'."""
    pattern = r"FINAL:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

from rlm.prompts import SYSTEM_INSTRUCTION, USER_PROMPT_PIPELINE, build_rlm_system_prompt, build_user_prompt, RLM_SYSTEM_PROMPT, QueryMetadata

def orchestrator_node(state: RLMState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Root Orchestrator Node.
    Compiles sandbox state metadata and execution logs, feeds them to the LLM,
    and returns the LLM's response.
    """
    model = config.get("configurable", {}).get("model")

    if not model:
        raise ValueError("Model must be provided in config['configurable']['model']")
        
    sandbox = state["sandbox"]
    variables_summary = sandbox.get_variables_summary()
    
    current_depth = getattr(sandbox, "current_depth", 0)
    max_depth = getattr(sandbox, "max_depth", 3)
    current_retry = state["step_count"] + 1
    max_retries = state["max_steps"]
    
    # Format system prompt with current sandbox state
    system_content = SYSTEM_INSTRUCTION.format(
        context_len=state["context_len"],
        query=state["query"],
        variables_summary=variables_summary,
        current_depth=current_depth,
        max_depth=max_depth,
        current_retry=current_retry,
        max_retries=max_retries
    )
    
    # Build messages list in conversational multi-turn format
    prompt_messages = list(state["messages"])
    if not prompt_messages:
        # Create QueryMetadata
        metadata = QueryMetadata(context_type="document", context_total_length=state["context_len"])
        initial_prompts = build_rlm_system_prompt(
            system_prompt=RLM_SYSTEM_PROMPT,
            query_metadata=metadata,
            root_prompt=state["query"],
            orchestrator=True
        )
        for p in initial_prompts:
            if p["role"] == "system":
                prompt_messages.append(SystemMessage(content=system_content))
            else:
                prompt_messages.append(HumanMessage(content=p["content"]))
        
        # Turn 1 user prompt
        turn_prompt = build_user_prompt(
            root_prompt=state["query"],
            iteration=0,
            max_iterations=state["max_steps"]
        )
        prompt_messages.append(HumanMessage(content=turn_prompt["content"]))
    else:
        # Replace the first message (SystemMessage) with the latest formatted system content
        prompt_messages[0] = SystemMessage(content=system_content)
        
        # Subsequent turns: append next user prompt (Turn i/N)
        turn_prompt = build_user_prompt(
            root_prompt=state["query"],
            iteration=state["step_count"],
            max_iterations=state["max_steps"]
        )
        prompt_messages.append(HumanMessage(content=turn_prompt["content"]))
    
    # Format user prompt with the execution logs from the history (kept for logs/compatibility)
    user_content = USER_PROMPT_PIPELINE.format(
        history_logs=state["history_logs"] if state["history_logs"] else "No previous code execution has occurred."
    )
    
    # Invoke Root LLM
    response = model.invoke(prompt_messages)
    
    # Print orchestrator response to terminal stdout for debugging
    import sys
    sys.__stdout__.write(f"\n--- Turn {current_retry} Orchestrator Response ---\n{response.content}\n--------------------------------------------\n")
    sys.__stdout__.flush()
    
    # Extract token usage details
    tokens = extract_token_usage(response)
    if tokens["total"] == 0:
        input_text = system_content + user_content
        output_text = response.content
        p = len(input_text) // 4
        c = len(output_text) // 4
        tokens = {
            "prompt": p,
            "completion": c,
            "total": p + c
        }
    
    # Update messages tracking
    updated_messages = list(prompt_messages)
    updated_messages.append(response)
    
    # Parse final answer immediately if it's there
    final_ans = parse_final_answer(response.content)
    
    if hasattr(sandbox, "callback") and sandbox.callback:
        sandbox.callback({
            "type": "orchestrator",
            "depth": getattr(sandbox, "current_depth", 0),
            "content": response.content,
            "final_answer": final_ans,
            "variables_summary": variables_summary,
            "tokens": tokens
        })
    
    return {
        "messages": updated_messages,
        "final_answer": final_ans,
        "status": "success" if final_ans else "running"
    }

def executor_node(state: RLMState) -> Dict[str, Any]:
    """
    Stateful Execution Node.
    Parses python code from the orchestrator, runs it inside the persistent sandbox,
    and logs stdout, stderr, or traceback back into the state history.
    """
    last_msg = state["messages"][-1].content
    code = parse_code_block(last_msg)
    
    sandbox = state["sandbox"]
    turn_num = state["step_count"] + 1
    new_logs = f"--- Turn {turn_num} Execution ---\n"
    
    repl_contents = []
    
    if code:
        new_logs += f"Executing Code:\n```python\n{code}\n```\n\n"
        # Run code in sandbox
        res = sandbox.run_code(code)
        
        # Print sandbox outputs to terminal stdout for debugging
        import sys
        sys.__stdout__.write(f"\n--- Turn {turn_num} Sandbox Execution ---\n")
        if res["stdout"]:
            sys.__stdout__.write(f"Stdout:\n{res['stdout']}\n")
        if res["stderr"]:
            sys.__stdout__.write(f"Stderr:\n{res['stderr']}\n")
        if res["exception"]:
            sys.__stdout__.write(f"Exception:\n{res['exception']}\n")
        sys.__stdout__.write("-----------------------------------------\n")
        sys.__stdout__.flush()
        
        # Log outputs
        if res["stdout"]:
            new_logs += f"Stdout:\n{res['stdout']}\n"
            repl_contents.append(f"Stdout:\n{res['stdout']}")
        if res["stderr"]:
            new_logs += f"Stderr:\n{res['stderr']}\n"
            repl_contents.append(f"Stderr:\n{res['stderr']}")
        
        if res["success"]:
            new_logs += "Execution Status: Success\n"
        else:
            new_logs += f"Execution Status: Failed with exception:\n{res['exception']}\n"
            repl_contents.append(f"Execution failed with exception:\n{res['exception']}")
            if res["traceback"]:
                new_logs += f"Traceback:\n{res['traceback']}\n"
                repl_contents.append(f"Traceback:\n{res['traceback']}")
                
        if hasattr(sandbox, "callback") and sandbox.callback:
            sandbox.callback({
                "type": "executor",
                "depth": getattr(sandbox, "current_depth", 0),
                "code": code,
                "stdout": res["stdout"],
                "stderr": res["stderr"],
                "success": res["success"],
                "exception": res["exception"]
            })
    else:
        err_msg = "Execution Status: Failed. No valid ```python ``` or ```repl ``` code block found in your previous response.\nPlease write Python code inside markdown tags or output FINAL: <your final answer>.\n"
        new_logs += err_msg
        repl_contents.append(err_msg)
        
        if hasattr(sandbox, "callback") and sandbox.callback:
            sandbox.callback({
                "type": "executor",
                "depth": getattr(sandbox, "current_depth", 0),
                "code": None,
                "stdout": "",
                "stderr": "",
                "success": False,
                "exception": "No valid ```python ``` or ```repl ``` code block found in previous response."
            })
        
    repl_text = "\n\n".join(repl_contents)
    if not repl_text:
        repl_text = "Executed successfully with no output."
        
    # Append to running history log
    updated_history = state["history_logs"]
    if updated_history:
        updated_history += "\n" + new_logs
    else:
        updated_history = new_logs
        
    # Check if sandbox has final answer set via the answer dictionary
    final_ans = state["final_answer"]
    if "answer" in sandbox.local_vars and isinstance(sandbox.local_vars["answer"], dict):
        if sandbox.local_vars["answer"].get("ready"):
            final_ans = str(sandbox.local_vars["answer"].get("content", ""))
            
    # Append to message list tracking for multi-turn chat history
    updated_messages = list(state["messages"])
    updated_messages.append(HumanMessage(content=repl_text))
        
    return {
        "messages": updated_messages,
        "history_logs": updated_history,
        "step_count": turn_num,
        "final_answer": final_ans,
        "status": "success" if final_ans is not None else "running"
    }

def should_continue(state: RLMState) -> str:
    """Routing function to decide between looping, success termination, or step limit exit."""
    if state["final_answer"] is not None:
        return "end"
    sandbox = state["sandbox"]
    if "answer" in sandbox.local_vars and isinstance(sandbox.local_vars["answer"], dict):
        if sandbox.local_vars["answer"].get("ready"):
            return "end"
    if state["step_count"] >= state["max_steps"]:
        return "end"
    return "continue"

def build_rlm_graph() -> StateGraph:
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(RLMState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("executor", executor_node)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Set edges
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue,
        {
            "continue": "executor",
            "end": END
        }
    )
    workflow.add_edge("executor", "orchestrator")
    
    return workflow
