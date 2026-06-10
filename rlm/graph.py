import re
from typing import Any, Dict, List, TypedDict, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from rlm.sandbox import Sandbox
from rlm.prompts import SYSTEM_INSTRUCTION, USER_PROMPT

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
    """Helper to extract the python code block inside ```python ... ``` tags."""
    pattern = r"```python\s*(.*?)\s*```"
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
    
    # Format system prompt with current sandbox state
    system_content = SYSTEM_INSTRUCTION.format(
        context_len=state["context_len"],
        query=state["query"],
        variables_summary=variables_summary
    )
    
    # Format user prompt with the execution logs from the history
    user_content = USER_PROMPT.format(
        history_logs=state["history_logs"] if state["history_logs"] else "No previous code execution has occurred."
    )
    
    # Prepare message window
    prompt_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ]
    
    # Invoke Root LLM
    response = model.invoke(prompt_messages)
    
    # Update messages tracking
    updated_messages = list(state["messages"])
    updated_messages.append(response)
    
    # Parse final answer immediately if it's there
    final_ans = parse_final_answer(response.content)
    
    if hasattr(sandbox, "callback") and sandbox.callback:
        sandbox.callback({
            "type": "orchestrator",
            "depth": getattr(sandbox, "current_depth", 0),
            "content": response.content,
            "final_answer": final_ans,
            "variables_summary": variables_summary
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
    
    if code:
        new_logs += f"Executing Code:\n```python\n{code}\n```\n\n"
        # Run code in sandbox
        res = sandbox.run_code(code)
        
        # Log outputs
        if res["stdout"]:
            new_logs += f"Stdout:\n{res['stdout']}\n"
        if res["stderr"]:
            new_logs += f"Stderr:\n{res['stderr']}\n"
        
        if res["success"]:
            new_logs += "Execution Status: Success\n"
        else:
            new_logs += f"Execution Status: Failed with exception:\n{res['exception']}\n"
            if res["traceback"]:
                new_logs += f"Traceback:\n{res['traceback']}\n"
                
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
        new_logs += "Execution Status: Failed. No valid ```python ``` code block found in your previous response.\n"
        new_logs += "Please write Python code inside markdown tags or output FINAL: <your final answer>.\n"
        
        if hasattr(sandbox, "callback") and sandbox.callback:
            sandbox.callback({
                "type": "executor",
                "depth": getattr(sandbox, "current_depth", 0),
                "code": None,
                "stdout": "",
                "stderr": "",
                "success": False,
                "exception": "No valid ```python ``` code block found in previous response."
            })
        
    # Append to running history log
    updated_history = state["history_logs"]
    if updated_history:
        updated_history += "\n" + new_logs
    else:
        updated_history = new_logs
        
    return {
        "history_logs": updated_history,
        "step_count": turn_num
    }

def should_continue(state: RLMState) -> str:
    """Routing function to decide between looping, success termination, or step limit exit."""
    if state["final_answer"] is not None:
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
