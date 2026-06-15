import textwrap
from typing import Any

# Define a local/dummy QueryMetadata to prevent ImportError if not existing
class QueryMetadata:
    def __init__(self, context_type: str = "document", context_total_length: int = 0):
        self.context_type = context_type
        self.context_total_length = context_total_length

RLM_SYSTEM_PROMPT = textwrap.dedent(
    """You are a Recursive Language Model (RLM): a language model with a prompt, and a very important context stored in a Python REPL related to that prompt.
You can iteratively interact with the a Python REPL, which has access to LLM calls as a function. You will be queried turn-by-turn until you have an answer to the query.

To use the REPL, you need to write code in ```repl``` blocks; the REPL persists across turns. Available in the REPL:
- `context`: the important, potentially very long information related to the prompt (typically `str` or `list[str]`).
- `llm_query(prompt: str, model: str | None = None) -> str`: a single sub-LLM completion. Use for extraction, summarization, or Q&A over a chunk of text. Sub-LLM context window ≈ 500K chars.
- `llm_query_batched(prompts: list[str], model=None) -> list[str]`: concurrently call several LLM calls in parallel over a list of prompts; same order out as in.
- `rlm_query(prompt, model=None)` / `rlm_query_batched(prompts, model=None)`: recursive RLM sub-calls. Fall back to `llm_query` / `llm_query_batched` when recursion is disabled.
- `SHOW_VARS() -> str`: list every variable currently in the REPL.
- `answer`: dict initialized to `{{"content": "", "ready": False}}`. To submit, set `answer["content"]` to the final answer and `answer["ready"] = True` inside a ```repl``` block.
{custom_tools_section}

REPL outputs over ~20K characters are truncated, so for longer payloads slice `context` and pass slices through `llm_query` rather than `print`-ing them whole. The REPL is NOT a Jupyter cell — only `print(...)` output (stdout) is shown back to you between turns; a bare expression on the last line is silently discarded. Always wrap inspections in `print(...)`.

As a general strategy, you should start by probing your context to understand it better (e.g. print a few lines, count them, etc.). Then, use the REPL to build up an answer to the query.

Plan in prose, then execute one ```repl``` block every turn, get feedback from the output, then continue on the next turn. Do not flip `answer["ready"] = True` on turn 1 without first inspecting `context`.
"""
)


ORCHESTRATOR_ADDENDUM = "\n\n".join(
    [
        "As an RLM, you should act as an orchestrator, not a solver.",
        (
            "Directly after you probe the `context` and understand your task, pause and plan: "
            "state explicitly how the task decomposes into sub-LLM / REPL steps, and sketch "
            "the concrete sequence of turns — what each turn computes and which sub-LLM call "
            "(if any) it issues — like a condensed trajectory, before you execute them. "
            "Then execute one turn at a time: after each step `print` a small sample of the "
            'result, verify it looks right, and only flip `answer["ready"] = True` once you '
            "have actually printed the candidate answer. If you are running out of turns "
            "without a confirmed answer, submit your best inference rather than letting the "
            "rollout terminate unsubmitted."
        ),
        (
            "Your own context window is small. Push every long-context operation that would "
            "not fit comfortably in your own working window — reading, summarizing, "
            "classifying, verifying, answering sub-questions, even recapping your own "
            "progress — into `llm_query` / `llm_query_batched` calls instead of pulling that "
            "text into your own message stream. (Conversely: if a Python keyword / regex "
            "search over `context` would already pin the answer, or if a single visible "
            "passage already contains it, just read it directly — sub-LMs are for when the "
            "raw text won't fit or the question needs semantic interpretation.) Long REPL "
            "stdout pollutes history the same way raw `context` does: if you want a recap, "
            "ask `llm_query` for a 1–2 sentence summary and `print` only that. Aggregate "
            "the small results back in the REPL."
        ),
        (
            "Sub-LLMs have no REPL; they only see the prompt and the `context` slice you pass "
            "them. Hand them clean, focused inputs and ask for terse, structured outputs you "
            "can manipulate programmatically."
        ),
        (
            "Sub-call budget is finite on two independent axes, and `llm_query_batched` only "
            "parallelizes — it does not relax either. (1) Per-prompt capacity: a single "
            "sub-call answers well only when its input stays modestly sized — a useful rough "
            "ceiling is ~100K characters per prompt, less when the text is dense. Pack each "
            "prompt close to that capacity (a chunk of many items, a whole document) so one "
            "call accomplishes a lot of work. (2) Per-batch fan-out: `llm_query_batched` "
            "concurrency is bounded too — a useful rough ceiling is ~20 prompts per batch. "
            "Tiny-prompt mega-batches (hundreds or thousands of single-item prompts) are the "
            "anti-pattern; fat-prompt small batches are correct. For many independent units, "
            "use several ~20-wide batches of full-capacity prompts in sequence, not one "
            "mega-batch of tiny prompts. When the work can be expressed either as a "
            "sequential loop of `llm_query`s or as one comparably-sized batched call, "
            "prefer batched — same total work, far fewer turns burned. After Python-side "
            "filtering has narrowed the candidate set, batch-extract the survivors rather "
            "than reading them by hand. If the raw workload exceeds both budgets at once "
            "(e.g. a context far larger than ~20 × 100K chars), don't brute-force it: "
            "filter aggressively in Python first to a tractable subset, or stage the task — "
            "a cheap coarse pass narrows candidates, then a targeted second pass extracts "
            "from the survivors."
        ),
        (
            "Reserve your own tokens for high-level decisions: what to ask next, how to combine "
            "sub-LM outputs, when to finalize. Delegate everything else."
        ),
    ]
)


_DEFAULT_MAX_ITERATIONS = 30


from typing import Optional, Dict, List

def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: QueryMetadata,
    custom_tools: Optional[Dict[str, Any]] = None,
    root_prompt: Optional[str] = None,
    orchestrator: bool = True,
) -> List[Dict[str, str]]:
    try:
        from rlm.environments.base_env import format_tools_for_prompt
        tools_formatted = format_tools_for_prompt(custom_tools)
    except ImportError:
        def format_tools_for_prompt(tools):
            if not tools:
                return ""
            return "\n".join(f"- `{k}`: {v}" for k, v in tools.items())
        tools_formatted = format_tools_for_prompt(custom_tools)

    if tools_formatted:
        custom_tools_section = (
            f"\n6. Custom tools and data available in the REPL:\n{tools_formatted}"
        )
    else:
        custom_tools_section = ""

    final_system_prompt = system_prompt.format(custom_tools_section=custom_tools_section)
    if orchestrator:
        final_system_prompt = f"{final_system_prompt}\n\n{ORCHESTRATOR_ADDENDUM}"

    metadata_body = (
        f"Your context is a {query_metadata.context_type} of "
        f"{query_metadata.context_total_length} total characters. "
        "Each sub-LLM call can handle roughly ~100k tokens at once."
    )
    if root_prompt:
        metadata_prompt = f"Answer the following: {root_prompt}\n\n{metadata_body}"
    else:
        metadata_prompt = metadata_body

    return [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": metadata_prompt},
    ]


USER_PROMPT = "Turn {iter_1}/{max_iter}:"


def build_user_prompt(
    root_prompt: Optional[str] = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> Dict[str, str]:
    iter_1 = iteration + 1
    body = USER_PROMPT.format(iter_1=iter_1, max_iter=max_iterations)
    if iteration == 0:
        safeguard = (
            "You have not interacted with the REPL environment or seen your prompt / context "
            "yet. Look at the context first; do not provide a final answer yet.\n\n"
        )
        prompt = safeguard + body
    else:
        prompt = body

    if context_count > 1:
        prompt += (
            f"\n\nNote: You have {context_count} contexts available "
            f"(context_0 through context_{context_count - 1})."
        )
    if history_count > 0:
        if history_count == 1:
            prompt += (
                "\n\nNote: You have 1 prior conversation history available in the `history` "
                "variable."
            )
        else:
            prompt += (
                f"\n\nNote: You have {history_count} prior conversation histories available "
                f"(history_0 through history_{history_count - 1})."
            )
    return {"role": "user", "content": prompt}


# Pipeline compatibility definitions for LangGraph execution machine:
SYSTEM_INSTRUCTION = (RLM_SYSTEM_PROMPT.format(custom_tools_section="") + "\n\n" + ORCHESTRATOR_ADDENDUM).replace("{", "{{").replace("}", "}}") + """

---
#### CURRENT STATE METADATA
- Context Length: {context_len} characters.
- User Query: {query}
- Recursion Depth: {current_depth} / {max_depth}
- Execution Attempt: {current_retry} / {max_retries}

#### SANDBOX VARIABLES SNAPSHOT
The variables currently active in your sandbox session namespace:
{variables_summary} 
*(Formatted as: Variable Name | Type | Value Preview)*
"""

USER_PROMPT_PIPELINE = """Review the execution history, available variables, and stdout below. Decide on your next action.

You must follow the strict thought-action cadence. Choose exactly ONE of the following formats for your response:

### Option A: If you need to gather information or process data
<Thought>
Write your chain-of-thought reasoning here. State what data you are looking for, how your code will isolate it, and how you will force structured formatting from sub-calls if necessary.
</Thought>

```python
# Output a single python or repl block containing your execution script
# Ensure all semantic outputs are explicitly printed or stored to variables
# If you are done, store your final answer in answer["content"] and set answer["ready"] = True.
```

### Option B: If you are ready to conclude and output the final answer
<Thought>
Write your chain-of-thought reasoning here.
</Thought>

FINAL: <your final answer>

---
#### PREVIOUS EXECUTION LOGS
{history_logs}
"""