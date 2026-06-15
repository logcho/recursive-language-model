import os
import argparse
from dotenv import load_dotenv

from rlm.models import get_model, MockChatModel
from rlm.engine import RLMEngine

def setup_simulation_mock() -> MockChatModel:
    """Configures a MockChatModel with response trees for a simulated document review."""
    mock = MockChatModel()
    mock.responses = {
        "Identify major projects": [
            # Root Orchestrator Turn 1
            "```python\n"
            "# We split the document into paragraphs to analyze projects\n"
            "paragraphs = [p.strip() for p in context.split('\\n\\n') if p.strip()]\n"
            "project_details = []\n"
            "for idx, p in enumerate(paragraphs):\n"
            "    # We recursively invoke a child RLM to summarize project names/budgets\n"
            "    summary = rlm_query('Identify project name and budget in text', p)\n"
            "    project_details.append(f'Chunk {idx+1}: {summary}')\n"
            "print('EXTRACTED_DATA:\\n' + '\\n'.join(project_details))\n"
            "```",
            # Root Orchestrator Turn 2
            "FINAL: The two major projects identified in the report are Project Apollo ($5M budget) and Project Titan ($12M budget)."
        ],
        "Identify project name and budget in text": [
            # Child Orchestrator (Depth 1) Turn 1
            "```python\n"
            "# Call leaf LLM model to query details\n"
            "name = llm_query('What is the project name?', context)\n"
            "budget = llm_query('What is the budget?', context)\n"
            "print(f'NAME: {name} | BUDGET: {budget}')\n"
            "```",
            # Child Orchestrator Turn 2
            "FINAL: Project Apollo ($5M) and Project Titan ($12M) found in text chunk."
        ]
    }
    
    # Configure leaf node returns
    mock.default_responses = [
        "FINAL: Project Apollo/Titan detail extracted."
    ]
    
    return mock

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Recursive Language Model (RLM) CLI - MIT CSAIL Architecture over Long Contexts"
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "openai", "anthropic", "google"],
        default="mock",
        help="LLM provider to use (default: mock simulation)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name for orchestrator and leaf (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run over the context"
    )
    parser.add_argument(
        "--context-file",
        type=str,
        help="Path to a text file containing the isolated context"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum recursive depth (default: 3)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps per loop execution (default: 10)"
    )
    parser.add_argument(
        "--environment",
        choices=["local", "ipython", "docker", "modal", "prime", "daytona", "e2b"],
        default="local",
        help="Sandbox environment to run code inside (default: local)"
    )
    
    args = parser.parse_args()
    
    # Check if we should run the default simulation demo
    if args.provider == "mock" and not args.query and not args.context_file:
        print("=== RUNNING RLM SIMULATION DEMO ===")
        print("No arguments provided. Running default nested simulation task...\n")
        
        sample_context = (
            "Annual Operations Summary Report:\n\n"
            "Project Apollo was approved in March. The engineering team has completed phase 1. "
            "The initial financial allocation is $5,000,000 for server deployment and testing.\n\n"
            "Project Titan is focused on database modernization. The budget is $12,000,000. "
            "Milestones include cloud migration and containerization."
        )
        sample_query = "Identify major projects and their budgets mentioned in the report."
        
        mock_model = setup_simulation_mock()
        
        engine = RLMEngine(
            model=mock_model,
            leaf_model=mock_model,
            max_depth=args.max_depth,
            max_steps=args.max_steps,
            verbose=True,
            environment=args.environment
        )
        
        result = engine.run(sample_query, sample_context)
        print("\n=== FINAL ANSWER ===")
        print(result)
        return
        
    # Validate arguments for real execution
    if not args.query:
        parser.error("--query is required when running in non-simulation mode.")
    if not args.context_file:
        parser.error("--context-file is required when running in non-simulation mode.")
        
    if not os.path.exists(args.context_file):
        print(f"Error: Context file '{args.context_file}' does not exist.")
        return
        
    if args.context_file.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(args.context_file)
            pages_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            context_content = "\n".join(pages_text)
        except ImportError:
            print("Error: pypdf is required to parse PDF files. Run 'pip install pypdf'.")
            return
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return
    else:
        with open(args.context_file, "r", encoding="utf-8") as f:
            context_content = f.read()
        
    print(f"=== RUNNING RLM RUNNER ===")
    print(f"Provider:   {args.provider}")
    print(f"Model:      {args.model}")
    print(f"Query:      {args.query}")
    print(f"Context:    {args.context_file} ({len(context_content)} characters)\n")
    
    # Load model
    try:
        model = get_model(args.provider, args.model)
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Make sure you have set the appropriate API keys in your environment or .env file.")
        return
        
    engine = RLMEngine(
        model=model,
        leaf_model=model,
        max_depth=args.max_depth,
        max_steps=args.max_steps,
        verbose=True,
        environment=args.environment
    )
    
    try:
        result = engine.run(args.query, context_content)
        print("\n=== FINAL ANSWER ===")
        print(result)
    except Exception as e:
        print(f"\nExecution error: {e}")

if __name__ == "__main__":
    main()
