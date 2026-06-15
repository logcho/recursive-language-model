import os
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class MockChatModel(BaseChatModel):
    """
    A Mock Chat Model used to simulate neural orchestrator decisions in tests.
    It matches queries using substrings and returns pre-configured sequences of responses.
    """
    responses: Dict[str, List[str]] = {}
    default_responses: List[str] = []
    
    # Store dynamic call counts mapped to a hash of messages or a count
    _query_counters: Dict[str, int] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._query_counters = {}

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Find the query in the messages (either User Query or Sub-Query labels)
        user_query = ""
        for msg in messages:
            if isinstance(msg.content, str):
                lines = msg.content.split("\n")
                for line in lines:
                    if "user query:" in line.lower() or "sub-query:" in line.lower():
                        user_query = line.split(":", 1)[1].strip()
                        break
                if user_query:
                    break
                    
        # Fall back to using the human message content directly if no headers found
        if not user_query:
            for msg in messages:
                if msg.type in ("human", "user"):
                    user_query = msg.content
                    break
        
        # Determine which response list to use
        response_list = self.default_responses
        matched_key = None
        
        for key, resp in self.responses.items():
            if key.lower() in user_query.lower():
                response_list = resp
                matched_key = key
                break
        
        # Determine the current turn index statelessly by counting assistant replies in the message history,
        # or checking execution log turn headers.
        turn_idx = sum(1 for msg in messages if msg.type in ("ai", "assistant"))
        if turn_idx == 0:
            for msg in messages:
                if msg.type in ("human", "user") and isinstance(msg.content, str):
                    turn_idx = max(turn_idx, msg.content.count("--- Turn "))

        if response_list and turn_idx < len(response_list):
            content = response_list[turn_idx]
        else:
            content = "FINAL: Simulation complete (no mock response configured)."

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

def get_model(provider: str, model_name: str, temperature: float = 0.0, **kwargs) -> BaseChatModel:
    """
    Factory function to retrieve a chat model from different providers.
    Supports 'mock', 'openai', 'anthropic', 'google'.
    """
    provider = provider.lower()
    if provider == "mock":
        return MockChatModel(**kwargs)
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature, **kwargs)
        
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature, **kwargs)
        
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, **kwargs)
        
    else:
        raise ValueError(f"Unsupported provider: {provider}")
