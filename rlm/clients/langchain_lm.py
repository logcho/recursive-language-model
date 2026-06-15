from typing import Any, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

class LangChainLM(BaseLM):
    """
    Adapter client to run LangChain models as BaseLM classes.
    Makes them compatible with socket-based LMHandler.
    """
    def __init__(
        self,
        langchain_model: BaseChatModel,
        model_name: str,
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.langchain_model = langchain_model
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def _convert_prompt_to_messages(self, prompt: Union[str, list[dict[str, Any]]]):
        if isinstance(prompt, str):
            return [HumanMessage(content=prompt)]
        
        # Convert list of OpenAI-style dicts to LangChain message formats
        messages = []
        for msg in prompt:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def _track_cost(self, response) -> None:
        self.total_calls += 1
        from rlm.graph import extract_token_usage
        usage = extract_token_usage(response)
        
        self.last_prompt_tokens = usage.get("prompt", 0)
        self.last_completion_tokens = usage.get("completion", 0)
        
        # Fallback if 0 tokens extracted
        if self.last_prompt_tokens == 0 and self.last_completion_tokens == 0:
            self.last_prompt_tokens = 500
            self.last_completion_tokens = 100
            
        self.total_prompt_tokens += self.last_prompt_tokens
        self.total_completion_tokens += self.last_completion_tokens

    def completion(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = self._convert_prompt_to_messages(prompt)
        response = self.langchain_model.invoke(messages)
        self._track_cost(response)
        return response.content

    async def acompletion(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = self._convert_prompt_to_messages(prompt)
        response = await self.langchain_model.ainvoke(messages)
        self._track_cost(response)
        return response.content

    def get_usage_summary(self) -> UsageSummary:
        model_usage = ModelUsageSummary(
            total_calls=self.total_calls,
            total_input_tokens=self.total_prompt_tokens,
            total_output_tokens=self.total_completion_tokens,
            total_cost=0.0
        )
        return UsageSummary(model_usage_summaries={self.model_name: model_usage})

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            total_cost=0.0
        )
