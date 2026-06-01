import asyncio
import httpx
import os
from agentskg.agents.base import AgentConfig, BaseAgent
from agentskg.prompts.describe_prompt import DESCRIBE_RELATION_PROMPT


class RelationDescriptionAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(timeout=120.0)

    async def execute(self, context: str, relation: str) -> str:
        api_url = os.getenv("API_URL")
        api_key = os.getenv("API_KEY")
        if not api_url or not api_key:
            print("Error: RelationDescriptionAgent failed to load API_URL or API_KEY.")
            return None

        api_url = f"{api_url.rstrip('/')}/chat/completions"
        user_content = (
            f"{DESCRIBE_RELATION_PROMPT}\n"
            f"--- Input Triple ---\n{context}\n"
            f"--- Target ---\n{relation}"
        )
        payload = {
            "messages": [{"role": "user", "content": user_content}],
            "model": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            response = await self.client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            if "choices" in response_data and response_data["choices"]:
                message = response_data["choices"][0].get("message", {})
                content = message.get("content")
                if content:
                    return content.strip()
            return None

        except httpx.HTTPStatusError as e:
            print(
                f"HTTP error in RelationDescriptionAgent: {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            print(f"An unexpected error occurred in RelationDescriptionAgent: {e}")
            return None

    async def validate(self, result: str) -> bool:
        return isinstance(result, str) and bool(result)

    async def close(self):
        await self.client.aclose()
