import asyncio
import json
import os
import re
import httpx
from typing import List, Dict, Any, Optional
from agentskg.agents.base import AgentConfig, BaseAgent
from agentskg.prompts.verify_fact_prompt import VERIFY_OUTPUT_INSTRUCTION


class VerifyFactLLMAgent(BaseAgent):

    def __init__(self, config: AgentConfig, enabled: bool = True):
        self.config = config
        self.enabled = enabled
        self.client = httpx.AsyncClient(timeout=120.0)
        if enabled:
            self.agent_roles = [
                "You are a Knowledge Graph Auditor. Verify that the triplet is structurally sound: entities must be atomic (no long sentences), and the predicate must be a concise, valid property.",
                "You are a Domain Expert. Verify that the technical terms and concepts are used accurately and make logical sense within the given context.",
                "You are a strict Fact-Checker. Verify that the triplet is explicitly supported by the provided source text and contains NO hallucinations.",
            ]

    async def execute(self, triplets: List[Dict], original_text: str):
        batch_size = 10
        batches = [
            triplets[i : i + batch_size] for i in range(0, len(triplets), batch_size)
        ]

        final_results = []
        for agent_role in self.agent_roles:
            agent_results = []
            for batch in batches:
                triplets_json = json.dumps(batch, ensure_ascii=False, indent=2)
                user_content = (
                    f"Evaluate each triplet based on your assigned System Role.\n"
                    f"## Source Text\n"
                    f"{original_text}\n\n"
                    f"## Input Triplets\n"
                    f"{triplets_json}\n\n"                    
                    f"{VERIFY_OUTPUT_INSTRUCTION}"
                )
                prompt = {
                    "user_content": user_content,
                    "system_role": agent_role,
                }
                answer = await self.execute_prompt(prompt)
                if answer is None:
                    print("LLM response is empty, automatically marking this batch as all zeros")
                    agent_results.extend([0] * len(batch))
                    continue
                try:
                    cleaned_answer = (
                        answer.strip()
                        .replace("`", "")
                        .replace("[", "")
                        .replace("]", "")
                    )
                    results = [
                        int(x.strip())
                        for x in cleaned_answer.split(",")
                        if x.strip() in ["0", "1"]
                    ]
                    while len(results) < len(batch):
                        results.append(0)
                    results = results[: len(batch)]

                except (ValueError, AttributeError) as e:
                    print(f"Error parsing result: {e}")
                    results = [0] * len(batch)

                agent_results.extend(results)

            final_results.append(agent_results)

        return final_results

    async def execute_prompt(self, context):
        api_url = os.getenv("API_URL")
        api_key = os.getenv("API_KEY")
        if not api_url or not api_key:
            print(
                "Error: VerifyFactLLMAgent failed to load API_URL or API_KEY from environment variables."
            )
            print("Please check your .env file.")
            return None
        api_url = f"{api_url.rstrip('/')}/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": context["system_role"]},
                {"role": "user", "content": context["user_content"]},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "model": self.config.model_name,
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

            # Safely extract the content
            if "choices" in response_data and response_data["choices"]:
                message = response_data["choices"][0].get("message", {})
                content = message.get("content")
                if content:
                    return content

            print(f"Error: Invalid LLM response structure: {response_data}")
            return None

        except Exception as e:
            print(f"Verify Agent API Error: {e}")
            return None

    async def validate(self, result: List[List[int]]) -> bool:
        """
        Validates the output of the execute method.
        Checks if the result is a list of lists of integers (0 or 1).
        """
        if not isinstance(result, list):
            return False
        for sublist in result:
            if not isinstance(sublist, list):
                return False
            for item in sublist:
                if item not in [0, 1]:
                    return False
        return True

    async def close(self):
        await self.client.aclose()
        print("VerifyFactLLMAgent's HTTP client closed.")
