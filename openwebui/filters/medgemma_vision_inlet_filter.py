"""
title: MedGemma Multimodal Vision Bridge Filter
author: spark
version: 1.0.0
description: >
    OpenWebUI Inlet Filter that intercepts image uploads (CT, MRI, X-rays) for text-only 
    models, routes them secretly to a serving instance of the multimodal `medgemma-4b-it`,
    and inserts the generated clinical-grade visual description into the text prompt.
    This prevents downstream API crashes on text-only models like your fine-tuned `medgemma-27b-text`.
"""

import base64
import requests
import logging
from typing import Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Filter:
    class Valves(BaseModel):
        # --- Servicing Endpoint Config ---
        vision_api_url: str = Field(
            default="http://10.5.10.100:8090/v1/chat/completions",
            description="The OpenAI-compatible chat completions endpoint running google/medgemma-4b-it."
        )
        vision_model_id: str = Field(
            default="medgemma-4b-it",
            description="The model identifier registered in the serving engine (e.g., vLLM or Ollama)."
        )
        vision_api_key: str = Field(
            default="",
            description="Bearer token/API key if authorization is enabled on the serving host."
        )

        # --- Operational Flags ---
        strip_original_images: bool = Field(
            default=True,
            description="Remove the raw image objects prior to forwarding context to the main text model to prevent API structural crashes."
        )
        enable_logging: bool = Field(
            default=True,
            description="Enable verbose step-by-step logs in the container daemon."
        )

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __request__=None,
    ) -> dict:
        if self.valves.enable_logging:
            logger.info(f"[MedGemmaBridge] Evaluating request for model: {body.get('model')}")

        messages = body.get("messages", [])
        if not messages:
            return body

        last_message = messages[-1]
        raw_content = last_message.get("content", "")
        images = last_message.get("images", [])

        # ── 1. Structural Parsing ──
        # Handle cases where upstream uses OpenAI multimodal array structure for content
        text_query = ""
        inline_images = []

        if isinstance(raw_content, list):
            for part in raw_content:
                if part.get("type") == "text":
                    text_query = part.get("text", "")
                elif part.get("type") == "image_url":
                    url_data = part.get("image_url", {}).get("url", "")
                    if url_data:
                        inline_images.append(url_data)
        else:
            text_query = str(raw_content)

        # Collect standard base64 structures attached in OpenWebUI's image field
        for img in images:
            img_url = img.get("url", "")
            if img_url:
                inline_images.append(img_url)

        if not inline_images:
            if self.valves.enable_logging:
                logger.info("[MedGemmaBridge] No medical visual uploads detected on current user turn.")
            return body

        logger.info(f"[MedGemmaBridge] Found {len(inline_images)} visual files attached. Invoking `medgemma-4b-it` pipeline...")

        # ── 3. Vision API Secret Routing ──
        # Ensure we hit the chat completions endpoint regardless of how the valves are configured
        api_url = self.valves.vision_api_url.strip()
        if not api_url.endswith("/chat/completions"):
            api_url = api_url.rstrip("/")
            if api_url.endswith("/v1"):
                api_url = f"{api_url}/chat/completions"
            else:
                api_url = f"{api_url}/v1/chat/completions"

        summaries = []
        headers = {
            "Content-Type": "application/json"
        }
        if self.valves.vision_api_key:
            headers["Authorization"] = f"Bearer {self.valves.vision_api_key}"

        for idx, img_url in enumerate(inline_images):
            try:
                # Build the multi-modal payload for the google/medgemma-4b-it assistant completion
                # We skip sending the system message here so that OpenWebUI or the serving host
                # can natively apply the preconfigured System Prompt configured for the 'cancer-image-analysis' model.
                payload = {
                    "model": self.valves.vision_model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please provide a complete visual report and clinical analysis of this image."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": img_url
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1024
                }

                if self.valves.enable_logging:
                    logger.info(f"[MedGemmaBridge] Submitting image #{idx+1} to vision serving host: {api_url}")

                response = requests.post(
                    api_url, 
                    json=payload, 
                    headers=headers, 
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    interpretation = result["choices"][0]["message"]["content"]
                    
                    summaries.append(
                        f"### [CLINICAL IMAGING ANALYSIS - DOCUMENT #{idx+1}]\n"
                        f"{interpretation.strip()}\n"
                    )
                else:
                    logger.error(f"[MedGemmaBridge] Vision service failed with status {response.status_code}: {response.text}")
                    summaries.append(f"### [CLINICAL IMAGING ANALYSIS - DOCUMENT #{idx+1} ERROR]\n*(Vision engine failed to process: Status {response.status_code})*\n")
            
            except Exception as ex:
                logger.error(f"[MedGemmaBridge] Connection exception while processing image: {str(ex)}")
                summaries.append(f"### [CLINICAL IMAGING ANALYSIS - DOCUMENT #{idx+1} FAILURE]\n*(System Exception: {str(ex)})*\n")

        # ── 3. Prompt Context Re-Writing ──
        if summaries:
            consolidated_report = "\n".join(summaries)
            updated_content = (
                f"{consolidated_report}"
                f"\n---\n"
                f"**Downstream Clinical Task:**\n"
                f"The user has supplied clinical imaging. Above is the clinical-grade image description/interpretation generated by the `medgemma-4b` multimodal sub-agent.\n"
                f"Using the findings described above, answer the patient's inquiry:\n"
                f"> {text_query}"
            )

            # Assign text prompt back to the last message turn
            last_message["content"] = updated_content

            # ── 4. Crash Prevention / Cleanup ──
            # Strip original multi-modal objects so the backend text model doesn't block/crash on images
            if self.valves.strip_original_images:
                if "images" in last_message:
                    last_message["images"] = []
                if isinstance(raw_content, list):
                    last_message["content"] = updated_content # Flatten structured JSON back to string content

            if self.valves.enable_logging:
                logger.info("[MedGemmaBridge] Context re-writing complete. Safely stripped visual payload and routed text-only summaries downstream.")

        return body
