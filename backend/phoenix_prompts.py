"""Phoenix prompt lifecycle management.

Handles prompt registration and retrieval for version control and A/B testing.
"""
import os
import logging
from typing import Dict, Any, Optional

from .prompts import (
    ROUTER_PROMPT, QUERY_REWRITER_PROMPT, ANSWER_PROMPT,
    CHITCHAT_PROMPT, UNSUPPORTED_PROMPT, VALIDATOR_PROMPT, CONTEXTUAL_PROMPT
)

logger = logging.getLogger(__name__)

PROMPT_REGISTRY = {
    "router": {
        "template": ROUTER_PROMPT,
        "description": "Classifies incoming queries by type"
    },
    "query-rewriter": {
        "template": QUERY_REWRITER_PROMPT,
        "description": "Optimizes queries for better retrieval"
    },
    "answer": {
        "template": ANSWER_PROMPT,
        "description": "Generates cited answers from context"
    },
    "chitchat": {
        "template": CHITCHAT_PROMPT,
        "description": "Handles casual conversation"
    },
    "unsupported": {
        "template": UNSUPPORTED_PROMPT,
        "description": "Responds to out-of-scope queries"
    },
    "validator": {
        "template": VALIDATOR_PROMPT,
        "description": "Checks answer grounding"
    },
    "contextual": {
        "template": CONTEXTUAL_PROMPT,
        "description": "Handles follow-up questions"
    }
}

_cache: Dict[str, Dict[str, Any]] = {}
_phoenix_client = None
_initialized = False


def _get_client():
    """Get Phoenix client for prompt management."""
    global _phoenix_client
    
    if _phoenix_client is not None:
        return _phoenix_client
    
    try:
        from phoenix.client import Client
        
        endpoint = os.getenv("PHOENIX_ENDPOINT", "http://phoenix:4317")
        http_endpoint = endpoint.replace(":4317", ":6006")
        
        _phoenix_client = Client(base_url=http_endpoint)
        logger.info(f"Connected to Phoenix at {http_endpoint}")
        return _phoenix_client
        
    except Exception as e:
        logger.warning(f"Phoenix client unavailable: {e}")
        return None


def init_prompts() -> None:
    """Register all prompts with Phoenix."""
    global _cache, _initialized
    
    if _initialized:
        return
    
    client = _get_client()
    
    for name, data in PROMPT_REGISTRY.items():
        _cache[name] = {
            "template": data["template"],
            "description": data["description"],
            "version": "1.0.0"
        }
        
        if client:
            try:
                from phoenix.client.types import PromptVersion
                
                # Phoenix requires PromptVersion with messages format
                client.prompts.create(
                    name=name,
                    version=PromptVersion(
                        [{"role": "user", "content": data["template"]}],
                        model_name="llama3.1",
                    ),
                    prompt_description=data["description"]
                )
                logger.info(f"Registered prompt: {name} v1.0.0")
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    logger.info(f"Prompt exists: {name}")
                else:
                    logger.debug(f"Could not register {name}: {e}")
        else:
            logger.info(f"Cached prompt: {name} v1.0.0")
    
    _initialized = True
    logger.info(f"Initialized {len(_cache)} prompts")


def get_prompt(name: str, **kwargs) -> str:
    """Get a formatted prompt by name.
    
    Tries Phoenix first, falls back to local cache.
    """
    if not _initialized:
        init_prompts()
    
    # Map internal names to Phoenix names (Phoenix doesn't allow underscores at start)
    phoenix_name = name.replace("_", "-")
    
    client = _get_client()
    template = None
    
    if client:
        try:
            prompt = client.prompts.get(prompt_identifier=phoenix_name)
            if prompt:
                # Extract template from prompt messages
                messages = getattr(prompt, 'messages', None)
                if messages and len(messages) > 0:
                    template = messages[0].get('content', None)
        except Exception:
            pass
    
    if template is None:
        if name not in _cache:
            raise ValueError(f"Unknown prompt: {name}")
        template = _cache[name]["template"]
    
    if kwargs:
        return template.format(**kwargs)
    return template


def get_all_prompts() -> Dict[str, Any]:
    """Get all registered prompts."""
    if not _initialized:
        init_prompts()
    return _cache
