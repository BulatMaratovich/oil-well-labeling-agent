from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class AppConfig:
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-nemo")
    mistral_api_base: str = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
    mistral_timeout_sec: float = float(os.getenv("MISTRAL_TIMEOUT_SEC", "30"))

    @property
    def mistral_configured(self) -> bool:
        return bool(self.mistral_api_key.strip())

    @property
    def mistral_resolved_model(self) -> str:
        normalized = self.mistral_model.strip()
        aliases = {
            "mistral-nemo": "open-mistral-nemo-2407",
            "open-mistral-nemo": "open-mistral-nemo-2407",
            "nemo": "open-mistral-nemo-2407",
        }
        return aliases.get(normalized, normalized)


settings = AppConfig()
