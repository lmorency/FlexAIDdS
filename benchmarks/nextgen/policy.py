
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple

CLEAN_ROOM_POLICY_VERSION = "DockingAtHOME-NG-CR-1.0"
CLEAN_ROOM_POLICY_SUMMARY = (
    "Clean-room orchestration policy: external GPL or copyleft docking projects may be "
    "used only as high-level architectural inspiration and benchmark comparators; no code, "
    "configs, prompts, generated patches, or non-trivial implementation text may be copied, "
    "translated, paraphrased into code, or used as training material for this runner."
)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    retryable_error_substrings: Tuple[str, ...] = (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "resource busy",
        "no space left",
        "i/o error",
    )

    def is_retryable(self, error: str) -> bool:
        lowered = (error or "").lower()
        return any(token in lowered for token in self.retryable_error_substrings)


@dataclass(frozen=True)
class CleanRoomPolicy:
    version: str = CLEAN_ROOM_POLICY_VERSION
    name: str = "DockingAtHOME-inspired clean-room next-generation policy"
    source_reference: str = "OpenPeerAI/DockingAtHOME README"
    source_license: str = "GPL-3.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "source_reference": self.source_reference,
            "source_license": self.source_license,
            "summary": CLEAN_ROOM_POLICY_SUMMARY,
        }
