"""Agent-facing typed message contracts."""

from .envelopes import (
    AgentClarificationEnvelope,
    AgentFinalEnvelope,
    Envelope,
    ToolCallEnvelope,
    ToolResultEnvelope,
)
from .trajectory import TRAJECTORY_SCHEMA, validate_trajectory

__all__ = [
    "AgentClarificationEnvelope",
    "AgentFinalEnvelope",
    "Envelope",
    "ToolCallEnvelope",
    "ToolResultEnvelope",
    "TRAJECTORY_SCHEMA",
    "validate_trajectory",
]
