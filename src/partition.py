"""
Compatibility shim: route all partition imports to the LLaMA-only implementation.
This avoids stale code paths that could pass missing position_ids and break rotary embedding.
"""

from .llama_partition import Stage0, StageSegment, StageLast, load_stage_model

__all__ = ["Stage0", "StageSegment", "StageLast", "load_stage_model"]
