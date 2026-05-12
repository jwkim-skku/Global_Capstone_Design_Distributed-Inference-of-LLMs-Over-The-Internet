"""
벤치마크·논문용 지표 계산 (recovery, stall, throughput 구간, 공정성).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def jain_fairness_index(counts: List[float]) -> float:
    """
    Jain's fairness index in [0, 1]; 1 = perfectly fair (all equal).
    Uses only strictly positive counts (idle servers excluded).
    """
    positives = [float(c) for c in counts if c and float(c) > 0]
    if len(positives) <= 1:
        return 1.0
    s = sum(positives)
    s2 = sum(c * c for c in positives)
    if s2 <= 0:
        return 1.0
    n = len(positives)
    return (s * s) / (n * s2)


def load_fairness_stats(counts_by_peer: Dict[str, int]) -> Dict[str, Any]:
    """클라이언트가 관측한 원격 forward 횟수로 공정성 지표."""
    counts = [int(v) for v in counts_by_peer.values()]
    total = sum(counts)
    positives = [c for c in counts if c > 0]
    if not positives:
        return {
            "total_forwards": 0,
            "num_peers_observed": 0,
            "jain_fairness_index": 1.0,
            "imbalance_ratio_max_min": None,
            "coefficient_of_variation": None,
        }
    jain = jain_fairness_index([float(x) for x in positives])
    mx, mn = max(positives), min(positives)
    imbalance = float(mx / mn) if mn > 0 else None
    mean = total / len(counts) if counts else 0.0
    var = sum((c - mean) ** 2 for c in counts) / len(counts) if counts else 0.0
    cv = (var**0.5 / mean) if mean > 1e-9 else None
    return {
        "total_forwards": total,
        "num_peers_observed": len([c for c in counts if c > 0]),
        "jain_fairness_index": round(jain, 6),
        "imbalance_ratio_max_min": round(imbalance, 4) if imbalance is not None else None,
        "coefficient_of_variation": round(cv, 4) if cv is not None else None,
    }


def throughput_segments_from_decode(
    *,
    token_to_token_intervals_s: List[float],
    had_recovery_per_decode_step: List[bool],
) -> Dict[str, Any]:
    """
    token_to_token_intervals_s[i]: i번째 디코드 스텝(토큰 i -> i+1) 구간의 경과 시간(초).
    had_recovery_per_decode_step[i]: 그 스텝에서 FT 복구가 발생했는지.

    첫 복구 이전을 pre_failure, 첫 복구 직후 구간부터 끝까지를 post_first_recovery 로 집계.
    복구가 없으면 post_first_recovery 는 비어 있음.
    """
    n = len(token_to_token_intervals_s)
    if n == 0:
        return {"had_any_recovery": False, "note": "no decode steps recorded"}
    if n != len(had_recovery_per_decode_step):
        raise ValueError("intervals and had_recovery must have same length")

    first_recovery_idx: Optional[int] = None
    for i, h in enumerate(had_recovery_per_decode_step):
        if h:
            first_recovery_idx = i
            break

    stall_times_s = [
        token_to_token_intervals_s[i]
        for i in range(n)
        if had_recovery_per_decode_step[i]
    ]

    if first_recovery_idx is None:
        pre_tokens = n
        pre_time = sum(token_to_token_intervals_s)
        return {
            "had_any_recovery": False,
            "first_recovery_decode_step_index": None,
            "pre_failure": {
                "decode_steps": pre_tokens,
                "wall_time_s": round(pre_time, 6),
                "tokens_per_s": round(pre_tokens / pre_time, 4) if pre_time > 1e-9 else None,
            },
            "post_first_recovery": None,
            "token_stall_times_after_failure_s": [round(x, 6) for x in stall_times_s],
        }

    pre_intervals = token_to_token_intervals_s[:first_recovery_idx]
    post_intervals = token_to_token_intervals_s[first_recovery_idx + 1 :]
    pre_time = sum(pre_intervals)
    post_time = sum(post_intervals)
    pre_steps = len(pre_intervals)
    post_steps = len(post_intervals)

    return {
        "had_any_recovery": True,
        "first_recovery_decode_step_index": first_recovery_idx,
        "pre_failure": {
            "decode_steps": pre_steps,
            "wall_time_s": round(pre_time, 6),
            "tokens_per_s": round(pre_steps / pre_time, 4) if pre_time > 1e-9 else None,
        },
        "post_first_recovery": {
            "decode_steps": post_steps,
            "wall_time_s": round(post_time, 6),
            "tokens_per_s": round(post_steps / post_time, 4) if post_time > 1e-9 else None,
        },
        "token_stall_times_after_failure_s": [round(x, 6) for x in stall_times_s],
    }
