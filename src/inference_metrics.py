"""
벤치마크·논문용 지표 계산 (recovery, stall, throughput 구간, 공정성).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


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


def _stall_time_summary(stall_times_s: Sequence[float]) -> Dict[str, Any]:
    if not stall_times_s:
        return {"count": 0, "sum_s": 0.0, "mean_s": None, "max_s": None}
    xs = [float(x) for x in stall_times_s]
    s = sum(xs)
    return {
        "count": len(xs),
        "sum_s": round(s, 6),
        "mean_s": round(s / len(xs), 6),
        "max_s": round(max(xs), 6),
    }


def recovery_latency_summary(recovery_events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """rpc_transport.recovery_events 집계 (스텝 내 RPC 재시도~성공 구간)."""
    latencies: List[float] = []
    for ev in recovery_events:
        v = ev.get("recovery_latency_s")
        if v is not None:
            try:
                latencies.append(float(v))
            except (TypeError, ValueError):
                continue
    if not latencies:
        return {
            "count": 0,
            "sum_s": 0.0,
            "mean_s": None,
            "max_s": None,
            "min_s": None,
        }
    s = sum(latencies)
    return {
        "count": len(latencies),
        "sum_s": round(s, 6),
        "mean_s": round(s / len(latencies), 6),
        "max_s": round(max(latencies), 6),
        "min_s": round(min(latencies), 6),
    }


def decode_step_latency_recovery_stretch(
    *,
    per_decode_wall_time_s: Sequence[float],
    had_recovery_per_decode_step: Sequence[bool],
) -> Dict[str, Any]:
    """
    복구가 있었던 디코드 스텝 vs 없던 스텝의 end-to-end(클라이언트 관측) 디코드 스텝 지연 비교.
    """
    if len(per_decode_wall_time_s) != len(had_recovery_per_decode_step):
        return {"note": "per_decode_wall_time_s 와 had_recovery_per_decode_step 길이가 다릅니다."}
    with_r: List[float] = []
    without_r: List[float] = []
    for t, h in zip(per_decode_wall_time_s, had_recovery_per_decode_step):
        if h:
            with_r.append(float(t))
        else:
            without_r.append(float(t))
    mw = sum(with_r) / len(with_r) if with_r else None
    mo = sum(without_r) / len(without_r) if without_r else None
    ratio = None
    if mw is not None and mo is not None and mo > 1e-9:
        ratio = mw / mo
    return {
        "mean_decode_step_latency_with_recovery_s": round(mw, 6) if mw is not None else None,
        "mean_decode_step_latency_without_recovery_s": round(mo, 6) if mo is not None else None,
        "decode_latency_stretch_ratio_with_over_without_recovery": round(ratio, 6) if ratio is not None else None,
        "num_decode_steps_with_recovery": len(with_r),
        "num_decode_steps_without_recovery": len(without_r),
    }


def build_baseline_comparison(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """
    동일 워크로드로 저장한 베이스라인 JSON과 비교해 FT 오버헤드·LB 이득을 채움.
    - FT: baseline 이 fault_tolerance_enabled=false, current 가 true
    - LB: baseline 이 use_load_balancing=false, current 가 true
    """
    out: Dict[str, Any] = {}

    ft: Dict[str, Any] = {}
    if baseline.get("fault_tolerance_enabled") is False and current.get("fault_tolerance_enabled") is True:
        tb = float(baseline.get("comparison_end_to_end_time_s") or baseline.get("total_time_s") or 0.0)
        tc = float(current.get("comparison_end_to_end_time_s") or current.get("total_time_s") or 0.0)
        if tb > 1e-9:
            ft["relative_time_overhead_vs_no_ft"] = round((tc - tb) / tb, 6)
        ft["baseline_total_time_s"] = baseline.get("total_time_s")
        ft["current_total_time_s"] = current.get("total_time_s")
        e2e_b = baseline.get("end_to_end_tokens_per_s")
        e2e_c = current.get("end_to_end_tokens_per_s")
        if e2e_b is not None and e2e_c is not None:
            fb, fc = float(e2e_b), float(e2e_c)
            if fb > 1e-9:
                ft["end_to_end_throughput_ratio_ft_over_no_ft"] = round(fc / fb, 6)
    else:
        ft["note"] = (
            "FT 오버헤드는 베이스라인이 fault_tolerance_enabled=false 이고 "
            "현재 실행이 true일 때만 relative_time_overhead_vs_no_ft 등이 채워집니다."
        )
    out["fault_tolerance_overhead"] = ft

    lb: Dict[str, Any] = {}
    if baseline.get("use_load_balancing") is False and current.get("use_load_balancing") is True:
        thr_b = baseline.get("end_to_end_tokens_per_s")
        thr_c = current.get("end_to_end_tokens_per_s")
        if thr_b is not None and thr_c is not None:
            fb, fc = float(thr_b), float(thr_c)
            if fb > 1e-9:
                lb["throughput_gain_lb_over_no_lb"] = round(fc / fb, 6)
        lat_b = float(baseline.get("comparison_end_to_end_time_s") or baseline.get("total_time_s") or 0.0)
        lat_c = float(current.get("comparison_end_to_end_time_s") or current.get("total_time_s") or 0.0)
        if lat_b > 1e-9:
            lb["latency_reduction_fraction"] = round((lat_b - lat_c) / lat_b, 6)
        lb["baseline_total_time_s"] = baseline.get("total_time_s")
        lb["current_total_time_s"] = current.get("total_time_s")
    else:
        lb["note"] = (
            "LB 이득은 베이스라인이 use_load_balancing=false 이고 현재가 true일 때만 "
            "throughput_gain_lb_over_no_lb, latency_reduction_fraction 이 채워집니다."
        )
    out["load_balancing_gain"] = lb

    return out


def aggregate_trial_metrics(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """failover_trials 등 반복 실행 요약."""
    n = len(trials)
    ok = [t for t in trials if t.get("trial_completed") is True and not t.get("error")]
    k = len(ok)
    totals = [float(t["total_time_s"]) for t in ok if t.get("total_time_s") is not None]
    decodes = [float(t["decode_time_s"]) for t in ok if t.get("decode_time_s") is not None]
    out: Dict[str, Any] = {
        "n_trials": n,
        "completed_trials": k,
        "failover_success_rate": round(k / n, 6) if n else None,
        "note": (
            "failover_success_rate 는 예외 없이 끝난 트라이얼 비율입니다. "
            "강제 종료 실험 시 트라이얼 사이에 외부에서 스테이지를 kill 한 뒤 동일 옵션으로 N회 실행하면 됩니다."
        ),
    }
    if totals:
        out["average_end_to_end_latency_s"] = round(sum(totals) / len(totals), 6)
        out["stdev_end_to_end_latency_s"] = round(
            (sum((x - sum(totals) / len(totals)) ** 2 for x in totals) / len(totals)) ** 0.5, 6
        ) if len(totals) > 1 else 0.0
    else:
        out["average_end_to_end_latency_s"] = None
        out["stdev_end_to_end_latency_s"] = None
    if decodes:
        out["average_decode_latency_s"] = round(sum(decodes) / len(decodes), 6)
    else:
        out["average_decode_latency_s"] = None
    return out


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
        return {
            "had_any_recovery": False,
            "note": "no decode steps recorded",
            "steady_state_vs_post_failure": None,
            "token_stall_time_after_failure_summary_s": _stall_time_summary([]),
        }
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
    stall_summary = _stall_time_summary(stall_times_s)

    if first_recovery_idx is None:
        pre_tokens = n
        pre_time = sum(token_to_token_intervals_s)
        pre_tps = round(pre_tokens / pre_time, 4) if pre_time > 1e-9 else None
        return {
            "had_any_recovery": False,
            "first_recovery_decode_step_index": None,
            "steady_state_vs_post_failure": {
                "steady_state_tokens_per_s": pre_tps,
                "post_failure_tokens_per_s": None,
                "post_over_steady_throughput_ratio": None,
                "note": "디코드 중 FT 복구가 없어 장애 전후 처리량 비교를 할 수 없습니다.",
            },
            "pre_failure": {
                "decode_steps": pre_tokens,
                "wall_time_s": round(pre_time, 6),
                "tokens_per_s": pre_tps,
            },
            "post_first_recovery": None,
            "token_stall_times_after_failure_s": [round(x, 6) for x in stall_times_s],
            "token_stall_time_after_failure_summary_s": stall_summary,
        }

    pre_intervals = token_to_token_intervals_s[:first_recovery_idx]
    post_intervals = token_to_token_intervals_s[first_recovery_idx + 1 :]
    pre_time = sum(pre_intervals)
    post_time = sum(post_intervals)
    pre_steps = len(pre_intervals)
    post_steps = len(post_intervals)
    pre_tps = round(pre_steps / pre_time, 4) if pre_time > 1e-9 else None
    post_tps = round(post_steps / post_time, 4) if post_time > 1e-9 else None
    ratio = None
    if pre_tps is not None and post_tps is not None and pre_tps > 1e-9:
        ratio = round(float(post_tps) / float(pre_tps), 6)

    return {
        "had_any_recovery": True,
        "first_recovery_decode_step_index": first_recovery_idx,
        "steady_state_vs_post_failure": {
            "steady_state_tokens_per_s": pre_tps,
            "post_failure_tokens_per_s": post_tps,
            "post_over_steady_throughput_ratio": ratio,
            "note": "steady=첫 복구 이전 디코드 구간, post=첫 복구 직후~종료 구간의 토큰/wall 기준 처리량.",
        },
        "pre_failure": {
            "decode_steps": pre_steps,
            "wall_time_s": round(pre_time, 6),
            "tokens_per_s": pre_tps,
        },
        "post_first_recovery": {
            "decode_steps": post_steps,
            "wall_time_s": round(post_time, 6),
            "tokens_per_s": post_tps,
        },
        "token_stall_times_after_failure_s": [round(x, 6) for x in stall_times_s],
        "token_stall_time_after_failure_summary_s": stall_summary,
    }
