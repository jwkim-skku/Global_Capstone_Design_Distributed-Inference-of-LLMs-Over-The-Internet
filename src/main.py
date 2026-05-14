import argparse
import asyncio
import json
import os
from uuid import uuid4
import time

import torch
from transformers import AutoTokenizer
import logging
from hivemind import DHT, get_dht_time
from hivemind.p2p import P2P
from hivemind.utils.logging import get_logger

# Import 경로 처리: 패키지로 실행되거나 직접 실행될 때 모두 지원
try:
    from .llama_partition import load_stage_model, Stage0, StageSegment, StageLast
    from .rpc_transport import RpcTransport
    from .rpc_handler import StageConnectionHandler
    from .inference_metrics import (
        aggregate_trial_metrics,
        build_baseline_comparison,
        decode_step_latency_recovery_stretch,
        recovery_latency_summary,
        throughput_segments_from_decode,
    )
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src.llama_partition import load_stage_model, Stage0, StageSegment, StageLast
    from src.rpc_transport import RpcTransport
    from src.rpc_handler import StageConnectionHandler
    from src.inference_metrics import (
        aggregate_trial_metrics,
        build_baseline_comparison,
        decode_step_latency_recovery_stretch,
        recovery_latency_summary,
        throughput_segments_from_decode,
    )

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)


def build_masks(seq_len: int, device, dtype=None):
    mask_dtype = dtype if dtype is not None else torch.float
    attn = torch.ones(1, seq_len, device=device, dtype=mask_dtype)
    pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    return attn, pos


def parse_splits(splits_str: str):
    return [int(x.strip()) for x in splits_str.split(",")]


def _format_initial_peers(dht_initial_peers: str) -> list:
    if not dht_initial_peers:
        return []
    peers = [p.strip() for p in dht_initial_peers.split(",") if p.strip()]
    return peers


def _get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@torch.inference_mode()
def run_rank0(args, device, splits):
    """Run Stage0 (client side)."""

    try:
        from transformers.cache_utils import Cache  # type: ignore
    except Exception:
        Cache = None

    def _describe_past(past):
        if past is None:
            return "None"
        if Cache is not None and isinstance(past, Cache):
            try:
                l0 = past.get_seq_length(0)
            except Exception:
                l0 = "err"
            return f"{type(past).__name__}(len={l0})"
        if isinstance(past, (list, tuple)):
            if len(past) == 0:
                return "empty tuple"
            first = past[0]
            if isinstance(first, (list, tuple)) and first and first[0] is not None:
                return f"tuple(len={len(past)}, first_shape={tuple(first[0].shape)})"
            return f"tuple(len={len(past)}, first={first})"
        return str(type(past))

    # ✅ 너의 4-stage splits 구조에서는 Stage0가 첫 구간(splits[0])까지 로컬로 계산하고,
    #    나머지 블록(splits[0]..total_blocks-1)을 petals:module 라우팅으로 처리
    stage0_end = splits[0]  # ✅ LB여도 Stage0는 첫 구간(splits[0])까지 로컬로 계산

    full = load_stage_model(args.model, device, role="stage0", end=stage0_end, dtype=args.dtype)
    s0 = Stage0(full, stage0_end).to(device)

    dht_peers = _format_initial_peers(args.dht_initial_peers)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # total_blocks 기본값 추정
    total_blocks = args.total_blocks
    if total_blocks is None:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(args.model)
            total_blocks = int(getattr(config, "num_hidden_layers", 32))
        except Exception:
            total_blocks = 32

    try:
        from .utils import normalize_cache
    except ImportError:
        from src.utils import normalize_cache

    n_trials = max(1, int(getattr(args, "failover_trials", 1) or 1))

    def _run_rank0_single_session(trial_index: int) -> dict:
        tx = None
        try:
            tx = RpcTransport(
                device=device,
                stage=0,
                dht_initial_peers=dht_peers,
                dht_port=args.dht_port,
                rpc_port=args.rpc_port,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stage_keys=[f"mini_petals:stage{i}" for i in range(1, 4)],
                routing=("module" if args.use_load_balancing else "stage"),
                model_name=args.model,
                total_blocks=total_blocks,
                start_block=stage0_end,
                fault_tolerance=not getattr(args, "no_fault_tolerance", False),
                enable_metrics=not getattr(args, "no_metrics", False),
            )

            prompt = args.prompt
            input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            L = input_ids.size(1)

            attn, pos = None, torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)

            past0 = None
            prompt_ids = input_ids.clone()

            t_prefill_start = time.perf_counter()

            hidden, past0 = s0(input_ids, pos, attn, past0, use_cache=True)
            past0 = normalize_cache(past0)

            session_id = str(uuid4())
            max_length = L + args.max_new_tokens

            tx.send_prefill(L, hidden, session_id=session_id, max_length=max_length)
            next_id = tx.recv_token()
            generated = [next_id]
            t_prefill_end = time.perf_counter()
            prefill_time = t_prefill_end - t_prefill_start

            t_after_last_token_wall = time.perf_counter()
            decode_token_to_token_intervals_s: list[float] = []
            had_recovery_per_decode_step: list[bool] = []

            logger.info(f"Prefill completed in {prefill_time:.3f}s")
            logger.info(f"Generated token: {next_id} ({tok.decode([next_id], skip_special_tokens=True)})")

            cur_len = L + 1
            t_decode_start = time.perf_counter()
            eos_token_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id

            consecutive_repeat_count = 0
            last_token = None
            decode_total_times: list[float] = []

            for _ in range(args.max_new_tokens - 1):
                need_full_recompute = (past0 is None) or (isinstance(past0, (list, tuple)) and len(past0) == 0)

                if need_full_recompute:
                    logger.warning("Stage0 cache missing; recomputing from full history.")
                    full_input = torch.cat([prompt_ids, torch.tensor([generated], device=device, dtype=torch.long)], dim=1)
                    full_pos = torch.arange(full_input.shape[1], device=device, dtype=torch.long).unsqueeze(0)
                    attn = None
                    hidden_full, past0 = s0(full_input, full_pos, attn, None, use_cache=True)
                    past0 = normalize_cache(past0)
                    hidden = hidden_full[:, -1:, :]
                else:
                    new_input = torch.tensor([[next_id]], device=device, dtype=torch.long)
                    if isinstance(past0, (list, tuple)) and isinstance(past0[0], (list, tuple)) and past0[0][0] is not None:
                        past_len = past0[0][0].shape[-2]
                    elif Cache is not None and isinstance(past0, Cache):
                        past_len = past0.get_seq_length(0)
                    else:
                        past_len = cur_len - 1
                        logger.warning(
                            f"Stage0 past tuple missing first entry, fallback past_len={past_len}, "
                            f"past summary: {_describe_past(past0)}"
                        )

                    attn = None
                    pos = torch.tensor([[past_len]], device=device, dtype=torch.long)
                    hidden, past0 = s0(new_input, pos, attn, past0, use_cache=True)
                    past0 = normalize_cache(past0)

                tx.send_decode_step(cur_len, hidden, session_id=session_id, max_length=max_length, generated_tokens=generated)
                next_id = tx.recv_token()
                t_now_wall = time.perf_counter()
                decode_token_to_token_intervals_s.append(t_now_wall - t_after_last_token_wall)
                had_recovery_per_decode_step.append(bool(getattr(tx, "last_decode_had_recovery", False)))
                t_after_last_token_wall = t_now_wall

                if eos_token_id is not None and next_id == eos_token_id:
                    logger.info("EOS token generated, stopping generation")
                    break

                if next_id == last_token:
                    consecutive_repeat_count += 1
                    if consecutive_repeat_count >= 5:
                        next_token_text = tok.decode([next_id], skip_special_tokens=True)
                        logger.warning(
                            f"Consecutive repetition detected (token {next_id}='{next_token_text}'), stopping generation"
                        )
                        break
                else:
                    consecutive_repeat_count = 0
                    last_token = next_id

                if tx.last_decode_total is not None:
                    decode_total_times.append(tx.last_decode_total)

                generated.append(next_id)
                cur_len += 1

            t_decode_end = time.perf_counter()
            decode_time = t_decode_end - t_decode_start
            total_time = t_decode_end - t_prefill_start

            generated_text = tok.decode(generated, skip_special_tokens=True)
            if n_trials == 1:
                print(f"\n{'='*80}")
                print(f"PROMPT: {prompt}")
                print(f"GENERATED: {generated_text}")
                print(f"{'='*80}\n")
            else:
                logger.info(f"Trial {trial_index + 1}/{n_trials} GENERATED ({len(generated)} tokens): {generated_text[:200]}...")

            logger.info(f"\nDecode completed in {decode_time:.3f}s")
            logger.info(f"Total time: {total_time:.3f}s")
            logger.info(f"TTFT (Time to First Token): {prefill_time:.3f}s")

            rpc_summary = tx.get_metrics_summary()
            recovery_events = rpc_summary.get("recovery_events") or []
            throughput_seg = throughput_segments_from_decode(
                token_to_token_intervals_s=decode_token_to_token_intervals_s,
                had_recovery_per_decode_step=had_recovery_per_decode_step,
            )

            metrics_report: dict = {
                "model": args.model,
                "use_load_balancing": bool(getattr(args, "use_load_balancing", False)),
                "fault_tolerance_enabled": not getattr(args, "no_fault_tolerance", False),
                "metrics_enabled": not getattr(args, "no_metrics", False),
                "trial_index": trial_index,
                "trial_completed": True,
                "prefill_time_s": round(prefill_time, 6),
                "decode_time_s": round(decode_time, 6),
                "total_time_s": round(total_time, 6),
                "end_to_end_latency_s": round(total_time, 6),
                "output_token_count": len(generated),
                "end_to_end_tokens_per_s": round(len(generated) / total_time, 6) if total_time > 1e-9 else None,
                "decode_only_tokens_per_s": round((len(generated) - 1) / decode_time, 6)
                if len(generated) > 1 and decode_time > 1e-9
                else None,
                "recovery_time_and_latency": {
                    "recovery_events_aggregate_s": recovery_latency_summary(recovery_events),
                    "last_prefill_recovery_latency_s": rpc_summary.get("last_prefill_recovery_latency_s"),
                    "last_decode_recovery_latency_s": rpc_summary.get("last_decode_recovery_latency_s"),
                    "last_prefill_had_recovery": rpc_summary.get("last_prefill_had_recovery"),
                    "last_decode_had_recovery": rpc_summary.get("last_decode_had_recovery"),
                    "note": (
                        "recovery_latency_s 는 RPC 실패 시점부터 같은 스텝에서 재시도로 성공할 때까지입니다. "
                        "디코드 스텝 전체 관점에서는 decode_step_recovery_vs_normal_latency 와 "
                        "per_decode_rpc_compute_latency_s 를 함께 보세요."
                    ),
                },
                "steady_state_and_post_failure_throughput": throughput_seg.get("steady_state_vs_post_failure"),
                "decode_step_recovery_vs_normal_latency": decode_step_latency_recovery_stretch(
                    per_decode_wall_time_s=decode_total_times,
                    had_recovery_per_decode_step=had_recovery_per_decode_step,
                ),
                "load_distribution_fairness": rpc_summary.get("load_fairness"),
                "rpc_transport": rpc_summary,
                "decode_token_to_token_wall": {
                    "intervals_s": [round(x, 6) for x in decode_token_to_token_intervals_s],
                    "had_recovery_per_step": had_recovery_per_decode_step,
                    "throughput_segments": throughput_seg,
                },
                "per_decode_rpc_compute_latency_s": [round(x, 6) for x in decode_total_times],
                "average_per_decode_rpc_latency_s": round(sum(decode_total_times) / len(decode_total_times), 6)
                if decode_total_times
                else None,
                "note_ft_overhead_ab": (
                    "장애가 없을 때 FT 순수 오버헤드: --baseline_metrics_json 에 "
                    "동일 조건의 --no_fault_tolerance 실행 결과를 넘기면 "
                    "baseline_comparison.fault_tolerance_overhead 가 채워집니다."
                ),
                "note_lb_gain_baseline": (
                    "LB 이득: 베이스라인 JSON(use_load_balancing=false) 경로를 "
                    "--baseline_metrics_json 으로 넘기면 baseline_comparison.load_balancing_gain 이 채워집니다."
                ),
                "generated_text_preview": (generated_text[:800] + "…") if len(generated_text) > 800 else generated_text,
            }
            return metrics_report
        except Exception as e:
            logger.exception("Stage0 inference trial %s failed", trial_index)
            return {
                "model": args.model,
                "use_load_balancing": bool(getattr(args, "use_load_balancing", False)),
                "fault_tolerance_enabled": not getattr(args, "no_fault_tolerance", False),
                "metrics_enabled": not getattr(args, "no_metrics", False),
                "trial_index": trial_index,
                "trial_completed": False,
                "error": str(e),
            }
        finally:
            if tx is not None:
                try:
                    tx.shutdown()
                except Exception as ex:
                    logger.warning("RpcTransport shutdown after trial: %s", ex)

    trial_reports: list[dict] = []
    for ti in range(n_trials):
        logger.info("Inference trial %s/%s", ti + 1, n_trials)
        trial_reports.append(_run_rank0_single_session(ti))

    successes = [t for t in trial_reports if t.get("trial_completed") is True]
    if not successes:
        raise RuntimeError("모든 inference trial 이 실패했습니다. 로그의 error 필드를 확인하세요.")

    metrics_report = dict(successes[-1])
    agg = aggregate_trial_metrics(trial_reports)
    metrics_report["failover_eval_trials"] = agg
    metrics_report["each_trial_compact"] = [
        {
            "trial_index": t.get("trial_index"),
            "trial_completed": t.get("trial_completed"),
            "total_time_s": t.get("total_time_s"),
            "recovery_event_count": (t.get("rpc_transport") or {}).get("recovery_event_count"),
            "error": t.get("error"),
        }
        for t in trial_reports
    ]
    avg_e2e = agg.get("average_end_to_end_latency_s")
    metrics_report["average_end_to_end_latency_s"] = avg_e2e
    metrics_report["comparison_end_to_end_time_s"] = avg_e2e if avg_e2e is not None else metrics_report.get("total_time_s")

    baseline_path = (getattr(args, "baseline_metrics_json", None) or "").strip()
    if baseline_path:
        try:
            with open(baseline_path, encoding="utf-8") as bf:
                baseline_obj = json.load(bf)
            metrics_report["baseline_comparison"] = build_baseline_comparison(metrics_report, baseline_obj)
        except Exception as e:
            logger.warning("baseline_metrics_json 로드 실패 (%s): %s", baseline_path, e)
            metrics_report["baseline_comparison"] = {"error": str(e), "path": baseline_path}

    metrics_report["failover_success_rate"] = agg.get("failover_success_rate")

    if n_trials > 1:
        prev = successes[-1].get("generated_text_preview") or ""
        print(
            f"\n{'='*80}\n(multi-trial) 마지막 성공 trial_index={successes[-1].get('trial_index')} "
            f"failover_success_rate={metrics_report.get('failover_success_rate')}\n"
            f"generated_text_preview: {prev[:400]}{'…' if len(prev) > 400 else ''}\n{'='*80}\n"
        )

    metrics_json = getattr(args, "metrics_json", None) or ""
    if metrics_json:
        out_path = metrics_json.strip()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote metrics JSON to {out_path}")

    print("\n" + "=" * 80)
    print("METRICS_SUMMARY (see also rpc_transport.recovery_events & load_fairness)")
    print(json.dumps(metrics_report, indent=2, ensure_ascii=False))
    print("=" * 80 + "\n")


@torch.inference_mode()
def run_stage_server(args, device, splits):
    """Run a server stage (1, 2, or 3) with optional Load Balancing."""
    use_load_balancing = getattr(args, 'use_load_balancing', False)
    num_blocks = getattr(args, 'num_blocks', None)
    total_blocks = getattr(args, 'total_blocks', None)

    if use_load_balancing:
        return run_stage_server_with_load_balancing(args, device, splits, num_blocks, total_blocks)

    return run_stage_server_fixed(args, device, splits)


def run_stage_server_fixed(args, device, splits):
    """Run a server stage with fixed splits (original implementation)."""
    use_cpu_offload = getattr(args, 'use_cpu_offload', False)
    keep_layers_on_gpu = getattr(args, 'keep_layers_on_gpu', 0)

    if args.stage == 1:
        start, end = splits[0], splits[1]
        full = load_stage_model(
            args.model, device, role="segment",
            start=start, end=end, dtype=args.dtype,
            use_cpu_offload=use_cpu_offload
        )
        stage_model = StageSegment(full, start, end, gpu_device=device, keep_layers_on_gpu=keep_layers_on_gpu)
        final_stage = False
    elif args.stage == 2:
        start, end = splits[1], splits[2]
        full = load_stage_model(
            args.model, device, role="segment",
            start=start, end=end, dtype=args.dtype,
            use_cpu_offload=use_cpu_offload
        )
        stage_model = StageSegment(full, start, end, gpu_device=device, keep_layers_on_gpu=keep_layers_on_gpu)
        final_stage = False
    elif args.stage == 3:
        start = splits[2]
        full = load_stage_model(
            args.model, device, role="last",
            start=start, dtype=args.dtype,
            use_cpu_offload=use_cpu_offload
        )
        stage_model = StageLast(full, start, gpu_device=device, keep_layers_on_gpu=keep_layers_on_gpu)
        final_stage = True
    else:
        raise ValueError("stage must be 1, 2, or 3 for server")

    return _setup_and_run_server(args, device, stage_model, final_stage)


def run_stage_server_with_load_balancing(args, device, splits, num_blocks, total_blocks):
    """Run a server stage with Load Balancing (논문식 Full Load Balancing)."""
    try:
        from .load_balancing import choose_best_blocks, should_choose_other_blocks, ServerState
        from .dht_utils import get_remote_module_infos
        from .throughput_measurement import get_server_throughput
    except ImportError as e:
        logger.error(f"Failed to import Load Balancing modules: {e}")
        logger.error("Falling back to fixed splits mode")
        return run_stage_server_fixed(args, device, splits)

    initial_peers_list = _format_initial_peers(args.dht_initial_peers)
    local_ip = _get_local_ip()
    announce_ip = args.public_ip if args.public_ip else local_ip
    public_dht_port = args.public_dht_port if args.public_dht_port is not None else args.dht_port

    if args.public_ip:
        host_maddrs = [f"/ip4/0.0.0.0/tcp/{args.dht_port}"]
        announce_maddrs = [f"/ip4/{args.public_ip}/tcp/{public_dht_port}"]
    else:
        host_maddrs = [f"/ip4/{local_ip}/tcp/{args.dht_port}"]
        announce_maddrs = None

    dht = DHT(
        start=True,
        initial_peers=initial_peers_list if initial_peers_list else None,
        host_maddrs=host_maddrs,
        announce_maddrs=announce_maddrs,
    )

    visible = dht.get_visible_maddrs()
    dht_peer_id = str(dht.peer_id)
    if args.public_ip:
        visible_str = [str(m) for m in visible] if visible else []
        has_public_ip = any(args.public_ip in str(m) for m in visible_str)
        if not has_public_ip:
            public_maddr = f"/ip4/{args.public_ip}/tcp/{public_dht_port}/p2p/{dht_peer_id}"
            logger.warning(
                f"DHT visible multiaddrs do not contain public IP {args.public_ip}. "
                f"Use this multiaddr for --dht_initial_peers: {public_maddr}"
            )
            logger.info(f"DHT visible multiaddrs (may contain private IP): {visible}")
        else:
            logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")
    elif visible:
        logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")

    if num_blocks is None:
        num_blocks = 4
    if total_blocks is None:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(args.model)
            total_blocks = int(getattr(config, "num_hidden_layers", 32))
        except Exception:
            total_blocks = 32

    # ✅ Stage0가 splits[0]까지 로컬로 계산하므로, LB 서버는 그 다음 블록부터만 담당
    lb_min_block = int(splits[0])

    balance_quality = getattr(args, 'balance_quality', 0.75)
    mean_balance_check_period = getattr(args, 'mean_balance_check_period', 120.0)

    while True:
        try:
            module_infos = []
            max_retries = 3
            retry_delay = 2.0

            for retry in range(max_retries):
                module_infos = get_remote_module_infos(dht, args.model, total_blocks)
                logger.info(f"Retrieved {len(module_infos)} module infos from DHT (attempt {retry+1}/{max_retries})")

                if len(module_infos) > 0 or retry == max_retries - 1:
                    break

                logger.info(f"No existing servers found, waiting {retry_delay}s for DHT propagation...")
                time.sleep(retry_delay)
                retry_delay *= 1.5

            if len(module_infos) == 0:
                start = lb_min_block
                end = min(lb_min_block + num_blocks, total_blocks)
                block_indices = list(range(start, end))
                logger.info(f"No existing servers found, selecting blocks from {start} to {end-1}: {block_indices}")
            else:
                block_indices = choose_best_blocks(num_blocks, module_infos, total_blocks, min_block=lb_min_block)
                logger.info(f"Load balancing selected blocks: {block_indices}")

            start_block = min(block_indices)
            end_block = max(block_indices) + 1
            logger.info(f"Selected blocks: {block_indices} (start={start_block}, end={end_block})")

            use_cpu_offload = getattr(args, 'use_cpu_offload', False)
            keep_layers_on_gpu = getattr(args, 'keep_layers_on_gpu', 0)

            if end_block >= total_blocks:
                full = load_stage_model(
                    args.model, device, role="last",
                    start=start_block, dtype=args.dtype,
                    use_cpu_offload=use_cpu_offload
                )
                stage_model = StageLast(full, start_block, gpu_device=device, keep_layers_on_gpu=keep_layers_on_gpu)
                final_stage = True
            else:
                full = load_stage_model(
                    args.model, device, role="segment",
                    start=start_block, end=end_block, dtype=args.dtype,
                    use_cpu_offload=use_cpu_offload
                )
                stage_model = StageSegment(full, start_block, end_block, gpu_device=device, keep_layers_on_gpu=keep_layers_on_gpu)
                final_stage = False

            try:
                hidden_size = getattr(stage_model.config, 'hidden_size', 4096)
                throughput = get_server_throughput(
                    stage_model, device, num_blocks=len(block_indices),
                    hidden_size=hidden_size, dtype=args.dtype,
                    network_bandwidth_mbps=getattr(args, 'network_bandwidth_mbps', None),
                )
            except Exception as e:
                logger.warning(f"Failed to measure throughput, using default: {e}")
                throughput = 10.0

            should_rebalance = _setup_and_run_server_with_rebalancing(
                args, device, stage_model, final_stage, dht,
                block_indices, throughput, balance_quality, mean_balance_check_period,
                args.model, total_blocks, lb_min_block
            )

            if not should_rebalance:
                logger.info("Server shutting down normally")
                break

            logger.info("Re-evaluating block assignment...")
            time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutting down Load Balancing server...")
            break
        except Exception as e:
            logger.error(f"Error in Load Balancing loop: {e}", exc_info=True)
            time.sleep(5)


def _setup_and_run_server(args, device, stage_model, final_stage):
    """기존 서버 설정 및 실행 로직 (고정 splits용)."""
    initial_peers_list = _format_initial_peers(args.dht_initial_peers)
    local_ip = _get_local_ip()
    announce_ip = args.public_ip if args.public_ip else local_ip

    public_dht_port = args.public_dht_port if args.public_dht_port is not None else args.dht_port
    public_rpc_port = args.public_rpc_port if args.public_rpc_port is not None else args.rpc_port

    if args.public_ip:
        host_maddrs = [f"/ip4/0.0.0.0/tcp/{args.dht_port}"]
        announce_maddrs = [f"/ip4/{args.public_ip}/tcp/{public_dht_port}"]
    else:
        host_maddrs = [f"/ip4/{local_ip}/tcp/{args.dht_port}"]
        announce_maddrs = None

    dht = DHT(
        start=True,
        initial_peers=initial_peers_list if initial_peers_list else None,
        host_maddrs=host_maddrs,
        announce_maddrs=announce_maddrs,
    )

    visible = dht.get_visible_maddrs()
    peer_id = str(dht.peer_id)

    if args.public_ip:
        visible_str = [str(m) for m in visible] if visible else []
        has_public_ip = any(args.public_ip in str(m) for m in visible_str)
        if not has_public_ip:
            public_maddr = f"/ip4/{args.public_ip}/tcp/{public_dht_port}/p2p/{peer_id}"
            logger.warning(
                f"DHT visible multiaddrs do not contain public IP {args.public_ip}. "
                f"Use this multiaddr for --dht_initial_peers: {public_maddr}"
            )
            logger.info(f"DHT visible multiaddrs (may contain private IP): {visible}")
        else:
            logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")
    elif visible:
        logger.info(f"DHT visible multiaddrs (use for --dht_initial_peers): {visible}")

    handler = StageConnectionHandler(
        dht=dht,
        stage_model=stage_model,
        device=device,
        request_timeout=args.request_timeout,
        final_stage=final_stage,
    )

    async def setup_and_run():
        p2p = None
        hb_task = None
        try:
            logger.info(f"Initializing P2P for Stage{args.stage}...")

            if args.public_ip:
                p2p_host_maddrs = [f"/ip4/0.0.0.0/tcp/{args.rpc_port}"]
            else:
                p2p_host_maddrs = [f"/ip4/{local_ip}/tcp/{args.rpc_port}"]

            p2p = await P2P.create(host_maddrs=p2p_host_maddrs)
            logger.info(f"P2P initialized successfully, PeerID: {p2p.peer_id}")

            visible_maddrs = await p2p.get_visible_maddrs()
            p2p_maddrs = [str(m) for m in visible_maddrs] if visible_maddrs else []

            if args.public_ip:
                public_p2p_maddr = f"/ip4/{args.public_ip}/tcp/{public_rpc_port}/p2p/{p2p.peer_id}"

                # If the daemon didn't report a public multiaddr, still add it for convenience.
                if not any(args.public_ip in m for m in p2p_maddrs):
                    logger.warning(
                        f"Stage{args.stage} P2P visible maddrs do not contain public IP {args.public_ip}. "
                        f"Use this multiaddr: {public_p2p_maddr}"
                    )
                    p2p_maddrs.append(public_p2p_maddr)

                # ✅ DHT에는 'public IP + public port' multiaddr만 남기기
                # (10.x / 127.0.0.1 / 0.0.0.0 등 내부 주소가 먼저 선택되어 연결 실패 → peer 제외되는 문제 방지)
                p2p_maddrs = [m for m in p2p_maddrs if f"/ip4/{args.public_ip}/" in m]
                if public_p2p_maddr not in p2p_maddrs:
                    p2p_maddrs.append(public_p2p_maddr)
                # dedupe (preserve order)
                p2p_maddrs = list(dict.fromkeys(p2p_maddrs))

            if p2p_maddrs:
                logger.info(f"Stage{args.stage} P2P listen maddrs: {p2p_maddrs}")
            else:
                p2p_maddrs = [f"/ip4/{announce_ip}/tcp/{public_rpc_port}/p2p/{p2p.peer_id}"]
                logger.warning(f"Stage{args.stage} P2P listen maddrs unknown; using fallback {p2p_maddrs}")

            peer_info = {"peer_id": str(p2p.peer_id), "timestamp": get_dht_time(), "stage": args.stage, "p2p_maddrs": p2p_maddrs}
            STAGE_KEY = f"mini_petals:stage{args.stage}"
            SUBKEY = str(p2p.peer_id)
            TTL = 45

            def _store_once():
                peer_info["timestamp"] = get_dht_time()
                dht.store(key=STAGE_KEY, subkey=SUBKEY, value=peer_info, expiration_time=get_dht_time() + TTL)

            _store_once()
            logger.info(f"Stage{args.stage} registered in DHT: key={STAGE_KEY}, subkey={SUBKEY[:8]}..., ttl={TTL}s")

            async def heartbeat():
                while True:
                    try:
                        _store_once()
                    except Exception as e:
                        logger.warning(f"Stage{args.stage} heartbeat failed: {e}")
                    await asyncio.sleep(TTL / 3)

            hb_task = asyncio.create_task(heartbeat())

            await handler.add_p2p_handlers(p2p)
            logger.info(f"Stage{args.stage} handlers registered, waiting for requests...")

            await asyncio.Event().wait()

        except KeyboardInterrupt:
            logger.info(f"Stage{args.stage} shutting down...")
        finally:
            if hb_task:
                hb_task.cancel()
            if p2p:
                try:
                    await p2p.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down P2P: {e}")

    asyncio.run(setup_and_run())


def _setup_and_run_server_with_rebalancing(
    args,
    device,
    stage_model,
    final_stage,
    dht,
    block_indices,
    init_throughput,
    balance_quality,
    mean_balance_check_period,
    model_name,
    total_blocks,
    lb_min_block: int,
) -> bool:
    """
    Load Balancing 서버 실행 (주기적 재조정 포함)

    Full LB 핵심:
    - mini_petals:stageX heartbeat
    - petals:server:* + petals:module:* heartbeat (subkey 저장)
    - peer_id는 반드시 p2p.peer_id 사용
    """
    try:
        from .load_balancing import should_choose_other_blocks, ServerState
        from .dht_utils import register_server_on_dht, register_blocks_on_dht, get_remote_module_infos
        from .throughput_measurement import get_server_throughput
    except ImportError as e:
        logger.error(f"Failed to import Load Balancing modules: {e}")
        return False

    import random

    handler = StageConnectionHandler(
        dht=dht,
        stage_model=stage_model,
        device=device,
        request_timeout=args.request_timeout,
        final_stage=final_stage,
    )

    should_rebalance = False

    async def setup_and_run_with_rebalancing():
        nonlocal should_rebalance
        stop_event = asyncio.Event()

        p2p = None
        hb_task = None
        rebalance_task = None

        # throughput은 주기적으로 갱신될 수 있으므로 nonlocal로 관리
        current_throughput = {"value": float(init_throughput)}

        try:
            local_ip = _get_local_ip()
            public_rpc_port = args.public_rpc_port if args.public_rpc_port is not None else args.rpc_port

            if args.public_ip:
                p2p_host_maddrs = [f"/ip4/0.0.0.0/tcp/{args.rpc_port}"]
            else:
                p2p_host_maddrs = [f"/ip4/{local_ip}/tcp/{args.rpc_port}"]

            p2p = await P2P.create(host_maddrs=p2p_host_maddrs)
            logger.info(f"P2P initialized for Load Balancing server, PeerID: {p2p.peer_id}")

            # p2p_maddrs 계산
            visible_maddrs = await p2p.get_visible_maddrs()
            p2p_maddrs = [str(m) for m in visible_maddrs] if visible_maddrs else []

            if args.public_ip:
                public_p2p_maddr = f"/ip4/{args.public_ip}/tcp/{public_rpc_port}/p2p/{p2p.peer_id}"

                # If the daemon didn't report a public multiaddr, still add it for convenience.
                if not any(args.public_ip in m for m in p2p_maddrs):
                    logger.warning(
                        f"Stage{args.stage} P2P visible maddrs do not contain public IP {args.public_ip}. "
                        f"Use this multiaddr: {public_p2p_maddr}"
                    )
                    p2p_maddrs.append(public_p2p_maddr)

                # ✅ DHT에는 'public IP + public port' multiaddr만 남기기
                # (10.x / 127.0.0.1 / 0.0.0.0 등 내부 주소가 먼저 선택되어 연결 실패 → peer 제외되는 문제 방지)
                p2p_maddrs = [m for m in p2p_maddrs if f"/ip4/{args.public_ip}/" in m]
                if public_p2p_maddr not in p2p_maddrs:
                    p2p_maddrs.append(public_p2p_maddr)
                # dedupe (preserve order)
                p2p_maddrs = list(dict.fromkeys(p2p_maddrs))

            if p2p_maddrs:
                logger.info(f"Stage{args.stage} P2P listen maddrs: {p2p_maddrs}")

            STAGE_KEY = f"mini_petals:stage{args.stage}"
            SUBKEY = str(p2p.peer_id)
            TTL = 45

            start_block = int(min(block_indices))
            end_block = int(max(block_indices) + 1)

            def _store_once():
                # 1) mini_petals
                peer_info = {
                    "peer_id": str(p2p.peer_id),
                    "timestamp": get_dht_time(),
                    "stage": args.stage,
                    "blocks": block_indices,
                    "throughput": float(current_throughput["value"]),
                    "p2p_maddrs": p2p_maddrs,
                }
                dht.store(key=STAGE_KEY, subkey=SUBKEY, value=peer_info, expiration_time=get_dht_time() + TTL)

                # 2) petals:server + petals:module (Full LB)
                exp = get_dht_time() + TTL
                register_server_on_dht(
                    dht=dht,
                    peer_id=p2p.peer_id,
                    start_block=start_block,
                    end_block=end_block,
                    throughput=float(current_throughput["value"]),
                    model_name=model_name,
                    server_address=args.public_ip or local_ip,
                    p2p_maddrs=p2p_maddrs,
                    final_stage=final_stage,
                    state=ServerState.ONLINE,
                    expiration_time=exp,
                )
                register_blocks_on_dht(
                    dht=dht,
                    peer_id=p2p.peer_id,
                    block_indices=block_indices,
                    model_name=model_name,
                    p2p_maddrs=p2p_maddrs,
                    start_block=start_block,
                    end_block=end_block,
                    throughput=float(current_throughput["value"]),
                    final_stage=final_stage,
                    state=ServerState.ONLINE,
                    expiration_time=exp,
                )

            _store_once()
            logger.info(f"Load Balancing server published to DHT (TTL={TTL}s): blocks={block_indices}, thr={current_throughput['value']:.2f}")

            async def heartbeat():
                while not stop_event.is_set():
                    try:
                        _store_once()
                    except Exception as e:
                        logger.warning(f"Heartbeat failed: {e}")
                    await asyncio.sleep(TTL / 3)

            hb_task = asyncio.create_task(heartbeat())

            async def rebalance_check():
                nonlocal should_rebalance
                while not stop_event.is_set():
                    try:
                        timeout = random.random() * 2 * mean_balance_check_period
                        await asyncio.sleep(timeout)
                        if stop_event.is_set():
                            break

                        # 처리량 측정 갱신
                        try:
                            hidden_size = getattr(stage_model.config, 'hidden_size', 4096)
                            new_thr = get_server_throughput(
                                stage_model, device, num_blocks=len(block_indices),
                                hidden_size=hidden_size, dtype=args.dtype,
                            )
                            current_throughput["value"] = float(new_thr)
                        except Exception as e:
                            logger.debug(f"Failed to update throughput: {e}")

                        # 재조정 필요 여부 확인
                        module_infos = get_remote_module_infos(dht, model_name, total_blocks)
                        if should_choose_other_blocks(p2p.peer_id, module_infos, balance_quality, total_blocks, min_block=lb_min_block):
                            logger.info("Load balancing detected imbalance, will rebalance blocks")
                            should_rebalance = True
                            stop_event.set()
                            break

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.warning(f"Rebalance check failed: {e}")
                        await asyncio.sleep(10)

            rebalance_task = asyncio.create_task(rebalance_check())

            await handler.add_p2p_handlers(p2p)
            logger.info(f"Load Balancing server ready, blocks={block_indices}, throughput={current_throughput['value']:.2f} rps")

            await stop_event.wait()

        except KeyboardInterrupt:
            logger.info("Shutting down Load Balancing server...")
        except Exception as e:
            logger.error(f"Error in Load Balancing server: {e}", exc_info=True)
        finally:
            if hb_task:
                hb_task.cancel()
            if rebalance_task:
                rebalance_task.cancel()
            if p2p:
                try:
                    await p2p.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down P2P: {e}")

    try:
        asyncio.run(setup_and_run_with_rebalancing())
    except KeyboardInterrupt:
        logger.info("Load Balancing server interrupted")
        return False

    return should_rebalance


def main():
    parser = argparse.ArgumentParser(description="Mini Petals: Distributed Inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True,
                        help="Comma-separated cut points for 4-stage pipeline, e.g., 10,20,30")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="Model dtype: fp16 (default), bf16, fp32")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Input prompt for text generation")
    parser.add_argument("--dht_initial_peers", type=str, default="",
                        help="Comma-separated list of initial DHT peers (full multiaddrs recommended)")
    parser.add_argument("--public_ip", type=str, default="",
                        help="Public IP address for DHT announcement (required for cross-instance connections)")
    parser.add_argument("--public_dht_port", type=int, default=None,
                        help="Public DHT port. If not provided, uses --dht_port")
    parser.add_argument("--public_rpc_port", type=int, default=None,
                        help="Public RPC port. If not provided, uses --rpc_port")
    parser.add_argument("--dht_port", type=int, default=8000)
    parser.add_argument("--rpc_port", type=int, default=8001)
    parser.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3],
                        help="Stage number (0=client, 1/2 mid, 3 final server)")
    parser.add_argument("--request_timeout", type=float, default=30.0,
                        help="Timeout for RPC requests in seconds")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.92, help="Nucleus sampling p")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0=disabled)")
    parser.add_argument("--use_cpu_offload", action="store_true",
                        help="Enable CPU offloading")
    parser.add_argument("--keep_layers_on_gpu", type=int, default=0,
                        help="Number of recent layers to keep on GPU when using CPU offloading")

    # Load Balancing 옵션
    parser.add_argument("--use_load_balancing", action="store_true",
                        help="Enable Full Load Balancing (논문식)")
    parser.add_argument("--num_blocks", type=int, default=None,
                        help="Number of blocks to serve (for Load Balancing, default: auto)")
    parser.add_argument("--total_blocks", type=int, default=None,
                        help="Total number of blocks in the model (for Load Balancing, default: auto)")
    parser.add_argument("--balance_quality", type=float, default=0.75,
                        help="Load balancing quality threshold (default: 0.75)")
    parser.add_argument("--mean_balance_check_period", type=float, default=120.0,
                        help="Mean period for balance check in seconds (default: 120)")
    parser.add_argument("--network_bandwidth_mbps", type=float, default=None,
                        help="Network bandwidth in Mbps for throughput estimation (default: auto estimate)")

    parser.add_argument(
        "--no_fault_tolerance",
        action="store_true",
        help="FT 비활성: 클라이언트 캐시·replay·피어 재탐색 없이 첫 RPC 실패 시 중단 (오버헤드 A/B 베이스라인)",
    )
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        help="recovery/peer 카운터 등 RpcTransport 계측 비활성화 (미세 벤치마크용)",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="",
        help="실행 후 지표를 이 경로에 JSON으로 저장 (비우면 파일만 stdout 요약)",
    )
    parser.add_argument(
        "--failover_trials",
        type=int,
        default=1,
        help="동일 추론을 N회 연속 실행해 failover_success_rate 및 평균 E2E 지연을 집계 (강제 kill 실험 시 트라이얼 사이에 수동/스크립트로 kill)",
    )
    parser.add_argument(
        "--baseline_metrics_json",
        type=str,
        default="",
        help="비교용 베이스라인 metrics JSON (--no_fault_tolerance 또는 LB off 등 이전 실행 결과)",
    )

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    args.dtype = dtype_map[args.dtype]

    splits = parse_splits(args.splits)
    if args.stage == 0:
        run_rank0(args, device, splits)
    else:
        run_stage_server(args, device, splits)


if __name__ == "__main__":
    main()
