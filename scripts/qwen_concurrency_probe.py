from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def gpu_snapshot() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        total, used, free, util = [float(value.strip()) for value in out.split(",")]
        return {
            "gpu_total_mib": total,
            "gpu_used_mib": used,
            "gpu_free_mib": free,
            "gpu_util_pct": util,
        }
    except Exception as exc:  # pragma: no cover - hardware dependent
        return {"gpu_error": repr(exc)}


def call_ollama(
    *,
    base_url: str,
    model: str,
    prompt: str,
    idx: int,
    workers: int,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": "Return only valid JSON. Do not explain. Do not include markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
        },
    }
    started = time.perf_counter()
    transport_ok = False
    json_ok = False
    status = None
    error = None
    json_error = None
    response_len = None
    total_duration_ns = None
    load_duration_ns = None
    eval_count = None
    eval_duration_ns = None
    prompt_eval_count = None
    prompt_eval_duration_ns = None
    try:
        response = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_s)
        status = response.status_code
        response.raise_for_status()
        data = response.json()
        raw = data.get("message", {}).get("content") or ""
        response_len = len(raw)
        total_duration_ns = data.get("total_duration")
        load_duration_ns = data.get("load_duration")
        eval_count = data.get("eval_count")
        eval_duration_ns = data.get("eval_duration")
        prompt_eval_count = data.get("prompt_eval_count")
        prompt_eval_duration_ns = data.get("prompt_eval_duration")
        transport_ok = True
        try:
            json.loads(raw)
            json_ok = True
        except Exception as exc:
            json_error = f"{type(exc).__name__}: {exc}"[:500]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
    elapsed = time.perf_counter() - started
    return {
        "workers": workers,
        "idx": idx,
        "ok": transport_ok,
        "transport_ok": transport_ok,
        "json_ok": json_ok,
        "http_status": status,
        "latency_s": elapsed,
        "response_len": response_len,
        "total_duration_ns": total_duration_ns,
        "load_duration_ns": load_duration_ns,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_ns": prompt_eval_duration_ns,
        "error": error,
        "json_error": json_error,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def run_setting(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    workers: int,
    max_tokens: int,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before = gpu_snapshot()
    peak_used = before.get("gpu_used_mib")
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(
                call_ollama,
                base_url=base_url,
                model=model,
                prompt=prompt,
                idx=idx,
                workers=workers,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            for idx, prompt in enumerate(prompts)
        }
        while pending:
            done, pending = futures.wait(pending, timeout=0.5, return_when=futures.FIRST_COMPLETED)
            for future in done:
                rows.append(future.result())
            snapshot = gpu_snapshot()
            if "gpu_used_mib" in snapshot:
                peak_used = max(float(peak_used or 0), float(snapshot["gpu_used_mib"]))
    wall = time.perf_counter() - start
    after = gpu_snapshot()
    ok_latencies = [float(row["latency_s"]) for row in rows if row["transport_ok"]]
    eval_tokens = [float(row["eval_count"]) for row in rows if row.get("eval_count")]
    eval_seconds = [float(row["eval_duration_ns"]) / 1e9 for row in rows if row.get("eval_duration_ns")]
    summary = {
        "workers": workers,
        "n": len(rows),
        "ok": sum(bool(row["transport_ok"]) for row in rows),
        "transport_ok": sum(bool(row["transport_ok"]) for row in rows),
        "json_ok": sum(bool(row["json_ok"]) for row in rows),
        "errors": sum(not bool(row["transport_ok"]) for row in rows),
        "json_errors": sum(bool(row["transport_ok"]) and not bool(row["json_ok"]) for row in rows),
        "wall_s": wall,
        "throughput_req_per_s": len(rows) / wall if wall else None,
        "throughput_ok_per_s": sum(bool(row["transport_ok"]) for row in rows) / wall if wall else None,
        "latency_median_s": statistics.median(ok_latencies) if ok_latencies else None,
        "latency_p90_s": percentile(ok_latencies, 0.9),
        "latency_mean_s": statistics.mean(ok_latencies) if ok_latencies else None,
        "eval_tokens_per_s_sum": sum(eval_tokens) / sum(eval_seconds) if sum(eval_seconds) else None,
        "gpu_before": before,
        "gpu_after": after,
        "gpu_peak_used_mib": peak_used,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Ollama qwen3.5:0.8b concurrency with real benchmark prompts.")
    parser.add_argument("--base-url", default="http://172.26.48.1:11434")
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/runs/ces_2024_swing_aggregate_benchmark_qwen08b_fast30_v1/prompts.parquet"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/runs/qwen35_08b_concurrency_probe_v1"))
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--n-per-setting", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_frame = pd.read_parquet(args.prompts)
    prompts = prompt_frame["prompt_text"].dropna().astype(str).tolist()
    if not prompts:
        raise SystemExit(f"No prompt_text rows found in {args.prompts}")
    selected = prompts[: args.n_per_setting]

    print("warmup_start", json.dumps(gpu_snapshot(), sort_keys=True), flush=True)
    warmup_rows = [
        call_ollama(
            base_url=args.base_url,
            model=args.model,
            prompt=selected[idx % len(selected)],
            idx=idx,
            workers=1,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )
        for idx in range(args.warmup)
    ]
    print(
        "warmup_done",
        json.dumps(
            {
                "ok": sum(row["transport_ok"] for row in warmup_rows),
                "json_ok": sum(row["json_ok"] for row in warmup_rows),
                "n": len(warmup_rows),
                "gpu": gpu_snapshot(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for workers in args.workers:
        print(f"start workers={workers}", json.dumps(gpu_snapshot(), sort_keys=True), flush=True)
        rows, summary = run_setting(
            base_url=args.base_url,
            model=args.model,
            prompts=selected,
            workers=workers,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print("summary", json.dumps(summary, sort_keys=True), flush=True)
        if args.stop_on_error and summary["errors"]:
            print(f"stopping_after_errors workers={workers}", flush=True)
            break

    pd.DataFrame(warmup_rows).to_parquet(args.out_dir / "warmup_rows.parquet", index=False)
    pd.DataFrame(all_rows).to_parquet(args.out_dir / "request_rows.parquet", index=False)
    summary_frame = pd.DataFrame(
        [
            {
                **summary,
                "gpu_before": json.dumps(summary["gpu_before"], sort_keys=True),
                "gpu_after": json.dumps(summary["gpu_after"], sort_keys=True),
            }
            for summary in summaries
        ]
    )
    summary_frame.to_parquet(args.out_dir / "summary.parquet", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
