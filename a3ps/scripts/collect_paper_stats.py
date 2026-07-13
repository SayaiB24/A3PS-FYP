"""
scripts/collect_paper_stats.py

Scans the A3PS repo for every parameter, setting, and evaluation result
needed to fill in Paper 2 (the main system paper) — and writes one
consolidated markdown file you can paste directly into the paper-generation
prompt in place of the [FILL IN] / [RESULT: ...] placeholders.

WHAT IT COLLECTS
----------------
1. Config / hyperparameters        <- configs/default.yaml
2. Hardware + environment          <- torch/platform introspection, run live
3. Dataset split sizes             <- data/nexar/index.csv (dev/train_traj/eval counts)
4. Per-stage latency (Table 3)     <- meta.json files across all processed clips
5. Anticipation results (Table 1)  <- eval/anticipation.md or eval/anticipation.csv,
                                       if scripts/eval_anticipation.py has been run
6. Forecasting ADE/FDE (for ablation Table 2) <- eval/forecast_table.md or .csv,
                                       if scripts/eval_forecast.py has been run
7. Ablation flags actually exercised <- scans configs used across dashboard/clips/*/meta.json
                                       to report which forecaster/threshold modes were tested

WHAT IT DOES NOT DO
--------------------
- It does NOT invent numbers. If a file is missing (e.g. you haven't run
  eval_anticipation.py yet), that section is reported as "NOT YET AVAILABLE"
  with the exact command to run to produce it. This mirrors the paper
  prompt's own rule: no fabricated results.

USAGE
-----
    python scripts/collect_paper_stats.py
    python scripts/collect_paper_stats.py --out eval/paper_stats_summary.md

Run this from the repo root (a3ps/). It only reads files; it changes nothing.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path
from statistics import mean, pstdev

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pandas as pd
except ImportError:
    pd = None


REPO_ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).parent.name == "scripts") else Path.cwd()


# --------------------------------------------------------------------------
# Section 1 — config / hyperparameters
# --------------------------------------------------------------------------

def collect_config(root: Path) -> dict:
    cfg_path = root / "configs" / "default.yaml"
    if not cfg_path.exists():
        return {"_status": f"NOT FOUND at {cfg_path}"}
    if yaml is None:
        return {"_status": "PyYAML not installed (`pip install pyyaml`) — cannot parse config"}

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


# --------------------------------------------------------------------------
# Section 2 — hardware / environment
# --------------------------------------------------------------------------

def collect_hardware() -> dict:
    info = {
        "python_version": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "unknown (fill in manually, e.g. Intel i7)",
        "gpu": "unknown — run `nvidia-smi --query-gpu=name,memory.total --format=csv` and paste here",
        "torch_version": "not checked (torch not importable in this environment)",
        "cuda_available": "not checked",
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
    except ImportError:
        pass
    return info


# --------------------------------------------------------------------------
# Section 3 — dataset split sizes
# --------------------------------------------------------------------------

def collect_dataset_splits(root: Path) -> dict:
    idx_path = root / "data" / "nexar" / "index.csv"
    if not idx_path.exists():
        return {"_status": f"NOT FOUND at {idx_path} — run scripts/prepare_nexar.py first"}
    if pd is None:
        return {"_status": "pandas not installed — cannot parse index.csv"}

    df = pd.read_csv(idx_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "split" not in df.columns or "label" not in df.columns:
        return {"_status": f"index.csv found but missing expected columns "
                           f"(need 'split' and 'label'); found: {list(df.columns)}"}

    out = {}
    for split_name in sorted(df["split"].dropna().unique()):
        sub = df[df["split"] == split_name]
        pos = int((sub["label"] == 1).sum())
        neg = int((sub["label"] == 0).sum())
        out[split_name] = {"positives": pos, "negatives": neg, "total": pos + neg}
    out["_grand_total_clips_in_index"] = len(df)
    return out


# --------------------------------------------------------------------------
# Section 4 — per-stage latency (Table 3), from meta.json across all clips
# --------------------------------------------------------------------------

def collect_latency(root: Path) -> dict:
    clips_dir = root / "dashboard" / "clips"
    if not clips_dir.exists():
        return {"_status": f"NOT FOUND at {clips_dir}"}

    meta_files = list(clips_dir.glob("*/meta.json"))
    if not meta_files:
        return {"_status": "no meta.json files found — run the pipeline on at least one clip"}

    per_stage: dict[str, list[float]] = {}
    clip_count = 0
    fps_values = []

    for mf in meta_files:
        try:
            data = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        clip_count += 1

        # Expected shape (adjust keys here if your meta.json uses different
        # names — this looks for a few common variants):
        timing = data.get("per_stage_ms") or data.get("timing") or data.get("stage_timing_ms")
        if isinstance(timing, dict):
            for stage, ms in timing.items():
                if isinstance(ms, (int, float)):
                    per_stage.setdefault(stage, []).append(float(ms))

        fps = data.get("fps") or data.get("config", {}).get("process_fps") if isinstance(data.get("config"), dict) else data.get("fps")
        if isinstance(fps, (int, float)):
            fps_values.append(fps)

    if not per_stage:
        return {"_status": f"found {clip_count} meta.json file(s) but no recognizable "
                           f"per-stage timing field. Check the exact key your pipeline.py "
                           f"writes (per_stage_ms / timing / stage_timing_ms) and adjust "
                           f"this script's collect_latency() if it differs."}

    summary = {"_clips_measured": clip_count}
    total_mean = 0.0
    for stage, values in per_stage.items():
        summary[stage] = {
            "mean_ms_per_frame": round(mean(values), 2),
            "std_ms_per_frame": round(pstdev(values), 2) if len(values) > 1 else 0.0,
            "n_samples": len(values),
        }
        total_mean += mean(values)
    summary["_end_to_end_mean_ms_per_frame"] = round(total_mean, 2)
    summary["_implied_fps"] = round(1000.0 / total_mean, 1) if total_mean > 0 else None
    return summary


# --------------------------------------------------------------------------
# Section 5 — anticipation results (Table 1), from eval/anticipation.{md,csv}
# --------------------------------------------------------------------------

def collect_anticipation_results(root: Path) -> dict:
    md_path = root / "eval" / "anticipation.md"
    csv_path = root / "eval" / "anticipation_per_clip.csv"

    if not md_path.exists() and not csv_path.exists():
        return {"_status": "NOT YET AVAILABLE — run: "
                           "python scripts/eval_anticipation.py "
                           "(requires the eval split from prepare_nexar.py)"}

    out = {}
    if md_path.exists():
        text = md_path.read_text()
        out["_raw_markdown_report"] = text
        # Try to pull a few common numeric patterns out for convenience —
        # this is a best-effort scrape, not a strict parser. The full
        # markdown report above is the authoritative source; use it directly
        # in the paper if these regexes don't match your report's exact
        # wording.
        patterns = {
            "detection_rate_pct": r"detection rate[:\s]+([\d.]+)%",
            "false_alarm_rate_pct": r"false.alarm rate[:\s]+([\d.]+)%",
            "mtta_seconds": r"mTTA[:\s]+([\d.]+)\s*s",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                out[key] = float(m.group(1))
    if csv_path.exists() and pd is not None:
        df = pd.read_csv(csv_path)
        out["_per_clip_csv_summary"] = {
            "n_clips": len(df),
            "columns": list(df.columns),
        }
    return out


# --------------------------------------------------------------------------
# Section 6 — forecasting ADE/FDE (Table 2 ablation), from eval/forecast_table.md
# --------------------------------------------------------------------------

def collect_forecast_results(root: Path) -> dict:
    md_path = root / "eval" / "forecast_table.md"
    if not md_path.exists():
        return {"_status": "NOT YET AVAILABLE — run: python scripts/eval_forecast.py "
                           "(only meaningful if the LSTM stretch goal was attempted)"}
    return {"_raw_markdown_report": md_path.read_text()}


# --------------------------------------------------------------------------
# Section 7 — which config variants were actually exercised
# --------------------------------------------------------------------------

def collect_exercised_configs(root: Path) -> dict:
    clips_dir = root / "dashboard" / "clips"
    if not clips_dir.exists():
        return {"_status": f"NOT FOUND at {clips_dir}"}

    forecasters_seen = set()
    forecast_spaces_seen = set()
    thresholds_seen = set()

    for mf in clips_dir.glob("*/meta.json"):
        try:
            data = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cfg = data.get("config", {})
        if isinstance(cfg, dict):
            if "forecaster" in cfg:
                forecasters_seen.add(str(cfg["forecaster"]))
            if "forecast_space" in cfg:
                forecast_spaces_seen.add(str(cfg["forecast_space"]))
            if "base_threshold" in cfg:
                thresholds_seen.add(cfg["base_threshold"])

    return {
        "forecaster_variants_run": sorted(forecasters_seen) or ["none recorded"],
        "forecast_space_variants_run": sorted(forecast_spaces_seen) or ["none recorded"],
        "base_thresholds_tested": sorted(thresholds_seen) or ["none recorded"],
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_markdown(sections: dict) -> str:
    lines = ["# A3PS — Paper 2 Stats Summary",
             "",
             "Auto-collected from the repo. Paste the relevant parts of this file into "
             "the Paper 2 generation prompt in place of [FILL IN] / [RESULT: ...] "
             "placeholders. Sections marked NOT YET AVAILABLE need their listed "
             "command run first — do not hand-fill fabricated numbers.",
             ""]

    def dump(obj, indent=0):
        pad = "  " * indent
        out_lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    out_lines.append(f"{pad}- **{k}**:")
                    out_lines.extend(dump(v, indent + 1))
                else:
                    out_lines.append(f"{pad}- **{k}**: {v}")
        elif isinstance(obj, list):
            for item in obj:
                out_lines.append(f"{pad}- {item}")
        else:
            out_lines.append(f"{pad}{obj}")
        return out_lines

    titles = {
        "config": "## 1. Configuration / Hyperparameters (configs/default.yaml)",
        "hardware": "## 2. Hardware & Environment",
        "dataset_splits": "## 3. Dataset Split Sizes (dev / train_traj / eval)",
        "latency": "## 4. Per-Stage Latency — Table 3 material",
        "anticipation": "## 5. Anticipation Results — Table 1 material",
        "forecast": "## 6. Forecasting ADE/FDE — Table 2 ablation material",
        "exercised_configs": "## 7. Config Variants Actually Run",
    }

    for key, title in titles.items():
        lines.append(title)
        lines.append("")
        section = sections.get(key, {"_status": "not collected"})
        if "_raw_markdown_report" in section:
            lines.append("Raw report found on disk — copy directly into the paper:")
            lines.append("")
            lines.append("```")
            lines.append(section["_raw_markdown_report"].strip())
            lines.append("```")
            remainder = {k: v for k, v in section.items() if k != "_raw_markdown_report"}
            if remainder:
                lines.extend(dump(remainder))
        else:
            lines.extend(dump(section))
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="Repo root (default: auto-detected)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output markdown path (default: eval/paper_stats_summary.md)")
    args = parser.parse_args()

    root = args.root
    out_path = args.out or (root / "eval" / "paper_stats_summary.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] scanning repo at: {root}")

    sections = {
        "config": collect_config(root),
        "hardware": collect_hardware(),
        "dataset_splits": collect_dataset_splits(root),
        "latency": collect_latency(root),
        "anticipation": collect_anticipation_results(root),
        "forecast": collect_forecast_results(root),
        "exercised_configs": collect_exercised_configs(root),
    }

    report = render_markdown(sections)
    out_path.write_text(report)

    print(f"[done] wrote summary to: {out_path}")
    print()
    print("Sections still marked NOT YET AVAILABLE need their listed command run first.")
    print("Open the file and paste the relevant sections into the Paper 2 prompt.")


if __name__ == "__main__":
    main()