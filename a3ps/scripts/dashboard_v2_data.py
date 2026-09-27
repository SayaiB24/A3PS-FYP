"""Read-only data layer for the v2 results dashboard (scripts/serve_dashboard_v2.py).

This module never writes anything and never scores a checkpoint. It only reads
artifacts the pipeline already produced, through the repo's own readers where
one exists:

- ``a3ps.common.schema.ClipResult``          -> clip events.json (tracks, events)
- ``a3ps.risk.temporal.RiskGRU.load``        -> checkpoint ``extra`` (config,
                                                best epoch, selection-time val)
- ``eval_anticipation.compute_metrics`` /
  ``average_precision`` / ``frame_prob_timeline`` -> baseline metrics, clip AP,
                                                per-frame threshold-system risk
- ``build_dashboard_demo.load_index`` /
  ``resolve_video``                          -> ground-truth times, video paths
- ``a3ps.common.splits.load_freeze``         -> frozen split membership

There is no run registry in this repo. A "run" is the naming tag shared by
``notebooks/models/risk_gru_<tag>.pt`` and ``eval/risk_gru_history_<tag>.json``;
evaluation records are attached to a run by the checkpoint path each artifact
names in its own header, not by its filename. See the dashboard README section
in README.md for how to add a run.

All paths are relative to the a3ps project root (the server chdirs there, which
is also what the reused readers assume).
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from a3ps.common.schema import ClipResult  # noqa: E402
from a3ps.common.splits import DEFAULT_FREEZE_PATH, load_freeze, normalize_clip_id  # noqa: E402

import eval_anticipation as ea  # noqa: E402

MODELS_DIR = os.path.join("notebooks", "models")
EVAL_DIR = "eval"
LEGACY_CLIPS_DIR = os.path.join("dashboard", "clips")
ANTICIPATION_DIR = os.path.join("eval", "anticipation")
INDEX_FULL = os.path.join("eval", "nexar_index_full.csv")
PER_CLIP_CSV = os.path.join("eval", "anticipation_per_clip.csv")
VIDEOS_ROOT = os.path.join("data", "nexar")

# Split vocabulary. Feature directory -> dashboard split id.
FEATURE_DIR_SPLIT = {
    "eval": "eval_v1",
    "cache_eval": "eval_v1",
    "train_val": "train_val",
    "train_val_v2": "train_val_v2",
    "eval_v2": "eval_v2",
}
SPLITS = {
    "eval_v1": {
        "label": "eval (v1 split)",
        "note": "v1 eval split. Every pre-'clean' checkpoint was early-stopped on it, "
                "and its alert-to-event windows are length-biased "
                "(docs/status/eval_leakage_audit.md, docs/status/label_window_audit.md).",
    },
    "train_val": {
        "label": "train_val (v1 clean)",
        "note": "Validation carve-out of the v1 partition (clean protocol).",
    },
    "train_val_v2": {
        "label": "train_val_v2",
        "note": "Validation split of the v2 partition; all selection happens here.",
    },
    "eval_v2": {
        "label": "eval_v2 (held out)",
        "note": "Frozen held-out split. Read once for the committed checkpoint "
                "(Step 11); the dashboard never re-scores it.",
    },
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _num(v):
    """float or None; NaN/blank -> None so JSON stays valid."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _clean(obj):
    """Recursively replace NaN/inf with None (json.dumps would emit NaN)."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _tag_from_checkpoint(path: str) -> Optional[str]:
    base = os.path.basename(str(path).replace("\\", "/"))
    m = re.match(r"risk_gru_(.+)\.pt$", base)
    return m.group(1) if m else None


def _split_from_features(features: str) -> Optional[str]:
    leaf = os.path.basename(str(features).replace("\\", "/").rstrip("/"))
    return FEATURE_DIR_SPLIT.get(leaf)


def parse_tag(tag: str) -> Dict[str, Any]:
    """Decode the run-naming convention. Checkpoint ``extra`` wins where present."""
    cfg: Dict[str, Any] = {}
    m = re.search(r"k(\d+)p(\d+)", tag)
    if m:
        cfg["kappa"] = float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"paw(\d+)p(\d+)", tag)
    if m:
        cfg["pre_alert_weight"] = float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"_s(\d{4})", tag)
    cfg["seed"] = int(m.group(1)) if m else 1234
    m = re.search(r"cap_h(\d+)", tag)
    if m:
        cfg["hidden"] = int(m.group(1))
    if tag.startswith("abl_"):
        cfg["ablate"] = tag[4:]
    cfg["partition"] = "v2" if "_v2" in tag else ("clean" if "clean" in tag else "v1")
    cfg["selector"] = "fixed (selfix)" if "selfix" in tag else "original"
    if tag.endswith("_valid"):
        cfg["variant"] = "valid"
    return cfg


def run_family(tag: str) -> str:
    if tag.startswith("abl_"):
        return "ablation"
    return "learned"


def _git_added_dates() -> Dict[str, str]:
    """repo-relative path (from a3ps root) -> ISO date the file was first committed.

    Read-only ``git log``. Runs are dated by when their artifacts landed, since
    nothing in the pipeline records a run timestamp.
    """
    out: Dict[str, str] = {}
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20).stdout.strip()
        log = subprocess.run(
            ["git", "log", "--diff-filter=A", "--name-only", "--format=__%cI",
             "--", EVAL_DIR, MODELS_DIR],
            cwd=ROOT, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    prefix = os.path.relpath(ROOT, top).replace("\\", "/") if top else ""
    date = None
    for line in log.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("__"):
            date = line[2:]
            continue
        rel = line
        if prefix and prefix != "." and rel.startswith(prefix + "/"):
            rel = rel[len(prefix) + 1:]
        # git log is newest-first; keep overwriting so the oldest add wins.
        out[rel] = date
    return out


def _mtime_iso(path):
    import datetime
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).astimezone().isoformat()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# markdown sweep tables (no existing reader: sweep_operating_point.py only writes)
# ---------------------------------------------------------------------------

_HEADER_KEYS = {
    "threshold": "threshold", "confirm": "confirm", "useful": "useful_warning_rate",
    "too early": "n_too_early", "false alarm": "false_alarm_rate",
    "mean lead (s)": "mean_lead_s", "mean ap": "mean_AP",
}


def parse_sweep_md(path: str) -> Optional[Dict[str, Any]]:
    """Parse an operating_point_sweep*.md / final_eval_read_*.md table."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"checkpoint:\s*`([^`]+)`(?:\s*\(best epoch (\d+)\))?", text)
    if not m:
        return None
    ckpt, best = m.group(1), m.group(2)
    s = re.search(r"scored on:\s*`([^`]+)`\s*[^0-9]*(\d+) clips,\s*(\d+) positive", text)
    features = s.group(1) if s else None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return None
    header = [h.strip().strip("*").strip().lower() for h in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        row = {}
        for h, c in zip(header, cells):
            key = _HEADER_KEYS.get(h)
            if key is None:
                continue
            c = c.replace("*", "").replace("✅", "").strip()
            row[key] = int(c) if key in ("confirm", "n_too_early") and c.isdigit() else _num(c)
        if row.get("threshold") is not None:
            rows.append(row)
    return {
        "checkpoint": ckpt, "tag": _tag_from_checkpoint(ckpt),
        "best_epoch": int(best) if best else None,
        "features": features, "split": _split_from_features(features or ""),
        "n_clips": int(s.group(2)) if s else None, "n_pos": int(s.group(3)) if s else None,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# per-clip rows -> derived metrics
# ---------------------------------------------------------------------------

def outcome_of(label: int, fired: bool) -> str:
    if label == 1:
        return "TP" if fired else "FN"
    return "FP" if fired else "TN"


def derived_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Clip-level confusion, precision/recall/F1 and a PR curve by peak_prob.

    DERIVED, not a pipeline metric: "positive prediction" = the decision rule
    fired anywhere in the clip; the PR curve ranks clips by whole-clip peak
    probability. The pipeline's own metrics (useful-warning rate, FA, lead,
    cutoff AP) are time-aware and are reported separately, unchanged.
    """
    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for r in rows:
        cm[r["outcome"]] += 1
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None

    scored = [(r["peak_prob"], r["label"]) for r in rows if r.get("peak_prob") is not None]
    n_pos = sum(1 for _, y in scored if y == 1)
    curve = []
    if n_pos:
        # Consume tie groups together, as ea.average_precision does.
        for thr in sorted({p for p, _ in scored}, reverse=True):
            sel = [y for p, y in scored if p >= thr]
            tp_k = sum(sel)
            curve.append({"threshold": thr, "recall": tp_k / n_pos,
                          "precision": tp_k / len(sel)})
    ap = ea.average_precision([p for p, _ in scored], [y for _, y in scored]) if n_pos else None
    return _clean({
        "confusion": cm, "precision": prec, "recall": rec, "f1": f1,
        "pr_curve": curve, "clip_AP_peak": ap,
        "note": "Derived clip-level view: fired anywhere = predicted positive; "
                "PR ranks clips by whole-clip peak_prob.",
    })


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------

class DashboardStore:
    """Discovers runs once, then answers API queries from memory."""

    def __init__(self, load_checkpoints: bool = True):
        self._lock = threading.Lock()
        self._clip_cache: "OrderedDict[str, Any]" = OrderedDict()
        self._clip_cache_size = 4
        self._resolve_video = None
        self._index = None
        self.warnings: List[str] = []
        self.freeze = load_freeze(DEFAULT_FREEZE_PATH) or {}
        self.dates = _git_added_dates()
        self.runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.rows: Dict[str, List[Dict[str, Any]]] = {}   # eval_id -> per-clip rows
        self._discover(load_checkpoints)

    # ---- discovery ---------------------------------------------------------

    def _date_for(self, *paths):
        for p in paths:
            if not p:
                continue
            rel = p.replace("\\", "/")
            if rel in self.dates:
                return self.dates[rel], "git (first commit)"
        for p in paths:
            if p and os.path.exists(p):
                return _mtime_iso(p), "file mtime"
        return None, None

    def _add_eval(self, run, ev):
        ev.setdefault("threshold", None)
        ev.setdefault("confirm", None)
        split = ev.get("split")
        ev["split_label"] = SPLITS.get(split, {}).get("label", split or "unknown")
        ev["held_out"] = split == "eval_v2"
        ev["metrics"] = _clean(ev.get("metrics", {}))
        run["evals"].append(ev)

    def _discover(self, load_checkpoints):
        extras = self._load_checkpoint_extras() if load_checkpoints else {}

        # 1. learned runs: one per history json and/or checkpoint.
        tags = set()
        for p in glob.glob(os.path.join(EVAL_DIR, "risk_gru_history*.json")):
            base = os.path.basename(p)[len("risk_gru_history"):-len(".json")]
            tags.add(base.lstrip("_") or "v1")
        for p in glob.glob(os.path.join(MODELS_DIR, "risk_gru_*.pt")):
            tags.add(_tag_from_checkpoint(p))
        tags.discard(None)

        for tag in sorted(tags):
            hist_path = os.path.join(EVAL_DIR, "risk_gru_history.json" if tag == "v1"
                                     else f"risk_gru_history_{tag}.json")
            ckpt = os.path.join(MODELS_DIR, f"risk_gru_{tag}.pt")
            log = os.path.join(EVAL_DIR, f"train_log_{tag}.txt")
            history = _read_json(hist_path) if os.path.exists(hist_path) else []
            extra = extras.get(tag) or {}
            cfg = parse_tag(tag)
            for k in ("kappa", "pre_alert_weight", "ablate"):
                if extra.get(k) is not None:
                    cfg[k] = extra[k]
            if extra.get("threshold") is not None:
                cfg["train_threshold"] = extra["threshold"]
            if extra.get("n_clips_with_ego") is not None:
                cfg["n_train_clips"] = extra["n_clips_with_ego"]
            date, date_src = self._date_for(hist_path, ckpt)
            run = {
                "id": tag, "label": tag, "family": run_family(tag),
                "checkpoint": ckpt.replace("\\", "/") if os.path.exists(ckpt) else None,
                "history_path": hist_path.replace("\\", "/") if history else None,
                "train_log": log.replace("\\", "/") if os.path.exists(log) else None,
                "date": date, "date_source": date_src,
                "config": cfg, "best_epoch": extra.get("best_epoch"),
                "history": history, "evals": [], "notes": [],
            }
            if extra.get("degenerate_checkpoint"):
                run["notes"].append("Checkpoint flagged degenerate at selection time.")
            val = extra.get("val")
            if val:
                n = val.get("n_clips")
                split = ("train_val_v2" if cfg["partition"] == "v2" else
                         "train_val" if cfg["partition"] == "clean" else
                         "eval_v1" if n == 120 else None)
                self._add_eval(run, {
                    "id": f"{tag}:selection_val", "kind": "selection_val",
                    "title": "best-epoch validation (checkpoint record)",
                    "source": run["checkpoint"], "split": split,
                    "threshold": extra.get("threshold"), "confirm": None,
                    "metrics": {k: v for k, v in val.items() if k != "AP"},
                })
                if split == "eval_v1":
                    run["notes"].append("Selected on the v1 eval split: its validation "
                                        "numbers are not held out.")
            self.runs[tag] = run

        # 2. operating-point sweeps and v1 final reads (markdown).
        for p in sorted(glob.glob(os.path.join(EVAL_DIR, "operating_point_sweep*.md")) +
                        glob.glob(os.path.join(EVAL_DIR, "final_eval_read_*.md"))):
            sw = parse_sweep_md(p)
            if not sw or sw["tag"] not in self.runs:
                continue
            run = self.runs[sw["tag"]]
            kind = "final_read" if "final_eval_read" in p else "sweep"
            for r in sw["rows"]:
                self._add_eval(run, {
                    "id": f"{sw['tag']}:{os.path.basename(p)}:{r['threshold']}/{r.get('confirm')}",
                    "kind": kind, "title": os.path.basename(p),
                    "source": p.replace("\\", "/"), "split": sw["split"],
                    "threshold": r["threshold"], "confirm": r.get("confirm"),
                    "metrics": {k: v for k, v in r.items() if k not in ("threshold", "confirm")},
                })
            if run["best_epoch"] is None and sw["best_epoch"]:
                run["best_epoch"] = sw["best_epoch"]

        # 3. the held-out eval_v2 read(s) with per-clip rows.
        for p in sorted(glob.glob(os.path.join(EVAL_DIR, "final_eval_read_*.json"))):
            d = _read_json(p)
            tag = _tag_from_checkpoint(d.get("checkpoint", ""))
            if tag not in self.runs:
                continue
            eid = f"{tag}:{os.path.basename(p)}"
            rows = [self._learned_row(r) for r in d.get("rows", [])]
            self.rows[eid] = rows
            metrics = {k: v for k, v in d.items()
                       if k not in ("rows", "AP", "checkpoint", "features", "threshold", "confirm")}
            self._add_eval(self.runs[tag], {
                "id": eid, "kind": "final_read", "title": os.path.basename(p),
                "source": p.replace("\\", "/"), "split": _split_from_features(d.get("features", "")),
                "threshold": d.get("threshold"), "confirm": d.get("confirm"),
                "metrics": metrics, "has_rows": True, "derived": derived_metrics(rows),
            })

        # 4. ablation re-run at a common operating point.
        for p in glob.glob(os.path.join(EVAL_DIR, "ablation_rerun_*", "*.json")):
            d = _read_json(p)
            for r in d.get("rows", []):
                tag = _tag_from_checkpoint(r.get("checkpoint", ""))
                if tag not in self.runs:
                    continue
                self._add_eval(self.runs[tag], {
                    "id": f"{tag}:{os.path.basename(p)}", "kind": "ablation_rerun",
                    "title": os.path.basename(p), "source": p.replace("\\", "/"),
                    "split": _split_from_features(d.get("features", "")),
                    "threshold": d.get("threshold"), "confirm": d.get("confirm"),
                    "metrics": {k: v for k, v in r.items()
                                if isinstance(v, (int, float)) and k != "trained_best_epoch"},
                })

        # 5. ensemble / calibration variants (their own runs; totals only).
        comp = os.path.join(EVAL_DIR, "ensemble_calibration", "comparison.json")
        if os.path.exists(comp):
            d = _read_json(comp)
            date, src = self._date_for(comp)
            for name, v in d.items():
                ev = {"id": f"ensemble:{name}", "kind": "ensemble_grid",
                      "title": "ensemble_calibration/comparison.json",
                      "source": comp.replace("\\", "/"), "split": "train_val_v2",
                      "threshold": v.get("threshold"), "confirm": v.get("confirm"),
                      "metrics": {k: x for k, x in v.items()
                                  if isinstance(x, (int, float)) and k not in ("threshold", "confirm")}}
                if name.startswith("single:") and "k1p0_v2_selfix_s" + name[7:] in self.runs:
                    self._add_eval(self.runs["k1p0_v2_selfix_s" + name[7:]], ev)
                    continue
                run = {"id": f"ensemble_{name}", "label": f"ensemble · {name}",
                       "family": "ensemble", "checkpoint": None, "history_path": None,
                       "train_log": None, "date": date, "date_source": src,
                       "config": {"partition": "v2", "variant": name}, "best_epoch": None,
                       "history": [], "evals": [], "notes": [
                           "Ensemble of the k1p0_v2_selfix seeds; see "
                           "docs/status/ensemble_calibration_results.md."]}
                self._add_eval(run, ev)
                self.runs[run["id"]] = run

        # 6. pre-Phase-IV baselines with per-clip rows (threshold system, reactive).
        if os.path.exists(PER_CLIP_CSV):
            self._add_baselines()

        # Headline eval per run: held-out read > final read > ablation > selection val.
        order = {"final_read": 0, "ablation_rerun": 1, "ensemble_grid": 2,
                 "per_clip": 2, "selection_val": 3, "sweep": 4}
        for run in self.runs.values():
            evs = sorted(run["evals"], key=lambda e: (not e["held_out"], order.get(e["kind"], 9)))
            run["headline_eval"] = evs[0]["id"] if evs else None

    def _load_checkpoint_extras(self) -> Dict[str, Dict[str, Any]]:
        """Checkpoint metadata via the repo's own RiskGRU.load (needs torch)."""
        try:
            from a3ps.risk.temporal import RiskGRU
        except Exception as e:  # torch missing: degrade to history-only runs.
            self.warnings.append(f"torch unavailable ({e}); checkpoint metadata skipped")
            return {}
        out = {}
        for p in glob.glob(os.path.join(MODELS_DIR, "risk_gru_*.pt")):
            try:
                _, extra = RiskGRU.load(p)
                out[_tag_from_checkpoint(p)] = extra
            except Exception as e:
                self.warnings.append(f"could not read {p}: {e}")
        return out

    def _learned_row(self, r):
        cid = normalize_clip_id(r["clip_id"])
        fired = r.get("first_fire_t") is not None
        return _clean({
            "clip_id": cid, "label": int(r["label"]),
            "alert_t": r.get("alert_t"), "event_t": r.get("event_t"),
            "first_fire_t": r.get("first_fire_t"), "fired": fired,
            "outcome": outcome_of(int(r["label"]), fired),
            "verdict": r.get("verdict"), "verdict_v2": r.get("verdict_v2"),
            "lead_s": r.get("lead_s"), "peak_prob": _num(r.get("peak_prob")),
            "peak_t": _num(r.get("peak_t")), "premature": r.get("premature"),
        })

    def _add_baselines(self):
        with open(PER_CLIP_CSV, newline="", encoding="utf-8-sig") as fh:
            raw = list(csv.DictReader(fh))
        date, src = self._date_for(PER_CLIP_CSV)
        for prefix, rid, label in (("a3ps", "baseline_threshold_system",
                                    "threshold system (pre-Phase IV)"),
                                   ("reactive", "baseline_reactive", "reactive ADAS baseline")):
            ea_rows, rows = [], []
            for r in raw:
                lab = 1 if r["label"] in ("pos", "1") else 0
                fa = _num(r.get(f"{prefix}_first_alert_t"))
                peak = _num(r.get("peak_prob")) if prefix == "a3ps" else None
                ea_rows.append({
                    "clip_id": r["clip_id"], "label": lab,
                    "event_time_s": _num(r.get("event_time_s")),
                    "alert_time_s": _num(r.get("alert_time_s")),
                    "first_alert_t": fa, "peak_prob": peak or 0.0,
                    "processed": r.get("processed") in ("1", "True", "true"),
                })
                rows.append(_clean({
                    "clip_id": normalize_clip_id(r["clip_id"]), "label": lab,
                    "alert_t": _num(r.get("alert_time_s")), "event_t": _num(r.get("event_time_s")),
                    "first_fire_t": fa, "fired": fa is not None,
                    "outcome": outcome_of(lab, fa is not None),
                    "verdict": r.get(f"{prefix}_outcome"), "verdict_v2": None,
                    "lead_s": _num(r.get(f"{prefix}_tta_s")), "peak_prob": peak, "peak_t": None,
                }))
            # The eval code's own aggregation, fed the CSV it wrote.
            summary, _ = ea.compute_metrics(ea_rows)
            if prefix == "reactive":
                summary.pop("AP", None)   # no reactive score; AP would be meaningless
            eid = f"{rid}:per_clip"
            self.rows[eid] = rows
            run = {"id": rid, "label": label, "family": "baseline", "checkpoint": None,
                   "history_path": None, "train_log": None, "date": date, "date_source": src,
                   "config": {"partition": "v1"}, "best_epoch": None, "history": [],
                   "evals": [], "notes": ["Scored on the GPU laptop's eval index "
                                          "(eval/nexar_index_gpu.csv), see eval/README.md."]}
            self._add_eval(run, {
                "id": eid, "kind": "per_clip", "title": "anticipation_per_clip.csv",
                "source": PER_CLIP_CSV.replace("\\", "/"), "split": "eval_v1",
                "metrics": {k: v for k, v in summary.items() if isinstance(v, (int, float))},
                "has_rows": True, "derived": derived_metrics(rows) if prefix == "a3ps"
                else {**derived_metrics(rows), "pr_curve": [], "clip_AP_peak": None},
            })
            self.runs[rid] = run

    # ---- API views --------------------------------------------------------

    def runs_summary(self):
        out = []
        for r in self.runs.values():
            out.append({k: v for k, v in r.items() if k != "history"} |
                       {"n_epochs": len(r["history"])})
        return _clean({"runs": out, "splits": SPLITS, "warnings": self.warnings})

    def run_detail(self, run_id):
        r = self.runs.get(run_id)
        return _clean(r) if r else None

    def eval_rows(self, eval_id):
        rows = self.rows.get(eval_id)
        if rows is None:
            return None
        return [dict(r, **self.clip_availability(r["clip_id"])) for r in rows]

    # ---- clips -------------------------------------------------------------

    def _index_rows(self):
        if self._index is None:
            try:
                import build_dashboard_demo as bdd  # imports torch at module load
                self._index = bdd.load_index(INDEX_FULL)
                self._resolve_video = bdd.resolve_video
            except Exception as e:
                self.warnings.append(f"build_dashboard_demo unavailable ({e}); "
                                     "video lookup limited to dashboard/clips")
                self._index = {}
        return self._index

    def video_path(self, cid):
        cid = normalize_clip_id(cid)
        for name in (cid, cid.zfill(5)):
            p = os.path.join(LEGACY_CLIPS_DIR, name, "raw.mp4")
            if os.path.isfile(p):
                return p
        idx = self._index_rows()
        if self._resolve_video:
            return self._resolve_video(cid, idx.get(cid, {}), VIDEOS_ROOT)
        return None

    def clip_availability(self, cid):
        cid = normalize_clip_id(cid)
        return {
            "has_tracks": os.path.isfile(os.path.join(ANTICIPATION_DIR, cid, "events.json"))
                          or os.path.isfile(os.path.join(LEGACY_CLIPS_DIR, cid, "events.json")),
            "has_risk_curve": os.path.isfile(os.path.join(LEGACY_CLIPS_DIR, cid, "risk.json")),
            "split_v2": (self.freeze.get("clips", {}).get(cid) or {}).get("split"),
        }

    def _load_clip_result(self, path):
        with self._lock:
            if path in self._clip_cache:
                self._clip_cache.move_to_end(path)
                return self._clip_cache[path]
        cr = ClipResult.load_json(path)
        with self._lock:
            self._clip_cache[path] = cr
            while len(self._clip_cache) > self._clip_cache_size:
                self._clip_cache.popitem(last=False)
        return cr

    def manifest(self):
        """dashboard/clips/manifest.json, normalised exactly as the legacy app.js does."""
        path = os.path.join(LEGACY_CLIPS_DIR, "manifest.json")
        raw = _read_json(path) if os.path.isfile(path) else []
        clips = ([{"id": c, "group": "", "label": c} for c in raw] if isinstance(raw, list)
                 else list((raw or {}).get("clips", [])))
        for c in clips:
            d = self._legacy_dir(c["id"])
            c["has_video"] = bool(d and os.path.isfile(os.path.join(d, "raw.mp4")))
            c["has_risk"] = bool(d and os.path.isfile(os.path.join(d, "risk.json")))
        return {"clips": clips, "source": path.replace("\\", "/")}

    @staticmethod
    def _legacy_dir(raw_id):
        """dashboard/clips/<dir> for an id written any of the project's three ways."""
        raw = str(raw_id).strip()
        for name in (raw, normalize_clip_id(raw), normalize_clip_id(raw).zfill(5)):
            d = os.path.join(LEGACY_CLIPS_DIR, name)
            if name and os.path.isdir(d):
                return d
        return None

    def _ground_plane(self, raw_id, width, height):
        """The pipeline's own image->ground homography (a3ps.pipeline._fill_bev).

        Default config + the per-clip override the pipeline would use, so a
        projected position is what the pipeline itself writes to centroid_bev.
        """
        from a3ps.common.geometry import GroundPlane, load_ground_override
        from a3ps.pipeline import load_config
        cfg = load_config(os.path.join("configs", "default.yaml"))
        d = self._legacy_dir(raw_id)
        override_path = cfg.get("ground_plane_override")
        for name in ("ground.yaml", "ground.yml", "ground.json"):
            if d and os.path.isfile(os.path.join(d, name)):
                override_path = os.path.join(d, name)
        return GroundPlane.from_config(cfg, width, height, load_ground_override(override_path))

    def _fill_ground(self, payload, raw_id):
        """Give every track a ground position and every forecast a ground path.

        Positions the pipeline already wrote are kept and tagged "pipeline".
        Missing ones (the slim demo overlays drop them) are projected with the
        same GroundPlane the pipeline uses and tagged "projected". Forecast
        paths use the pipeline's mean_bev when present, else its image-space
        forecast (mean_img, the Kalman-CV output) projected onto the ground.
        """
        frames = payload["frames"]
        if not frames:
            return
        w = payload["meta"].get("width") or 1280
        h = payload["meta"].get("height") or 720
        try:
            gp = self._ground_plane(raw_id, w, h)
        except Exception as e:  # cv2/yaml missing: leave BEV as stored
            payload["bev_note"] = f"ground-plane projection unavailable ({e})"
            return

        def ok(p):
            return -40.0 <= p[0] <= 40.0 and -2.0 <= p[1] <= 90.0

        # Batch every point into ONE homography call (per-point calls are ~50x slower).
        jobs, pts = [], []
        n_proj = n_pipe = 0
        for f in frames:
            if not f.get("corridor_bev") and f.get("corridor_img"):
                jobs.append((f, "corridor_bev", len(pts), len(f["corridor_img"]), False))
                pts.extend(f["corridor_img"])
            for t in f["tracks"]:
                if t.get("c_bev"):
                    t["bev_src"] = "pipeline"
                    n_pipe += 1
                elif t.get("c_img"):
                    t["bev_src"] = "projected"
                    n_proj += 1
                    jobs.append((t, "c_bev", len(pts), 1, True))
                    pts.append(t["c_img"])
                if not t.get("pred_bev") and t.get("pred_img"):
                    t["pred_bev_src"] = "projected mean_img"
                    jobs.append((t, "pred_bev", len(pts), len(t["pred_img"]), True))
                    pts.extend(t["pred_img"])
        out = gp.img_to_bev(pts) if pts else []
        for obj, key, i0, n, filt in jobs:
            seg = [[round(x, 2), round(y, 2)] for x, y in out[i0:i0 + n]]
            if key == "c_bev":
                obj[key] = seg[0] if ok(seg[0]) else None
            else:
                obj[key] = [q for q in seg if ok(q)] if filt else seg
        payload["bev_counts"] = {"pipeline": n_pipe, "projected": n_proj}

    def clip_detail(self, cid, overlay=None, hz=10.0, mask_points=16):
        """Replay payload for one clip. Downsampled; never the raw 90 MB file."""
        raw_id = str(cid)
        cid = normalize_clip_id(cid)
        sources = OrderedDict()
        full = os.path.join(ANTICIPATION_DIR, cid, "events.json")
        legacy_dir = self._legacy_dir(raw_id)
        slim = os.path.join(legacy_dir, "events.json") if legacy_dir else ""
        if os.path.isfile(full):
            sources["full"] = full
        if os.path.isfile(slim):
            sources["legacy"] = slim
        chosen = overlay if overlay in sources else next(iter(sources), None)

        idx_row = self._index_rows().get(cid, {})
        gt = {"alert_t": _num(idx_row.get("alert_time_s")),
              "event_t": _num(idx_row.get("event_time_s")),
              "label": int(idx_row["label"]) if str(idx_row.get("label", "")).isdigit() else None}

        payload: Dict[str, Any] = {
            "clip_id": cid, "gt": gt, "overlay_sources": list(sources),
            "overlay_source": chosen,
            "overlay_path": sources.get(chosen, "").replace("\\", "/") or None,
            "video_url": f"/media/video/{raw_id}" if self.video_path(raw_id) else None,
            "meta": {}, "frames": [], "events": [], "curves": [], "decision": None,
            "split_v2": (self.freeze.get("clips", {}).get(cid) or {}).get("split"),
        }

        risk_path = os.path.join(legacy_dir or os.path.join(LEGACY_CLIPS_DIR, cid), "risk.json")
        if os.path.isfile(risk_path):
            risk = _read_json(risk_path)
            lr = risk.get("learned") or {}
            payload["curves"].append({
                "id": "learned", "name": lr.get("name", "learned head"),
                "source": risk_path.replace("\\", "/"), "checkpoint": lr.get("checkpoint"),
                "points": lr.get("curve", [])})
            payload["decision"] = {
                "operating_point": risk.get("operating_point"), "window": risk.get("window"),
                "event": lr.get("event"), "brake_event": lr.get("brake_event"),
                "verdict": lr.get("verdict"), "verdict_v2": lr.get("verdict_v2"),
                "threshold_system": {k: (risk.get("threshold_system") or {}).get(k)
                                     for k in ("verdict", "first_alert_t", "n_events")}}
            m = risk.get("markers") or {}
            gt["alert_t"] = gt["alert_t"] if gt["alert_t"] is not None else m.get("time_of_alert")
            gt["event_t"] = gt["event_t"] if gt["event_t"] is not None else m.get("time_of_event")
            if gt["label"] is None:
                gt["label"] = risk.get("label")
            payload["meta"]["duration_s"] = risk.get("duration_s")

        if chosen:
            cr = self._load_clip_result(sources[chosen])
            payload["meta"].update({k: cr.meta.get(k) for k in ("fps", "width", "height", "n_frames")})
            if cr.frames:
                payload["meta"]["duration_s"] = payload["meta"].get("duration_s") or cr.frames[-1].t
            payload["meta"]["config"] = cr.meta.get("config")
            payload["meta"]["overlay_note"] = cr.meta.get("overlay_note")
            payload["frames"] = self._slim_frames(cr, hz, mask_points)
            payload["events"] = [e.to_dict() for e in cr.events]
            self._fill_ground(payload, raw_id)
            if chosen == "full":
                tl = ea.frame_prob_timeline(cr)
                step = max(1, int(round((cr.meta.get("fps") or 30) / hz)))
                payload["curves"].append({
                    "id": "threshold_system",
                    "name": "threshold system · per-frame max collision prob",
                    "source": sources[chosen].replace("\\", "/"),
                    "points": [[round(t, 3), round(p, 4)] for t, p in tl[::step]]})
        return _clean(payload)

    @staticmethod
    def _slim_frames(cr, hz, mask_points):
        fps = cr.meta.get("fps") or 30.0
        frames = cr.frames
        if len(frames) > 1:
            src_hz = 1.0 / max(1e-6, (frames[-1].t - frames[0].t) / (len(frames) - 1))
        else:
            src_hz = fps
        step = max(1, int(round(src_hz / hz)))
        out = []

        def rp(p, nd=1):
            return None if p is None else [round(float(p[0]), nd), round(float(p[1]), nd)]

        for f in frames[::step]:
            tracks = []
            for t in f.tracks:
                pred = t.prediction
                mask = t.mask_poly
                if mask and len(mask) > mask_points:
                    k = len(mask) / mask_points
                    mask = [mask[int(i * k)] for i in range(mask_points)]
                tracks.append({
                    "id": t.id, "cls": t.cls, "bbox": [round(v) for v in t.bbox],
                    "c_img": rp(t.centroid_img), "c_bev": rp(t.centroid_bev, 2),
                    "v_bev": rp(t.velocity_bev, 2), "lvl": t.risk_level,
                    "mask": [[round(x), round(y)] for x, y in mask] if mask else None,
                    "trail": [rp(p) for p in (t.history_img or [])[-10:]],
                    "p": None if not pred or pred.collision_prob is None
                    else round(float(pred.collision_prob), 4),
                    "ttc": None if not pred else pred.ttc_s,
                    "pred_img": [rp(p) for p in (pred.mean_img[::2] if pred else [])],
                    "pred_bev": [rp(p, 2) for p in (pred.mean_bev or [])[::2]] if pred else [],
                    "std_bev": [rp(p, 2) for p in (pred.std_bev or [])[::2]] if pred else [],
                    "horizon_s": pred.horizon_s if pred else None,
                    "ttc_s": pred.ttc_s if pred else None,
                })
            ego = f.ego or {}
            out.append({"t": round(f.t, 3), "i": f.frame_idx, "tracks": tracks,
                        "corridor_img": ego.get("corridor_poly_img"),
                        "corridor_bev": ego.get("corridor_poly_bev"),
                        "ctx": f.context})
        return out
