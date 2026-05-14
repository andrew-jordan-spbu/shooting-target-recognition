#!/usr/bin/env python3
"""
pipeline.py — unified training CLI
────────────────────────────────────
Usage:
  python pipeline.py --step dataset  --api-key YOUR_KEY
  python pipeline.py --step train
  python pipeline.py --step evaluate
  python pipeline.py --step report
  python pipeline.py --step inference --input ./samples/
  python pipeline.py --step all      --api-key YOUR_KEY

API key can also be set via env var: ROBOFLOW_API_KEY=xxx python pipeline.py ...
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

CONFIG_PATH  = Path(__file__).parent / "config.yaml"
STATE_DIR    = Path(__file__).parent / ".pipeline_state"   # stores resolved paths between steps


def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _save_state(key: str, value: str) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    (STATE_DIR / key).write_text(value)


def _load_state(key: str) -> str:
    p = STATE_DIR / key
    if not p.exists():
        print(f"[ERROR] State '{key}' not found. Did you run the previous step?")
        sys.exit(1)
    return p.read_text().strip()


# ── Steps ────────────────────────────────────────────────────────────────────

def step_dataset(cfg: dict, api_key: str) -> None:
    from src.merge import merge_datasets, verify_merged

    print("\n═══ STEP 1 — Dataset (download + merge) ═══════════════════")
    sources    = cfg["datasets"]
    raw_dir    = Path(cfg.get("raw_data_dir", "./data/raw"))
    merged_dir = Path(cfg.get("merge_output_dir", "./data/merged"))

    print(f"  Sources to merge: {len(sources)}")
    for s in sources:
        print(f"    • {s['name']:<14s} → {s['workspace']}/{s['project']} v{s['version']}")

    data_yaml = merge_datasets(
        sources_cfg=sources,
        output_dir=merged_dir,
        api_key=api_key,
        raw_dir=raw_dir,
    )

    print("\n[VERIFY] Merged dataset integrity:")
    counts = verify_merged(data_yaml)

    # ── Sanity checks on merged dataset ────────────────────────────────────
    MIN_IMAGES = {"train": 200, "valid": 50, "test": 50}
    print("\n[CHECK] Merged dataset split sizes:")
    any_small = False
    for split, min_n in MIN_IMAGES.items():
        n = counts.get(split, 0)
        ok = "✓" if n >= min_n else "⚠ "
        print(f"  {ok}  {split:6s}: {n:4d} images  (recommended ≥ {min_n})")
        if n < min_n:
            any_small = True
    if any_small:
        print("\n  ⚠  Some splits are still small after merge.")
        print("     Consider adding more source datasets to config.yaml → datasets[].")

    _save_state("data_yaml", data_yaml)
    _save_state("data_location", str(Path(data_yaml).parent))
    print(f"\n✓ Step 1 complete.  Next: python pipeline.py --step train")


def step_train(cfg: dict) -> None:
    from src.train import run

    print("\n═══ STEP 2 — Training ══════════════════════════════════════")
    data_yaml = _load_state("data_yaml")
    best_pt   = run(cfg, data_yaml)
    _save_state("best_pt", best_pt)
    print(f"\n✓ Step 2 complete.  Next: python pipeline.py --step evaluate")


def step_evaluate(cfg: dict) -> None:
    from src.evaluate import run

    print("\n═══ STEP 3 — Evaluation ════════════════════════════════════")
    best_pt   = _load_state("best_pt")
    data_yaml = _load_state("data_yaml")
    run(cfg, best_pt, data_yaml)
    print(f"\n✓ Step 3 complete.  Next: python pipeline.py --step report")


def step_report(cfg: dict) -> None:
    from src.report import build

    print("\n═══ STEP 4 — Report ════════════════════════════════════════")
    out    = cfg["output"]
    run_dir = Path(out["project"]) / out["run_name"]
    output  = run_dir / "training_report.html"
    build(str(run_dir), str(output))
    print(f"\n✓ Step 4 complete.  Open: {output}")


def step_inference(cfg: dict, input_path: str) -> None:
    from src.inference import run

    print("\n═══ STEP 5 — Inference ═════════════════════════════════════")
    best_pt = _load_state("best_pt")
    out     = cfg["output"]
    out_dir = Path(out["project"]) / out["run_name"] / "inference_results"
    run(best_pt, input_path, str(out_dir), cfg["inference"]["conf"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shooting target model — training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step", required=True,
        choices=["dataset", "train", "evaluate", "report", "inference", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Input path for the inference step",
    )
    args = parser.parse_args()

    cfg = _load_cfg()

    if args.step in ("dataset", "all"):
        if not args.api_key:
            print("[ERROR] --api-key is required for the dataset step.")
            print("  Get a free key at https://roboflow.com → Account → API keys")
            sys.exit(1)
        step_dataset(cfg, args.api_key)

    if args.step in ("train", "all"):
        step_train(cfg)

    if args.step in ("evaluate", "all"):
        step_evaluate(cfg)

    if args.step in ("report", "all"):
        step_report(cfg)

    if args.step == "inference":
        if not args.input:
            print("[ERROR] --input is required for the inference step.")
            sys.exit(1)
        step_inference(cfg, args.input)

    if args.step == "all":
        print("\n" + "═" * 60)
        print("  All steps complete!")
        out = cfg["output"]
        best = Path(out["project"]) / out["run_name"] / "weights" / "best.pt"
        report = Path(out["project"]) / out["run_name"] / "training_report.html"
        print(f"  Model  → {best}")
        print(f"  Report → {report}")
        print("═" * 60)


if __name__ == "__main__":
    main()
