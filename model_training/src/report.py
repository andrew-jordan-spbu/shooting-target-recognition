"""
src/report.py
─────────────
Build a self-contained HTML training report.
All plots are embedded as base64 — the file can be shared without extra assets.
"""
import base64
import csv
import datetime
import json
from pathlib import Path

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Training Report — Shooting Target</title>
<style>
  body{{font-family:'Helvetica Neue',sans-serif;background:#0f1117;color:#e0e0e0;
        padding:28px;max-width:1100px;margin:auto;line-height:1.6}}
  h1{{color:#facc15;margin-bottom:4px}}
  h2{{color:#60a5fa;margin:28px 0 10px;border-bottom:1px solid #2a2d3a;padding-bottom:6px}}
  .meta{{color:#6b7280;font-size:.9rem;margin-bottom:24px}}
  .grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
  .card{{background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;
         padding:18px;text-align:center}}
  .val{{font-size:2.2rem;font-weight:700;color:#facc15}}
  .lbl{{font-size:.78rem;color:#9ca3af;margin-top:4px;text-transform:uppercase;letter-spacing:.05em}}
  table{{border-collapse:collapse;width:100%;margin-bottom:18px}}
  th,td{{padding:9px 14px;border-bottom:1px solid #2a2d3a;text-align:left;font-size:.88rem}}
  th{{background:#1a1d27;color:#9ca3af;font-weight:600;font-size:.78rem;
      text-transform:uppercase;letter-spacing:.05em}}
  tr:hover td{{background:#161923}}
  .tag{{display:inline-block;padding:2px 9px;border-radius:10px;font-size:.75rem;font-weight:700}}
  .good{{background:#16a34a30;color:#4ade80}}
  .ok  {{background:#d9770630;color:#fb923c}}
  .bad {{background:#dc262630;color:#f87171}}
  .plots{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:10px}}
  .plots img{{max-width:100%;border-radius:8px;border:1px solid #2a2d3a}}
  .bar-wrap{{background:#1a1d27;border-radius:4px;overflow:hidden;height:10px;min-width:120px}}
  .bar{{height:100%;background:#3b82f6;border-radius:4px}}
  .info{{background:#1e293b;border-left:3px solid #3b82f6;padding:12px 16px;
         border-radius:0 6px 6px 0;font-size:.88rem;color:#cbd5e1;margin-top:10px}}
  .info li{{margin-top:5px}}
  @media(max-width:700px){{.grid4{{grid-template-columns:1fr 1fr}}.plots{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>🎯 Training Report — Shooting Target Detection</h1>
<div class="meta">Model: {model} &nbsp;·&nbsp; Run: {run_name} &nbsp;·&nbsp; {timestamp}</div>

<h2>Key Metrics</h2>
<div class="grid4">
  <div class="card"><div class="val">{map50:.3f}</div><div class="lbl">mAP@50</div></div>
  <div class="card"><div class="val">{map5095:.3f}</div><div class="lbl">mAP@50-95</div></div>
  <div class="card"><div class="val">{precision:.3f}</div><div class="lbl">Precision</div></div>
  <div class="card"><div class="val">{recall:.3f}</div><div class="lbl">Recall</div></div>
</div>

<h2>Assessment</h2>
<table>
<tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th><th>Notes</th></tr>
{assessment_rows}
</table>

<h2>Per-Class mAP@50</h2>
<table>
<tr><th>Class (zone)</th><th>mAP@50</th><th>Visual</th></tr>
{class_rows}
</table>

<h2>Training Curves</h2>
<div class="plots">{plot_imgs}</div>

<h2>How to read these results</h2>
<div class="info"><ul>
  <li><b>mAP@50 &gt; 0.80</b> — model reliably detects holes in all 10 scoring zones.</li>
  <li><b>Recall &gt; 0.75</b> — priority: missing a hole means wrong score for the shooter.</li>
  <li><b>Low AP on class "10"</b> — expected (smallest zone, fewest labeled samples). Acceptable.</li>
  <li><b>mAP &lt; 0.60?</b> — try: more epochs, larger model (yolov8m), lower conf threshold.</li>
  <li><b>Confusion matrix</b> — adjacent zone confusion (8↔9) is acceptable for overall scoring.</li>
</ul></div>
</body>
</html>
"""


def _tag(v: float, g: float, o: float) -> str:
    if v >= g: return '<span class="tag good">✓ Good</span>'
    if v >= o: return '<span class="tag ok">~ OK</span>'
    return '<span class="tag bad">✗ Low</span>'


def _embed(path: Path) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    ext  = "png" if path.suffix == ".png" else "jpeg"
    return f'<img src="data:image/{ext};base64,{data}" alt="{path.name}">'


def _read_csv_metrics(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    m = {"map50": 0.0, "map5095": 0.0, "precision": 0.0, "recall": 0.0}
    if not csv_path.exists():
        return m
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return m
    last = rows[-1]
    mapping = {
        "map50":     "metrics/mAP50(B)",
        "map5095":   "metrics/mAP50-95(B)",
        "precision": "metrics/precision(B)",
        "recall":    "metrics/recall(B)",
    }
    for key, col in mapping.items():
        for k, v in last.items():
            if k.strip() == col.strip():
                try: m[key] = float(v)
                except ValueError: pass
    return m


def build(run_dir: str, output: str) -> None:
    p = Path(run_dir)

    # Prefer eval_metrics.json (from evaluate step), fallback to results.csv
    eval_json = p / "eval_metrics.json"
    if eval_json.exists():
        metrics = json.loads(eval_json.read_text())
        per_class = metrics.get("per_class", {})
    else:
        metrics   = _read_csv_metrics(p)
        per_class = {}

    thresholds = [
        ("mAP@50",    metrics["map50"],     0.80, 0.60, "&gt; 0.80", "Main quality indicator"),
        ("mAP@50-95", metrics["map5095"],   0.55, 0.40, "&gt; 0.55", "Stricter localisation"),
        ("Precision",  metrics["precision"], 0.80, 0.65, "&gt; 0.80", "False-positive rate"),
        ("Recall",     metrics["recall"],    0.75, 0.55, "&gt; 0.75", "Missed holes (priority)"),
    ]
    assessment_rows = "\n".join(
        f"<tr><td>{n}</td><td>{v:.4f}</td><td>{t}</td>"
        f"<td>{_tag(v,g,o)}</td><td style='color:#6b7280'>{note}</td></tr>"
        for n, v, g, o, t, note in thresholds
    )

    if per_class:
        class_rows = "\n".join(
            f"<tr><td>{cls}</td><td>{ap:.4f}</td>"
            f"<td><div class='bar-wrap'><div class='bar' style='width:{int(ap*100)}%'></div></div></td></tr>"
            for cls, ap in sorted(per_class.items(), key=lambda x: -x[1])
        )
    else:
        class_rows = "<tr><td colspan='3' style='color:#6b7280'>Run evaluate step to generate per-class data.</td></tr>"

    plot_files = ["results.png", "confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png"]
    plot_imgs  = "".join(_embed(p / pf) for pf in plot_files)
    if not plot_imgs:
        plot_imgs = "<p style='color:#6b7280'>No plots found. Check run directory.</p>"

    # Try to read model name from args.yaml
    model_name = "YOLOv8s"
    args_yaml = p / "args.yaml"
    if args_yaml.exists():
        try:
            import yaml
            a = yaml.safe_load(args_yaml.read_text())
            model_name = a.get("model", model_name)
        except Exception:
            pass

    html = _HTML.format(
        model=model_name,
        run_name=p.name,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        map50=metrics["map50"],
        map5095=metrics["map5095"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        assessment_rows=assessment_rows,
        class_rows=class_rows,
        plot_imgs=plot_imgs,
    )

    Path(output).write_text(html, encoding="utf-8")
    print(f"[OK]  Report saved → {output}")
