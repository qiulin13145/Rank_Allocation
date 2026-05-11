import html
import json
import os
import re
from collections import defaultdict
from datetime import datetime

from para_eff_pt.pt_rank_allocation_lora.rank_allocation_lora import _module_kind


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_name(value):
    value = str(value or "rank_allocation_lora")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "rank_allocation_lora"


def _short_module_name(name):
    parts = str(name).split(".")
    if len(parts) <= 4:
        return str(name)
    return ".".join(parts[-4:])


def _layer_index(name):
    match = re.search(r"(?:layers|h|blocks)\.(\d+)", str(name))
    if match:
        return int(match.group(1))
    return -1


def build_rank_allocation_event(modules, result, summary, *, step, loss_value, lr):
    module_rows = []
    for module in modules:
        old_rank = result.old_ranks.get(module, module.rank)
        new_rank = result.new_ranks.get(module, module.rank)
        module_rows.append(
            {
                "name": module.module_name,
                "short_name": _short_module_name(module.module_name),
                "kind": _module_kind(module),
                "layer": _layer_index(module.module_name),
                "old_rank": int(old_rank),
                "new_rank": int(new_rank),
                "delta": int(new_rank - old_rank),
                "score": _safe_float(module.score),
                "credit_eff": _safe_float(module.eff_ema),
                "probe": _safe_float(module.probe_ema),
                "energy": _safe_float(module.energy_ema),
            }
        )

    move_rows = []
    for move in result.moves:
        remove_module = move.remove_module
        add_module = move.add_module
        move_rows.append(
            {
                "remove_module": remove_module.module_name,
                "remove_kind": _module_kind(remove_module),
                "remove_layer": _layer_index(remove_module.module_name),
                "remove_old_rank": int(result.old_ranks[remove_module]),
                "remove_new_rank": int(result.new_ranks[remove_module]),
                "remove_score": _safe_float(remove_module.score),
                "add_module": add_module.module_name,
                "add_kind": _module_kind(add_module),
                "add_layer": _layer_index(add_module.module_name),
                "add_old_rank": int(result.old_ranks[add_module]),
                "add_new_rank": int(result.new_ranks[add_module]),
                "add_score": _safe_float(add_module.score),
                "delta_rank": int(move.delta_rank),
            }
        )

    return {
        "step": int(step),
        "loss": _safe_float(loss_value),
        "lr": _safe_float(lr),
        "metrics": dict(summary.get("metrics", {})),
        "modules": module_rows,
        "moves": move_rows,
    }


def _final_module_rows(modules):
    rows = []
    for module in modules:
        rows.append(
            {
                "name": module.module_name,
                "short_name": _short_module_name(module.module_name),
                "kind": _module_kind(module),
                "layer": _layer_index(module.module_name),
                "rank": int(module.rank),
                "initial_rank": int(module.initial_rank),
                "delta": int(module.rank - module.initial_rank),
                "score": _safe_float(module.score),
                "credit_eff": _safe_float(module.eff_ema),
                "probe": _safe_float(module.probe_ema),
                "energy": _safe_float(module.energy_ema),
            }
        )
    return rows


def _rank_bar_svg(rows):
    if not rows:
        return "<p>No RankAllocationLoRA modules found.</p>"
    rows = sorted(rows, key=lambda row: (row["layer"], row["name"]))
    width = max(1200, len(rows) * 22 + 160)
    height = 430
    left = 60
    bottom = 105
    top = 30
    chart_h = height - bottom - top
    max_rank = max(row["rank"] for row in rows) or 1
    bar_w = max(8, min(18, (width - left - 30) / max(len(rows), 1) - 2))
    colors = {"attention": "#2f80ed", "mlp": "#27ae60", "other": "#9b51e0"}

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Final rank by module">',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-25}" y2="{height-bottom}" stroke="#b7bec8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#b7bec8"/>',
        f'<text x="8" y="{top+8}" font-size="12" fill="#52606d">rank</text>',
    ]
    for idx, row in enumerate(rows):
        x = left + 10 + idx * (bar_w + 4)
        h = chart_h * row["rank"] / max_rank
        y = height - bottom - h
        color = colors.get(row["kind"], "#7f8c8d")
        title = html.escape(f'{row["name"]}: rank {row["rank"]} ({row["delta"]:+d})')
        label = html.escape(row["short_name"])
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"><title>{title}</title></rect>')
        if idx % 2 == 0:
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{height-bottom+14}" transform="rotate(60 {x+bar_w/2:.1f},{height-bottom+14})" font-size="10" text-anchor="start" fill="#52606d">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _flow_svg(events):
    matrix = defaultdict(int)
    for event in events:
        for move in event["moves"]:
            matrix[(move["remove_kind"], move["add_kind"])] += move["delta_rank"]
    kinds = ["attention", "mlp", "other"]
    max_value = max(matrix.values()) if matrix else 0
    if max_value <= 0:
        return "<p>No rank movement was recorded.</p>"

    cell = 92
    left = 105
    top = 45
    width = left + len(kinds) * cell + 35
    height = top + len(kinds) * cell + 55
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Rank flow heatmap">']
    parts.append(f'<text x="{left}" y="22" font-size="13" fill="#344054">to module type</text>')
    parts.append(f'<text x="10" y="{top-15}" font-size="13" fill="#344054">from</text>')
    for j, kind in enumerate(kinds):
        parts.append(f'<text x="{left+j*cell+cell/2}" y="{top-14}" font-size="12" text-anchor="middle" fill="#52606d">{kind}</text>')
    for i, from_kind in enumerate(kinds):
        parts.append(f'<text x="{left-10}" y="{top+i*cell+cell/2+5}" font-size="12" text-anchor="end" fill="#52606d">{from_kind}</text>')
        for j, to_kind in enumerate(kinds):
            value = matrix[(from_kind, to_kind)]
            alpha = 0.08 + 0.82 * value / max_value if value else 0.04
            x = left + j * cell
            y = top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-8}" height="{cell-8}" rx="6" fill="rgba(47,128,237,{alpha:.3f})" stroke="#d0d5dd"/>')
            parts.append(f'<text x="{x+(cell-8)/2}" y="{y+(cell-8)/2+5}" font-size="16" text-anchor="middle" fill="#101828">{value}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _html_table(rows, columns, limit=None):
    selected = rows[:limit] if limit else rows
    header = "".join(f"<th>{html.escape(title)}</th>" for title, _ in columns)
    body = []
    for row in selected:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4e}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_rank_allocation_html_report(
    *,
    events,
    modules,
    args,
    final_eval_loss=None,
    final_eval_ppl=None,
):
    report_dir = args.rank_allocation_report_dir
    if report_dir is None:
        report_dir = os.path.join(args.save_dir, "rank_allocation_report")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = _safe_name(getattr(args, "model_name", None))
    update_step = events[-1]["step"] if events else getattr(args, "num_training_steps", "final")
    report_path = os.path.join(report_dir, f"rank_allocation_report_{model_name}_step{update_step}_{timestamp}.html")

    final_rows = _final_module_rows(modules)
    total_rank = sum(row["rank"] for row in final_rows)
    initial_total_rank = sum(row["initial_rank"] for row in final_rows)
    rank_moved = sum(move["delta_rank"] for event in events for move in event["moves"])
    changed_modules = sum(1 for row in final_rows if row["delta"] != 0)

    by_kind = defaultdict(list)
    for row in final_rows:
        by_kind[row["kind"]].append(row["rank"])
    kind_rows = [
        {
            "kind": kind,
            "modules": len(ranks),
            "mean_rank": sum(ranks) / len(ranks),
            "min_rank": min(ranks),
            "max_rank": max(ranks),
        }
        for kind, ranks in sorted(by_kind.items())
    ]

    top_gain = sorted(final_rows, key=lambda row: row["delta"], reverse=True)[:12]
    top_loss = sorted(final_rows, key=lambda row: row["delta"])[:12]
    all_moves = []
    for event in events:
        for move in event["moves"]:
            move_row = dict(move)
            move_row["step"] = event["step"]
            all_moves.append(move_row)

    event_rows = []
    for event in events:
        metrics = event["metrics"]
        event_rows.append(
            {
                "step": event["step"],
                "loss": event["loss"],
                "lr": event["lr"],
                "rank_moved": metrics.get("rank_allocation/rank_moved", 0),
                "changed_modules": metrics.get("rank_allocation/changed_modules", 0),
                "rank_min": metrics.get("rank_allocation/rank_min", 0),
                "rank_mean": metrics.get("rank_allocation/rank_mean", 0),
                "rank_max": metrics.get("rank_allocation/rank_max", 0),
                "score_median": metrics.get("rank_allocation/score_median", 0),
                "probe_median": metrics.get("rank_allocation/probe_median", 0),
                "credit_eff_median": metrics.get("rank_allocation/credit_eff_median", 0),
            }
        )

    payload = {
        "model_name": getattr(args, "model_name", None),
        "save_dir": args.save_dir,
        "generated_at": timestamp,
        "events": events,
        "final_modules": final_rows,
    }

    payload_json = json.dumps(payload).replace("</", "<\\/")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RankAllocationLoRA Report - {html.escape(str(getattr(args, "model_name", "")))}</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #101828; background: #f6f7f9; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin-top: 32px; font-size: 20px; }}
.muted {{ color: #667085; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 22px 0; }}
.card {{ background: white; border: 1px solid #e4e7ec; border-radius: 8px; padding: 14px; }}
.card .label {{ color: #667085; font-size: 12px; }}
.card .value {{ margin-top: 6px; font-size: 24px; font-weight: 650; }}
section {{ background: white; border: 1px solid #e4e7ec; border-radius: 8px; padding: 18px; margin: 18px 0; overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border-bottom: 1px solid #eaecf0; padding: 7px 8px; text-align: left; white-space: nowrap; }}
th {{ color: #475467; background: #f9fafb; position: sticky; top: 0; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
.legend span {{ display: inline-flex; align-items: center; margin-right: 16px; color: #475467; font-size: 13px; }}
.dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; margin-right: 6px; }}
code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
<h1>RankAllocationLoRA Report</h1>
<div class="muted">model: <code>{html.escape(str(getattr(args, "model_name", "")))}</code> · save_dir: <code>{html.escape(str(args.save_dir))}</code> · generated: {timestamp}</div>

<div class="cards">
<div class="card"><div class="label">restart events</div><div class="value">{len(events)}</div></div>
<div class="card"><div class="label">rank moved</div><div class="value">{rank_moved}</div></div>
<div class="card"><div class="label">changed modules</div><div class="value">{changed_modules}</div></div>
<div class="card"><div class="label">total rank</div><div class="value">{total_rank}</div></div>
<div class="card"><div class="label">initial total rank</div><div class="value">{initial_total_rank}</div></div>
<div class="card"><div class="label">final eval loss</div><div class="value">{'' if final_eval_loss is None else f'{final_eval_loss:.4f}'}</div></div>
</div>

<section>
<h2>Final Rank By Module</h2>
<div class="legend"><span><i class="dot" style="background:#2f80ed"></i>attention</span><span><i class="dot" style="background:#27ae60"></i>mlp</span><span><i class="dot" style="background:#9b51e0"></i>other</span></div>
{_rank_bar_svg(final_rows)}
</section>

<section>
<h2>Rank Flow By Module Type</h2>
{_flow_svg(events)}
</section>

<div class="grid2">
<section>
<h2>Top Rank Gains</h2>
{_html_table(top_gain, [("module", "name"), ("kind", "kind"), ("initial", "initial_rank"), ("final", "rank"), ("delta", "delta"), ("score", "score"), ("probe", "probe"), ("credit_eff", "credit_eff")])}
</section>
<section>
<h2>Top Rank Losses</h2>
{_html_table(top_loss, [("module", "name"), ("kind", "kind"), ("initial", "initial_rank"), ("final", "rank"), ("delta", "delta"), ("score", "score"), ("probe", "probe"), ("credit_eff", "credit_eff")])}
</section>
</div>

<section>
<h2>Restart Timeline</h2>
{_html_table(event_rows, [("step", "step"), ("loss", "loss"), ("lr", "lr"), ("rank_moved", "rank_moved"), ("changed", "changed_modules"), ("rank_min", "rank_min"), ("rank_mean", "rank_mean"), ("rank_max", "rank_max"), ("score_median", "score_median"), ("probe_median", "probe_median"), ("credit_eff_median", "credit_eff_median")])}
</section>

<section>
<h2>Rank Moves</h2>
{_html_table(all_moves, [("step", "step"), ("remove_module", "remove_module"), ("remove_kind", "remove_kind"), ("remove_rank", "remove_new_rank"), ("add_module", "add_module"), ("add_kind", "add_kind"), ("add_rank", "add_new_rank"), ("delta", "delta_rank"), ("remove_score", "remove_score"), ("add_score", "add_score")])}
</section>

<section>
<h2>Final Module Table</h2>
{_html_table(final_rows, [("module", "name"), ("kind", "kind"), ("layer", "layer"), ("initial", "initial_rank"), ("final", "rank"), ("delta", "delta"), ("score", "score"), ("probe", "probe"), ("credit_eff", "credit_eff"), ("energy", "energy")])}
</section>

<section>
<h2>Module Type Summary</h2>
{_html_table(kind_rows, [("kind", "kind"), ("modules", "modules"), ("mean_rank", "mean_rank"), ("min_rank", "min_rank"), ("max_rank", "max_rank")])}
</section>

<script type="application/json" id="rank-allocation-data">{payload_json}</script>
</main>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return report_path
