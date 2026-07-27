#!/usr/bin/env python3
"""
make_tables.py — parse PARC sweep results into the paper's LaTeX tables.

Reads every *.jsonl produced by run_sweep.sh, groups by (method, partition,
seed), computes the final-round metrics and the forgetting metric, aggregates
mean +/- std across seeds, and prints LaTeX-ready tables.

USAGE:
    python3 make_tables.py /var/fmrag/results
    python3 make_tables.py /var/fmrag/results --metric auroc

Outputs (to stdout and to <results_dir>/tables/):
    table_main.tex        — method x metric (final round, global eval)
    table_forgetting.tex  — round-1 vs final AUROC, forgetting gap
    table_cost.tex        — comm MB/round, train s/round, params
    summary.csv           — everything, for your own plotting
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def load_records(results_dir):
    """Load all JSONL records, tagging each with its source filename."""
    records = []
    for path in glob.glob(os.path.join(results_dir, "*.jsonl")):
        run = os.path.basename(path)[:-6]   # strip .jsonl
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    r["_run"] = run
                    records.append(r)
                except json.JSONDecodeError:
                    continue
    return records


def parse_run_tag(run):
    """dataset__method__partition_aA__cN__seedS -> dict."""
    out = {"dataset": "", "method": "", "partition": "", "alpha": "",
           "clients": "", "seed": ""}
    parts = run.split("__")
    if len(parts) >= 4:
        out["dataset"]   = parts[0]
        out["method"]    = parts[1]
        pa = parts[2]                      # e.g. dirichlet_a0.5 or iid_a0
        if "_a" in pa:
            out["partition"], out["alpha"] = pa.rsplit("_a", 1)
        else:
            out["partition"] = pa
        # remaining parts may include cN and seedS in any of the trailing slots
        for p in parts[3:]:
            if p.startswith("c") and p[1:].isdigit():
                out["clients"] = p[1:]
            elif p.startswith("seed"):
                out["seed"] = p.replace("seed", "")
    return out


def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def fmt(m, s):
    if m != m:   # NaN
        return "--"
    return f"{m:.3f} $\\pm$ {s:.3f}"


def collect(records):
    """
    Build per-run summaries:
      final global metrics, round-1 global AUROC, cost averages.
    Keyed by run tag.
    """
    by_run = defaultdict(lambda: {"global": [], "local": [], "cost": [], "summary": []})
    for r in records:
        scope = r.get("scope")
        if scope in by_run[r["_run"]]:
            by_run[r["_run"]][scope].append(r)

    runs = {}
    for run, scopes in by_run.items():
        g = sorted(scopes["global"], key=lambda x: x.get("round", 0))
        c = scopes["cost"]
        if not g:
            continue
        meta = parse_run_tag(run)
        final = g[-1]
        first = g[0]

        # ── Forgetting measure (standard continual-learning definition) ──
        # Forgetting = (best AUROC ever reached) - (final AUROC).
        # This is fair across all baselines: each is measured against its OWN
        # peak, so it captures how much a method DROPS from its best, not how
        # it compares to a near-chance round-1 value. Positive = forgot;
        # ~0 = retained; negative shouldn't happen (final can't beat peak).
        aurocs = [x.get("auroc") for x in g
                  if x.get("auroc") is not None and x.get("auroc") == x.get("auroc")]
        early_auroc = first.get("auroc")
        final_auroc = final.get("auroc")
        if aurocs:
            peak_auroc = max(aurocs)
            peak_round = next((x.get("round") for x in g
                               if x.get("auroc") == peak_auroc), None)
            forgetting = (peak_auroc - final_auroc
                          if (final_auroc is not None and final_auroc == final_auroc)
                          else None)
            # also the round-1->final drop, kept for reference/robustness
            drop_from_r1 = (early_auroc - final_auroc
                            if (early_auroc is not None and final_auroc is not None
                                and early_auroc == early_auroc
                                and final_auroc == final_auroc) else None)
        else:
            peak_auroc = peak_round = forgetting = drop_from_r1 = None

        # ── Robust cost aggregation with outlier filtering ──────────────
        # If the VM is paused/suspended mid-run, wall-clock keeps counting and
        # a single train_s/eval_s can be absurd (e.g. 253000s = 70 hours for
        # one eval). Those are artifacts, not real compute. Filter any value
        # more than 5x the median before averaging, so the cost table reflects
        # true per-round compute.
        def _robust_mean(vals):
            vals = [v for v in vals if v is not None and v == v and v > 0]
            if not vals:
                return None
            svals = sorted(vals)
            med = svals[len(svals) // 2]
            kept = [v for v in vals if v <= 5 * med] or vals
            return sum(kept) / len(kept)

        comm    = [(x.get("down_mb", 0) + x.get("up_mb", 0)) for x in c]
        train_s = [x.get("train_s", 0) for x in c]
        runs[run] = {
            **meta,
            "final_auroc": final.get("auroc"),
            "final_auprc": final.get("auprc"),
            "final_f1":    final.get("f1"),
            "final_acc":   final.get("accuracy"),
            "round1_auroc": early_auroc,
            "peak_auroc":  peak_auroc,
            "peak_round":  peak_round,
            "forgetting":  forgetting,          # peak - final (the headline)
            "drop_from_r1": drop_from_r1,       # round1 - final (reference)
            "comm_mb_per_round": _robust_mean(comm),
            "train_s_per_round": _robust_mean(train_s),
            "rounds": len(g),
        }
    return runs


def group_by(runs, keys):
    groups = defaultdict(list)
    for run, d in runs.items():
        gk = tuple(d.get(k, "") for k in keys)
        groups[gk].append(d)
    return groups


def table_main(runs, out_dir):
    """Method x {AUROC, AUPRC, F1} aggregated over seeds, for IID."""
    groups = group_by(runs, ["method", "partition"])
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Mortality prediction (final-round global model, mean$\\pm$std over seeds).}",
        "\\label{tab:main}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Method & Partition & AUROC & AUPRC & F1 \\\\",
        "\\midrule",
    ]
    for (method, partition), ds in sorted(groups.items()):
        au = mean_std([d["final_auroc"] for d in ds])
        ap = mean_std([d["final_auprc"] for d in ds])
        f1 = mean_std([d["final_f1"] for d in ds])
        lines.append(f"{method} & {partition} & {fmt(*au)} & {fmt(*ap)} & {fmt(*f1)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    text = "\n".join(lines)
    _write(out_dir, "table_main.tex", text)
    return text


def table_forgetting(runs, out_dir):
    """Peak vs final AUROC and the forgetting measure — the headline.
    Compares ALL methods on the same standard metric (peak - final)."""
    groups = group_by(runs, ["method", "partition"])
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Catastrophic forgetting across methods. Forgetting = peak "
        "$-$ final AUROC (standard measure); lower is better. Mean$\\pm$std "
        "over seeds. PARC variants should show the smallest forgetting.}",
        "\\label{tab:forgetting}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Method & Partition & Peak AUROC & Final AUROC & Forgetting $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for (method, partition), ds in sorted(groups.items()):
        pk = mean_std([d["peak_auroc"] for d in ds])
        fn = mean_std([d["final_auroc"] for d in ds])
        fg = mean_std([d["forgetting"] for d in ds])
        lines.append(f"{method} & {partition} & {fmt(*pk)} & {fmt(*fn)} & {fmt(*fg)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    text = "\n".join(lines)
    _write(out_dir, "table_forgetting.tex", text)
    return text


def table_cost(runs, out_dir):
    groups = group_by(runs, ["method"])
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Communication and computation cost (mean over seeds/rounds).}",
        "\\label{tab:cost}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & Comm (MB/round) & Train (s/round) \\\\",
        "\\midrule",
    ]
    for (method,), ds in sorted(groups.items()):
        comm = mean_std([d["comm_mb_per_round"] for d in ds])
        tr   = mean_std([d["train_s_per_round"] for d in ds])
        lines.append(f"{method} & {fmt(*comm)} & {fmt(*tr)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    text = "\n".join(lines)
    _write(out_dir, "table_cost.tex", text)
    return text


def table_scalability(runs, out_dir):
    """AUROC by method x client-count — the 2/4/6 scalability story."""
    groups = group_by(runs, ["method", "clients"])
    # collect the set of client counts present
    client_counts = sorted({d.get("clients", "") for d in runs.values() if d.get("clients")},
                           key=lambda x: int(x) if x.isdigit() else 0)
    lines = [
        "\\begin{table}[t]\\centering",
        "\\caption{Scalability: final AUROC by number of clients. Mean$\\pm$std "
        "over seeds (non-IID Dirichlet).}",
        "\\label{tab:scalability}",
        "\\begin{tabular}{l" + "c" * len(client_counts) + "}",
        "\\toprule",
        "Method & " + " & ".join(f"{c} clients" for c in client_counts) + " \\\\",
        "\\midrule",
    ]
    methods = sorted({d["method"] for d in runs.values()})
    for method in methods:
        cells = []
        for c in client_counts:
            ds = [d for d in runs.values()
                  if d["method"] == method and d.get("clients") == c
                  and d.get("partition") == "dirichlet"]
            cells.append(fmt(*mean_std([d["final_auroc"] for d in ds])))
        lines.append(f"{method} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    text = "\n".join(lines)
    _write(out_dir, "table_scalability.tex", text)
    return text


def write_forgetting_curves(records, runs, out_dir):
    """
    Export per-round AUROC trajectories (averaged over seeds) for each
    method+partition, so you can PLOT the forgetting curves — the most
    convincing visual evidence. A method that forgets shows AUROC rising then
    falling; PARC should rise and hold.
    Output: forgetting_curves.csv (method,partition,round,auroc_mean,auroc_std)
    """
    import csv
    from collections import defaultdict
    by_key = defaultdict(list)   # (method,partition,round) -> [auroc,...]
    run_meta = {r: {"method": d["method"], "partition": d["partition"]}
                for r, d in runs.items()}
    for rec in records:
        if rec.get("scope") != "global":
            continue
        meta = run_meta.get(rec.get("_run"))
        if not meta:
            continue
        au = rec.get("auroc")
        if au is None or au != au:
            continue
        by_key[(meta["method"], meta["partition"], rec.get("round"))].append(au)

    path = os.path.join(out_dir, "forgetting_curves.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "partition", "round", "auroc_mean", "auroc_std"])
        for key in sorted(by_key.keys(), key=lambda k: (k[0], k[1], k[2] or 0)):
            m, s = mean_std(by_key[key])
            w.writerow([key[0], key[1], key[2], f"{m:.4f}", f"{s:.4f}"])
    print(f"  wrote {path}  (plot these to SHOW forgetting)")


def write_csv(runs, out_dir):
    import csv
    cols = ["dataset", "method", "partition", "alpha", "clients", "seed",
            "rounds", "round1_auroc", "peak_auroc", "peak_round",
            "final_auroc", "forgetting", "drop_from_r1", "final_auprc",
            "final_f1", "final_acc", "comm_mb_per_round", "train_s_per_round"]
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for d in runs.values():
            w.writerow(d)
    print(f"  wrote {path}")


def _write(out_dir, name, text):
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    args = ap.parse_args()

    records = load_records(args.results_dir)
    if not records:
        print(f"No .jsonl records found in {args.results_dir}")
        return
    runs = collect(records)
    print(f"Parsed {len(runs)} runs from {len(records)} records.\n")

    out_dir = os.path.join(args.results_dir, "tables")
    os.makedirs(out_dir, exist_ok=True)

    print("=== TABLE: main ===")
    print(table_main(runs, out_dir), "\n")
    print("=== TABLE: forgetting (headline) ===")
    print(table_forgetting(runs, out_dir), "\n")
    print("=== TABLE: cost ===")
    print(table_cost(runs, out_dir), "\n")
    print("=== TABLE: scalability (2/4/6 clients) ===")
    print(table_scalability(runs, out_dir), "\n")
    write_csv(runs, out_dir)
    write_forgetting_curves(records, runs, out_dir)
    print("\nDone. LaTeX tables + curves + summary.csv in", out_dir)


if __name__ == "__main__":
    main()
