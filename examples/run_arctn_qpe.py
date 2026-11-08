"""ArcTN vs our 26-full on the fresh-QPE matrix, under the same K = 2**28 element cap.

Run inside tnlab-benchmark:v5-kahypar-arctn with:
  /t  = directory holding only the screen's qpe_j26_trace.py (read-only, iterative scorer)
  /q  = results/qpe_fresh_20261105T054446Z_33fbb46a/generated (read-only)
  /w  = arctn_qpe_compare/runs (writable)

    python run_arctn_qpe.py NAME PRESET MODE SEED

PRESET in {heavy, light}; MODE in {dynamic, fixed}. One JSON per run.
Every reported number is recomputed by the same iterative exact integer scorer
(qpe_j26_trace.collect_native_exact_summary) that scored the fixed-matrix screen,
so it uses the same cost model as the 26-full results.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/t")
K = 2 ** 28
W = Path("/w")


def ssa_to_linear(n, ssa):
    """Convert an SSA contraction path to a linear (recycled-position) path."""
    alive = list(range(n))
    nxt = n
    linear = []
    for a, b in ssa:
        i, j = alive.index(a), alive.index(b)
        linear.append([min(i, j), max(i, j)])
        for k in sorted((i, j), reverse=True):
            alive.pop(k)
        alive.append(nxt)
        nxt += 1
    return linear


def main():
    name, preset, mode, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    out = W / f"{name}--arctn-{preset}-{mode}-s{seed}.json"
    if out.exists():
        raise SystemExit("refusing to overwrite " + str(out))
    case = json.load(open(f"/q/{name}.json"))["case"]
    inputs = [[str(x) for x in t] for t in case["inputs"]]
    output = [str(x) for x in case["output"]]
    sizes = {str(k): int(v) for k, v in case["sizes"].items()}
    import arctn
    import qpe_j26_trace as trace
    assert trace.__file__ == "/t/qpe_j26_trace.py", trace.__file__

    kwargs = dict(preset=preset, seed=seed, target_size=K, slicing_mode=mode, max_time=3600.0,
                  use_ssa=True, flops_weight=1.0, read_write_weight=0.0, rate_enabled=False)
    started = time.perf_counter()
    report = dict(arctn.arctn_schedule(inputs, output, sizes, **kwargs))
    seconds = time.perf_counter() - started
    path = ssa_to_linear(len(inputs), report["path"])
    # ArcTN reports slices as dense integer leg IDs assigned in first-appearance
    # order over inputs then output (arctn._to_int_labels); map back to labels.
    order = {}
    for term in inputs + [output]:
        for lab in term:
            order.setdefault(lab, len(order))
    label_of_id = {i: lab for lab, i in order.items()}
    raw_legs = report.get("sliced_legs") or []
    sliced = [label_of_id[int(x)] for x in raw_legs]
    # report["sliced"] is a bool flag that may be True with an empty leg list
    # when no slicing was needed; the authoritative slice set is sliced_legs.
    exact = trace.collect_native_exact_summary(inputs, output, sizes, path, sliced)
    record = {
        "network": name, "method": f"arctn-{preset}-{mode}", "seed": seed,
        "arctn_kwargs": kwargs, "seconds": round(seconds, 3),
        "inputs": len(inputs), "sliced": sliced, "n_sliced": len(sliced),
        "exact_total_work": str(exact["total_work"]),
        "exact_result_only_peak": str(exact["result_only_peak"]),
        "within_cap": exact["result_only_peak"] <= K,
        "log2_total_work": round(math.log2(exact["total_work"]), 6),
        "arctn_reported": {k: report.get(k) for k in (
            "log10_flops", "sliced_log10_flops_total", "log2_n_slices", "sliced_legs",
            "max_intermediate_log2_elements_per_slice", "memory_target_log2_elements")},
        "path": path,
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record) + "\n", encoding="utf-8")
    tmp.rename(out)
    print(json.dumps({k: record[k] for k in ("network", "method", "seconds", "n_sliced",
                                             "exact_total_work", "exact_result_only_peak", "within_cap",
                                             "log2_total_work")} | {"arctn_sliced_log10": report.get("sliced_log10_flops_total")}))


if __name__ == "__main__":
    main()
