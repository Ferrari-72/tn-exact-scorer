"""Rescore saved fresh-QPE task paths with the same scorer used for ArcTN runs."""
import json
import sys

sys.path.insert(0, "/t")
import qpe_j26_trace as trace

for task in sys.argv[1:]:
    req = json.load(open(f"/tasks/{task}/request.json"))
    out = json.load(open(f"/tasks/{task}/stdout.json"))
    inputs = [[str(x) for x in t] for t in req["inputs"]]
    output = [str(x) for x in req["output"]]
    sizes = {str(k): int(v) for k, v in req["sizes"].items()}
    exact = trace.collect_native_exact_summary(inputs, output, sizes, out["path"], [str(x) for x in out["sliced"]])
    print(task, req.get("name"), exact["total_work"], exact["result_only_peak"], len(out["sliced"]))
