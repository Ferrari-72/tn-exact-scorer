# tn-exact-scorer

An exact, arbitrary-precision integer scorer for sliced tensor-network contraction plans, plus a wrapper that re-scores ArcTN plans with it.

Different planners (cotengra, ArcTN, OMECO, …) report costs with their own float models and conventions. This scorer takes a binary contraction path and a slice set and returns **exact integers**:

- `total_work`: total multiply–add work summed over all slices
- `per_slice_work`, `multiplicity` (number of slices)
- `result_only_peak`: largest single intermediate result, in elements
- `contractions`

It is iterative (no recursion limit), handles hyperedges and recycled linear paths, supports up to 20,000 inputs, and checks full leaf coverage, label incidence and that slices are internal.

## Usage

```python
from tn_exact_scorer.qpe_j26_trace import collect_native_exact_summary

exact = collect_native_exact_summary(inputs, output, sizes, path, slices)
print(exact["total_work"], exact["result_only_peak"])
```

`path` is a linear (opt_einsum-style) path of index pairs; `slices` is a list of index labels.

## Example: ArcTN on QPE networks at the same cap

`examples/run_arctn_qpe.py` runs ArcTN (`arctn_schedule`, Dynamic slicing, `target_size=2**28`, FLOPs-only objective), converts its SSA path, maps its integer leg IDs back to labels, and re-scores the plan with this scorer. `examples/rescore_saved.py` re-scores saved plans from another planner the same way.

Results on [qpe-tn-benchmark](https://github.com/Ferrari-72/qpe-tn-benchmark) structures are in `results/arctn_qpe/`. Total sliced work at K = 2^28, lower is better:

| structure | our earlier planner (26-full) | ArcTN Light | ArcTN Heavy |
|---|---:|---:|---:|
| N64 (1018 tensors) | 1.450×10^12 | 4.82×10^10 | 4.71×10^10 |
| N96 (2274 tensors) | 1.659×10^17 | 1.07×10^16 | 6.34×10^15 |

Before scoring ArcTN, the scorer reproduced the saved 26-full totals exactly; ArcTN's own reported sliced cost agrees with these numbers to four significant digits. The runs used one seed and 1 CPU, so they are not a strict equal-time comparison.

ArcTN itself is **not** included. Install it separately from its own source; the examples assume `import arctn` works.

## Notes

The module name `qpe_j26_trace` is kept from the research code so that results can be traced back. It also contains instrumentation for a frozen planner that is not used by `collect_native_exact_summary`.

## License

MIT (see `LICENSE`).
