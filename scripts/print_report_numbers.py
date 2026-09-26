#!/usr/bin/env python3
"""Format results/summary.json into report-ready lines (avoids hand transcription)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S = json.load(open(ROOT / "results" / "summary.json"))


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


print("### datasets/models")
print("datasets:", S["datasets"], "| models:", S["models"],
      "| gap rows:", S["n_gap_rows"])

for key, label in [("H1_vs_forward", "vs forward_expanding"),
                   ("H1_vs_last", "vs last_session"),
                   ("H1_vs_first", "vs first_session")]:
    h = S.get(key, {})
    print(f"\n### H1 {label}")
    print(f"  mean Delta = {g(h,'mean'):+.4f}  (subject mean {g(h,'subject_mean'):+.4f}, "
          f"n={g(h,'n')}, subjects={g(h,'n_subjects')})")
    print(f"  subject t-test one-sided p = {g(h,'subj_p_onesided')}")
    print(f"  Wilcoxon one-sided p = {g(h,'subj_wilcoxon_p')}")
    print(f"  LMM intercept = {g(h,'lmm_intercept'):+.4f}, one-sided p = {g(h,'lmm_p_onesided')}")

print("\n### H1 by model (vs forward)")
for m, v in S.get("H1_by_model", {}).items():
    print(f"  {m}: mean Delta = {v.get('mean'):+.4f}, LMM p1 = {v.get('lmm_p_onesided')}")

print("\n### H2 primary model")
hm = S.get("H2_model", {})
print("  formula:", hm.get("formula"), "| backend:", hm.get("backend"),
      "| n:", hm.get("n"), "subjects:", hm.get("n_subjects"))
print(f"  drift_z beta = {hm.get('drift_beta'):+.5f}, p = {hm.get('drift_p')}, CI = {hm.get('drift_ci')}")
if "params" in hm:
    print("  target_session beta =", round(hm["params"].get("target_session", float('nan')), 5),
          "p =", round(hm["pvalues"].get("target_session", float('nan')), 4))

print("\n### H2 mechanism model")
mm = S.get("H2_mechanism", {})
print("  formula:", mm.get("formula"), "| backend:", mm.get("backend"))
print(f"  drift_z beta = {mm.get('drift_z_beta'):+.5f}, p = {mm.get('drift_z_p')}")
print(f"  sessions_after beta = {mm.get('sessions_after_beta'):+.5f}, p = {mm.get('sessions_after_p')}")

print("\n### H2 per-dataset drift slopes")
for r in S.get("H2_per_dataset", []):
    print(f"  {r['dataset']}: mean Delta={r.get('mean_delta'):+.4f}, slope={r.get('slope')}, "
          f"p={r.get('slope_p')}, n={r.get('n')}, subj={r.get('n_subjects')}")

print("\n### transfer symmetry (future - past donor, matched size, within test session)")
ts = S.get("transfer_symmetry", {})
print(f"  mean future-past = {g(ts,'mean_future_minus_past'):+.4f}, "
      f"t={g(ts,'t')}, p2={g(ts,'p_twosided')}; "
      f"subject-level p2={g(ts,'subj_p_twosided')} (n_subj={g(ts,'n_subjects')})")

print("\n### protocol mean AUC (interior sessions)")
for p, v in sorted(S.get("protocol_means", {}).items(), key=lambda kv: -kv[1]):
    print(f"  {p:20s} {v:.4f}")

print("\n### model ranking")
for r in S.get("model_ranking", []):
    print(f"  {r['protocol']:20s} {r['ranking']}")

# theory grid
tg = ROOT / "results" / "theory_grid.csv"
if tg.exists():
    print("\n### theory grid")
    print(pd.read_csv(tg)[["name", "tau", "mu", "mse_loso", "mse_forward",
                           "ratio", "frac_reps_loso_better"]].round(3).to_string(index=False))
