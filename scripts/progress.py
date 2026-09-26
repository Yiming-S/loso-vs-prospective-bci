#!/usr/bin/env python3
"""Print size-matched run progress (subjects done per dataset) + ETA."""
import glob, os, re, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
TARGET = {"BNCI2014_004": 9, "Zhou2016": 4, "Zhou2020": 20,
          "Kumar2024": 18, "Ma2020": 25, "Stieger2021": 62}
PACE = {"BNCI2014_004": 10, "Zhou2016": 8, "Zhou2020": 120,
        "Kumar2024": 12, "Ma2020": 182, "Stieger2021": 250}

done = {}
for f in sorted(glob.glob("results/raw/auc_matched_*.csv")):
    ds = os.path.basename(f)[len("auc_matched_"):-len(".csv")]
    done[ds] = pd.read_csv(f).subject.nunique()

tot = rem_s = 0
for ds, n in TARGET.items():
    k = done.get(ds, 0); tot += k
    rem_s += (n - k) * PACE[ds]
    bar = "#" * int(20 * k / n) + "." * (20 - int(20 * k / n))
    print(f"  {ds:14s} [{bar}] {k:2d}/{n}")
print(f"  TOTAL {tot}/138   ETA ~{rem_s/3600:.1f} h")

log_path = "logs/matched_run.log"
if os.path.exists(log_path):
    txt = open(log_path).read()
    cur = re.findall(r"\[(\w+) S(\d+)\]", txt)
    if cur:
        print(f"  currently: {cur[-1][0]} S{cur[-1][1]}")
