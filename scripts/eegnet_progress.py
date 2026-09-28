#!/usr/bin/env python3
"""Progress + ETA for the long EEGNet run. Usage: python scripts/eegnet_progress.py"""
import os, re, time
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
TARGET = {"Ma2020": 25, "Stieger2021": 62}
for ds, n in TARGET.items():
    try:
        d = pd.read_csv(f"results/raw/auc_{ds}.csv")
        k = d[d.model == "eegnet"].subject.nunique()
    except Exception:
        k = 0
    bar = "#" * int(20 * k / n) + "." * (20 - int(20 * k / n))
    print(f"  {ds:12s} [{bar}] {k:2d}/{n}")
txt = open("logs/eegnet_longT.log").read()
st = [float(x) for x in re.findall(r"Stieger2021 S\d+ \['eegnet'\] done in ([\d.]+)s", txt)]
if st:
    pace = sum(st[-5:]) / len(st[-5:])
    left = 62 - len(st)
    eta = left * pace / 3600
    print(f"  Stieger pace {pace/60:.0f} min/subj | {left} left | ETA {eta:.1f} h "
          f"-> {time.strftime('%m-%d %H:%M', time.localtime(time.time()+eta*3600))}")
