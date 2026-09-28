#!/bin/zsh
# Wait for the Kumar2024 zip to finish downloading, verify size, then run it.
cd "/Users/yiming/Documents/WORK/Research/Leave-One-Session-Out Evaluation" || exit 1
ZIP="cache/mne_data/MNE-kumar2024-data/zenodo/10694880/content"
EXPECTED=4470654230

echo "[watch] waiting for curl to finish..."
while pgrep -f "curl.*Online_Offline" >/dev/null 2>&1; do sleep 20; done
sleep 3
SZ=$(stat -f%z "$ZIP" 2>/dev/null || echo 0)
echo "[watch] curl done. size=$SZ expected=$EXPECTED"
if [ "$SZ" -lt "$EXPECTED" ]; then
  echo "[watch] INCOMPLETE download ($SZ < $EXPECTED) — aborting Kumar run."
  exit 2
fi
echo "[watch] download complete. Running Kumar2024 classical + transfer..."
python3 scripts/run_experiment.py --datasets Kumar2024 --models csp_lda ts_lr --transfer
echo "[watch] Kumar classical done. Running EEGNet on first 6 Kumar subjects..."
python3 scripts/run_experiment.py --datasets Kumar2024 --models eegnet --subjects-limit 6
echo "[watch] KUMAR ALL DONE"
