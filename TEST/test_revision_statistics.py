"""Checks for estimand weighting and dependence-aware result-level summaries."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("revision_analysis", ROOT / "scripts/revision_analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.fixture(scope="module")
def classical_rows():
    frame = pd.read_csv(ROOT / "results/gap_wide.csv")
    return frame[frame["model"].isin(analysis.CLASSICAL)].copy()


def test_structural_zero_scaling_is_participant_specific(classical_rows):
    frame = classical_rows
    assert (frame.loc[frame["is_structural_zero"], "delta"] == 0).all()
    overall = frame.groupby("subject_uid")["delta"].mean()
    conditional = frame[~frame["is_structural_zero"]].groupby("subject_uid")["delta"].mean()
    lengths = frame.groupby("subject_uid")["T"].first()
    np.testing.assert_allclose(overall, conditional * (lengths - 2) / (lengths - 1), atol=1e-15)
    assert lengths.nunique() > 1


def test_common_support_additivity_and_coverage(classical_rows):
    conditional = classical_rows[~classical_rows["is_structural_zero"]]
    strict = conditional.dropna(subset=["future_matched", "loso_matched"])
    assert len(conditional) == 2002
    assert len(strict) == 1120
    for frame, expected_mean in ((conditional, 0.034762254188677566), (strict, 0.052936187150016877)):
        total = analysis.contrast_summary(frame, "loso", "forward_expanding")
        count = analysis.contrast_summary(frame, "loso", "loso_matched")
        within = analysis.contrast_summary(frame, "loso_matched", "forward_expanding")
        assert total["n"] == count["n"] == within["n"] == 138
        assert total["mean"] == pytest.approx(expected_mean)
        assert total["mean"] == pytest.approx(count["mean"] + within["mean"])


def test_observed_flips_and_bilateral_margins(classical_rows):
    for frame, flips, robust, directions in (
        (classical_rows, 14, 1, (10, 4)),
        (classical_rows[~classical_rows["is_structural_zero"]], 19, 6, (15, 4)),
    ):
        result = analysis.operational_impact(frame)
        assert result["model_selection_changes"] == flips
        assert (result["csp_to_ts"], result["ts_to_csp"]) == directions
        counts = [row["n"] for row in result["bilateral_margin_sensitivity"]]
        assert counts[0] == flips
        assert counts[2] == robust
        assert counts == sorted(counts, reverse=True)


def test_mc_triangle_plugin_handles_shared_draws_and_equal_participants():
    # Subject 1 has one target and subject 2 has two. Each has two decoders.
    # Marginal SEs are 1 for subject 1 and 3 for subject 2. Perfectly correlated
    # draw errors attain the triangle expression: .5*1 + .5*3 = 2.
    rows = []
    for subject, targets, marginal_se in ((1, [1], 1.0), (2, [1, 2], 3.0)):
        for target in targets:
            for model in analysis.CLASSICAL:
                rows.append(dict(dataset="synthetic", subject=subject, model=model,
                                 target_session=target, protocol="loso_matched",
                                 auc_sd_draws=marginal_se * np.sqrt(4), n_draws=5,
                                 subject_uid=f"synthetic/{subject}"))
    frame = pd.DataFrame(rows)
    result = analysis.monte_carlo_summary(frame, {"example": frame})
    record = result["by_support"]["example"]
    assert record["triangle_se_bound_plugin"] == pytest.approx(2.0)
    assert record["participant_targets"] == 3
    assert record["model_target_rows"] == 6
    assert record["max_marginal_se"] == pytest.approx(3.0)
    assert "aggregate_mcse" not in record
    # A missing decoder/target variance cannot silently change aggregation weights.
    with pytest.raises(ValueError, match="Missing draw variability"):
        analysis.monte_carlo_summary(frame.iloc[:-1], {"example": frame})


def test_exact_decoder_ties_are_not_counted_as_ordering_flips():
    rows = pd.DataFrame([
        dict(subject_uid="p1", model="csp_lda", loso=0.7, forward_expanding=0.6, delta=0.1),
        dict(subject_uid="p1", model="ts_lr", loso=0.7, forward_expanding=0.8, delta=-0.1),
    ])
    margins = analysis.decoder_margin_frame(rows)
    assert not margins.loc["p1", "flip"]


def test_hksj_sensitivity_leaves_observed_average_direction_positive(classical_rows):
    for conditional, expected in ((False, (0.0089160996, 0.0393485473)),
                                  (True, (0.0138960214, 0.0468024198))):
        frame = classical_rows[~classical_rows["is_structural_zero"]] if conditional else classical_rows
        _, result = analysis.random_effects_meta(analysis.participant_frame(frame))
        np.testing.assert_allclose(result["hksj_sensitivity"]["ci"], expected, atol=1e-8)
        assert result["prediction_interval"][0] < 0 < result["prediction_interval"][1]
