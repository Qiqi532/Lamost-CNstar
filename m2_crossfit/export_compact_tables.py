"""Re-export compact review tables from the completed M2 cache."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_dr13_all_cache import load_dr13_all_cache
from m2_crossfit.m2_crossfit_engine import exact_recall_threshold


RESULTS_DIR = PROJECT_ROOT / "m2_crossfit" / "results"
ARRAYS_PATH = RESULTS_DIR / "m2_all_scores.npz"

BASIC_COLUMNS = [
    "uid",
    "ra",
    "dec",
    "teff",
    "logg",
    "feh",
    "rv",
    "snru",
    "snrg",
    "mag_ps_g",
    "lmjd",
    "planid",
    "spid",
    "fiberid",
    "obsid",
]

CANDIDATE_COLUMNS = [
    "candidate_rank",
    *BASIC_COLUMNS,
    "m2_oof_score",
    "final_model_score",
]


def build_candidate_table(
    stars: pd.DataFrame,
    status: np.ndarray,
    oof_score: np.ndarray,
    final_score: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Return only unlabeled objects above one recall threshold."""

    selected = (status == 0) & (oof_score >= threshold)
    table = stars.loc[selected, BASIC_COLUMNS].copy()
    table["m2_oof_score"] = oof_score[selected]
    table["final_model_score"] = final_score[selected]
    table = table.sort_values("m2_oof_score", ascending=False).reset_index(drop=True)
    table.insert(0, "candidate_rank", np.arange(1, len(table) + 1))
    return table[CANDIDATE_COLUMNS]


def main() -> None:
    stars = pd.DataFrame(load_dr13_all_cache()["stars_clean"]).reset_index(drop=True)
    with np.load(ARRAYS_PATH, allow_pickle=False) as archive:
        status = archive["status"]
        oof_score = archive["score_mean"]
        final_score = archive["final_model_score"]
        cached_uid = archive["uid"].astype(str)

    if len(stars) != len(status):
        raise ValueError("母样本行数与 M2 缓存不一致")
    if not np.array_equal(stars["uid"].astype(str).to_numpy(), cached_uid):
        raise ValueError("母样本 UID 顺序与 M2 缓存不一致")

    positive_score = oof_score[status == 1]
    threshold90, required90, recall90 = exact_recall_threshold(positive_score, 0.90)
    threshold95, required95, recall95 = exact_recall_threshold(positive_score, 0.95)

    old_positive = stars["label"].eq(1).to_numpy()
    if int(old_positive.sum()) != 91:
        raise ValueError("原始证认星数量不是 91")

    known91 = stars.loc[old_positive, BASIC_COLUMNS].copy()
    known91["label"] = (oof_score[old_positive] >= threshold90).astype(int)
    known91 = known91.sort_values(["label", "uid"], ascending=[True, True]).reset_index(
        drop=True
    )

    candidates90 = build_candidate_table(
        stars, status, oof_score, final_score, threshold90
    )
    candidates95 = build_candidate_table(
        stars, status, oof_score, final_score, threshold95
    )

    outputs = {
        RESULTS_DIR / "known91_recognition_labels.csv": known91,
        RESULTS_DIR / "m2_candidates_recall90.csv": candidates90,
        RESULTS_DIR / "m2_candidates_recall95.csv": candidates95,
    }
    for path, table in outputs.items():
        table.to_csv(path, index=False, encoding="utf-8-sig")

    print(
        {
            "known91": {"rows": len(known91), "labels": known91["label"].value_counts().to_dict()},
            "recall90": {
                "threshold": threshold90,
                "required_positive": required90,
                "achieved_recall": recall90,
                "candidates": len(candidates90),
            },
            "recall95": {
                "threshold": threshold95,
                "required_positive": required95,
                "achieved_recall": recall95,
                "candidates": len(candidates95),
            },
        }
    )


if __name__ == "__main__":
    main()
