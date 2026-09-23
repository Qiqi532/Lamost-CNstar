"""Index screening method: standard CN3839 / CN4142 / CH4300 spectral indices.

Independent of the area method.  Loads the frozen XGB threshold candidates and the
DR13-all cache, computes the paper-standard index for every target plus the project
wide-band definition as a strictly separated ``project_`` sensitivity control, assigns
the 1.0/1.5/2.0/2.5 sigma tiers, and exports only three artifacts:
  * cn_index_candidates.csv     -- candidates passing the 1 sigma primary screen
  * top30_index_candidates.png  -- shared Top-30 high-score spectrum plot
  * cn_index_summary.json       -- config, band defs, tier counts, known-CN recall

Run directly:  python cn_index_screening.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import cn_common as cc

RESULTS_DIR = cc.RESULTS_INDEX_DIR


def run_index_screening(output_dir: Path | None = None, verbose: bool = True) -> Dict[str, Any]:
    inputs = cc.load_and_validate_inputs()
    cache = cc.load_cache()
    ctx = cc.prepare_targets(inputs.candidates, cache, verbose=verbose)
    cand_df, known_df = cc.build_index_table(ctx)

    out_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = cc.build_method_summary(cand_df, known_df, "index", inputs.manifest, ctx, include_project=True)

    cand_path = cc.write_candidate_csv(cand_df, out_dir / "cn_index_candidates.csv")
    plot_path = cc.plot_top30(
        ctx, cand_df, out_dir / "top30_index_candidates.png",
        suptitle="CN index screening - top 30 candidates versus matched normal spectra",
    )
    summary_path = out_dir / "cn_index_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if verbose:
        print(f"[index] candidates={len(cand_df)}  known={len(known_df)}")
        for label in ("1p0", "1p5", "2p0", "2p5"):
            print(f"  tier>={label}: candidates={summary['tier_counts'][label]['candidate_cn_dual_pass']}  "
                  f"known_recall={summary['known_recall'][label]['recall']:.3f} "
                  f"(n_pass={summary['known_recall'][label]['n_pass']}/{summary['known_recall'][label]['n_pass'] and len(known_df)})")
        print(f"[index] wrote {cand_path.name}, {plot_path.name}, {summary_path.name}")

    return {
        "candidates": cand_df,
        "known": known_df,
        "summary": summary,
        "paths": {"candidates": cand_path, "plot": plot_path, "summary": summary_path},
    }


if __name__ == "__main__":
    res = run_index_screening()
    sys.exit(0)
