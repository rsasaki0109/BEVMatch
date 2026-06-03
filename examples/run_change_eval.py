"""Change-detection benchmark (§11, §13.4).

    python examples/run_change_eval.py

1. Persistence: a multi-frame query burst with stable added/removed objects and
   moving dynamic objects -> persistence keeps the stable ones, filters dynamics.
2. Occlusion: a query occluder hides reference objects -> occlusion-aware
   comparable region avoids reporting them as "removed".
Saves a before/after change viewer figure if matplotlib is available.
"""

from __future__ import annotations

import json
from pathlib import Path

from bevmatch.alignment import SE2Aligner
from bevmatch.change import ChangeConfig, detect_changes_detailed, detect_persistent_changes
from bevmatch.datasets import make_occlusion_case, make_synthetic_change_case
from bevmatch.eval import change_prf, false_changes
from bevmatch.viz import save_change_figure

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    aligner = SE2Aligner()

    # 1. Persistence / dynamic filtering.
    case = make_synthetic_change_case(seed=3)
    changes = detect_persistent_changes(
        case.query_frames, case.reference, aligner, ChangeConfig(use_occlusion=False)
    )
    actionable = [c for c in changes if c.actionable]
    dynamic = [c for c in changes if c.category == "dynamic"]
    add_prf = change_prf(actionable, case.added_centers, "added")
    rem_prf = change_prf(actionable, case.removed_centers, "removed")

    print("=== Persistence (dynamic filtering) ===")
    print(f"frames={len(case.query_frames)}  GT added={len(case.added_centers)} "
          f"removed={len(case.removed_centers)} dynamic/frame={len(case.dynamic_centers_per_frame[0])}")
    print(f"actionable: added P/R={add_prf.precision:.2f}/{add_prf.recall:.2f} "
          f"removed P/R={rem_prf.precision:.2f}/{rem_prf.recall:.2f}  dynamic filtered={len(dynamic)}")
    print(f"false actionable changes={false_changes(actionable, case.added_centers, case.removed_centers)}")

    # 2. Occlusion vs removal.
    occ = make_occlusion_case(seed=5)
    a = aligner.align(occ.query, occ.reference)
    print("\n=== Occlusion vs removal ===")
    for use in (False, True):
        res = detect_changes_detailed(
            occ.query.primary().xy(), occ.reference.primary().xy(),
            a.relative_pose, ChangeConfig(use_occlusion=use),
        )
        rem = change_prf(res.removed(), occ.removed_centers, "removed")
        fp_occluded = change_prf(res.removed(), occ.occluded_centers, "removed").tp
        print(f"use_occlusion={use}: removed={len(res.removed())} "
              f"(true removed recall={rem.recall:.2f}, occluded mis-reported={fp_occluded}) "
              f"comparable={res.comparable_ratio:.2f} occluded={res.occluded_ratio:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "change_eval.json").write_text(
        json.dumps({"added": add_prf.to_dict(), "removed": rem_prf.to_dict(),
                    "dynamic_filtered": len(dynamic)}, indent=2),
        encoding="utf-8",
    )
    fig = save_change_figure(
        case.reference.primary().xy(), case.query_frames[0].primary().xy(),
        aligner.align(case.query_frames[0], case.reference).relative_pose,
        changes, OUT_DIR / "change_evidence.png",
    )
    print(f"\nReport written to: {OUT_DIR / 'change_eval.json'}")
    if fig:
        print(f"Change figure written to: {fig}")


if __name__ == "__main__":
    main()
