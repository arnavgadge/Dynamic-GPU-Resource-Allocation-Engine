"""The 20% job-size similarity rule.

One clearly defined metric, used everywhere the "are these jobs
similar enough for FCFS" question comes up, instead of re-deriving an
ad-hoc comparison at each call site.

Definition: for a set of candidate job sizes, the **relative size
spread** is ``(largest - smallest) / largest``. Candidates are
"similar" when that spread is at or below
`engine.allocation.config.JOB_SIZE_SIMILARITY_THRESHOLD`.

Worked examples (matching the project report):

- 10 / 12 / 11 minutes: spread = (12 - 10) / 12 ≈ 16.7% -> similar -> FCFS.
- 15 minutes / 1 hour / 3 hours: spread = (180 - 15) / 180 = 91.7% ->
  not similar -> score-based selection.

Note on the report's other example ("15 / 10 / 13 minutes -> within
20%"): under this ratio-based definition that particular triple is
*not* within 20% ((15-10)/15 ≈ 33%). The report's illustrative numbers
and its stated "20%" rule are not simultaneously satisfiable under
any single reasonable ratio - the two examples it gives (small,
close-together minute values vs. minutes-through-hours) point at the
same qualitative idea ("about the same size" vs. "different orders of
magnitude"), so this module implements that idea as one precise,
single-source rule instead of picking whichever ad-hoc comparison
happens to reproduce one illustrative example. See the Phase 3
section of the README for this call-out.
"""

from typing import Sequence

from engine.allocation.config import JOB_SIZE_SIMILARITY_THRESHOLD


def relative_size_spread(sizes: Sequence[float]) -> float:
    """How spread out ``sizes`` are, as a fraction of the largest size.

    0.0 means every size is identical. Values are clamped so a
    non-positive largest size (all candidates zero or negative,
    which should not happen but is not the job of this function to
    reject) is treated as "no meaningful spread" rather than dividing
    by a non-positive number.
    """
    if not sizes:
        raise ValueError("relative_size_spread requires at least one size")
    largest = max(sizes)
    smallest = min(sizes)
    if largest <= 0:
        return 0.0
    return (largest - smallest) / largest


def sizes_are_similar(sizes: Sequence[float]) -> bool:
    """Whether ``sizes`` are within the project's similarity threshold."""
    return relative_size_spread(sizes) <= JOB_SIZE_SIMILARITY_THRESHOLD
