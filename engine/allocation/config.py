"""The project's fixed allocation policy constants.

One home for every "magic number" the allocation policy depends on,
so the 60/40 weighting and the 20% similarity threshold are each
defined exactly once and imported everywhere they're used - never
re-typed as a literal in `scoring.py`, `similarity.py`, or
`engine.py`.

These are policy inputs, not tunables discovered by the code -
matching the project report exactly:

    Allocation Score = 0.6 x Priority + 0.4 x Job Size Inverse
    Jobs within 20% of each other in size -> FCFS
"""

#: Weight given to (normalized) priority in the allocation score.
#: Priority is weighted higher than size because a missed company
#: deadline is worse than slightly inefficient GPU usage.
PRIORITY_WEIGHT: float = 0.6

#: Weight given to the job-size-inverse component of the allocation
#: score. Smaller jobs still matter - they finish faster and free
#: GPU capacity sooner for whoever is waiting next.
SIZE_WEIGHT: float = 0.4

#: Two waiting jobs are treated as "similar enough in size" to use
#: FCFS instead of score-based selection when their relative size
#: spread (see `engine.allocation.similarity.relative_size_spread`)
#: is at or below this fraction.
JOB_SIZE_SIMILARITY_THRESHOLD: float = 0.20

assert abs((PRIORITY_WEIGHT + SIZE_WEIGHT) - 1.0) < 1e-9, (
    "PRIORITY_WEIGHT and SIZE_WEIGHT must sum to 1.0 - the project's "
    "formula is a weighted average, not two independent terms."
)

#: Starvation-prevention aging (100-scenario validation, Phase 2). A
#: waiting job's *effective* score gains a small, capped bonus for
#: every minute it has waited - on top of, never in place of, the
#: 0.6/0.4 priority+size formula above. Deliberately small per-minute
#: (0.01), so a job that has waited only a few minutes barely moves -
#: aging never "immediately" overturns a fresh priority/size decision
#: (Phase 2's own requirement).
#:
#: The cap is deliberately set *above* 1.0 - the highest base score
#: any job (fresh or not) can ever have (`PRIORITY_WEIGHT +
#: SIZE_WEIGHT == 1.0`) - specifically so the guarantee is real, not
#: approximate: once a job has waited `AGING_MAX_CONTRIBUTION /
#: AGING_RATE_PER_MINUTE` minutes (110 at these defaults, ~1h50m), its
#: total score exceeds *any* possible fresh arrival's maximum possible
#: score (base 1.0 + zero aging), regardless of that job's own base
#: score being as low as 0. That is what actually prevents a
#: continuous stream of new work from starving a job forever,
#: deterministically - not merely "usually". Both numbers are
#: ordinary module-level constants, exactly like `PRIORITY_WEIGHT`
#: above - change them here, nowhere else.
AGING_RATE_PER_MINUTE: float = 0.01
AGING_MAX_CONTRIBUTION: float = 1.1
