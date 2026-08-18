# Results
 
Regenerable output. Nothing here is tracked by git, as files would be too large / pointless.
 
- `holdout/<family>/chunk_*.csv` — per-chunk metrics, written as the run goes
- `holdout/full_<family>_lp<..>_la<..>.csv` — the chunks concatenated
- `holdout/manifest.json` — pair count, draws, seed, noise point, wall time
Produced by:
 
    python scripts/run_holdout.py --family all --n 5000 --draws 256
 
Chunks already present are skipped, so an interrupted run resumes safely. The noise
draw and the flow sampling are seeded on the chunk's first row index, so a
resumed run agrees with an uninterrupted one.

