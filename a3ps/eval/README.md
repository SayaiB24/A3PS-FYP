# eval/ — evaluation artifacts

## nexar_index_gpu.csv — the split the eval numbers refer to

`data/nexar/index.csv` is gitignored (it lives under `data/`), and the two
laptops do **not** have the same one. The GPU laptop was given extra negative
clips (`../docs/runbooks/GPU_HANDOFF.md` §4) and `prepare_nexar.py` was re-run there, so its
eval split contains ~46 clips that do not exist in this laptop's index at all.

`eval/anticipation.md` and the cached per-clip output under `eval/anticipation/`
were produced against the **GPU laptop's** index. Scoring them with this
laptop's `data/nexar/index.csv` silently drops to 69 of 120 clips (all 60
positives, but only 9 of 60 negatives), which makes the false-alarm rate
meaningless without any error being raised.

This file is a copy of that GPU-laptop index, kept here so the split survives
outside gitignored `data/`. Score against it explicitly:

    python scripts/eval_anticipation.py --index eval/nexar_index_gpu.csv \
        --split eval --clips-dir eval/anticipation

Expect `clips scored: 120 (60 positive, 60 negative)`. If you see 69, you are
using the wrong index.

Splits it defines: eval 120, dev 15, train_traj 220 (355 rows total).
