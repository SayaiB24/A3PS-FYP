"""Tests for the frozen dev/eval split guard (a3ps/common/splits.py).

These are the tests that matter most in this repo: the eval split is drawn by a
pool-size-dependent shuffle, so a silent re-draw is both easy to cause and
invisible in review. Every test here is a scenario that must NOT be allowed to
pass quietly.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.splits import (  # noqa: E402
    SplitFreezeError,
    apply_freeze,
    build_freeze,
    load_freeze,
    normalize_clip_id,
    save_freeze,
    verify_freeze,
)


def _rec(clip_id, label, split):
    return {"clip_id": clip_id, "label": label, "split": split}


def _pool():
    """A small stand-in for the real index: 2 dev, 4 eval, 2 train_traj."""
    return [
        _rec("1", 0, "dev"), _rec("2", 1, "dev"),
        _rec("10", 1, "eval"), _rec("11", 1, "eval"),
        _rec("12", 0, "eval"), _rec("13", 0, "eval"),
        _rec("20", 0, "train_traj"), _rec("21", 0, "train_traj"),
    ]


# ---------------------------------------------------------------------------
# id normalisation -- the project writes clip ids three different ways
# ---------------------------------------------------------------------------

def test_normalize_clip_id_handles_the_three_project_spellings():
    assert normalize_clip_id("00584") == "584"
    assert normalize_clip_id("584") == "584"
    assert normalize_clip_id(584) == "584"
    assert normalize_clip_id(" 584 ") == "584"
    assert normalize_clip_id("dev01") == "dev01"     # non-numeric passes through
    assert normalize_clip_id("") == ""


def test_freeze_matches_zero_padded_ids_against_bare_ids():
    freeze = build_freeze([_rec("00010", 1, "eval")])
    ok, problems = verify_freeze([_rec("10", 1, "eval")], freeze)
    assert ok, problems


# ---------------------------------------------------------------------------
# build / round-trip
# ---------------------------------------------------------------------------

def test_build_freeze_records_only_dev_and_eval():
    freeze = build_freeze(_pool())
    assert set(freeze["clips"]) == {"1", "2", "10", "11", "12", "13"}
    assert freeze["counts"]["dev"] == {"pos": 1, "neg": 1}
    assert freeze["counts"]["eval"] == {"pos": 2, "neg": 2}


def test_freeze_round_trips_through_disk(tmp_path):
    path = str(tmp_path / "freeze.json")
    save_freeze(build_freeze(_pool()), path)
    loaded = load_freeze(path)
    ok, problems = verify_freeze(_pool(), loaded)
    assert ok, problems


def test_load_freeze_missing_path_returns_none():
    assert load_freeze(None) is None
    assert load_freeze("does/not/exist.json") is None


def test_load_freeze_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"schema_version": 99, "clips": {}}', encoding="utf-8")
    with pytest.raises(SplitFreezeError, match="schema_version"):
        load_freeze(str(path))


# ---------------------------------------------------------------------------
# the actual hazard: growing the pool must not move the held-out set
# ---------------------------------------------------------------------------

def test_new_negatives_join_train_traj_and_never_the_held_out_set():
    freeze = build_freeze(_pool())
    grown = _pool() + [_rec(str(100 + i), 0, "eval") for i in range(5)]
    report = apply_freeze(grown, freeze)

    # "unpinned" counts every negative absent from the freeze, which
    # includes the two clips already in train_traj -- 5 new + 2 existing.
    assert report["n_unpinned_negatives"] == 7
    # The five newcomers wanted "eval"; the freeze reassigns them.
    for rec in grown:
        if rec["clip_id"] in {"100", "101", "102", "103", "104"}:
            assert rec["split"] == "train_traj"
    # Held-out membership is unchanged in size and identity.
    assert {r["clip_id"] for r in grown if r["split"] == "eval"} == {"10", "11", "12", "13"}
    assert {r["clip_id"] for r in grown if r["split"] == "dev"} == {"1", "2"}


def test_apply_freeze_restores_a_reshuffled_split():
    """The exact failure mode: prepare_nexar.py re-draws, the freeze undoes it."""
    freeze = build_freeze(_pool())
    reshuffled = [
        _rec("1", 0, "train_traj"),    # was dev
        _rec("2", 1, "eval"),          # was dev
        _rec("10", 1, "eval"),
        _rec("11", 1, "train_traj"),   # was eval  <-- leaked into training
        _rec("12", 0, "eval"),
        _rec("13", 0, "dev"),          # was eval
        _rec("20", 0, "eval"),         # was train_traj <-- new clip in held-out set
        _rec("21", 0, "train_traj"),
    ]
    ok, problems = verify_freeze(reshuffled, freeze)
    assert not ok and len(problems) >= 4

    apply_freeze(reshuffled, freeze)
    ok, problems = verify_freeze(reshuffled, freeze)
    assert ok, problems


def test_new_positives_are_left_unassigned_not_auto_added():
    """Positives are scarce; they must be placed by a decision, not a default."""
    freeze = build_freeze(_pool())
    grown = _pool() + [_rec("200", 1, ""), _rec("201", 1, "")]
    report = apply_freeze(grown, freeze)

    assert report["n_unpinned_positives"] == 2
    assert report["unpinned_positives"] == ["200", "201"]
    for rec in grown:
        if rec["clip_id"] in {"200", "201"}:
            assert rec["split"] == "", "a new positive must not silently join a split"


def test_apply_freeze_raises_when_a_frozen_clip_disappears():
    freeze = build_freeze(_pool())
    shrunk = [r for r in _pool() if r["clip_id"] != "12"]
    with pytest.raises(SplitFreezeError, match="absent from the index"):
        apply_freeze(shrunk, freeze)


def test_apply_freeze_raises_when_a_frozen_clip_changes_label():
    freeze = build_freeze(_pool())
    relabelled = _pool()
    for r in relabelled:
        if r["clip_id"] == "12":
            r["label"] = 1          # was a negative
    with pytest.raises(SplitFreezeError, match="changed label"):
        apply_freeze(relabelled, freeze)


def test_verify_flags_an_index_that_enlarges_the_held_out_set():
    freeze = build_freeze(_pool())
    grown = _pool() + [_rec("300", 0, "eval")]
    ok, problems = verify_freeze(grown, freeze)
    assert not ok
    assert any("not in the freeze" in p for p in problems)


def test_verify_flags_a_label_change_without_mutating_records():
    freeze = build_freeze(_pool())
    records = _pool()
    for r in records:
        if r["clip_id"] == "10":
            r["label"] = 0
    ok, problems = verify_freeze(records, freeze)
    assert not ok
    assert any("frozen label" in p for p in problems)
    # verify_freeze is read-only.
    assert [r["split"] for r in records] == [r["split"] for r in _pool()]


# ---------------------------------------------------------------------------
# the real repository artifact
# ---------------------------------------------------------------------------

def test_committed_freeze_matches_the_index_it_was_drawn_from():
    """eval/split_freeze.json must agree with eval/nexar_index_gpu.csv.

    This is the guard that fires if someone re-runs prepare_nexar.py over a
    grown clip pool and commits the result.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    freeze_path = os.path.join(root, "eval", "split_freeze.json")
    index_path = os.path.join(root, "eval", "nexar_index_gpu.csv")
    if not (os.path.isfile(freeze_path) and os.path.isfile(index_path)):
        pytest.skip("freeze or index not present")

    import csv
    with open(index_path, newline="", encoding="utf-8-sig") as fh:
        records = [{"clip_id": r["clip_id"], "label": int(float(r["label"])),
                    "split": r.get("split", "") or ""}
                   for r in csv.DictReader(fh)]

    ok, problems = verify_freeze(records, load_freeze(freeze_path))
    assert ok, "held-out split drifted:\n" + "\n".join(problems[:20])
