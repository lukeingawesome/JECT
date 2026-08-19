"""GATE — no geometric augmentation anywhere in the training path.

54.6% of train and 57.7% of val finding sentences name a side ("right lower lobe"), so a
random flip trains the model to answer the opposite of what it was asked while the loss looks
fine. This is an executable assert rather than a comment because the first person who wants
more augmentation will reach for flips, and a comment does not stop them.

The same reasoning is why `predict_split.py` ships translation TTA and not flip TTA -- see
its module docstring, and tests/test_predict_split_tta.py for the inverse it must respect.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TRAINER = SCRIPTS / "train_voxtell.py"          # holds VoxTellPatches, the loader, and the loop
PREDICT = SCRIPTS / "predict_split.py"

# MONAI's geometric transforms, dict and array form. Intensity transforms are fine and expected.
FORBIDDEN = {
    "RandFlip", "RandFlipd", "RandAxisFlip", "RandAxisFlipd",
    "RandRotate", "RandRotated", "RandRotate90", "RandRotate90d",
    "RandAffine", "RandAffined", "RandZoom", "RandZoomd",
    "Flip", "Flipd", "Rotate90", "Rotate90d",
}


def _names(path: Path) -> set[str]:
    """Every identifier the module names, whether called, imported or referenced."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            found.update(a.name for a in node.names)
    return found


@pytest.mark.parametrize("path", [TRAINER, PREDICT], ids=lambda p: p.name)
def test_no_geometric_augmentation(path: Path) -> None:
    assert path.is_file(), f"{path} missing — the gate cannot protect a file that is not there"
    offenders = sorted(_names(path) & FORBIDDEN)
    assert not offenders, (
        f"{path.name} references geometric augmentation {offenders}. "
        "54.6% of train sentences name a side; a flip makes the target the mirror of the "
        "question. If a flip is genuinely wanted, the sentence must be flipped with it."
    )


def test_the_gate_can_actually_fail(tmp_path: Path) -> None:
    """A gate that cannot go red is decoration. Feed it a file that violates the rule."""
    bad = tmp_path / "bad_trainer.py"
    bad.write_text("from monai.transforms import Compose, RandFlipd\n"
                   "t = Compose([RandFlipd(keys=['image'], prob=0.5)])\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="RandFlipd"):
        test_no_geometric_augmentation(bad)
