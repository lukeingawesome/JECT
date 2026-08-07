"""Dataset locations and the input cube size.

Import this MODULE and read `paths.DATA` etc. -- never `from paths import DATA`, which binds the
value at import time so a later CLI override would not be seen.

There are no machine-specific defaults: an unset path fails loudly in train.py rather than silently
training on the wrong data.
  JECT_DATA : dir of per-volume .npz, each with `ct` and `rex`
  JECT_REX  : ReXGrounding CSV (object_id, rex_split, rex_findings)
  JECT_EMB  : .pt dict {phrase -> text embedding}
  JECT_RES  : cube side the volume is cropped to (default 224)
"""
import os

DATA = os.environ.get("JECT_DATA", "")
REX_CSV = os.environ.get("JECT_REX", "")
TEXT_EMB = os.environ.get("JECT_EMB", "")
_RES = int(os.environ.get("JECT_RES", "224"))
T = (_RES, _RES, _RES)
