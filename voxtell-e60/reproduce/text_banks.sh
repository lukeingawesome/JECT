#!/usr/bin/env bash
# The two text banks, built once. Everything after this reads the .npz/.npy on disk and never
# loads Qwen3 again -- which is why REXG_QWEN_DIR is only needed here.
#
# TWO banks, not one, and the second is not optional for arm E: the pooled bank is one vector
# per sentence (what the stock model consumes) and the token bank is the per-token sequence
# the PWAM fusion attends over. They must describe the SAME sentence set, which is why the
# token bank reads its sentence list off the pooled bank rather than recomputing it.
#
# TWO PAIRS, because a submission needs prompts a training bank does not have. The default
# bank covers findings with a mask (has_seg == 1); the test split has none, so its sentences
# are absent and `gate_bank_covers` would refuse at predict time -- before the GPU hours, not
# after. `--bank-splits train,val,test` is what produces the `_all` pair.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
DEV=${DEVICE:-cuda:0}
PAIRED=$("$PY" -c "
import sys; sys.path.insert(0, 'src')
from rexg_data.config import load_data_config
print(load_data_config('configs/rexg_native.yaml').paired_root)")

echo "banks -> $PAIRED"

# training / val: the sentences that have a mask
"$PY" scripts/train_voxtell.py --build-text-bank-only  --device "$DEV"
"$PY" scripts/train_voxtell.py --build-token-bank-only --device "$DEV"

# submission: every split's sentences. The token bank derives its own name from the pooled
# one (token_bank_path_for), so passing --text-bank to both is what keeps the pair together.
ALL="$PAIRED/text_emb_qwen3_all.npz"
"$PY" scripts/train_voxtell.py --build-text-bank-only  --device "$DEV" \
    --text-bank "$ALL" --bank-splits train,val,test
"$PY" scripts/train_voxtell.py --build-token-bank-only --device "$DEV" --text-bank "$ALL"

echo "done: $PAIRED/text_emb_qwen3.npz and $ALL, each with its token bank"
