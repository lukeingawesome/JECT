# CLAUDE.md — JECT

Rules for changing code in this repo. Engineering discipline comes from
[`my_agent/CLAUDE.md`](my_agent/CLAUDE.md) (its own repo, vendored here); this file states how it
applies to JECT and adds the one rule that is specific to us: **where a change is allowed to land.**

---

## 1. Two standards, two directories

| path | in git | standard |
|---|---|---|
| `local/` | **no** (gitignored) | none. Scratch, notes, one-off analysis, cached results. Messy is fine. |
| `grounding/`, `docker/` | yes | published. Every line is read by someone reproducing a number. |
| `my_agent/` | no (separate repo) | do not edit from here — change it upstream. |

**New work goes to `local/` by default.** `grounding/` changes only when the change is *crucial*,
and then it lands *clean*. Exploration, probes, ensembling and TTA sweeps, threshold studies,
miss analyses, per-session handoffs — those live in `local/` and stay there, even when they
produce the headline number. `local/tools/ensemble_eval.py` produced the best result on record
(0.3316) and still belongs in `local/`, because nothing in the training path depends on it.

### The promotion test — `local/` → `grounding/`

All four must hold. If any fails, it stays in `local/`.

1. **Load-bearing.** Something reported, reproduced, or launched depends on it. "It might be
   useful later" is not a reason; that is what `local/` is for.
2. **Defaults unchanged.** Every prior result still reproduces with no environment set. New
   behaviour is opt-in — a flag defaulting off, an env var with the old value as its default.
   Where a change touches the model, verify the default is **bit-identical** to the previous
   commit (same state_dict keys and shapes, same forward output from a fixed seed) and say so in
   the commit message, as `72f5920` did.
3. **Carries its own reason.** A comment stating *why*, with the measurement that motivated it —
   the house style in `grounding/reproduce/*.sh`. A reader must be able to tell an empirical
   finding from a guess without asking.
4. **Reviewed.** `/codex:adversarial-review --wait` before it counts as complete. User-invoked;
   name the command, do not try to run it.

### Corollary

Do not tidy `grounding/` on the way past. Renaming, reformatting, adding hooks "for consistency",
or parameterising a recipe nobody is running are all `local/`-grade motivations applied to
published code. If a recipe needs an env hook, add it in the commit that needs it.

---

## 2. How to change code

Full text in `my_agent/CLAUDE.md`. The four that bite hardest here:

- **Think before coding.** State assumptions; if two readings exist, surface both rather than
  picking silently. Ask when the answer changes the work.
- **Simplicity first.** Nothing speculative. No configurability that was not requested — every
  extra knob in a compose file or a recipe is a knob someone must later understand.
- **Surgical changes.** Every changed line traces to the request. Do not improve adjacent code.
  Match the surrounding style even where you would write it differently.
- **Verify, then claim.** Report what you actually ran. `bash -n`, a parse, a real invocation
  against real paths — evidence produced this session, not "should work".

Two more that come from this project's own scar tissue:

- **One variable per run.** Unstacking two recall biases in the same run drove Dice to 0.0000 and
  cost a training cycle to attribute. Change one thing, in its own run.
- **`--epochs` is fixed at launch.** `OneCycleLR` derives its whole schedule from it and cannot
  resume or extend. Measure one epoch, then set it. This is why containers never auto-restart.

---

## 3. Standing facts

- Two runs may be detached (`ppid=1`) on the GPUs at any time. **Check before launching anything;
  never kill one you did not start.** State lives in `local/HANDOFF.md`.
- Data, checkpoints and run logs are **outside this repo**, under `/opt/project/cepa2/art/`.
  Nothing is hardcoded: `JECT_DATA`, `JECT_REX`, `JECT_EMB`, `JECT_INIT` are required and fail
  loudly when unset, deliberately.
- Volumes are centre-cropped **in voxels**. Pairing a 224 run with the 384³ data trains happily on
  the wrong field of view and merely reports a worse number. Pair `--res` with the matching data
  and backbone every time — `grounding/SETUP.md` §2, checked by `docker/preflight.py`.
- `my_agent`'s skills are not installed into `.claude/skills/` yet:
  `./my_agent/install.sh . --copy` (copy, not symlink — `my_agent/` is gitignored, so symlinks
  would dangle for anyone cloning JECT).
