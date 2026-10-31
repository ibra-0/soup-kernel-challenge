"""Soup's rewind recorder without the extra forward pass.

Why: TRL's SFTTrainer.compute_loss computes the loss with a fused LM head (no logits). When
Soup's RewindTrainer asks it for `return_outputs=True`, TRL re-runs the whole model under
no_grad to build the full (B, T, V) logits ("recomputed with an extra forward pass"), and
Soup then takes per-row cross-entropy from those logits. That is one extra forward per
step and a ~0.6 GB logits tensor alive while the training graph is still in memory.

What this does: the fused forward already returns per-token log-probabilities
(`outputs.log_probs`, zero on ignored positions). A forward hook captures that output
object and the per-row mean supervised-token cross-entropy is computed from it.
rewind.jsonl keeps the same format and the same quantity; only its source changes.
Nothing else of Soup's outputs (loss, entropy, accuracy, num_tokens) is touched.

Safety: the fast path is used only for a real CUDA torch model and only if the captured
log-probs line up with the labels (zero on every ignored position). Otherwise it falls back
to the original behaviour (the extra forward), so results are never silently wrong.
Counters are printed at exit as "[fast_rewind] ..." so a run shows which path was used.
"""

import atexit
import functools

import torch

STATS = {"fast": 0, "fallback": 0, "orig": 0}
_NOTES = []


def _note(text):
    if text not in _NOTES and len(_NOTES) < 8:
        _NOTES.append(text)


def _report():
    print(f"[fast_rewind] steps: {STATS}; notes: {_NOTES}", flush=True)


def _fast_row_losses(log_probs, labels):
    """Per-row mean CE from per-token log-probs, or None if they do not line up."""
    lp = log_probs.detach()
    if lp.ndim != 2 or lp.shape[0] != labels.shape[0]:
        _note(f"log_probs shape {tuple(lp.shape)} vs labels {tuple(labels.shape)}")
        return None
    supervised = labels[:, 1:] != -100  # label t+1 is predicted at position t
    if lp.shape[1] == labels.shape[1]:
        expected = torch.zeros_like(labels, dtype=torch.bool)
        expected[:, :-1] = supervised
    elif lp.shape[1] == labels.shape[1] - 1:
        expected = supervised
    else:
        _note(f"log_probs shape {tuple(lp.shape)} vs labels {tuple(labels.shape)}")
        return None
    # log_probs must be zero on every ignored position, otherwise the alignment is different
    if bool(((lp != 0) & ~expected).any()):
        _note("log_probs nonzero on ignored positions: alignment differs")
        return None
    tokens = expected.sum(dim=1)
    totals = (-lp.float()).sum(dim=1)
    means = totals / tokens.clamp(min=1).to(totals.dtype)
    return [float(m) for m in means], [int(t) for t in tokens]


def patch_fast_rewind():
    import soup_cli.trainer.rewind_hf as rh

    if getattr(rh, "_fast_rewind_patched", False):
        return

    original_factory = rh.make_rewind_trainer_class  # lru_cache'd: one class per base

    @functools.lru_cache(maxsize=None)
    def make_rewind_trainer_class(base_cls):
        cls = original_factory(base_cls)
        original_compute_loss = cls.compute_loss
        _note("factory used")

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            state = getattr(self, rh._STATE_ATTR, None)
            if (
                state is None
                or state.failed
                or return_outputs
                or not isinstance(model, torch.nn.Module)
                or not getattr(model, "training", False)
                or "labels" not in inputs
                or "shift_labels" in inputs
                or not torch.is_tensor(inputs["labels"])
                or not inputs["labels"].is_cuda
            ):
                STATS["orig"] += 1
                return original_compute_loss(
                    self, model, inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            labels = inputs["labels"]
            captured = []

            def hook(module, args, kwargs, output):
                if kwargs.get("fused_lm_head") and not captured:
                    captured.append(output)

            handle = model.register_forward_hook(hook, with_kwargs=True)
            try:
                # base trainer's compute_loss (TRL), skipping the rewind override
                loss = super(cls, self).compute_loss(
                    model, inputs,
                    return_outputs=False,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                handle.remove()

            try:
                result = None
                out = captured[0] if captured else None
                log_probs = getattr(out, "log_probs", None)
                if log_probs is None:
                    _note("no captured log_probs (hook saw no fused forward)")
                else:
                    result = _fast_row_losses(log_probs, labels)
                if result is None:
                    STATS["fallback"] += 1
                    with torch.no_grad():
                        logits_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                        logits = model(**logits_inputs).logits
                    result = rh.row_losses(logits, labels)
                else:
                    STATS["fast"] += 1
                row_loss, row_tokens = result
                state.on_micro_batch(
                    step=int(self.state.global_step) + 1,
                    row_loss=row_loss,
                    row_tokens=row_tokens,
                )
            except Exception as exc:  # never raise into training
                state.fail(exc)
            return loss

        cls.compute_loss = compute_loss
        return cls

    rh.make_rewind_trainer_class = make_rewind_trainer_class
    rh._fast_rewind_patched = True
    atexit.register(_report)
