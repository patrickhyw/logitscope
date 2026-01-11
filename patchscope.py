from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transformer_lens.hook_points import HookPoint
from torch import Tensor
import torch


def next_token_patchscope(
    model: HookedTransformer,
    prompt_src: str,
    prompt_tgt: str,
    layer_src: int,
    layer_tgt: int,
    pos_src: int,
    pos_tgt: int,
    pos_pred: int,
) -> tuple[float, float, Tensor]:
    """
    Patch the residual activation from a layer and position in the source prompt to a
    layer and position in the target prompt. Evaluates whether the predicted next token
    after patching (at pos_pred in prompt_tgt) matches the predicted next token in the
    source prompt (at pos_src in prompt_src).

    Returns precision@1, surprisal, and the logits as a 1D vector.

    Credit: code adapted from the repo of https://arxiv.org/pdf/2401.06102 to use
    TransformerLens instead of PyTorch hooks.
    """
    logits_src, cache = model.run_with_cache(prompt_src)
    logits_src = logits_src[0, pos_src]
    probs_src = torch.softmax(logits_src, dim=-1)
    pred_src = torch.argmax(probs_src)

    def hook(act: Tensor, hook: HookPoint):
        act[:, pos_tgt] = cache["resid_post", layer_src][:, pos_src]
        return act

    logits_tgt = model.run_with_hooks(
        prompt_tgt, fwd_hooks=[(get_act_name("resid_post", layer_tgt), hook)]
    )
    logits_tgt = logits_tgt[0, pos_pred]
    probs_tgt = torch.softmax(logits_tgt, dim=-1)
    pred_tgt = torch.argmax(probs_tgt)

    prec1 = float((pred_src == pred_tgt).item())
    surprisal = -torch.log(probs_tgt[pred_src]).item()
    return prec1, surprisal, logits_tgt
