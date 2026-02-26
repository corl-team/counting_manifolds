import os
import textwrap
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyrallis
import torch
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from loguru import logger
from sae_lens import SAE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding


@dataclass
class Config:
    seed: int

    model_name: str
    dataset_name: str
    output_path: str
    batch_size: int
    num_workers: int

    line_length: int
    num_samples: int
    min_lines: int
    n_components: int
    max_seq_len: int
    use_chat: bool

    log_first_n: int
    sae_top_k_for_save: int

    pca_n_omit: int
    pca_n_components: int

    num_features_for_sae_span: int

    run_best_layer_only: bool
    best_layer: int

    def __post_init__(self):
        model_name = os.path.basename(self.model_name.strip("/"))
        dataset_name = os.path.basename(self.dataset_name.strip("/"))
        self.output_path = os.path.join(self.output_path, model_name, dataset_name)
        os.makedirs(self.output_path, exist_ok=True)


def get_mean_hiddens(hiddens, character_counts):
    mean_hiddens = []
    for count in np.unique(character_counts).tolist():
        mean_hiddens.append(hiddens[character_counts == count].mean(axis=0))
    mean_hiddens = np.vstack(mean_hiddens)

    return mean_hiddens


def get_num_layers(config: Config, model):
    if (
        "llama" in config.model_name
        or "Qwen3" in config.model_name
        or "gemma-2" in config.model_name
        or "gpt2" in config.model_name
        or "pythia" in config.model_name
    ):
        logger.info("It is Llama or Qwen3")
        num_layers = model.config.num_hidden_layers
    elif "gemma-3" in config.model_name:
        logger.info("It is gemma")
        cfg = model.config
        text_cfg = (
            cfg.get_text_config()
            if hasattr(cfg, "get_text_config")
            else getattr(cfg, "text_config", cfg)
        )
        num_layers = text_cfg.num_hidden_layers
    else:
        raise ValueError(f"Unknown model name: {config.model_name}")

    logger.info(f"Num Layers: {num_layers}")

    return num_layers


def resolve_max_seq_len(config, model, tokenizer) -> int:
    model_max = int(
        getattr(model, "max_seq_len", 0)
        or getattr(getattr(model, "config", None), "max_position_embeddings", 0)
        or getattr(tokenizer, "model_max_length", 0)
        or 0
    )
    req = config.max_seq_len

    assert req > 0, (
        "config must define max_seq_len / max_seq_length / max_length / seq_len"
    )
    assert model_max > 0, "could not infer model max sequence length"
    assert req <= model_max, f"config max_seq_len={req} exceeds model limit={model_max}"
    return req


def assert_lines_token_limit(text: str, tokenizer, line_length: int):
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    for line in text.split("\n"):
        ids = tokenizer.encode(line, add_special_tokens=False)
        assert ids.count(nl) == 0, "Unexpected newline token inside a line"
        assert len(ids) <= line_length, (len(ids), line_length, line[:200])


def wrap_preserve_newlines(text: str, width: int, **wrap_kwargs) -> str:
    wrapper = textwrap.TextWrapper(width=width, **wrap_kwargs)

    out_lines = []
    for line in text.splitlines(keepends=False):
        if line.strip() == "":  # preserve empty/whitespace-only lines
            out_lines.append(line)
        else:
            out_lines.extend(wrapper.wrap(line))
    return "\n".join(out_lines)


def make_line_wrapper(
    line_length: int, text_key: str = "text", out_key: str = "text_lines"
):
    def _fn(ex):
        ex[out_key] = wrap_preserve_newlines(ex[text_key], width=line_length)
        return ex

    return _fn


def assert_chars_since_nl_map(line_length: int, key: str = "chars_since_nl"):
    def _fn(batch):
        bad = [
            (i, m)
            for i, seq in enumerate(batch[key])
            if (m := max(seq, default=0)) > line_length
        ]
        assert not bad, (
            f"chars_since_nl > line_length={line_length} for {len(bad)} seqs; first={bad[0]}"
        )
        return batch

    return _fn


def make_forward_inputs_with_chars_since_nl(
    tokenizer,
    max_seq_len: int,
    use_chat: bool,
    text_key: str = "text_lines",
    out_chars_key: str = "chars_since_nl",
    add_generation_prompt: bool = False,
):
    """
    Returns a batched function suitable for Dataset.map(batched=True) that outputs:
      - input_ids
      - attention_mask
      - chars_since_nl  (same length as input_ids; special/template tokens => 0)

    chars_since_nl is computed on the ORIGINAL text (batch[text_key]) and ignores
    any chat-template-added tokens.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("This requires a *fast* tokenizer to get offset_mapping.")

    special_ids = set(getattr(tokenizer, "all_special_ids", ()))

    def render_and_span(x: str) -> Tuple[str, int, int]:
        """Return (rendered_text, start, end) where [start:end] corresponds to x."""
        if not use_chat:
            return x, 0, len(x)

        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": x}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        start = rendered.find(x)
        if start == -1:
            # Fallback: if something odd happens, treat everything as template (all zeros).
            # (Better than silently misaligning.)
            return rendered, 0, 0
        return rendered, start, start + len(x)

    def _fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        xs: List[str] = batch[text_key]
        rendered_list, spans = [], []
        for x in xs:
            rendered, s, e = render_and_span(x)
            rendered_list.append(rendered)
            spans.append((s, e))

        enc = tokenizer(
            rendered_list,
            add_special_tokens=not use_chat,  # chat template already includes needed markers
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        out_chars: List[List[int]] = []
        for x, (span_s, span_e), ids, offs, spmask in zip(
            xs,
            spans,
            enc["input_ids"],
            enc["offset_mapping"],
            enc["special_tokens_mask"],
        ):
            last_nl = -1
            cols: List[int] = []

            for tid, (s, e), is_sp in zip(ids, offs, spmask):
                tid = int(tid)

                # Anything special OR outside the user-text span => 0
                if is_sp or tid in special_ids or not (span_s <= s and e <= span_e):
                    cols.append(0)
                    continue

                # Map rendered offsets -> original text offsets
                s0, e0 = s - span_s, e - span_s

                # Update last newline if a '\n' appears within this token span
                nl = x.rfind("\n", s0, e0)
                if nl != -1:
                    last_nl = nl

                # Chars since last '\n', INCLUDING current token span
                cols.append(e0 - last_nl - 1)

            out_chars.append(cols)

        enc[out_chars_key] = out_chars
        # Return only what you need (keep offsets/masks if you want debugging)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            out_chars_key: enc[out_chars_key],
            # Optional, useful for debugging:
            # "offset_mapping": enc["offset_mapping"],
            # "special_tokens_mask": enc["special_tokens_mask"],
            # "rendered_text": rendered_list,
        }

    return _fn


def collect_layer_hiddens(
    dataset,
    model,  # compiled outside
    tokenizer,
    layer_idx: int,
    batch_size: int,
    num_workers: int = 0,
    max_seq_len: Optional[int] = None,  # pass config.max_seq_len for fixed shapes
    pad_to_multiple_of: Optional[int] = None,  # used only if max_seq_len is None
) -> List[torch.Tensor]:
    """
    Returns:
      layer_hiddens: list(len(dataset)) of [seq_len, hidden_dim] CPU tensors
      p_newline:     list(len(dataset)) of [seq_len] CPU tensors, where p_newline[t]
                     is P(next_token == "\\n" | prefix up to token t)

    Notes:
      - hidden_states = (embeddings, layer0, ..., layerN-1), so we map layer_idx -> +1.
      - seq_len is the *unpadded* length per sample (from attention_mask sum).
    """
    model.eval()
    device = next(model.parameters()).device

    # Ensure we can pad
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer has neither pad_token_id nor eos_token_id (cannot pad)."
        )

    idx = layer_idx if layer_idx < 0 else layer_idx + 1

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length" if max_seq_len is not None else "longest",
        max_length=max_seq_len,
        pad_to_multiple_of=(pad_to_multiple_of if max_seq_len is None else None),
        return_tensors="pt",
    )

    def collate_fn(examples):
        # Only model keys; ignore chars_since_nl/offset_mapping/etc
        feats = []
        for ex in examples:
            ids = ex["input_ids"]
            am = ex["attention_mask"]
            feats.append(
                {
                    "input_ids": ids.tolist() if torch.is_tensor(ids) else ids,
                    "attention_mask": am.tolist() if torch.is_tensor(am) else am,
                }
            )
        batch = collator(feats)
        batch["lengths"] = batch["attention_mask"].sum(dim=1, dtype=torch.long)
        return batch

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    layer_out: List[torch.Tensor] = []

    with torch.inference_mode():
        for batch in tqdm(dl, total=len(dl), desc="Collecting layer hiddens + p(\\n)"):
            lengths = batch.pop("lengths").tolist()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            outputs = model(
                **batch, output_hidden_states=True, return_dict=True, use_cache=False
            )
            hs = outputs.hidden_states
            if not (-len(hs) <= idx < len(hs)):
                raise IndexError(
                    f"layer_idx={layer_idx} -> hidden_states[{idx}] out of range (len={len(hs)})"
                )

            layer_h = hs[idx].detach().cpu()  # [B, T, H]

            for i, L in enumerate(lengths):
                if tokenizer.padding_side == "left":
                    layer_out.append(layer_h[i, -L:].contiguous())
                else:
                    layer_out.append(layer_h[i, :L].contiguous())

    assert len(layer_out) == len(dataset), (len(layer_out), len(dataset))

    return layer_out


def mean_hidden_by_chars_since_nl(layer_hiddens, chars_since_nl, max_chars: int):
    """
    layer_hiddens: list of [T, H] tensors
    chars_since_nl: list of length-T int sequences/tensors
    max_chars: line_length (upper bound for chars_since_nl)
    """
    H = int(layer_hiddens[0].shape[-1])
    sums = torch.zeros((max_chars + 1, H), dtype=torch.float32)
    counts = torch.zeros((max_chars + 1,), dtype=torch.long)

    for h, c in zip(layer_hiddens, chars_since_nl):
        h = h.detach().to("cpu", dtype=torch.float32)
        c = torch.as_tensor(c, dtype=torch.long).view(-1)
        assert h.shape[0] == c.numel(), (h.shape, c.shape)

        sums.index_add_(0, c, h)
        counts += torch.bincount(c, minlength=max_chars + 1)

    keys = (counts > 0).nonzero(as_tuple=False).squeeze(1)
    means = sums[keys] / counts[keys].unsqueeze(1).to(torch.float32)
    return means, keys


def save_token_scores(
    dataset,
    scores,
    tokenizer,
    use_chat: bool,
    log_first_n: int,
    out_dir: str | Path,
    text_key: str = "text_lines",
):
    os.makedirs(out_dir, exist_ok=True)

    special_ids = set(getattr(tokenizer, "all_special_ids", ()))

    def _render_with_span(x: str) -> Tuple[str, int, int]:
        """Return (rendered_text, start, end) span for original text within rendered."""
        if not use_chat:
            return x, 0, len(x)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": x}],
            tokenize=False,
            add_generation_prompt=False,
        )
        start = rendered.find(x)
        if start == -1:
            # If something strange happens, treat everything as template; no tokens kept.
            return rendered, 0, 0
        return rendered, start, start + len(x)

    def _keep_mask_for_ids_and_text(ids: List[int], text: str) -> List[bool]:
        """
        Compute which positions to KEEP:
          - drop all special tokens
          - if using a chat template, also drop any tokens outside the user text span
        """
        ids = list(ids)
        if not ids:
            return []

        # Non-chat models: simply drop special tokens.
        if not use_chat:
            return [tid not in special_ids for tid in ids]

        rendered, span_s, span_e = _render_with_span(text)

        enc = tokenizer(
            rendered,
            add_special_tokens=not use_chat,
            truncation=True,
            max_length=len(ids),
            padding=False,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        enc_ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        spmask = enc["special_tokens_mask"]

        # If anything is inconsistent, fall back to dropping only specials.
        if len(enc_ids) < len(ids):
            return [tid not in special_ids for tid in ids]

        keep = []
        for tid, (s, e), is_sp in zip(enc_ids, offs, spmask):
            # Align to dataset ids length
            if len(keep) >= len(ids):
                break

            tid = int(tid)
            # Drop anything special OR outside the user-text span.
            if is_sp or tid in special_ids or not (span_s <= s and e <= span_e):
                keep.append(False)
            else:
                keep.append(True)

        # Pad/truncate to exactly len(ids)
        if len(keep) < len(ids):
            keep.extend([False] * (len(ids) - len(keep)))
        elif len(keep) > len(ids):
            keep = keep[: len(ids)]

        return keep

    n = min(int(log_first_n), len(dataset))
    for i in range(n):
        ex = dataset[i]
        ids = ex["input_ids"].tolist()
        score = scores[i]

        # Convert score to a simple 1D list and align lengths.
        if isinstance(score, torch.Tensor):
            score_list = score.flatten().tolist()
        elif isinstance(score, np.ndarray):
            score_list = score.reshape(-1).tolist()
        else:
            score_list = list(score)

        L = min(len(ids), len(score_list))
        ids = ids[:L]
        score_list = score_list[:L]

        text = ex.get(text_key, "")
        keep_mask = _keep_mask_for_ids_and_text(ids, text)
        keep_mask = keep_mask[:L]

        kept_ids = [tid for tid, k in zip(ids, keep_mask) if k]
        kept_scores = [v for v, k in zip(score_list, keep_mask) if k]

        # Decode kept token ids to strings.
        toks = [tokenizer.decode([tid], skip_special_tokens=False) for tid in kept_ids]

        assert len(toks) == len(kept_scores), (len(toks), len(kept_scores))

        data = pd.DataFrame({"tokens": toks, "values": kept_scores})
        data.to_csv(os.path.join(out_dir, f"sample_data_{i:02d}.csv"), index=False)


def fit_linear_regression_chars_since_nl(
    layer_hiddens: Sequence[torch.Tensor],
    chars_since_nl: Sequence[Union[torch.Tensor, List[int]]],
) -> Dict[str, float]:
    """
    Fits y = X w + b (least squares) where:
      X: all token hidden states stacked  [N, H]
      y: chars_since_nl stacked           [N]

    Returns:
      metrics: {"r2": ..., "rmse": ..., "mae": ...}
      coef: numpy array of shape [H+1] (w followed by bias b)
    """
    X = torch.cat(
        [h.detach().to("cpu", dtype=torch.float32) for h in layer_hiddens], dim=0
    )  # [N,H]
    y = torch.cat(
        [torch.as_tensor(c, dtype=torch.float32).view(-1) for c in chars_since_nl],
        dim=0,
    )  # [N]

    assert X.shape[0] == y.shape[0], (X.shape, y.shape)

    # Add bias column
    ones = torch.ones((X.shape[0], 1), dtype=X.dtype)
    Xb = torch.cat([X, ones], dim=1)  # [N, H+1]

    # Least squares solve
    beta = torch.linalg.lstsq(Xb, y).solution  # [H+1]
    y_hat = Xb @ beta

    resid = y_hat - y
    rmse = torch.sqrt(torch.mean(resid**2)).item()
    mae = torch.mean(torch.abs(resid)).item()

    ss_res = torch.sum((y - y_hat) ** 2)
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot.item() > 0 else float("nan")

    metrics = {"r2": float(r2), "rmse": float(rmse), "mae": float(mae)}
    return metrics


def get_sae(model_name: str, layer: int):
    if "llama3.1-8b" == os.path.basename(model_name):
        logger.info(f"Loading Llama SAE for layer {layer}")
        sae = SAE.from_pretrained(
            release="llama_scope_lxr_32x", sae_id=f"l{layer}r_32x", device="cpu"
        )
    elif "gemma-2-9b" == os.path.basename(model_name):
        logger.info(f"Loading Gemma SAE for layer {layer}")
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-9b-pt-res-canonical",
            sae_id=f"layer_{layer}/width_131k/canonical",  # pick one of the available L0s
            device="cpu",
        )
    elif "gpt2" == os.path.basename(model_name):
        logger.info(f"Loading GPT2 SAE for layer {layer}")
        sae = SAE.from_pretrained(
            release="gpt2-small-resid-post-v5-128k",
            sae_id=f"blocks.{layer}.hook_resid_post",
            device="cpu",
        )
    else:
        logger.warning(f"Model {model_name} not supported")
        sae = None

    return sae


def get_sae_acts(
    sae,
    all_char_counts: torch.Tensor,
    all_hiddens: torch.Tensor,
    device: torch.device,
    batch_size: int = 8192,
):
    mean_acts = []
    frac_acts = []

    wdec_norm = torch.linalg.norm(sae.W_dec, dim=1).to(device)

    for count in tqdm(torch.unique(all_char_counts).tolist()):
        # ignore bos token
        if count == 0:
            continue
        count_hiddens = all_hiddens[all_char_counts == count]
        n = count_hiddens.shape[0]
        n_f = torch.tensor(float(n), device=device)

        act_sum = None
        nz_cnt = None

        with torch.no_grad():
            for batch in count_hiddens.split(batch_size):
                batch = batch.to(device, non_blocking=True)

                acts = sae.encode(batch) * wdec_norm
                acts = acts.float()

                bs = acts.sum(dim=0, dtype=torch.float32)
                bnz = (acts != 0).sum(dim=0, dtype=torch.int64)

                if act_sum is None:
                    act_sum = bs
                    nz_cnt = bnz
                else:
                    act_sum += bs
                    nz_cnt += bnz

        # mean over all samples
        mean = act_sum / n_f

        nz_cnt_f = nz_cnt.to(torch.float32)
        frac = nz_cnt_f / n_f

        mean_acts.append(mean.cpu().numpy())
        frac_acts.append(frac.cpu().numpy())

    mean_acts = np.vstack(mean_acts)
    frac_acts = np.vstack(frac_acts)

    return mean_acts, frac_acts


def top_by_std(sae_acts: np.ndarray, top_k: int):
    std = sae_acts.std(axis=1)  # [L, F]
    top_idx = np.argsort(std, axis=1)[:, ::-1][:, :top_k]  # [L, top_k]
    top_acts = np.take_along_axis(
        sae_acts, top_idx[:, None, :], axis=2
    )  # [L, C, top_k]
    return top_acts, top_idx


def get_newline_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    input_ids: [B, T] token ids
    returns:   [B, T] mask (0/1 by default; bool if return_int=False)

    Safe against -100 and ids >= len(tokenizer).
    Uses tokenizer.batch_decode once to build a cached lookup of which ids decode to a string containing '\n'.
    """
    assert input_ids.ndim == 2, input_ids.shape
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()

    N = len(tokenizer)  # includes added tokens; safer than tokenizer.vocab_size

    # Build/cache lookup on CPU
    lookup = getattr(tokenizer, "_contains_newline_lookup", None)
    if lookup is None or lookup.numel() != N:
        lookup_list = []
        chunk = 4096
        for start in range(0, N, chunk):
            ids = list(range(start, min(N, start + chunk)))
            decoded = tokenizer.batch_decode(
                [[i] for i in ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            lookup_list.extend(("\n" in s) for s in decoded)
        lookup = torch.tensor(lookup_list, dtype=torch.bool, device="cpu")
        tokenizer._contains_newline_lookup = lookup

    # Safe gather (avoid device-side assert)
    valid = (input_ids >= 0) & (input_ids < N)
    assert torch.all(input_ids >= 0), input_ids[input_ids < 0]
    assert torch.all(valid)
    safe_ids = input_ids.clamp(0, N - 1)
    mask = lookup.to(device=input_ids.device, non_blocking=True)[safe_ids] & valid

    return mask.to(dtype=torch.int64)


def eval_next_token_metrics(
    dataset,
    model,
    tokenizer,
    batch_size: int,
    num_workers: int,
    max_seq_len: int,
    log_first_n: int,
) -> Tuple[Dict[str, float], np.ndarray, List[np.ndarray]]:
    """Average next-token NLL loss + accuracy (ignores pads; handles left padding)."""
    model.eval()
    device = next(model.parameters()).device

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_seq_len,
        return_tensors="pt",
    )

    def collate_fn(examples):
        feats = []
        for ex in examples:
            ids = ex["input_ids"]
            am = ex["attention_mask"]
            feats.append(
                {
                    "input_ids": ids.tolist() if torch.is_tensor(ids) else ids,
                    "attention_mask": am.tolist() if torch.is_tensor(am) else am,
                }
            )
        return collator(feats)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    total_nll = 0.0
    total_correct = 0
    total_correct_if_newline = 0
    total_correct_if_newline_any = 0
    total_tokens = 0
    total_newline_tokens = 0

    newline_mass_probs: List[float] = []
    newline_vocab_ids = None  # lazily initialized (on correct V)
    newline_token_probs = []

    with torch.inference_mode():
        for batch in tqdm(dl, total=len(dl), desc="Eval next-token loss/acc"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)

            logits = model(
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=False,
                return_dict=True,
            ).logits  # [B,T,V]

            shift_logits = logits[:, :-1, :].float()  # predicts token t+1
            shift_labels = input_ids[:, 1:]  # gold token t+1
            # valid label positions + require previous token is also non-pad (important for left padding)
            valid = attn[:, 1:].bool() & attn[:, :-1].bool()  # [B,T-1]

            # build vocab ids that contain "\n" once (needs V)
            if newline_vocab_ids is None:
                V = len(tokenizer)
                all_ids_cpu = torch.arange(V, device="cpu").unsqueeze(0)  # [1,V]
                newline_vocab_mask = get_newline_mask(
                    input_ids=all_ids_cpu, tokenizer=tokenizer
                )[0]  # [V] bool
                newline_vocab_ids = (
                    newline_vocab_mask.nonzero(as_tuple=False).squeeze(-1).to(device)
                )

            if valid.any():
                logZ = torch.logsumexp(shift_logits, dim=-1)  # [B,T-1]
                gold = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                nll = logZ - gold  # -log p(gold)

                total_nll += nll[valid].sum().item()
                preds = shift_logits.argmax(dim=-1)
                total_correct += (preds.eq(shift_labels) & valid).sum().item()
                total_tokens += valid.sum().item()

                contains_newline = get_newline_mask(
                    input_ids=shift_labels, tokenizer=tokenizer
                )
                preds_contains_newline = get_newline_mask(
                    input_ids=preds, tokenizer=tokenizer
                )

                total_correct_if_newline += (
                    (preds.eq(shift_labels) & valid & contains_newline).sum().item()
                )
                total_correct_if_newline_any += (
                    (preds_contains_newline & valid & contains_newline).sum().item()
                )
                total_newline_tokens += (valid & contains_newline).sum().item()

                # probability mass on newline-tokens when the *gold next token* contains "\n"
                if newline_vocab_ids.numel() > 0:
                    log_mass = torch.logsumexp(
                        shift_logits[..., newline_vocab_ids], dim=-1
                    )  # [B,T-1]
                    p_mass = (log_mass - logZ).exp()  # [B,T-1]
                    sel = (valid & contains_newline).to(bool)

                    if sel.any():
                        newline_mass_probs.extend(p_mass[sel].float().cpu().tolist())

                    for sample_idx in range(p_mass.shape[0]):
                        if len(newline_token_probs) < log_first_n:
                            newline_token_probs.append(
                                np.array(
                                    p_mass[sample_idx][valid[sample_idx].to(bool)]
                                    .cpu()
                                    .tolist()
                                    + [0]
                                )
                            )

    newline_mass_probs = np.array(newline_mass_probs)
    denom = max(total_tokens, 1)

    return (
        {
            "lm_loss": float(total_nll / denom),
            "lm_acc": float(total_correct / denom),
            "lm_acc_if_newline": float(
                total_correct_if_newline / max(total_newline_tokens, 1)
            ),
            "lm_acc_if_newline_any": float(
                total_correct_if_newline_any / max(total_newline_tokens, 1)
            ),
            "num_tokens": int(total_tokens),
            "num_newline_tokens": int(total_newline_tokens),
            "mean_prob_if_newline": newline_mass_probs.mean(),
        },
        newline_mass_probs,
        newline_token_probs,
    )


def pca_per_layer(
    mean_hiddens: np.ndarray, n_omit: int, n_components: int
) -> Tuple[PCA, np.ndarray, float, np.ndarray]:
    if mean_hiddens.ndim != 2:
        raise ValueError(f"Expected (N,D), got {mean_hiddens.shape}")

    N, D = mean_hiddens.shape
    if not (0 <= n_omit < N):
        raise ValueError(f"n_omit must be in [0, N-1], got {n_omit=}, {N=}")

    X = mean_hiddens[n_omit:].astype(np.float32, copy=False)  # (N', D)
    Np = X.shape[0]
    if Np <= 1:
        raise ValueError(f"Too few samples after omitting: N'={Np}")

    K = min(Np, D)
    d = max(0, min(int(n_components), K))  # clip to [0, K]

    out = np.full((N, d), np.nan, dtype=np.float32)

    # Fit PCA with maximum possible components to get full explained_variance_ratio_
    pca = PCA(n_components=K, svd_solver="full")
    pca.fit(X)
    explained_variance_ratio_full = pca.explained_variance_ratio_.astype(
        np.float32, copy=False
    )

    if d == 0:
        return None, out, 0.0, explained_variance_ratio_full

    # Transform using only the first d components
    Z = pca.transform(X)[:, :d].astype(np.float32, copy=False)
    out[n_omit:] = Z

    varexp = float(explained_variance_ratio_full[:d].sum())
    return pca, out, varexp, explained_variance_ratio_full


def save_sae_feats(sae_acts, top_k_for_save: int, name: str, output_path: str):
    sae_acts = np.stack(sae_acts, axis=0)
    np.save(os.path.join(output_path, f"{name}_acts.npy"), sae_acts)

    sae_mean_top_acts_slice, sae_mean_indices_slice = top_by_std(
        sae_acts, top_k=top_k_for_save
    )
    np.save(os.path.join(output_path, f"{name}_top_acts.npy"), sae_mean_top_acts_slice)
    np.save(os.path.join(output_path, f"{name}_indices.npy"), sae_mean_indices_slice)


@pyrallis.wrap()
def run(config: Config):
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.cache_size_limit = 256

    # Model
    if not ("gpt2" in config.model_name or "pythia" in config.model_name):
        dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        dtype = torch.float32
        attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=dtype, attn_implementation=attn_implementation
    )
    model = model.to(device)
    model = model.eval()
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    model = torch.compile(model)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Qwen3" in config.model_name:
        tokenizer.padding_side = "left"  # <-- REQUIRED for Qwen3 + FlashAttention
    else:
        tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = get_num_layers(config, model)
    config.max_seq_len = resolve_max_seq_len(config, model, tokenizer)

    # dataset = load_dataset(config.dataset_name, split="train", streaming=True)
    # dataset = Dataset.from_list(list(dataset.take(10 * config.num_samples)))
    # Dataset
    # select samples where which will result in complete lines only
    stream = load_dataset(config.dataset_name, split="train", streaming=True)

    # Filter on-the-fly (cheap early rejection)
    stream = stream.filter(
        lambda x: (
            (len(x["text"]) > config.line_length * config.min_lines)
            and ("\n" not in x["text"])
        )
    )

    # If you only need to iterate: just use `stream` directly.
    # If you need a materialized Dataset: build only the first `num_samples` *after filtering*.
    def gen():
        yield from islice(stream, config.num_samples)

    dataset = Dataset.from_generator(gen)

    dataset = dataset.map(
        make_line_wrapper(
            line_length=config.line_length, text_key="text", out_key="text_lines"
        ),
        desc="Wrapping into char-limited lines",
    )
    dataset = dataset.map(
        make_forward_inputs_with_chars_since_nl(
            tokenizer,
            config.max_seq_len,
            use_chat=config.use_chat,
            text_key="text_lines",
        ),
        batched=True,
        desc="Making forward inputs",
    )
    dataset = dataset.with_format(
        "torch",
        columns=[
            c for c in dataset.column_names if c in ("input_ids", "attention_mask")
        ],
        output_all_columns=True,
    )
    dataset = dataset.map(
        assert_chars_since_nl_map(line_length=config.line_length),
        batched=True,
        desc="Checking everything is all right",
    )

    # Run over the dataset once, evaluating language modeling correctness.
    lm_metrics, newline_probs, newline_token_probs = eval_next_token_metrics(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_seq_len=config.max_seq_len,
        log_first_n=config.log_first_n,
    )
    np.save(os.path.join(config.output_path, "newline_probs.npy"), newline_probs)

    # Save a few samples for a later visualization
    save_token_scores(
        dataset,
        dataset["chars_since_nl"],
        tokenizer,
        use_chat=config.use_chat,
        log_first_n=config.log_first_n,
        out_dir=os.path.join(config.output_path, "counts"),
    )
    save_token_scores(
        dataset,
        newline_token_probs,
        tokenizer,
        use_chat=config.use_chat,
        log_first_n=config.log_first_n,
        out_dir=os.path.join(config.output_path, "newline_probs"),
    )
    logger.info(
        f"LM: loss={lm_metrics['lm_loss']:.4f} acc={lm_metrics['lm_acc']:.4%} acc_nl={lm_metrics['lm_acc_if_newline']:.4%} acc_nl_any={lm_metrics['lm_acc_if_newline_any']:.4%} tokens={lm_metrics['num_tokens']}"
    )
    pd.DataFrame([lm_metrics]).to_csv(
        os.path.join(config.output_path, "lm_metrics.csv"), index=False
    )

    all_metrics = []
    all_mean_hiddens = []
    all_sae_mean_acts = []
    all_sae_frac_acts = []
    all_pca_projs = []
    all_pca_sae_spanned_projs = []
    all_top_sae_features_projected = []
    all_top_sae_features_projected_scaled = []
    all_explained_variance_ratio = []

    if config.run_best_layer_only:
        custom_range = trange(config.best_layer, config.best_layer + 1)
    else:
        custom_range = trange(num_layers)

    for layer_idx in custom_range:
        layer_hiddens = collect_layer_hiddens(
            dataset,
            model,
            tokenizer,
            layer_idx=layer_idx,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            max_seq_len=config.max_seq_len,
        )
        layer_mean_hiddens, char_counts = mean_hidden_by_chars_since_nl(
            layer_hiddens,
            dataset["chars_since_nl"],
            max_chars=config.line_length,
        )
        assert (
            len(char_counts) == len(layer_mean_hiddens)
            and len(char_counts) == config.line_length + 1
        ), (len(char_counts), config.line_length, len(layer_mean_hiddens))

        # Calculate Linear-Regression Metrics
        layer_metrics = fit_linear_regression_chars_since_nl(
            layer_hiddens=layer_hiddens, chars_since_nl=dataset["chars_since_nl"]
        )
        layer_metrics["layer"] = layer_idx

        # Do a PCA projection
        pca, transformed, varexp_by_layer, explained_variance_ratio = pca_per_layer(
            mean_hiddens=layer_mean_hiddens.float().cpu().numpy(),
            n_omit=config.pca_n_omit,
            n_components=config.pca_n_components,
        )

        layer_metrics[f"pca_varexp_by_first_{config.pca_n_components}"] = (
            varexp_by_layer
        )

        # Calcualate sae features' activations
        sae = get_sae(config.model_name, layer_idx)

        if sae is not None:
            sae = sae.to(device)
            assert (
                np.array([len(x) for x in dataset["chars_since_nl"]])
                == np.array([len(x) for x in layer_hiddens])
            ).all()

            all_char_counts = torch.hstack(
                [torch.tensor(x) for x in dataset["chars_since_nl"]]
            )
            all_layer_hiddens = torch.vstack(layer_hiddens)

            sae_mean_acts, sae_frac_acts = get_sae_acts(
                sae=sae,
                all_char_counts=all_char_counts,
                all_hiddens=all_layer_hiddens,
                device=device,
            )

            all_sae_mean_acts.append(sae_mean_acts)
            all_sae_frac_acts.append(sae_frac_acts)

            top_indices = sae_mean_acts.std(0).argsort()[::-1][
                : config.sae_top_k_for_save
            ]
            top_indices = torch.from_numpy(top_indices.copy())

            top_feats = sae.W_dec[top_indices].float().cpu().numpy()
            sae_span, _ = np.linalg.qr(top_feats.T)

            transformed_sae_spanned = pca.transform(
                layer_mean_hiddens.float().cpu().numpy() @ sae_span @ sae_span.T
            )[:, : config.pca_n_components].astype(np.float32, copy=False)
            transformed_sae_spanned[np.isnan(transformed)] = np.nan
            all_pca_sae_spanned_projs.append(transformed_sae_spanned)

            all_top_sae_features_projected.append(
                pca.transform(top_feats)[:, : config.pca_n_components].astype(
                    np.float32, copy=False
                )
            )

            max_acts = sae_mean_acts[:, top_indices].max(0)[:, None]

            all_top_sae_features_projected_scaled.append(
                pca.transform(max_acts * top_feats)[
                    :, : config.pca_n_components
                ].astype(np.float32, copy=False)
            )
        all_pca_projs.append(transformed)
        all_mean_hiddens.append(layer_mean_hiddens)
        all_metrics.append(layer_metrics)
        all_explained_variance_ratio.append(explained_variance_ratio)

    if config.run_best_layer_only:
        all_metrics = [all_metrics[0] for _ in range(num_layers)]
        all_mean_hiddens = [all_mean_hiddens[0] for _ in range(num_layers)]
        all_pca_projs = [all_pca_projs[0] for _ in range(num_layers)]
        all_explained_variance_ratio = [
            all_explained_variance_ratio[0] for _ in range(num_layers)
        ]

        if sae is not None:
            all_sae_mean_acts = [all_sae_mean_acts[0] for _ in range(num_layers)]
            all_sae_frac_acts = [all_sae_frac_acts[0] for _ in range(num_layers)]
            all_pca_sae_spanned_projs = [
                all_pca_sae_spanned_projs[0] for _ in range(num_layers)
            ]
            all_top_sae_features_projected = [
                all_top_sae_features_projected[0] for _ in range(num_layers)
            ]
            all_top_sae_features_projected_scaled = [
                all_top_sae_features_projected_scaled[0] for _ in range(num_layers)
            ]

    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv(os.path.join(config.output_path, "metrics.csv"), index=False)
    print(f"Mean hiddens shapes: {[h.shape for h in all_mean_hiddens]}")
    all_mean_hiddens = np.stack(all_mean_hiddens, axis=0)
    np.save(os.path.join(config.output_path, "mean_hiddens.npy"), all_mean_hiddens)
    all_explained_variance_ratio = np.stack(all_explained_variance_ratio, axis=0)
    np.save(
        os.path.join(config.output_path, "explained_variance_ratio.npy"),
        all_explained_variance_ratio,
    )

    N, _ = layer_mean_hiddens.shape
    d = transformed.shape[1]
    start = config.pca_n_omit
    end = N - 1  # inclusive
    np.save(
        os.path.join(
            config.output_path, f"mean_hiddens_pca_slice_{start}_to_{end}_d{d}.npy"
        ),
        np.stack(all_pca_projs, axis=0),
    )

    if sae is not None:
        save_sae_feats(
            sae_acts=all_sae_mean_acts,
            top_k_for_save=config.sae_top_k_for_save,
            name="sae_mean",
            output_path=config.output_path,
        )
        save_sae_feats(
            sae_acts=all_sae_frac_acts,
            top_k_for_save=config.sae_top_k_for_save,
            name="sae_frac",
            output_path=config.output_path,
        )

        np.save(
            os.path.join(
                config.output_path,
                f"mean_hiddens_pca_sae_spanned_slice_{start}_to_{end}_d{d}.npy",
            ),
            np.stack(all_pca_sae_spanned_projs, axis=0),
        )
        np.save(
            os.path.join(config.output_path, "top_sae_features_projected.npy"),
            np.stack(all_top_sae_features_projected, axis=0),
        )
        np.save(
            os.path.join(config.output_path, "top_sae_features_projected_scaled.npy"),
            np.stack(all_top_sae_features_projected_scaled, axis=0),
        )


if __name__ == "__main__":
    run()
