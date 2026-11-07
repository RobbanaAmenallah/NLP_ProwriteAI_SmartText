"""
✨ ProwriteAI — Smart Text Improvement (Streamlit demo)

Interactive demo of the four Encoder–Decoder models trained FROM SCRATCH in
`ProwriteAI_SmartText_4Models_Training.ipynb`:

    A_LSTM · B_BiLSTM · C_Attention · D_Pointer

Everything the app needs is read from the saved checkpoints (`*.pt`): weights,
architecture configuration, vocabulary, tokenizer pattern and sequence-length limits.
The model classes, preprocessing and decoding code below are copied from the notebook
so that the checkpoints load with `strict=True`. No external API and no pretrained
model is used.

Run:   streamlit run app.py
Paths: PROWRITE_CHECKPOINT_DIR  (folder with A_LSTM.pt … D_Pointer.pt)
       PROWRITE_ARTIFACT_DIR    (folder with vocab.json / tokenizer.json, optional)
       PROWRITE_OUTPUT_DIR      (folder with test_metrics.csv, optional)
"""
from __future__ import annotations

import html
import json
import math
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# =============================================================================
# Configuration (all paths can be overridden with environment variables)
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_NAMES = ["A_LSTM", "B_BiLSTM", "C_Attention", "D_Pointer"]
MODEL_INFO = {
    "A_LSTM": {"label": "Classic LSTM", "emoji": "🧱", "color": "#60a5fa",
               "desc": "Embedding → LSTM encoder → final state → LSTM decoder. The baseline."},
    "B_BiLSTM": {"label": "Bidirectional LSTM", "emoji": "↔️", "color": "#34d399",
                 "desc": "Reads the message forwards and backwards, then decodes from one summary vector."},
    "C_Attention": {"label": "BiLSTM + Attention", "emoji": "🎯", "color": "#fbbf24",
                    "desc": "Bahdanau attention lets the decoder look back at every input word at each step."},
    "D_Pointer": {"label": "Pointer-Generator", "emoji": "🧲", "color": "#f472b6",
                  "desc": "Attention + a copy switch (p_gen) that can copy names and numbers from your message."},
}
EXAMPLES = [  # demonstration INPUTS only — outputs are always generated live
    "hey boss i cant come tomorrow im sick sorry",
    "hey can u send me the report tomorrow pls",
    "sorry i didnt reply earlier i was busy",
    "hi sir i need more time to finish the project",
    "can u send me the report today pls i need it",
    "the meeting was yesterday but nobody tell me",
    "hi sarah r u free for a call on friday at 3pm",
    "hi , were r u last night .",
    "im sick so i cant come to the meeting tmrw",
    "my order #884213 still didnt arrive can u check pls",
    "thx for helping me with the slides yesterday!!",
    "pls dont forget the team lunch is on thursday",
]
MAX_INPUT_CHARS = 600


def _first_existing(candidates, must_contain=None):
    for c in candidates:
        if c and Path(c).is_dir() and (must_contain is None or (Path(c) / must_contain).exists()):
            return Path(c)
    return Path(candidates[0]) if candidates and candidates[0] else APP_DIR


CKPT_DIR = _first_existing([os.environ.get("PROWRITE_CHECKPOINT_DIR"), APP_DIR / "prowrite_outputs" / "checkpoints",
                            APP_DIR / "checkpoints", APP_DIR], must_contain="A_LSTM.pt")
ARTIFACT_DIR = _first_existing([os.environ.get("PROWRITE_ARTIFACT_DIR"), APP_DIR / "prowrite_outputs" / "artifacts",
                                APP_DIR / "artifacts", APP_DIR], must_contain="vocab.json")
OUTPUT_DIR = _first_existing([os.environ.get("PROWRITE_OUTPUT_DIR"), APP_DIR / "prowrite_outputs", APP_DIR],
                             must_contain="test_metrics.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    # The models are small: one intra-op thread is as fast as several and avoids CPU contention
    # with Streamlit's own server threads (which caused latency spikes on small machines).
    torch.set_num_threads(max(1, int(os.environ.get("PROWRITE_TORCH_THREADS", "1"))))

# Special-token ids used by the notebook (verified against every checkpoint at load time)
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, "<SOS>": SOS_ID, "<EOS>": EOS_ID}

# =============================================================================
# Preprocessing — identical to notebook §3.1 (normalize_text)
# =============================================================================
TYPOGRAPHIC_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
                                 "“": '"', "”": '"', "„": '"', "″": '"',
                                 "–": "-", "—": "-", "−": "-", " ": " "})
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏⁠﻿]")


def normalize_text(text) -> str:
    """Unicode + whitespace normalization. Keeps case and punctuation."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(TYPOGRAPHIC_MAP)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n").replace(" ", "\n")
    text = CONTROL_CHARS.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TextPipeline:
    """Tokenizer + vocabulary rebuilt from the saved specification (checkpoint / tokenizer.json / vocab.json)."""

    def __init__(self, tokenizer_spec: dict, vocab_dict: dict, max_src_len: int, max_tgt_len: int):
        if tokenizer_spec.get("type") != "regex_word_tokenizer":
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_spec.get('type')!r}")
        flags = re.VERBOSE if "VERBOSE" in tokenizer_spec.get("flags", "") else 0
        self.token_re = re.compile(tokenizer_spec["pattern"], flags)
        self.no_space_before = set(tokenizer_spec["no_space_before"])
        self.no_space_after = set(tokenizer_spec["no_space_after"])
        self.itos = list(vocab_dict["itos"])
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        for tok, idx in SPECIAL_TOKENS.items():
            if self.stoi.get(tok) != idx:
                raise ValueError(f"Vocabulary special token {tok} has id {self.stoi.get(tok)}, expected {idx}")
        self.vocab_size = len(self.itos)
        self.is_word = [bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9'-]*", t)) for t in self.itos]
        self.max_src_len, self.max_tgt_len = int(max_src_len), int(max_tgt_len)

    def tokenize(self, text: str):
        return self.token_re.findall(normalize_text(text))

    def encode_source(self, tokens):
        return [self.stoi.get(t, UNK_ID) for t in tokens[: self.max_src_len - 1]] + [EOS_ID]

    def source_extended_ids(self, tokens):
        ext, oovs = [], []
        for t in tokens[: self.max_src_len - 1]:
            i = self.stoi.get(t)
            if i is None:
                if t not in oovs:
                    oovs.append(t)
                i = self.vocab_size + oovs.index(t)
            ext.append(i)
        return ext + [EOS_ID], oovs

    def decode(self, ids, oovs=None):
        toks = []
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            if i >= self.vocab_size:  # copied out-of-vocabulary word (pointer-generator)
                k = i - self.vocab_size
                toks.append(oovs[k] if oovs and k < len(oovs) else "<UNK>")
                continue
            tok = self.itos[i]
            if tok in ("<PAD>", "<SOS>", "<EOS>"):
                continue
            toks.append(tok)
        return toks

    def detokenize(self, tokens) -> str:
        out, prev, in_quote = "", None, False
        for tok in tokens:
            if tok == '"':
                if in_quote:
                    out += '"'; prev = '"close'
                else:
                    out += (" " if out else "") + '"'; prev = '"open'
                in_quote = not in_quote
                continue
            if not out:
                out = tok
            elif tok in self.no_space_before or prev in self.no_space_after or prev == '"open':
                out += tok
            else:
                out += " " + tok
            prev = tok
        return out


# =============================================================================
# Model architectures — identical to notebook §7 (required for strict loading)
# =============================================================================
class Encoder(nn.Module):
    """Embedding → (Bi)LSTM. Returns every hidden state and the final (h, c)."""
    def __init__(self, embedding, emb_dim, hid_dim, num_layers, dropout, bidirectional):
        super().__init__()
        self.embedding, self.bidirectional = embedding, bidirectional
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=src.size(1))
        return outputs, (h, c)


class Bridge(nn.Module):
    """Concatenate forward/backward final states (2H) and project them to the decoder size (H)."""
    def __init__(self, hid_dim):
        super().__init__()
        self.h_proj, self.c_proj = nn.Linear(2 * hid_dim, hid_dim), nn.Linear(2 * hid_dim, hid_dim)

    def forward(self, h, c):
        L2, B, H = h.shape
        cat = lambda x: x.view(L2 // 2, 2, B, H).permute(0, 2, 1, 3).reshape(L2 // 2, B, 2 * H)
        return torch.tanh(self.h_proj(cat(h))), self.c_proj(cat(c))


class BahdanauAttention(nn.Module):
    """Additive attention: e = vᵀ tanh(W_h h_j + W_s s_{t-1} + b), masked softmax, context = Σ α h."""
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def project_encoder(self, enc_out):
        return self.W_h(enc_out)

    def forward(self, s_prev, enc_out, enc_proj, mask):
        scores = self.v(torch.tanh(enc_proj + self.W_s(s_prev).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context, weights


class Seq2SeqBase(nn.Module):
    uses_copy = False
    uses_attention = False

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.vocab_size, self.emb_dim, self.hid_dim, self.num_layers = vocab_size, emb_dim, hid_dim, num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)

    def to_input_ids(self, ids):
        """Copied (extended) ids are fed back to the decoder as <UNK> (they have no embedding)."""
        return ids.masked_fill(ids >= self.vocab_size, UNK_ID)


class A_LSTM(Seq2SeqBase):
    description = "Embedding → LSTM encoder → final (h,c) → LSTM decoder → Linear"

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__(vocab_size, emb_dim, hid_dim, num_layers, dropout)
        self.encoder = Encoder(self.embedding, emb_dim, hid_dim, num_layers, dropout, bidirectional=False)
        self.decoder_rnn = nn.LSTM(emb_dim, hid_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out = nn.Linear(hid_dim, vocab_size)

    def encode(self, src, src_lengths, src_mask):
        enc_out, (h, c) = self.encoder(src, src_lengths)
        return {"enc_out": enc_out, "mask": src_mask, "state": (h, c)}

    def decode_step(self, y_prev, state, enc, src_ext=None, n_oov=0):
        x = self.dropout(self.embedding(y_prev)).unsqueeze(1)
        out, state = self.decoder_rnn(x, state)
        logits = self.out(self.dropout(out.squeeze(1)))
        return F.log_softmax(logits, dim=-1), state, {}


class B_BiLSTM(A_LSTM):
    description = "Embedding → BiLSTM encoder → bridge([h_fwd;h_bwd]) → LSTM decoder → Linear"

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__(vocab_size, emb_dim, hid_dim, num_layers, dropout)
        self.encoder = Encoder(self.embedding, emb_dim, hid_dim, num_layers, dropout, bidirectional=True)
        self.bridge = Bridge(hid_dim)

    def encode(self, src, src_lengths, src_mask):
        enc_out, (h, c) = self.encoder(src, src_lengths)
        return {"enc_out": enc_out, "mask": src_mask, "state": self.bridge(h, c)}


class C_Attention(Seq2SeqBase):
    description = "Embedding → BiLSTM encoder → Bahdanau attention → LSTM decoder([x;context]) → Linear"
    uses_attention = True

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__(vocab_size, emb_dim, hid_dim, num_layers, dropout)
        self.encoder = Encoder(self.embedding, emb_dim, hid_dim, num_layers, dropout, bidirectional=True)
        self.bridge = Bridge(hid_dim)
        self.attention = BahdanauAttention(2 * hid_dim, hid_dim, hid_dim)
        self.decoder_rnn = nn.LSTM(emb_dim + 2 * hid_dim, hid_dim, num_layers, batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
        self.pre_out = nn.Linear(hid_dim + 2 * hid_dim + emb_dim, hid_dim)
        self.out = nn.Linear(hid_dim, vocab_size)

    def encode(self, src, src_lengths, src_mask):
        enc_out, (h, c) = self.encoder(src, src_lengths)
        return {"enc_out": enc_out, "enc_proj": self.attention.project_encoder(enc_out), "mask": src_mask,
                "state": self.bridge(h, c)}

    def attend_and_step(self, y_prev, state, enc):
        x = self.dropout(self.embedding(y_prev))
        context, attn = self.attention(state[0][-1], enc["enc_out"], enc["enc_proj"], enc["mask"])
        out, state = self.decoder_rnn(torch.cat([x, context], -1).unsqueeze(1), state)
        s_t = out.squeeze(1)
        features = torch.tanh(self.pre_out(torch.cat([s_t, context, x], -1)))
        return self.out(self.dropout(features)), state, attn, context, s_t, x

    def decode_step(self, y_prev, state, enc, src_ext=None, n_oov=0):
        logits, state, attn, *_ = self.attend_and_step(y_prev, state, enc)
        return F.log_softmax(logits, dim=-1), state, {"attn": attn}


class CoverageAttention(BahdanauAttention):
    """Bahdanau attention whose scores also see the coverage vector u (attention already received by each source word)."""
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__(enc_dim, dec_dim, attn_dim)
        self.W_u = nn.Linear(1, attn_dim, bias=False)

    def forward(self, s_prev, enc_out, enc_proj, mask, coverage):
        scores = self.v(torch.tanh(enc_proj + self.W_s(s_prev).unsqueeze(1) + self.W_u(coverage.unsqueeze(-1)))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context, weights


class D_Pointer(C_Attention):
    description = "BiLSTM + coverage attention + pointer/copy: P = p_gen·P_vocab + (1−p_gen)·Σ attention (extended vocabulary)"
    uses_copy = True
    uses_coverage = True

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__(vocab_size, emb_dim, hid_dim, num_layers, dropout)
        self.attention = CoverageAttention(2 * hid_dim, hid_dim, hid_dim)
        self.p_gen_linear = nn.Linear(2 * hid_dim + hid_dim + emb_dim, 1)

    def encode(self, src, src_lengths, src_mask):
        enc = super().encode(src, src_lengths, src_mask)
        h, c = enc["state"]
        enc["state"] = (h, c, torch.zeros(src.size(0), src.size(1), device=src.device))
        return enc

    def _step(self, y_prev, state, enc):
        h, c, cov = state
        x = self.dropout(self.embedding(y_prev))
        context, attn = self.attention(h[-1], enc["enc_out"], enc["enc_proj"], enc["mask"], cov)
        out, (h, c) = self.decoder_rnn(torch.cat([x, context], -1).unsqueeze(1), (h, c))
        s_t = out.squeeze(1)
        logits = self.out(self.dropout(torch.tanh(self.pre_out(torch.cat([s_t, context, x], -1)))))
        p_gen = torch.sigmoid(self.p_gen_linear(torch.cat([context, s_t, x], -1)))
        cov_loss = torch.minimum(attn, cov).sum(-1)
        return logits, (h, c, cov + attn), attn, p_gen, cov_loss

    def decode_step(self, y_prev, state, enc, src_ext=None, n_oov=0):
        logits, state, attn, p_gen, cov_loss = self._step(y_prev, state, enc)
        vocab_dist = torch.softmax(logits, dim=-1)
        if n_oov > 0:
            vocab_dist = torch.cat([vocab_dist, vocab_dist.new_zeros(vocab_dist.size(0), n_oov)], dim=-1)
        final = (p_gen * vocab_dist).scatter_add(1, src_ext, (1 - p_gen) * attn)
        return torch.log(final.clamp_min(1e-12)), state, {"attn": attn, "p_gen": p_gen, "cov_loss": cov_loss}


def select_state(state, idx):
    """Reorder a decoder state (tuple of (L,B,H) tensors and, for coverage, a (B,S) tensor) along the batch axis."""
    return tuple(t[:, idx] if t.dim() == 3 else t[idx] for t in state)


MODEL_CLASSES = {"A_LSTM": A_LSTM, "B_BiLSTM": B_BiLSTM, "C_Attention": C_Attention, "D_Pointer": D_Pointer}
REQUIRED_CKPT_KEYS = {"model_name", "model_state_dict", "model_config", "vocab", "tokenizer",
                      "max_source_length", "max_target_length", "special_tokens"}


# =============================================================================
# Loading (cached) and decoding — decoding code mirrors notebook §14–§15
# =============================================================================
class AppError(Exception):
    """Error with a message that is safe and useful to show to the user."""


def _torch_load(path: Path):
    try:  # safe loader first: only tensors and plain Python containers are allowed
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            return torch.load(path, map_location=DEVICE, weights_only=True)
    except Exception:
        # Checkpoints written by the project notebook are trusted local files
        return torch.load(path, map_location=DEVICE, weights_only=False)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, ckpt_dir: str, mtime: float):
    """Load one trained checkpoint → (model, pipeline, metadata). Cached per file version."""
    path = Path(ckpt_dir) / f"{model_name}.pt"
    if not path.exists():
        raise AppError(f"Checkpoint not found: `{path}`. Run the training notebook or set PROWRITE_CHECKPOINT_DIR.")
    try:
        ckpt = _torch_load(path)
    except Exception as e:
        raise AppError(f"Could not read `{path.name}`: {e}") from e
    missing = REQUIRED_CKPT_KEYS - set(ckpt)
    if missing:
        raise AppError(f"`{path.name}` is not a ProwriteAI checkpoint (missing keys: {sorted(missing)}).")
    if ckpt["model_name"] != model_name or model_name not in MODEL_CLASSES:
        raise AppError(f"`{path.name}` contains model `{ckpt['model_name']}`, expected `{model_name}`.")
    if {k: int(v) for k, v in ckpt["special_tokens"].items()} != SPECIAL_TOKENS:
        raise AppError(f"`{path.name}` uses different special-token ids: {ckpt['special_tokens']}.")

    # Tokenizer/vocabulary: prefer the saved artifact files, verify they match the checkpoint
    tok_spec, vocab_dict, source = ckpt["tokenizer"], ckpt["vocab"], "checkpoint"
    tok_file, voc_file = ARTIFACT_DIR / "tokenizer.json", ARTIFACT_DIR / "vocab.json"
    if tok_file.exists() and voc_file.exists():
        with open(tok_file, encoding="utf-8") as fh:
            file_tok = json.load(fh)
        with open(voc_file, encoding="utf-8") as fh:
            file_voc = json.load(fh)
        if file_voc["itos"] != vocab_dict["itos"] or file_tok["pattern"] != tok_spec["pattern"]:
            raise AppError("tokenizer.json / vocab.json do not match the vocabulary stored in "
                           f"`{path.name}`. Use the artifact files produced by the same training run.")
        tok_spec, vocab_dict, source = file_tok, file_voc, "tokenizer.json + vocab.json"

    try:
        pipeline = TextPipeline(tok_spec, vocab_dict, ckpt["max_source_length"], ckpt["max_target_length"])
        mc = ckpt["model_config"]
        if mc["vocab_size"] != pipeline.vocab_size:
            raise ValueError(f"model vocab_size {mc['vocab_size']} ≠ vocabulary size {pipeline.vocab_size}")
        model = MODEL_CLASSES[model_name](mc["vocab_size"], mc["embedding_dim"], mc["hidden_dim"],
                                          mc["num_layers"], mc["dropout"])
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except Exception as e:
        raise AppError(f"Invalid model configuration in `{path.name}`: {e}") from e
    model.to(DEVICE).eval()
    try:  # warm-up pass so the first user request is not slowed by one-off initialisation
        with torch.inference_mode():
            warm, _ = prepare_input("thanks for the help", pipeline)
            greedy_decode(model, warm, pipeline)
    except Exception as e:
        raise AppError(f"`{path.name}` loaded but failed a test inference: {type(e).__name__}: {e}") from e
    meta = {"path": str(path), "best_epoch": ckpt.get("best_epoch"), "best_val_loss": ckpt.get("best_val_loss"),
            "epochs_trained": ckpt.get("epochs_trained"), "stop_reason": ckpt.get("stop_reason"),
            "parameters": sum(p.numel() for p in model.parameters()), "architecture": ckpt.get("architecture", ""),
            "saved_at": ckpt.get("saved_at", ""), "tokenizer_source": source, "vocab_size": pipeline.vocab_size,
            "max_src_len": pipeline.max_src_len, "max_tgt_len": pipeline.max_tgt_len}
    return model, pipeline, meta


def get_model(model_name: str):
    path = CKPT_DIR / f"{model_name}.pt"
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return load_model(model_name, str(CKPT_DIR), mtime)


def prepare_input(text: str, pipe: TextPipeline):
    """Normalize → tokenize → ids (+ extended ids for copying). Uses the source text only."""
    norm = normalize_text(text)
    toks = pipe.tokenize(norm)
    if not toks or not re.search(r"[A-Za-z0-9]", norm):
        raise AppError("The message contains no recognisable words — please write at least one word in English.")
    src_ids = pipe.encode_source(toks)
    src_ext, oovs = pipe.source_extended_ids(toks)
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    batch = {"src": src, "src_lengths": torch.tensor([len(src_ids)], dtype=torch.long),
             "src_mask": src != PAD_ID, "src_ext": torch.tensor([src_ext], dtype=torch.long, device=DEVICE),
             "max_oovs": len(oovs), "oovs": oovs}
    info = {"normalized": norm, "tokens": toks, "token_ids": src_ids, "unknown": oovs,
            "truncated": len(toks) > pipe.max_src_len - 1, "max_src_len": pipe.max_src_len}
    return batch, info


def finalize_text(text: str) -> str:
    """Capitalize the first letter of every sentence and the pronoun I (same rule as in the notebook)."""
    text = re.sub(r"\bi\b", "I", text)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def block_repeats(log_probs, prev_ids, pipe):
    """Decoding constraint used in the notebook: a word may not be generated twice in a row."""
    lp = log_probs.clone()
    for r, p in enumerate(prev_ids.tolist()):
        if p > EOS_ID and (p >= pipe.vocab_size or pipe.is_word[p]):
            lp[r, p] = float("-inf")
    return lp


def greedy_decode(model, batch, pipe):
    """y_t = argmax P(y | y_<t, x); stops at <EOS> or the maximum target length."""
    enc = model.encode(batch["src"], batch["src_lengths"], batch["src_mask"])
    state, y, out, logps = enc["state"], torch.tensor([SOS_ID], device=DEVICE), [], []
    for _ in range(pipe.max_tgt_len - 1):
        log_probs, state, _ = model.decode_step(y, state, enc, batch["src_ext"], batch["max_oovs"])
        log_probs = block_repeats(log_probs, torch.tensor([out[-1] if out else SOS_ID]), pipe)
        lp, nxt = log_probs.max(-1)
        nxt = int(nxt)
        out.append(nxt); logps.append(float(lp))
        if nxt == EOS_ID:
            break
        y = model.to_input_ids(torch.tensor([nxt], device=DEVICE))
    return out, sum(logps) / len(logps)


def beam_search_decode(model, batch, pipe, beam_width=3, alpha=0.7):
    """Beam search with length-normalized scores (score / len^alpha), as in notebook §15."""
    enc = model.encode(batch["src"], batch["src_lengths"], batch["src_mask"])
    state, seqs, scores, finished = enc["state"], [[]], torch.zeros(1, device=DEVICE), []
    for _ in range(pipe.max_tgt_len - 1):
        k = len(seqs)
        enc_k = {key: val.expand(k, *val.shape[1:]) for key, val in enc.items() if key != "state"}
        y = model.to_input_ids(torch.tensor([s[-1] if s else SOS_ID for s in seqs], device=DEVICE))
        log_probs, new_state, _ = model.decode_step(y, state, enc_k, batch["src_ext"].expand(k, -1), batch["max_oovs"])
        log_probs = block_repeats(log_probs, torch.tensor([s[-1] if s else SOS_ID for s in seqs]), pipe)
        cand = (scores.unsqueeze(1) + log_probs).view(-1)
        top_scores, top_idx = cand.topk(min(2 * beam_width, cand.numel()))
        v_ext = log_probs.size(1)
        new_seqs, new_scores, origin = [], [], []
        for sc, ix in zip(top_scores.tolist(), top_idx.tolist()):
            beam, tok = divmod(ix, v_ext)
            seq = seqs[beam] + [tok]
            if tok == EOS_ID:
                finished.append((seq, sc, sc / (len(seq) ** alpha)))
            else:
                new_seqs.append(seq); new_scores.append(sc); origin.append(beam)
            if len(new_seqs) == beam_width:
                break
        if len(finished) >= beam_width or not new_seqs:
            break
        seqs, scores = new_seqs, torch.tensor(new_scores, device=DEVICE)
        idx = torch.tensor(origin, device=DEVICE)
        state = select_state(new_state, idx)
    if not finished:
        finished = [(s, sc, sc / (len(s) ** alpha)) for s, sc in zip(seqs, scores.tolist())]
    best = max(finished, key=lambda f: f[2])
    return best[0], best[1] / len(best[0])


DECODERS = {"greedy": "Greedy decoding", "beam": "Beam search"}
DEFAULT_DECODING = "greedy"

# Confidence = mean log-probability of the generated tokens (greedy). Thresholds calibrated on 1,000 messages of
# validation.csv with calibrate_confidence.py (D_Pointer v3): "high" outputs reach sentence-chrF ≥ 80 in 88 % of cases;
# "low" flags 59 % of the bad outputs (chrF < 50) and only 2 % of the good ones.
CONFIDENCE_THRESHOLDS = {"high": -0.10, "low": -0.45}
CONFIDENCE_LABELS = {"high": ("🟢", "High confidence"), "medium": ("🟡", "Medium confidence"), "low": ("🔴", "Low confidence")}


def confidence_level(mean_logp: float) -> str:
    if mean_logp >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    return "medium" if mean_logp >= CONFIDENCE_THRESHOLDS["low"] else "low"


def best_model_name():
    """The model shown to users: lowest validation loss among the available checkpoints (selection on validation, not test)."""
    scores = {}
    for name in MODEL_NAMES:
        if (CKPT_DIR / f"{name}.pt").exists():
            try:
                scores[name] = get_model(name)[2]["best_val_loss"]
            except AppError:
                pass
    return min(scores, key=scores.get) if scores else None


def improve_text(text: str, model_name: str, decoding: str = "greedy", beam_width: int = 3) -> dict:
    """Full pipeline on one message with one trained checkpoint. Raises AppError on any failure."""
    if decoding not in DECODERS:
        raise AppError(f"Unsupported decoding method: {decoding!r}")
    if not text or not text.strip():
        raise AppError("Please type a message first ✍️")
    t_load = time.perf_counter()
    model, pipe, meta = get_model(model_name)
    load_ms = (time.perf_counter() - t_load) * 1000
    batch, info = prepare_input(text, pipe)
    try:
        model.eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            ids, mean_logp = greedy_decode(model, batch, pipe) if decoding == "greedy" else \
                beam_search_decode(model, batch, pipe, beam_width=beam_width)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        raise AppError(f"Inference failed for {model_name}: {type(e).__name__}: {e}") from e
    tokens = pipe.decode(ids, oovs=batch["oovs"])
    output = finalize_text(pipe.detokenize(tokens))
    if not output.strip():
        raise AppError(f"{model_name} produced an empty output for this message.")
    copied = [t for t in tokens if t in batch["oovs"]]
    return {"model": model_name, "text": output, "ms": ms, "mean_logp": mean_logp, "confidence": confidence_level(mean_logp), "load_ms": load_ms, "decoding": decoding,
            "beam_width": beam_width if decoding == "beam" else None, "n_tokens": len(tokens),
            "reached_eos": EOS_ID in ids, "copied_oov": copied, **info}


# =============================================================================
# Styling
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&family=Nunito:wght@400;600;700&display=swap');
:root { --pw-purple:#8b5cf6; --pw-pink:#ec4899; --pw-ink:#f5f3ff; --pw-muted:#c4b5fd; --pw-card:rgba(255,255,255,0.06); }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #3b1d6e 0%, transparent 60%),
                     radial-gradient(900px 500px at 110% 10%, #5b1a52 0%, transparent 55%),
                     linear-gradient(160deg, #0b0a24 0%, #15123a 45%, #1d1045 100%);
         color: var(--pw-ink); font-family: 'Nunito', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; color: var(--pw-ink) !important; }
p, label, span, li, .stMarkdown { color: var(--pw-ink); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #120f33 0%, #1a1244 100%); border-right: 1px solid rgba(196,181,253,.15); }

/* hero */
.pw-hero { text-align:center; padding: 1.4rem 1rem 0.6rem; animation: pwFade .7s ease-out; }
.pw-title { font-family:'Poppins',sans-serif; font-weight:800; font-size: clamp(2.4rem, 6vw, 4rem); line-height:1.05; margin:0;
            background: linear-gradient(90deg,#a78bfa,#f472b6,#fb7185,#a78bfa); background-size: 300% 100%;
            -webkit-background-clip:text; background-clip:text; color:transparent !important; -webkit-text-fill-color: transparent; animation: pwShine 8s linear infinite; }
.pw-sub { font-family:'Poppins',sans-serif; font-size: clamp(1rem, 2.4vw, 1.35rem); color:#e9d5ff; margin:.35rem 0 .4rem; font-weight:600; }
.pw-desc { color: var(--pw-muted); font-size: 1.02rem; margin-bottom: .8rem; }
.pw-badges { display:flex; gap:.5rem; justify-content:center; flex-wrap:wrap; margin-bottom: .6rem; }
.pw-badge { padding:.28rem .75rem; border-radius:999px; font-size:.82rem; font-weight:700; color:#fff;
            background: rgba(139,92,246,.18); border:1px solid rgba(196,181,253,.35); }

/* cards = bordered containers */
[class*="st-key-pw_card"] { background: var(--pw-card); border: 1px solid rgba(196,181,253,.18) !important;
            border-radius: 22px !important; box-shadow: 0 12px 40px rgba(10, 5, 35, .45); backdrop-filter: blur(6px); }
.pw-card-title { font-family:'Poppins',sans-serif; font-weight:700; font-size:1.25rem; margin:.1rem 0 .4rem; }

/* inputs */
.stTextArea textarea { background: rgba(15, 10, 45, .65) !important; color: var(--pw-ink) !important; border-radius: 16px !important;
            border: 1.5px solid rgba(167,139,250,.45) !important; font-size: 1.05rem !important; line-height:1.5; }
.stTextArea textarea:focus { border-color: #f472b6 !important; box-shadow: 0 0 0 3px rgba(244,114,182,.25) !important; }
.stTextArea textarea::placeholder { color: rgba(196,181,253,.6) !important; }
[data-baseweb="select"] > div { background: rgba(15,10,45,.65) !important; border-radius: 14px !important; border-color: rgba(167,139,250,.45) !important; }

/* buttons */
.stButton button, .stDownloadButton button { border-radius: 14px; font-weight: 700; border: 1px solid rgba(196,181,253,.35);
            background: rgba(139,92,246,.16); color: #fff; transition: transform .15s ease, box-shadow .15s ease, background .2s; }
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(236,72,153,.28); border-color:#f472b6; color:#fff; }
.stButton button[kind="primary"] { background: linear-gradient(90deg, #8b5cf6, #d946ef, #ec4899, #d946ef, #8b5cf6); border: none;
            font-size: 1.12rem; padding: .8rem 1rem; letter-spacing:.04em; box-shadow: 0 10px 28px rgba(217,70,239,.35);
            background-size: 300% 100%; animation: pwShine 9s linear infinite; }
.stButton button[kind="primary"]:hover { transform: translateY(-2px) scale(1.01); box-shadow: 0 14px 34px rgba(236,72,153,.5); }

/* result */
.pw-result { background: linear-gradient(135deg, rgba(139,92,246,.22), rgba(236,72,153,.16)); border:1px solid rgba(244,114,182,.35);
            border-radius: 18px; padding: 1.1rem 1.2rem; font-size: 1.12rem; line-height:1.6; color:#fff; animation: pwPop .45s ease-out; }
.pw-empty { border: 2px dashed rgba(196,181,253,.3); border-radius: 18px; padding: 2.2rem 1rem; text-align:center; color: var(--pw-muted); }
.pw-chips { display:flex; flex-wrap:wrap; gap:.45rem; margin:.7rem 0 .2rem; }
.pw-chip { font-size:.82rem; font-weight:700; padding:.25rem .65rem; border-radius:999px; background: rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); }
.pw-error { background: rgba(244,63,94,.14); border:1px solid rgba(251,113,133,.5); border-radius:16px; padding:.9rem 1rem; color:#ffe4e6; }

/* compare cards */
.pw-cmp { border-radius: 20px; padding: 1rem 1.1rem 1.1rem; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
          box-shadow: 0 10px 30px rgba(5,3,25,.45); height: 100%; animation: pwPop .5s ease-out both; }
.pw-cmp-head { display:flex; justify-content:space-between; align-items:center; gap:.5rem; margin-bottom:.55rem; }
.pw-cmp-name { font-family:'Poppins',sans-serif; font-weight:800; font-size:1.05rem; }
.pw-cmp-label { font-size:.8rem; color: var(--pw-muted); }
.pw-cmp-text { font-size:1.02rem; line-height:1.55; color:#fff; min-height: 3.2rem; }
.pw-cmp-foot { margin-top:.7rem; display:flex; gap:.4rem; flex-wrap:wrap; }
.pw-cmp:nth-child(2) { animation-delay: .08s; }

[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.55rem; }
.stButton button[kind="primary"] p { font-size: 1.12rem; font-weight: 800; }
.pw-footer { text-align:center; color: rgba(196,181,253,.7); font-size:.85rem; margin: 2rem 0 .5rem; }
[data-testid="stCode"] pre, [data-testid="stCodeBlock"] pre { border-radius: 14px !important; background: rgba(15,10,45,.75) !important; }
[data-testid="stExpander"] { border-radius: 16px !important; border-color: rgba(196,181,253,.2) !important; background: rgba(255,255,255,.03); }
@keyframes pwFade { from { opacity:0; transform: translateY(10px);} to { opacity:1; transform:none; } }
@keyframes pwPop { from { opacity:0; transform: translateY(8px) scale(.98);} to { opacity:1; transform:none; } }
@keyframes pwShine { 0% { background-position: 0% 50%; } 100% { background-position: 300% 50%; } }
@media (max-width: 640px) { .block-container { padding: 1rem .7rem; } .pw-result { font-size: 1.02rem; } }
</style>
"""


def esc(s) -> str:
    return html.escape(str(s))


def fmt_ms(ms: float) -> str:
    return f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"


def model_option_label(name: str) -> str:
    i = MODEL_INFO[name]
    return f"{i['emoji']}  {name} — {i['label']}"


# =============================================================================
# Session-state helpers (callbacks run before the widgets are redrawn)
# =============================================================================
def _set_text(value: str):
    st.session_state["input_text"] = value
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    st.session_state.pop("comparison", None)


def cb_random_example():
    current = st.session_state.get("input_text", "")
    _set_text(random.choice([e for e in EXAMPLES if e != current]))


def cb_clear():
    _set_text("")


def cb_example(text: str):
    _set_text(text)


@st.cache_data(show_spinner=False)
def read_scoreboard(path: str, mtime: float):
    import csv
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# =============================================================================
# Page
# =============================================================================
def main():
    st.set_page_config(page_title="ProwriteAI — Smart Text Improvement", page_icon="✨", layout="wide",
                       initial_sidebar_state="auto")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("input_text", "")

    available = {n: (CKPT_DIR / f"{n}.pt").exists() for n in MODEL_NAMES}
    device_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"

    # ---------------------------------------------------------------- hero
    st.markdown(f"""
    <div class="pw-hero">
      <h1 class="pw-title">✨ ProwriteAI</h1>
      <div class="pw-sub">From Casual to Clear, Correct &amp; Professional English</div>
      <div class="pw-desc">Type your message naturally and let our AI models improve its grammar, clarity, and professional tone!</div>
      <div class="pw-badges">
        <span class="pw-badge">🧠 4 Encoder–Decoder models</span>
        <span class="pw-badge">🌱 trained from scratch</span>
        <span class="pw-badge">🚫 no pretrained models or APIs</span>
        <span class="pw-badge">⚙️ running on {esc(device_name)}</span>
        <span class="pw-badge">📦 {sum(available.values())}/4 checkpoints found</span>
      </div>
    </div>""", unsafe_allow_html=True)

    if not any(available.values()):
        st.markdown(f"""<div class="pw-error">❌ <b>No trained checkpoints found</b> in <code>{esc(CKPT_DIR)}</code>.<br>
        Expected <code>A_LSTM.pt</code>, <code>B_BiLSTM.pt</code>, <code>C_Attention.pt</code> and <code>D_Pointer.pt</code>
        (written by the training notebook to <code>prowrite_outputs/checkpoints/</code>). Put them there or set the
        <code>PROWRITE_CHECKPOINT_DIR</code> environment variable, then reload the page.</div>""", unsafe_allow_html=True)

    # ---------------------------------------------------------------- automatic model + advanced settings
    model_name = best_model_name()
    with st.sidebar:
        with st.expander("⚙️ Advanced settings", expanded=False):
            decoding = st.radio("🧭 Decoding", list(DECODERS), index=list(DECODERS).index(DEFAULT_DECODING),
                                format_func=lambda k: DECODERS[k],
                                help="Greedy picks the single most likely word at every step. Beam search keeps several "
                                     "candidate sentences and returns the best length-normalized one.")
            beam_width = st.select_slider("🔦 Beam width", options=[3, 5], value=3, disabled=decoding != "beam",
                                          help="Number of candidate sentences kept by beam search.")

    # ---------------------------------------------------------------- main columns
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True, key="pw_card_input"):
            st.markdown('<div class="pw-card-title">✍️ Your Original Message</div>', unsafe_allow_html=True)
            text = st.text_area("Your message", key="input_text", height=190, max_chars=MAX_INPUT_CHARS,
                                placeholder="hey boss i cant come tomorrow im sick sorry", label_visibility="collapsed",
                                help="Write casually — abbreviations, typos and missing punctuation are welcome.")
            n_chars, n_words = len(text), len(text.split())
            st.markdown(f'<div class="pw-chips"><span class="pw-chip">🔤 {n_chars}/{MAX_INPUT_CHARS} characters</span>'
                        f'<span class="pw-chip">📝 {n_words} words</span></div>', unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            b1.button("🎲 Random example", on_click=cb_random_example, use_container_width=True,
                      help="Fill the box with a random demo message (the output is still generated live).")
            b2.button("🧹 Clear", on_click=cb_clear, use_container_width=True, help="Empty the message box.")
            improve_clicked = st.button("✨ IMPROVE MY TEXT", type="primary", use_container_width=True,
                                        disabled=model_name is None, help="Improve your message with the best trained model.")
            with st.expander("💡 Example messages (click to try)"):
                for i, ex in enumerate(EXAMPLES[:6]):
                    st.button(f"“{ex}”", key=f"ex_{i}", on_click=cb_example, args=(ex,), use_container_width=True)

    # ---------------------------------------------------------------- actions
    compare_clicked = st.session_state.pop("compare_requested", False)
    if improve_clicked:
        st.session_state.pop("comparison", None)
        if not text.strip():
            st.session_state["error"] = "Please type a message first ✍️ (or try 🎲 Random example)."
            st.session_state.pop("result", None)
        else:
            with st.spinner("✨ Polishing your words…"):
                try:
                    st.session_state["result"] = improve_text(text, model_name, decoding, beam_width)
                    st.session_state.pop("error", None)
                except AppError as e:
                    st.session_state["error"] = str(e); st.session_state.pop("result", None)
                except Exception as e:  # never hide failures or fall back to the input text
                    st.session_state["error"] = f"Unexpected error: {type(e).__name__}: {e}"
                    st.session_state.pop("result", None)

    if compare_clicked:
        if not text.strip():
            st.session_state["error"] = "Please type a message first ✍️ — then compare the four models."
            st.session_state.pop("comparison", None)
        else:
            results = {}
            with st.spinner("⚔️ The four models are competing on your message…"):
                for name in MODEL_NAMES:
                    try:
                        results[name] = improve_text(text, name, decoding, beam_width)
                    except AppError as e:
                        results[name] = {"error": str(e)}
                    except Exception as e:
                        results[name] = {"error": f"Unexpected error: {type(e).__name__}: {e}"}
            st.session_state["comparison"] = {"input": text, "decoding": decoding, "beam_width": beam_width, "results": results}
            st.session_state.pop("error", None)

    # ---------------------------------------------------------------- right column
    with right:
        with st.container(border=True, key="pw_card_output"):
            st.markdown('<div class="pw-card-title">✨ Your Improved Message</div>', unsafe_allow_html=True)
            err, res = st.session_state.get("error"), st.session_state.get("result")
            if err:
                st.markdown(f'<div class="pw-error">⚠️ {esc(err)}</div>', unsafe_allow_html=True)
            elif res:
                info = MODEL_INFO[res["model"]]
                dec = DECODERS[res["decoding"]] + (f" (k={res['beam_width']})" if res["beam_width"] else "")
                st.markdown(f'<div class="pw-result">{esc(res["text"])}</div>', unsafe_allow_html=True)
                icon, label = CONFIDENCE_LABELS[res["confidence"]]
                chips = [f"{icon} {label}", f"🤖 {res['model']} (best on validation)", f"⏱️ {fmt_ms(res['ms'])}", f"🧭 {dec}"]
                if res["copied_oov"]:
                    chips.append("🧲 copied: " + ", ".join(res["copied_oov"]))
                st.markdown('<div class="pw-chips">' + "".join(f'<span class="pw-chip">{esc(c)}</span>' for c in chips)
                            + "</div>", unsafe_allow_html=True)
                if res["confidence"] == "low":
                    st.warning("🔴 The model is not confident about this sentence. It may be a construction it did not "
                               "see during training — please check the output carefully.")
                elif res["confidence"] == "medium":
                    st.info("🟡 Medium confidence: the correction is probably right, but double-check it.")
                st.caption("📋 Copy-ready text (use the copy icon on the right):")
                st.code(res["text"], language=None, wrap_lines=True)
                if res["truncated"]:
                    st.warning(f"Your message is longer than the {res['max_src_len'] - 1} tokens the models accept, "
                               "so only its beginning was used.")
                if not res["reached_eos"]:
                    st.info("The model reached the maximum output length without finishing its sentence.")
                with st.expander("🔍 Under the hood"):
                    st.markdown(f"**Normalized input:** `{res['normalized']}`")
                    st.markdown("**Tokens:** " + " · ".join(f"`{t}`" for t in res["tokens"]))
                    st.markdown(f"**Token IDs** (+ `<EOS>`=3): `{res['token_ids']}`")
                    st.markdown("**Unknown words → `<UNK>`:** " + (", ".join(f"`{t}`" for t in res["unknown"]) or "none 🎉"))
                    st.markdown(f"**Confidence score** (mean log-probability of the generated words): `{res['mean_logp']:.3f}` "
                                f"(high ≥ {CONFIDENCE_THRESHOLDS['high']}, low < {CONFIDENCE_THRESHOLDS['low']})")
                    st.caption(f"Model inference time {fmt_ms(res['ms'])} (checkpoint lookup {fmt_ms(res['load_ms'])}, "
                               "cached after the first load). The reference text is never used — only your message.")
            else:
                st.markdown('<div class="pw-empty">🪄<br><b>Your polished message will appear here.</b><br>'
                            'Type something on the left and press <b>✨ IMPROVE MY TEXT</b>.</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------- academic comparison (optional)
    with st.expander("🔬 Academic view — compare the four architectures on this message"):
        st.caption("For the project presentation: the same message and decoding settings are run through the four trained "
                   "architectures (A_LSTM, B_BiLSTM, C_Attention, D_Pointer). Users of the app always get the best model.")
        if st.button("⚔️ Compare the 4 Models", use_container_width=True):
            st.session_state["compare_requested"] = True
            st.rerun()
    cmp = st.session_state.get("comparison")
    if cmp:
        dec = DECODERS[cmp["decoding"]] + (f" (k={cmp['beam_width']})" if cmp["decoding"] == "beam" else "")
        st.markdown(f"### ⚔️ Four-model comparison")
        st.caption(f"Same input, same decoding ({dec}). Input: “{cmp['input']}”")
        ok_times = {n: r["ms"] for n, r in cmp["results"].items() if "ms" in r}
        fastest = min(ok_times, key=ok_times.get) if ok_times else None
        cols = st.columns(4, gap="medium")
        for col, name in zip(cols, MODEL_NAMES):
            r, info = cmp["results"][name], MODEL_INFO[name]
            with col:
                if "error" in r:
                    body = f'<div class="pw-error">❌ {esc(r["error"])}</div>'
                    foot = ""
                else:
                    body = f'<div class="pw-cmp-text">{esc(r["text"])}</div>'
                    foot = f'<span class="pw-chip">⏱️ {fmt_ms(r["ms"])}</span>' + \
                           ('<span class="pw-chip">⚡ fastest</span>' if name == fastest else "") + \
                           (f'<span class="pw-chip">🧲 copied {esc(", ".join(r["copied_oov"]))}</span>' if r["copied_oov"] else "")
                st.markdown(f"""<div class="pw-cmp" style="border-top: 4px solid {info['color']};">
                    <div class="pw-cmp-head"><div><div class="pw-cmp-name">{info['emoji']} {name}</div>
                    <div class="pw-cmp-label">{esc(info['label'])}</div></div></div>
                    {body}<div class="pw-cmp-foot">{foot}</div></div>""", unsafe_allow_html=True)

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.markdown("## 🎓 About this demo")
        st.markdown("ProwriteAI rewrites informal English into clear, professional English with **four Encoder–Decoder "
                    "networks built and trained from scratch** in PyTorch for a university NLP project. No ChatGPT, Claude, "
                    "Gemini, T5, BART or GPT: every word on screen comes from these checkpoints.")
        st.markdown("### 🤖 Model in use")
        st.caption("Selected automatically: the checkpoint with the lowest validation loss.")
        if model_name:
            try:
                _, _, meta = get_model(model_name)
                st.markdown(f"**{MODEL_INFO[model_name]['emoji']} {model_name}**  \n{esc(meta['architecture'])}")
                m1, m2 = st.columns(2)
                m1.metric("Best epoch", meta["best_epoch"], help="Epoch with the lowest validation loss (restored weights).")
                m2.metric("Val. loss", f"{meta['best_val_loss']:.3f}" if meta["best_val_loss"] is not None else "–",
                          help="Token-level negative log-likelihood on validation.csv.")
                m1.metric("Parameters", f"{meta['parameters']/1e6:.1f}M")
                m2.metric("Vocabulary", f"{meta['vocab_size']:,}")
                st.caption(f"Trained {meta['epochs_trained']} epochs ({meta['stop_reason']}). Tokenizer from "
                           f"{meta['tokenizer_source']}. Max input {meta['max_src_len']} / output {meta['max_tgt_len']} tokens.")
            except AppError as e:
                st.error(str(e))
        else:
            st.error(f"No usable checkpoint found in `{CKPT_DIR}`.")

        metrics_path = OUTPUT_DIR / "test_metrics.csv"
        if metrics_path.exists():
            st.markdown("### 📊 Measured test scores")
            try:
                rows = read_scoreboard(str(metrics_path), metrics_path.stat().st_mtime)
                table = "| Model | BLEU | chrF |\n|---|---|---|\n" + "\n".join(
                    f"| {r['model'].replace('copy_input_baseline', 'copy input (baseline)')} | {float(r['BLEU']):.1f} | {float(r['chrF']):.1f} |"
                    for r in rows)
                st.markdown(table)
                st.caption("From `test_metrics.csv` written by the notebook (unseen test messages, greedy decoding). "
                           "Live outputs reflect these scores, so weaker models can produce off-topic sentences.")
            except Exception as e:
                st.caption(f"Could not read test_metrics.csv: {e}")

        st.caption("Override with PROWRITE_CHECKPOINT_DIR / PROWRITE_ARTIFACT_DIR / PROWRITE_OUTPUT_DIR.")

    st.markdown('<div class="pw-footer">✨ ProwriteAI · Smart Text Improvement · Encoder–Decoder models (LSTM, BiLSTM, '
                'Bahdanau attention, Pointer-Generator) trained from scratch · Streamlit + PyTorch</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
