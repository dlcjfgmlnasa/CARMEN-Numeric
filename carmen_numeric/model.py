"""UniTTab: one Transformer for time series of heterogeneous tabular rows.

Luetto, Garuti, Sangineto, Forni, Cucchiara, "One Transformer for All Time Series:
Representing and Training with Time-Dependent Heterogeneous Tabular Data" (arXiv
2302.06375). Architecture (paper Fig. 2b):

    field values ──► field embeddings f_1..f_k   (numerical: γ(v) → per-field Linear,
                                                   categorical: per-field lookup)
                 ──► Field Transformer (within a row, shared by all row types)
                 ──► concat (d·k_h) ──► W_h (row-type specific) ──► g' ∈ R^m
                 ──► Sequence Transformer (across rows, [CLS] + positions)
                 ──► z ∈ R^m ──► S_h ──► split into k_h vectors ──► per-field MLP head
                     (predicts the masked field: a category, or the bin of a number)

Rows of the same type always have the same fields in the same slots, so every stage
that depends on the row type runs once per type on a dense ``(n_h, k_h, ·)`` block —
no field-level padding inside the Field Transformer.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalize import Normalizer
from .schema import TableSchema


@dataclass
class UniTTabConfig:
    """Model hyper-parameters. Defaults are the paper's (Tab. 16)."""

    # data-dependent (filled by ``from_schema``)
    field_kinds: list[str] = field(default_factory=list)  # per field id; index 0 = ""
    vocab_sizes: list[int] = field(default_factory=list)  # output size per field id
    row_type_fields: list[list[int]] = field(default_factory=list)  # per row type id
    # Field Transformer
    d_field: int = 72
    field_layers: int = 1
    field_heads: int = 8
    # Sequence Transformer
    d_model: int = 1080
    seq_layers: int = 12
    seq_heads: int = 12
    max_rows: int = 512
    # misc
    num_freqs: int = 8  # L sine/cosine pairs
    ffn_mult: int = 4
    dropout: float = 0.1
    head_hidden: int | None = None  # hidden width of the per-field MLP heads

    @classmethod
    def from_schema(cls, schema: TableSchema, normalizer: Normalizer, **kw) -> UniTTabConfig:
        kinds = [""] + [f.kind for f in schema.fields]
        rows = [[]] + [schema.row_field_ids(i + 1) for i in range(schema.num_row_types)]
        return cls(
            field_kinds=kinds,
            vocab_sizes=normalizer.vocab_sizes(),
            row_type_fields=rows,
            **kw,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# Presets: "paper" matches Tab. 16; the smaller ones are for quick experiments.
PRESETS: dict[str, dict] = {
    "paper": dict(d_field=72, field_layers=1, field_heads=8, d_model=1080, seq_layers=12,
                  seq_heads=12),
    "base": dict(d_field=64, field_layers=1, field_heads=8, d_model=512, seq_layers=8,
                 seq_heads=8),
    "small": dict(d_field=48, field_layers=1, field_heads=4, d_model=256, seq_layers=4,
                  seq_heads=4),
    "tiny": dict(d_field=32, field_layers=1, field_heads=4, d_model=128, seq_layers=2,
                 seq_heads=4),
}


class FrequencyEncoding(nn.Module):
    """γ(v) = (sin(2^0 πv), cos(2^0 πv), …, sin(2^{L-1} πv), cos(2^{L-1} πv)) — Eq. (2)."""

    def __init__(self, num_freqs: int = 8) -> None:
        super().__init__()
        self.register_buffer(
            "freqs", (2.0 ** torch.arange(num_freqs)) * math.pi, persistent=False
        )

    @property
    def out_dim(self) -> int:
        return 2 * self.freqs.numel()

    def forward(self, v: torch.Tensor) -> torch.Tensor:  # (...) -> (..., 2L)
        x = v.unsqueeze(-1).float() * self.freqs
        return torch.stack([x.sin(), x.cos()], dim=-1).flatten(-2)


class Prediction(NamedTuple):
    """Logits for the masked values of one (row type, slot)."""

    field_id: int
    logits: torch.Tensor  # (n, V_f)
    target: torch.Tensor  # (n,)
    batch_idx: torch.Tensor  # (n,) position of each value in the batch
    row_idx: torch.Tensor  # (n,)
    slot: int


def _encoder(d: int, heads: int, layers: int, ffn_mult: int, dropout: float) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=d,
        nhead=heads,
        dim_feedforward=ffn_mult * d,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class UniTTab(nn.Module):
    """UniTTab encoder + masked-token heads.

    Batch layout (see ``data.collate``) — ``T`` rows, ``K`` field slots per row:

    * ``row_type`` ``(B, T)`` long, 1-based, 0 = padding row
    * ``num_value`` ``(B, T, K)`` float in [0, 1] (numerical slots)
    * ``cat_value`` ``(B, T, K)`` long (categorical slots)
    * ``missing`` ``(B, T, K)`` bool — the field exists but has no value
    """

    def __init__(self, cfg: UniTTabConfig) -> None:
        super().__init__()
        self.cfg = cfg
        n_fields = len(cfg.field_kinds) - 1
        d, m = cfg.d_field, cfg.d_model
        if n_fields < 1 or len(cfg.row_type_fields) < 2:
            raise ValueError("config has no fields / row types; build it with from_schema")

        # ── field embeddings ──
        self.freq = FrequencyEncoding(cfg.num_freqs)
        two_l = self.freq.out_dim
        # one Linear(2L, d) per numerical field, stored as stacked tensors
        self.num_weight = nn.Parameter(torch.empty(n_fields + 1, two_l, d))
        self.num_bias = nn.Parameter(torch.zeros(n_fields + 1, d))
        nn.init.uniform_(self.num_weight, -1 / math.sqrt(two_l), 1 / math.sqrt(two_l))
        # categorical: one table, field-specific offsets (row 0 = dummy for non-cat slots)
        offsets = [0] * (n_fields + 1)
        total = 1
        for fid in range(1, n_fields + 1):
            if cfg.field_kinds[fid] == "cat":
                offsets[fid] = total
                total += cfg.vocab_sizes[fid]
        self.register_buffer("cat_offset", torch.tensor(offsets), persistent=False)
        self.cat_emb = nn.Embedding(total, d)
        # special inputs, per field (so they also carry the field identity)
        self.mask_emb = nn.Embedding(n_fields + 1, d)
        self.missing_emb = nn.Embedding(n_fields + 1, d)
        is_num = torch.tensor([k == "num" for k in cfg.field_kinds])
        self.register_buffer("is_num", is_num, persistent=False)
        for h, fids in enumerate(cfg.row_type_fields):
            self.register_buffer(f"rt_fields_{h}", torch.tensor(fids, dtype=torch.long),
                                 persistent=False)

        # ── Field Transformer (shared by every row type) ──
        self.field_encoder = _encoder(d, cfg.field_heads, cfg.field_layers, cfg.ffn_mult,
                                      cfg.dropout)
        self.field_norm = nn.LayerNorm(d)

        # ── row-type specific interface W_h : R^{d·k_h} -> R^m  (Eq. 1) ──
        self.row_in = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(d * len(f), m) for f in cfg.row_type_fields[1:]]
        )

        # ── Sequence Transformer ──
        self.cls = nn.Parameter(torch.zeros(1, 1, m))
        self.pos_emb = nn.Embedding(cfg.max_rows + 1, m)  # +1 for [CLS]
        self.seq_drop = nn.Dropout(cfg.dropout)
        self.seq_encoder = _encoder(m, cfg.seq_heads, cfg.seq_layers, cfg.ffn_mult,
                                    cfg.dropout)
        self.seq_norm = nn.LayerNorm(m)

        # ── output side: S_h : R^m -> R^{d·k_h}, then one shallow MLP per field ──
        self.row_out = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(m, d * len(f)) for f in cfg.row_type_fields[1:]]
        )
        hid = cfg.head_hidden or d
        self.heads = nn.ModuleList([nn.Identity()])
        for fid in range(1, n_fields + 1):
            self.heads.append(nn.Sequential(
                nn.Linear(d, hid), nn.GELU(), nn.LayerNorm(hid),
                nn.Linear(hid, cfg.vocab_sizes[fid]),
            ))

        for emb in (self.cat_emb, self.mask_emb, self.missing_emb, self.pos_emb):
            nn.init.trunc_normal_(emb.weight, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    # ── helpers ────────────────────────────────────────────────────

    @property
    def num_row_types(self) -> int:
        return len(self.cfg.row_type_fields) - 1

    def _fields(self, h: int) -> torch.Tensor:
        return getattr(self, f"rt_fields_{h}")

    def _embed_fields(
        self,
        h: int,
        num_value: torch.Tensor,  # (n, k_h)
        cat_value: torch.Tensor,  # (n, k_h)
        missing: torch.Tensor,  # (n, k_h)
        masked: torch.Tensor | None,  # (n, k_h)
    ) -> torch.Tensor:  # (n, k_h, d)
        fids = self._fields(h)  # (k_h,)
        gamma = self.freq(num_value)  # (n, k_h, 2L)
        x_num = torch.einsum("nkl,kld->nkd", gamma, self.num_weight[fids]) + self.num_bias[fids]
        is_num = self.is_num[fids]  # (k_h,)
        cat_idx = torch.where(is_num, 0, self.cat_offset[fids] + cat_value)
        x_cat = self.cat_emb(cat_idx)
        x = torch.where(is_num[None, :, None], x_num, x_cat)
        x = torch.where(missing.unsqueeze(-1), self.missing_emb(fids).expand_as(x), x)
        if masked is not None:
            x = torch.where(masked.unsqueeze(-1), self.mask_emb(fids).expand_as(x), x)
        return x

    # ── forward ────────────────────────────────────────────────────

    def encode(
        self, batch: dict[str, torch.Tensor], masked: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Encode a batch.

        Returns ``rows`` ``(B, T, m)`` (final row embeddings z), ``cls`` ``(B, m)`` and
        ``row_mask`` ``(B, T)`` (valid rows).
        """
        row_type = batch["row_type"]
        b, t = row_type.shape
        if t > self.cfg.max_rows:
            raise ValueError(f"{t} rows > max_rows={self.cfg.max_rows}")
        m = self.cfg.d_model
        g = torch.zeros(b, t, m, device=row_type.device, dtype=self.cls.dtype)
        for h in range(1, self.num_row_types + 1):
            bi, ti = torch.nonzero(row_type == h, as_tuple=True)
            if bi.numel() == 0:
                continue
            k = self._fields(h).numel()
            x = self._embed_fields(
                h,
                batch["num_value"][bi, ti, :k],
                batch["cat_value"][bi, ti, :k],
                batch["missing"][bi, ti, :k],
                None if masked is None else masked[bi, ti, :k],
            )
            x = self.field_norm(self.field_encoder(x))  # (n, k_h, d)
            g[bi, ti] = self.row_in[h](x.flatten(1)).to(g.dtype)  # W_h

        row_mask = row_type > 0
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, g], dim=1)
        pos = torch.arange(t + 1, device=x.device)
        x = self.seq_drop(x + self.pos_emb(pos)[None])
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~row_mask], 1)
        z = self.seq_norm(self.seq_encoder(x, src_key_padding_mask=pad))
        return {"cls": z[:, 0], "rows": z[:, 1:], "row_mask": row_mask}

    def forward(
        self, batch: dict[str, torch.Tensor], masked: torch.Tensor
    ) -> dict[str, object]:
        """Masked-token forward: encode with ``masked`` slots hidden, predict them.

        Returns the ``encode`` outputs plus ``predictions``: a list of ``Prediction`` —
        one per (row type, slot) that has masked values in the batch.
        """
        out = self.encode(batch, masked)
        z = out["rows"]
        row_type = batch["row_type"]
        target = batch["target"]
        preds: list[Prediction] = []
        for h in range(1, self.num_row_types + 1):
            fids = self.cfg.row_type_fields[h]
            k = len(fids)
            sel = (row_type == h) & masked[..., :k].any(-1)
            bi, ti = torch.nonzero(sel, as_tuple=True)
            if bi.numel() == 0:
                continue
            zz = self.row_out[h](z[bi, ti]).view(-1, k, self.cfg.d_field)  # S_h, split
            mk = masked[bi, ti, :k]
            tg = target[bi, ti, :k]
            for j, fid in enumerate(fids):
                rows = torch.nonzero(mk[:, j], as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                preds.append(Prediction(fid, self.heads[fid](zz[rows, j]), tg[rows, j],
                                        bi[rows], ti[rows], j))
        out["predictions"] = preds
        return out


# ── pre-training objective ─────────────────────────────────────────


def sample_mask(
    batch: dict[str, torch.Tensor],
    is_time_field: torch.Tensor,  # (F+1,) bool lookup
    p_field: float = 0.15,
    p_row: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """UniTTab masking: field masking (p_f), row masking (p_r), joint time-stamp masking.

    Only fields that exist and have a value can be masked (a missing value has
    nothing to predict).
    """
    fid = batch["field_id"]
    can = (fid > 0) & ~batch["missing"]
    dev = fid.device
    m = torch.rand(fid.shape, generator=generator, device=dev) < p_field
    rows = torch.rand(fid.shape[:2], generator=generator, device=dev) < p_row
    m = m | rows.unsqueeze(-1)
    is_time = is_time_field.to(dev)[fid] & (fid > 0)
    any_time = (m & is_time).any(-1, keepdim=True)
    m = torch.where(is_time, any_time.expand_as(m), m)
    return m & can


@dataclass
class SpanMaskConfig:
    """Contiguous-span masking, added on top of the UniTTab masks (not in the paper).

    Minute-level vitals change slowly, so a value hidden alone is recovered by copying
    its neighbors; hiding a whole stretch forces the model to infer it from the other
    variables and from longer context.

    p:               expected fraction of dynamic rows covered by spans (0 disables).
    min_len/max_len: span length in rows (uniform).
    field_prob:      probability that a span hides a single field (a variable-level gap,
                     e.g. the arterial line for 20 minutes) instead of every field.
    keep_time:       leave the time fields of spanned rows visible (the model knows
                     *when* it has to fill in).
    """

    p: float = 0.0
    min_len: int = 5
    max_len: int = 30
    field_prob: float = 0.5
    keep_time: bool = True


def sample_span_mask(
    batch: dict[str, torch.Tensor],
    is_time_field: torch.Tensor,  # (F+1,) bool lookup
    cfg: SpanMaskConfig,
    context_row_types: torch.Tensor | None = None,  # row types never spanned (static)
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``(B, T, K)`` bool mask of row spans / single-field spans over the dynamic rows."""
    fid = batch["field_id"]
    row_type = batch["row_type"]
    b, t, _ = fid.shape
    dev = fid.device
    if cfg.p <= 0:
        return torch.zeros_like(fid, dtype=torch.bool)
    dyn = row_type > 0
    if context_row_types is not None and context_row_types.numel():
        dyn &= ~torch.isin(row_type, context_row_types.to(dev))
    is_time = is_time_field.to(dev)[fid]
    n_spans = max(1, math.ceil(cfg.p * t / (0.5 * (cfg.min_len + cfg.max_len))))
    # span starts among the dynamic rows of each sequence
    n_dyn = dyn.sum(1, keepdim=True).clamp(min=1)  # (B, 1)
    first = torch.argmax(dyn.int(), dim=1, keepdim=True)  # first dynamic row
    u = torch.rand(b, n_spans, generator=generator, device=dev)
    start = first + (u * n_dyn).long()  # (B, S)
    length = torch.randint(cfg.min_len, cfg.max_len + 1, (b, n_spans),
                           generator=generator, device=dev)
    rows = torch.arange(t, device=dev)[None, None]  # (1, 1, T)
    in_span = (rows >= start[..., None]) & (rows < (start + length)[..., None])
    in_span &= dyn[:, None, :]  # (B, S, T)
    # a field span hides one field id, drawn from the values present in the window
    observed = (fid > 0) & ~batch["missing"] & dyn[..., None] & ~is_time
    flat = observed.flatten(1).float()  # (B, T*K)
    flat = flat + (flat.sum(1, keepdim=True) == 0).float()  # avoid an all-zero row
    pick = torch.multinomial(flat, n_spans, replacement=True, generator=generator)
    span_fid = torch.gather(fid.flatten(1), 1, pick)  # (B, S)
    is_field_span = torch.rand(b, n_spans, generator=generator, device=dev) < cfg.field_prob
    hit_field = fid[:, None] == span_fid[..., None, None]  # (B, S, T, K)
    slot_sel = hit_field | ~is_field_span[..., None, None]
    m = (in_span[..., None] & slot_sel).any(1)  # (B, T, K)
    if cfg.keep_time:
        m &= ~is_time
    return m & (fid > 0) & ~batch["missing"]


@dataclass
class ForecastMaskConfig:
    """Hide the last rows of a window: predict the next minutes from the past only.

    Span masks still see both ends of a gap (interpolation); this regime does not.

    p:               probability that a sequence gets a forecast mask (0 disables).
    min_len/max_len: number of final dynamic rows hidden (uniform), at most half of the
                     sequence's dynamic rows.
    keep_time:       leave the time fields visible (the model knows *when* to predict).
    """

    p: float = 0.0
    min_len: int = 5
    max_len: int = 30
    keep_time: bool = True


def sample_forecast_mask(
    batch: dict[str, torch.Tensor],
    is_time_field: torch.Tensor,  # (F+1,) bool lookup
    cfg: ForecastMaskConfig,
    context_row_types: torch.Tensor | None = None,  # row types never hidden (static)
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``(B, T, K)`` bool mask hiding every value of the last ``L`` dynamic rows."""
    fid = batch["field_id"]
    row_type = batch["row_type"]
    b, t, _ = fid.shape
    dev = fid.device
    if cfg.p <= 0:
        return torch.zeros_like(fid, dtype=torch.bool)
    dyn = row_type > 0
    if context_row_types is not None and context_row_types.numel():
        dyn &= ~torch.isin(row_type, context_row_types.to(dev))
    n_dyn = dyn.sum(1)  # (B,)
    length = torch.randint(cfg.min_len, cfg.max_len + 1, (b,), generator=generator, device=dev)
    length = torch.minimum(length, n_dyn // 2)
    on = torch.rand(b, generator=generator, device=dev) < cfg.p
    rank_from_end = dyn.flip(1).long().cumsum(1).flip(1)  # 1 = last dynamic row
    rows = dyn & (rank_from_end <= length[:, None]) & on[:, None]  # (B, T)
    m = rows[..., None].expand_as(fid)
    if cfg.keep_time:
        m = m & ~is_time_field.to(dev)[fid]
    return m & (fid > 0) & ~batch["missing"]


def soft_targets(
    target: torch.Tensor, vocab: int, kind: str, eps: float = 0.1, radius: int = 5
) -> torch.Tensor:
    """Smoothed target distribution (n, vocab).

    Categorical — standard label smoothing, Eq. (3): ``1-ε`` on the label,
    ``ε/(V-1)`` elsewhere. Numerical — neighborhood label smoothing, Eq. (4): ``1-ε``
    on the bin, ``ε/10`` on each bin within ±5 (``radius``), 0 elsewhere. Near the ends
    of the range fewer neighbors exist; their share of ``ε`` is spread over the
    neighbors that do, so every row still sums to 1 (the paper does not say).
    """
    n = target.shape[0]
    p = torch.zeros(n, vocab, device=target.device)
    if vocab == 1 or eps == 0:
        p[torch.arange(n), target] = 1.0
        return p
    if kind == "cat":
        p.fill_(eps / (vocab - 1))
        p[torch.arange(n), target] = 1.0 - eps
        return p
    offs = torch.tensor([o for o in range(-radius, radius + 1) if o != 0], device=target.device)
    nb = target.unsqueeze(1) + offs  # (n, 2r)
    ok = (nb >= 0) & (nb < vocab)
    share = eps / ok.sum(1, keepdim=True).clamp(min=1)
    p.scatter_add_(1, nb.clamp(0, vocab - 1), torch.where(ok, share, 0.0))
    p[torch.arange(n), target] = 1.0 - eps
    no_nb = ok.sum(1) == 0  # cannot happen for vocab > 1, kept for safety
    p[no_nb] = 0.0
    p[no_nb, target[no_nb]] = 1.0
    return p


def masked_token_loss(
    predictions: list[Prediction],
    field_kinds: list[str],
    eps: float = 0.1,
    radius: int = 5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy with (neighborhood) label smoothing over every masked value — Eq. (5).

    Returns the mean loss over masked values and simple accuracies.
    """
    total = None
    n_total = 0
    hits = {"num": [], "num_pm2": [], "cat": []}
    counts = {"num": 0, "cat": 0}
    for pr in predictions:
        fid, logits, target = pr.field_id, pr.logits, pr.target
        kind = field_kinds[fid]
        logp = F.log_softmax(logits.float(), dim=-1)
        p = soft_targets(target, logits.shape[-1], kind, eps, radius)
        loss = -(p * logp).sum()
        total = loss if total is None else total + loss
        n = target.numel()
        n_total += n
        with torch.no_grad():  # kept as tensors: one host sync at the end, not one per field
            hit = logits.argmax(-1)
            if kind == "num":
                counts["num"] += n
                hits["num"].append((hit == target).sum())
                hits["num_pm2"].append(((hit - target).abs() <= 2).sum())
            else:
                counts["cat"] += n
                hits["cat"].append((hit == target).sum())
    if total is None:
        raise ValueError("nothing was masked in this batch")
    keys = [k for k in hits if hits[k]]
    vals = torch.stack([torch.stack(hits[k]).sum() for k in keys]).tolist() if keys else []
    got = dict(zip(keys, vals, strict=True))
    stats = {
        "n_num": counts["num"],
        "n_cat": counts["cat"],
        "acc_num": got.get("num", 0) / max(counts["num"], 1),
        "acc_num_pm2": got.get("num_pm2", 0) / max(counts["num"], 1),
        "acc_cat": got.get("cat", 0) / max(counts["cat"], 1),
    }
    return total / n_total, stats
