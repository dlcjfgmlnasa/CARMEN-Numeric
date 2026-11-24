import numpy as np
import pytest
import torch

from carmen_numeric import (
    Field,
    Normalizer,
    RowType,
    TableSchema,
    UniTTab,
    UniTTabConfig,
    masked_token_loss,
    sample_mask,
    soft_targets,
)
from carmen_numeric.data.dataset import WindowDataset, collate
from carmen_numeric.model import FrequencyEncoding


def toy_schema() -> TableSchema:
    return TableSchema(
        fields=[
            Field("t", "num", is_time=True, norm_range=(0, 100)),
            Field("a", "num"),
            Field("b", "num"),
            Field("c", "cat"),
            Field("age", "num"),
        ],
        row_types=[RowType("static", ("age", "c")), RowType("dyn", ("t", "a", "b"))],
    )


def toy_cases(schema: TableSchema, n_cases: int = 6, n_rows: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = schema.max_fields_per_row
    cases = {}
    for cid in range(1, n_cases + 1):
        v = np.full((n_rows + 1, k), np.nan, np.float32)
        v[0, 0] = rng.uniform(20, 80)
        v[0, 1] = rng.integers(0, 3)
        t = np.arange(n_rows, dtype=np.float32)
        v[1:, 0] = t
        v[1:, 1] = 50 + 10 * np.sin(t / 5) + rng.normal(0, 1, n_rows)
        v[1:, 2] = rng.normal(0, 1, n_rows)
        v[1::4, 2] = np.nan  # some missing values
        rt = np.array([1] + [2] * n_rows)
        cases[cid] = {"values": v, "row_type": rt, "subjectid": cid}
    return cases


def toy_normalizer(schema, cases):
    from carmen_numeric.data.dataset import collect_field_values

    vals = collect_field_values(cases, schema)
    return Normalizer.fit(schema, vals, {"c": 3}, num_bins=20)


def toy_model(schema, norm, **kw):
    cfg = UniTTabConfig.from_schema(
        schema, norm, d_field=16, field_heads=2, d_model=32, seq_layers=2, seq_heads=2,
        max_rows=64, dropout=0.0, **kw,
    )
    return UniTTab(cfg)


def test_frequency_encoding():
    enc = FrequencyEncoding(8)
    v = torch.tensor([0.0, 0.25, 1.0])
    g = enc(v)
    assert g.shape == (3, 16)
    assert torch.allclose(g[0, 1], torch.tensor(1.0))  # cos(0)
    assert torch.allclose(g[1, 0], torch.sin(torch.tensor(torch.pi / 4)))


@pytest.mark.parametrize("kind", ["num", "cat"])
@pytest.mark.parametrize("vocab", [1, 2, 7, 50])
def test_soft_targets_are_distributions(kind, vocab):
    target = torch.arange(vocab).repeat(3)
    p = soft_targets(target, vocab, kind, eps=0.1, radius=5)
    assert torch.allclose(p.sum(1), torch.ones(len(target)))
    assert (p >= 0).all()
    if vocab > 1:
        assert torch.allclose(p[torch.arange(len(target)), target], torch.tensor(0.9))


def test_neighborhood_smoothing_support():
    p = soft_targets(torch.tensor([20]), 50, "num", eps=0.1, radius=5)
    nz = torch.nonzero(p[0]).flatten().tolist()
    assert nz == list(range(15, 26))
    assert torch.isclose(p[0, 15], torch.tensor(0.01))


def test_normalizer_roundtrip(tmp_path):
    s = toy_schema()
    cases = toy_cases(s)
    n = toy_normalizer(s, cases)
    n.save(tmp_path / "n.json")
    n2 = Normalizer.load(s, tmp_path / "n.json")
    c = cases[1]
    a, b = n.transform(c["values"], c["row_type"]), n2.transform(c["values"], c["row_type"])
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert a["missing"][1:, 2].sum() == np.isnan(c["values"][1:, 2]).sum()
    assert (a["target"][a["missing"]] == -1).all()
    assert (a["num_value"] >= 0).all() and (a["num_value"] <= 1).all()
    dec = n.decode_bins(a["field_id"], a["target"])
    ok = (a["field_id"] == s.field_id["a"]) & ~a["missing"]
    assert np.nanmean(np.abs(dec[ok] - c["values"][ok])) < 3.0


def _batch(schema, max_rows=16):
    cases = toy_cases(schema)
    norm = toy_normalizer(schema, cases)
    ds = WindowDataset(cases, norm, max_rows=max_rows, random_crop=False)
    return norm, collate([ds[i] for i in range(len(ds))])


def test_mask_rules():
    s = toy_schema()
    _, batch = _batch(s)
    is_time = torch.tensor([False] + [f.is_time for f in s.fields])
    g = torch.Generator().manual_seed(0)
    m = sample_mask(batch, is_time, p_field=0.3, p_row=0.2, generator=g)
    assert not (m & batch["missing"]).any()
    assert not (m & (batch["field_id"] == 0)).any()
    assert m.any()


def test_forward_backward_and_padding_invariance():
    torch.manual_seed(0)
    s = toy_schema()
    norm, batch = _batch(s)
    model = toy_model(s, norm)
    is_time = torch.tensor([False] + [f.is_time for f in s.fields])
    m = sample_mask(batch, is_time, generator=torch.Generator().manual_seed(1))
    out = model(batch, m)
    loss, stats = masked_token_loss(out["predictions"], model.cfg.field_kinds)
    n_pred = sum(p.target.numel() for p in out["predictions"])
    assert n_pred == int(m.sum())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.num_weight.grad is not None and model.row_in[2].weight.grad.abs().sum() > 0

    # padding rows must not change the encoding of real rows
    model.eval()
    with torch.no_grad():
        full = model.encode(batch)["cls"]
        one = {k: v[:1] for k, v in batch.items()}
        n_real = int((one["row_type"][0] > 0).sum())
        one = {k: (v[:, :n_real] if v.dim() >= 2 else v) for k, v in one.items()}
        alone = model.encode(one)["cls"]
    assert torch.allclose(full[:1], alone, atol=1e-5)


def test_masked_values_do_not_leak():
    torch.manual_seed(0)
    s = toy_schema()
    norm, batch = _batch(s)
    model = toy_model(s, norm).eval()
    m = torch.zeros_like(batch["missing"])
    m[0, 3, 1] = True  # one numerical value
    b2 = {k: v.clone() for k, v in batch.items()}
    b2["num_value"][0, 3, 1] = 1.0 - b2["num_value"][0, 3, 1]
    with torch.no_grad():
        a = model(batch, m)["rows"]
        b = model(b2, m)["rows"]
    assert torch.allclose(a, b)


def test_gpu_batcher_matches_rows():
    from carmen_numeric.data.dataset import GPUWindowBatcher

    s = toy_schema()
    cases = toy_cases(s, n_rows=40)
    norm = toy_normalizer(s, cases)
    bt = GPUWindowBatcher(cases, norm, max_rows=16, context_types=(1,), device=torch.device("cpu"))
    b = bt.sample(8, torch.Generator().manual_seed(0))
    assert b["row_type"].shape == (8, 16)
    assert (b["row_type"][:, 0] == 1).all() and (b["row_type"][:, 1:] == 2).all()
    # time field of the dynamic rows is contiguous inside each window
    t_slot = b["num_value"][:, 1:, 0]
    d = torch.diff(t_slot, dim=1)
    assert torch.allclose(d, d[:, :1].expand_as(d), atol=1e-5)
    model = toy_model(s, norm)
    is_time = torch.tensor([False] + [f.is_time for f in s.fields])
    m = sample_mask(b, is_time, generator=torch.Generator().manual_seed(2))
    loss, _ = masked_token_loss(model(b, m)["predictions"], model.cfg.field_kinds)
    assert torch.isfinite(loss)


def test_span_mask_rules():
    from carmen_numeric import SpanMaskConfig, sample_span_mask

    s = toy_schema()
    _, batch = _batch(s, max_rows=40)
    is_time = torch.tensor([False] + [f.is_time for f in s.fields])
    cfg = SpanMaskConfig(p=0.3, min_len=4, max_len=8, field_prob=0.0, keep_time=True)
    m = sample_span_mask(batch, is_time, cfg, torch.tensor([1]),
                         generator=torch.Generator().manual_seed(0))
    assert m.any()
    assert not m[:, 0].any()  # the static (context) row is never spanned
    assert not (m & is_time[batch["field_id"]]).any()  # time stays visible
    assert not (m & batch["missing"]).any()
    # row spans: a spanned row hides every observed non-time value of the row
    rows = m.any(-1)
    vis = (batch["field_id"] > 0) & ~batch["missing"] & ~is_time[batch["field_id"]]
    assert ((m == vis) | ~rows[..., None]).all()

    cfg = SpanMaskConfig(p=0.3, min_len=4, max_len=8, field_prob=1.0, keep_time=True)
    m = sample_span_mask(batch, is_time, cfg, torch.tensor([1]),
                         generator=torch.Generator().manual_seed(1))
    per_seq_fields = [torch.unique(batch["field_id"][i][m[i]]).numel() for i in range(len(m))]
    n_spans = 2  # ceil(0.3 * 40 / 6)
    assert max(per_seq_fields) <= n_spans  # one field per span
