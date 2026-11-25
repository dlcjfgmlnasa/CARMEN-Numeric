"""Render the prediction figures from ``viz_predictions.py`` output (matplotlib).

    python scripts/plot_predictions.py viz_signal.json --out figs/

Writes forecast_examples.png, risk_timelines.png, roc.png and lead_error.png; with
``--finetune`` (and ``--scratch``) ft_roc.png and ft_risk_timelines.png instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-3, status critical, ink, chrome
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
METHODS = [("encoder", "신호 전용 encoder + 선형", BLUE, "-"),
           ("hand_gbm", "수작업 추세 특징 + GBM", ORANGE, "-"),
           ("map_only", "현재 MAP만", AQUA, "--")]
FIELD_LABEL = {"art_mbp": "동맥압 MAP (mmHg)", "hr": "HR (bpm)", "spo2": "SpO2 (%)",
               "nibp_mbp": "NIBP MAP (mmHg)", "etco2": "EtCO2 (mmHg)"}


def setup() -> None:
    for name in ("NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False,
        "axes.unicode_minus": False, "lines.linewidth": 2,
    })


def arr(x) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def forecast_examples(d: dict, out: Path) -> None:
    items = [(e, f) for e in d["examples"] for f in e["forecasts"]]
    fields = ["art_mbp", "hr", "spo2"]
    fig, axes = plt.subplots(len(items), 3, figsize=(13, 2.9 * len(items)), squeeze=False)
    for r, (e, f) in enumerate(items):
        s = e["series"]
        t = arr(s["t"])
        a = f["anchor_t"]
        for c, name in enumerate(fields):
            ax = axes[r, c]
            v = arr(s[name])
            past = (t <= a) & (t >= a - 45)
            fut = (t > a) & (t <= a + 16)
            ax.plot(t[past], v[past], color=INK2, lw=1.6, label="실측 (과거)")
            ax.plot(t[fut], v[fut], color=INK, lw=2, marker="o", ms=3, label="실측 (미래)")
            if name in f:
                p = f[name]
                ft = arr(p["t"])
                ax.fill_between(ft, arr(p["q10"]), arr(p["q90"]), color=BLUE, alpha=0.18,
                                lw=0, label="예측 10–90%")
                ax.plot(ft, arr(p["point"]), color=BLUE, lw=2, label="모델 예측")
                ax.plot(ft, arr(p["persist"]), color=ORANGE, lw=2, ls=":",
                        label="마지막 값 유지")
            ax.axvline(a, color=AXIS, lw=1)
            if name == "art_mbp":
                ax.axhline(65, color=CRITICAL, lw=1, ls="--")
                ax.text(a - 44, 65, " 65", color=CRITICAL, va="bottom", fontsize=8)
            ax.set_xlim(a - 45, a + 16)
            if r == 0:
                ax.set_title(FIELD_LABEL[name])
            if c == 0:
                kind = "저혈압 시작 10분 전" if f["kind"] == "before_onset" else "안정 구간"
                ax.set_ylabel(f"케이스 {e['caseid']}\n{kind}", color=INK, fontsize=9)
            if r == len(items) - 1:
                ax.set_xlabel("마취 시작 후 시간 (분) · 세로선 = 예측 시점")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("15분 미래 예측 — 처음 보는 환자 (무작위로 고른 test 케이스)", y=1.025,
                 color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "forecast_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def risk_timelines(d: dict, out: Path) -> None:
    ex = d["examples"]
    fig, axes = plt.subplots(2 * len(ex), 1, figsize=(13, 3.3 * len(ex)), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1] * len(ex)})
    for i, e in enumerate(ex):
        ax, ar = axes[2 * i, 0], axes[2 * i + 1, 0]
        t = arr(e["series"]["t"])
        mp = arr(e["series"]["art_mbp"])
        ax.plot(t, mp, color=INK2, lw=1.4)
        low = mp < 65
        ax.scatter(t[low], mp[low], color=CRITICAL, s=10, zorder=3, label="65 미만")
        ax.axhline(65, color=CRITICAL, lw=1, ls="--")
        for o in e["onsets_t"]:
            for a in (ax, ar):
                a.axvline(o, color=CRITICAL, lw=0.8, alpha=0.6)
        ax.set_ylabel("동맥압 MAP\n(mmHg)")
        tag = "저혈압 있음" if e["has_onset"] else "저혈압 없음"
        ax.set_title(f"케이스 {e['caseid']} ({tag}) — 세로선 = 저혈압 시작", loc="left")
        rt = arr(e["risk_t"])
        ar.plot(rt, arr(e["risk"]["5"]), color=BLUE, lw=1.6, label="5분 내 저혈압 점수")
        ar.plot(rt, arr(e["risk"]["10"]), color=ORANGE, lw=1.6, label="10분 내 저혈압 점수")
        ar.set_ylim(0, 1)
        ar.set_ylabel("위험 점수\n(상대값)")
        ar.set_xlim(ax.get_xlim())
        if i == 0:
            ar.legend(loc="upper right", ncol=2, fontsize=9)
            ax.legend(loc="upper right", fontsize=9)
    axes[-1, 0].set_xlabel("마취 시작 후 시간 (분)")
    fig.tight_layout()
    fig.savefig(out / "risk_timelines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def roc(d: dict, out: Path) -> None:
    hs = list(d["probe"])
    fig, axes = plt.subplots(1, len(hs) + 1, figsize=(4.2 * (len(hs) + 1), 4.3))
    for ax, h in zip(axes, hs, strict=False):
        pr = d["probe"][h]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
        for key, label, color, ls in METHODS:
            m = pr["methods"][key]
            ax.plot(arr(m["fpr"]), arr(m["tpr"]), color=color, ls=ls,
                    label=f"{label}  {m['auroc']:.3f}")
        ax.set_title(f"{h}분 내 저혈압 (발생률 {pr['prevalence'] * 100:.1f}%)")
        ax.set_xlabel("위양성률 (1 - 특이도)")
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8, title="AUROC", title_fontsize=8)
    axes[0].set_ylabel("민감도")
    ax = axes[-1]
    cal = d["probe"]["5"]["calibration"]
    lim = max(max(cal["encoder"]["pred"]), max(cal["encoder"]["obs"]), 0.3) * 1.05
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1)
    for key, label, color, _ in METHODS[:2]:
        c = cal[key]
        ax.plot(c["pred"], c["obs"], color=color, marker="o", ms=5, label=label)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title("보정 (5분, 예측 위험 10분위)")
    ax.set_xlabel("예측 위험 평균")
    ax.set_ylabel("실제 발생률")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("저혈압 예측 — test 환자 371명 · ROC와 보정", color=INK, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def lead_error(d: dict, out: Path) -> None:
    fields = [f for f in ("art_mbp", "hr", "spo2", "nibp_mbp") if f in d["leads"]]
    fig, axes = plt.subplots(1, len(fields), figsize=(4.0 * len(fields), 3.6))
    for ax, f in zip(axes, fields, strict=True):
        L = d["leads"][f]
        x = np.array(L["lead"])
        ax.plot(x, arr(L["model"]), color=BLUE, marker="o", ms=4, label="모델")
        ax.plot(x, arr(L["persist"]), color=ORANGE, marker="o", ms=4, ls=":",
                label="마지막 값 유지")
        ax.set_title(FIELD_LABEL[f])
        ax.set_xlabel("몇 분 뒤를 예측하나")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("평균 절대 오차")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("미래 예측 오차 — 예측 거리별 (test 환자)", color=INK, fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out / "lead_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def finetune_figs(viz: dict, ft: dict, scratch: dict | None, out: Path) -> None:
    """ROC / calibration and risk over time: fine-tuned vs. frozen encoder vs. baselines."""
    hs = list(ft["horizons"])
    lines = [("finetuned", ft, "사전학습 encoder 미세조정", BLUE, "-")]
    if scratch is not None:
        lines.append(("finetuned", scratch, "같은 구조, 처음부터 학습", "#6250d6", "-"))
    fig, axes = plt.subplots(1, len(hs) + 1, figsize=(4.2 * (len(hs) + 1), 4.4))
    for ax, h in zip(axes, hs, strict=False):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
        for key, src, label, color, ls in lines:
            m = src["horizons"][h]["methods"][key]
            ax.plot(arr(m["fpr"]), arr(m["tpr"]), color=color, ls=ls,
                    label=f"{label}  {m['auroc']:.3f}")
        fz = viz["probe"][h]["methods"]["encoder"]
        ax.plot(arr(fz["fpr"]), arr(fz["tpr"]), color=MUTED, ls="-", lw=1.5,
                label=f"고정 encoder + 선형  {fz['auroc']:.3f}")
        for key, label, color, ls in METHODS[1:]:
            m = ft["horizons"][h]["methods"][key]
            ax.plot(arr(m["fpr"]), arr(m["tpr"]), color=color, ls=ls, lw=1.5,
                    label=f"{label}  {m['auroc']:.3f}")
        ax.set_title(f"{h}분 내 저혈압 (발생률 {ft['horizons'][h]['prevalence'] * 100:.1f}%)")
        ax.set_xlabel("위양성률 (1 - 특이도)")
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=7.5, title="AUROC", title_fontsize=8)
    axes[0].set_ylabel("민감도")
    ax = axes[-1]
    cal = ft["horizons"]["5"]["calibration"]
    lim = max(max(cal["finetuned"]["pred"]), max(cal["finetuned"]["obs"]), 0.3) * 1.05
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1)
    ax.plot(cal["finetuned"]["pred"], cal["finetuned"]["obs"], color=BLUE, marker="o", ms=5,
            label="미세조정 encoder")
    ax.plot(cal["hand_gbm"]["pred"], cal["hand_gbm"]["obs"], color=ORANGE, marker="o", ms=5,
            label="수작업 추세 특징 + GBM")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title("보정 (5분, 예측 위험 10분위)")
    ax.set_xlabel("예측 위험 평균")
    ax.set_ylabel("실제 발생률")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("저혈압 예측 — encoder까지 함께 학습 (test 환자)", color=INK, fontsize=13,
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out / "ft_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ex = {e["caseid"]: e for e in viz["examples"]}
    fx = [e for e in ft["examples"] if e["caseid"] in ex]
    fig, axes = plt.subplots(2 * len(fx), 1, figsize=(13, 3.3 * len(fx)), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1] * len(fx)})
    for i, e in enumerate(fx):
        v = ex[e["caseid"]]
        ax, ar = axes[2 * i, 0], axes[2 * i + 1, 0]
        t = arr(v["series"]["t"])
        mp = arr(v["series"]["art_mbp"])
        ax.plot(t, mp, color=INK2, lw=1.4)
        low = mp < 65
        ax.scatter(t[low], mp[low], color=CRITICAL, s=10, zorder=3, label="65 미만")
        ax.axhline(65, color=CRITICAL, lw=1, ls="--")
        for o in v["onsets_t"]:
            for a in (ax, ar):
                a.axvline(o, color=CRITICAL, lw=0.8, alpha=0.6)
        ax.set_ylabel("동맥압 MAP\n(mmHg)")
        tag = "저혈압 있음" if v["has_onset"] else "저혈압 없음"
        ax.set_title(f"케이스 {e['caseid']} ({tag}) — 세로선 = 저혈압 시작", loc="left")
        ar.plot(arr(e["risk_t"]), arr(e["risk"]["5"]), color=BLUE, lw=1.6,
                label="미세조정 encoder (5분)")
        ar.plot(arr(v["risk_t"]), arr(v["risk"]["5"]), color=MUTED, lw=1.2,
                label="고정 encoder + 선형 (5분, 상대값)")
        ar.set_ylim(0, 1)
        ar.set_ylabel("5분 내\n저혈압 위험")
        ar.set_xlim(ax.get_xlim())
        if i == 0:
            ar.legend(loc="upper right", ncol=2, fontsize=9)
            ax.legend(loc="upper right", fontsize=9)
    axes[-1, 0].set_xlabel("마취 시작 후 시간 (분)")
    fig.tight_layout()
    fig.savefig(out / "ft_risk_timelines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--finetune", help="finetune_ioh.py result.json (pre-trained encoder)")
    ap.add_argument("--scratch", help="finetune_ioh.py result.json (--scratch)")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    if args.finetune:
        ft = json.loads(Path(args.finetune).read_text())
        sc = json.loads(Path(args.scratch).read_text()) if args.scratch else None
        finetune_figs(d, ft, sc, out)
        print(f"figures in {out}")
        return
    forecast_examples(d, out)
    risk_timelines(d, out)
    roc(d, out)
    lead_error(d, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
