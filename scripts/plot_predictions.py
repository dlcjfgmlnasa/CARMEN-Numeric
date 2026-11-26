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


def paired_fig(cmp: dict, out: Path, methods: list[tuple[str, str, str]],
               name: str = "ft_paired.png") -> None:
    """Paired AUROC / AUPRC differences vs. the reference, with patient-bootstrap 95 % CI."""
    hs = list(cmp["horizons"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 + 0.42 * len(methods) * len(hs)),
                             sharey=True)
    for ax, key, title in ((axes[0], "auroc", "AUROC 차이"), (axes[1], "auprc", "AUPRC 차이")):
        y = 0
        ticks, labels = [], []
        for m, label, color in methods:
            for h in hs:
                r = cmp["horizons"][h]["methods"][m]
                d, ci = r[f"d_{key}"], r[f"d_{key}_ci"]
                ax.plot(ci, [y, y], color=color, lw=2, solid_capstyle="round")
                ax.plot([d], [y], marker="o", ms=7, color=color, mec=SURFACE, mew=1.5)
                ticks.append(y)
                labels.append(f"{label} · {h}분")
                y += 1
            y += 0.6
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(f"{title} (기준: 수작업 추세 특징 + GBM)")
        ax.set_xlabel("차이 (0보다 크면 기준보다 좋음) · 선 = 95% 신뢰구간")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ticks, labels)
    axes[0].invert_yaxis()
    fig.suptitle("같은 test 환자에서 짝지은 비교", color=INK, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fusion_figs(viz: dict, runs: list[tuple[str, dict, str, str]], out: Path) -> None:
    """ROC per horizon and risk over time for several result.json files.

    ``runs``: (label, result.json dict, method key inside it, color).
    """
    ref = runs[0][1]
    hs = list(ref["horizons"])
    fig, axes = plt.subplots(1, len(hs), figsize=(4.4 * len(hs), 4.5))
    for ax, h in zip(axes, hs, strict=True):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
        for label, res, key, color in runs:
            m = res["horizons"][h]["methods"][key]
            ax.plot(arr(m["fpr"]), arr(m["tpr"]), color=color, lw=2 if key != "hand_gbm" else 1.5,
                    label=f"{label}  {m['auroc']:.3f}")
        mo = ref["horizons"][h]["methods"]["map_only"]
        ax.plot(arr(mo["fpr"]), arr(mo["tpr"]), color=AQUA, ls="--", lw=1.5,
                label=f"현재 MAP만  {mo['auroc']:.3f}")
        ax.set_title(f"{h}분 내 저혈압 (발생률 {ref['horizons'][h]['prevalence'] * 100:.1f}%)")
        ax.set_xlabel("위양성률 (1 - 특이도)")
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=7.5, title="AUROC", title_fontsize=8)
    axes[0].set_ylabel("민감도")
    fig.suptitle("저혈압 예측 — 수치 + CARMEN 파형 (test 환자)", color=INK, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fusion_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ex = {e["caseid"]: e for e in viz["examples"]}
    cases = [e["caseid"] for e in runs[0][1].get("examples", []) if e["caseid"] in ex]
    if not cases:
        return
    fig, axes = plt.subplots(2 * len(cases), 1, figsize=(13, 3.3 * len(cases)), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1] * len(cases)})
    for i, cid in enumerate(cases):
        v = ex[cid]
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
        ax.set_title(f"케이스 {cid} ({tag}) — 세로선 = 저혈압 시작, 점 = 5분마다 예측", loc="left")
        for label, res, key, color in runs:
            if key in ("hand_gbm", "map_only"):
                continue
            e = next((x for x in res.get("examples", []) if x["caseid"] == cid), None)
            if e is not None:
                ar.plot(arr(e["risk_t"]), arr(e["risk"]["5"]), color=color, lw=1.6, marker="o",
                        ms=3, label=label)
        ar.set_ylim(0, 1)
        ar.set_ylabel("5분 내\n저혈압 위험")
        ar.set_xlim(ax.get_xlim())
        if i == 0:
            ar.legend(loc="upper right", ncol=3, fontsize=9)
            ax.legend(loc="upper right", fontsize=9)
    axes[-1, 0].set_xlabel("마취 시작 후 시간 (분)")
    fig.tight_layout()
    fig.savefig(out / "fusion_risk_timelines.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def robust_fig(cmp: dict, out: Path, models: list[tuple[str, str, str, str]],
               reference: str = "numeric_ft", conds: list[tuple[str, str]] | None = None,
               delta: bool = False, name: str = "robust.png") -> None:
    """AUROC / AUPRC under missing-waveform conditions (compare_ioh output).

    ``models``: (prefix in ``compare_ioh --scores NAME``, label, color, linestyle).
    ``delta``: plot the paired difference to ``reference`` (same bootstrap draws) instead.
    """
    conds = conds or [("full", "전체"), ("no_abp", "동맥압 파형 없음\n(ECG+PPG)"),
                      ("abp_only", "동맥압 파형만"), ("no_wave", "파형 없음")]
    hs = list(cmp["horizons"])
    fig, axes = plt.subplots(2, len(hs), figsize=(4.6 * len(hs), 7.6), squeeze=False)
    x = np.arange(len(conds))
    for col, h in enumerate(hs):
        ms = cmp["horizons"][h]["methods"]
        for row, key in enumerate(("auroc", "auprc")):
            ax = axes[row, col]
            ref = ms[reference]
            k = f"d_{key}" if delta else key
            if delta:
                ax.axhline(0, color=MUTED, lw=1.5, ls="--", label="수치만 (미세조정) = 0")
            else:
                ax.axhspan(ref[f"{key}_ci"][0], ref[f"{key}_ci"][1], color=MUTED, alpha=0.15,
                           lw=0)
                ax.axhline(ref[key], color=MUTED, lw=1.5, ls="--", label="수치만 (미세조정)")
            for i, (prefix, label, color, ls) in enumerate(models):
                xs = x + (i - (len(models) - 1) / 2) * (0.07 if ls != "none" else 0.16)
                ys = [ms[f"{prefix}:{c}"][k] for c, _ in conds]
                lo = [ms[f"{prefix}:{c}"][f"{k}_ci"][0] for c, _ in conds]
                hi = [ms[f"{prefix}:{c}"][f"{k}_ci"][1] for c, _ in conds]
                ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], color=color,
                            ls=ls, marker="o", ms=6, lw=2, capsize=0, elinewidth=1.2,
                            label=label)
            ax.set_xticks(x, [c[1] for c in conds], fontsize=8.5)
            ax.set_title(f"{h}분 내 저혈압 · {'Δ' if delta else ''}{key.upper()}")
            if col == 0:
                ax.set_ylabel(f"{key.upper()} 차이 (vs 수치만)" if delta else key.upper())
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    title = ("파형 조합별 — 수치만 모델 대비 차이 (같은 test 환자, paired bootstrap 95% CI)"
             if delta else "파형이 빠졌을 때 — 같은 test 환자 (회색 띠 = 수치만 모델의 95% CI)")
    fig.suptitle(title, color=INK, fontsize=13, y=1.01)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


TASK_NAME = {"aki": "수술 후 AKI", "mortality": "원내 사망", "icu": "ICU 입실"}


def case_fig(cmps: dict[str, dict], out: Path, methods: list[tuple[str, str, str]]) -> None:
    """AUROC / AUPRC with 95 % CI per case-level task (one compare_ioh output per task)."""
    tasks = list(cmps)
    firsts = [next(iter(cmps[t]["horizons"].values()))["methods"] for t in tasks]
    # a "<model>:<condition>" group that scores identically in every task gets one row
    groups: dict[str, list[str]] = {}
    for m, _, _ in methods:
        if ":" in m and any(m in ms for ms in firsts):
            groups.setdefault(m.split(":")[0], []).append(m)
    rows = []
    for m, lab, c in methods:
        g = groups.get(m.split(":")[0], []) if ":" in m else []
        if len(g) > 1 and all(len({(ms[x]["auroc"], ms[x]["auprc"]) for x in g if x in ms}) <= 1
                              for ms in firsts):
            if m != g[0]:
                continue
            lab = lab.split(" (")[0] + " — 파형 조건과 무관하게 동일"
        rows.append((m, lab, c))
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.4 * len(tasks), 7.2), squeeze=False)
    for col, task in enumerate(tasks):
        r = next(iter(cmps[task]["horizons"].values()))
        ms = r["methods"]
        for row, key in enumerate(("auroc", "auprc")):
            ax = axes[row, col]
            present = [(m, lab, c) for m, lab, c in rows if m in ms]
            y = np.arange(len(present))
            for i, (m, _, color) in enumerate(present):
                v = ms[m]
                ax.plot(v[f"{key}_ci"], [i, i], color=color, lw=2, solid_capstyle="round")
                ax.plot([v[key]], [i], "o", color=color, ms=7, mec=SURFACE, mew=1.5)
                ax.text(v[f"{key}_ci"][1], i, f"  {v[key]:.3f}", va="center", fontsize=8,
                        color=INK2)
            if key == "auprc":
                ax.axvline(r["prevalence"], color=MUTED, lw=1, ls=":")
            ax.set_yticks(y, [lab for _, lab, _ in present] if col == 0 else [""] * len(y))
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            name = TASK_NAME.get(task, task)
            ax.set_title(f"{name} · {key.upper()}  (n={r['n']}, "
                         f"발생률 {r['prevalence'] * 100:.1f}%)", fontsize=10)
    fig.suptitle("케이스 단위 결과 예측 — test 환자 (선 = 95% CI, 점선 = 발생률)", color=INK,
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "case_tasks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def task_delta_fig(specs: list[tuple[str, dict, str, str, str]], out: Path,
                   conds: list[tuple[str, str]], reference: str = "numeric_ft") -> None:
    """Paired difference to ``reference`` per task, waveform condition and fusion design.

    ``specs``: (task label, compare_ioh output, horizon key, cross-attn prefix, residual prefix).
    """
    fig, axes = plt.subplots(2, len(specs), figsize=(3.9 * len(specs), 7.0), squeeze=False)
    x = np.arange(len(conds))
    designs = [("cross-attention 융합", BLUE, -0.13), ("잔차 융합 (수치 고정)", ORANGE, 0.13)]
    for col, (label, cmp, h, cross, res) in enumerate(specs):
        r = cmp["horizons"][h]
        ms = r["methods"]
        n_pos = round(r["n"] * r["prevalence"])
        for row, key in enumerate(("auroc", "auprc")):
            ax = axes[row, col]
            ax.axhline(0, color=MUTED, lw=1.5, ls="--")
            for (name, color, dx), prefix in zip(designs, (cross, res), strict=True):
                v = [ms[f"{prefix}:{c}"] for c, _ in conds]
                ys = np.array([m[f"d_{key}"] for m in v])
                lo = ys - np.array([m[f"d_{key}_ci"][0] for m in v])
                hi = np.array([m[f"d_{key}_ci"][1] for m in v]) - ys
                ax.errorbar(x + dx, ys, yerr=[lo, hi], fmt="o", color=color, ms=6, capsize=3,
                            lw=1.6, mec=SURFACE, mew=1.2, label=name if col == 0 else None)
            ax.set_xticks(x, [lab for _, lab in conds], fontsize=8)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{label} · Δ{key.upper()}\n(test n={r['n']}, 양성 {n_pos})",
                         fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{key.upper()} 차이 (vs 수치만)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("과제별 — 파형을 더했을 때 수치만 모델 대비 차이 "
                 "(같은 test 환자, paired bootstrap 95% CI)", color=INK, fontsize=13, y=1.01)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "task_delta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


ARR_CONDS = [("eap", "ABP\nECG\nPPG"), ("ep", "ECG\nPPG"), ("e", "ECG"), ("ap", "ABP\nPPG"),
             ("p", "PPG"), ("a", "ABP"), ("none", "파형\n없음")]
ARR_NAME = {"NSR": "정상 동리듬 (1분 전체)", "AF": "심방세동/조동", "VA": "심실성 부정맥",
            "BradyCond": "서맥·전도장애"}


def arr_figs(cmp: dict, out: Path, models: list[tuple[str, str, str]],
             reference: str = "numeric_ft", pair: str = "wave",
             ref_label: str = "수치만 (미세조정)", name: str = "arr") -> None:
    """Arrhythmia (pooled CV, ``cv_folds.py pool``): per class, each waveform condition.

    ``<name>_abs.png``: AUROC / AUPRC per model and condition, numeric-only as a band.
    ``<name>_delta.png``: models with numeric inputs minus the waveform-only model under the same
    condition (``d_*_vs_<pair>``, paired patient bootstrap).
    """
    labels = list(cmp["horizons"])
    first = cmp["horizons"][labels[0]]["methods"]
    conds = [(c, lab) for c, lab in ARR_CONDS if any(f"{m[0]}:{c}" in first for m in models)]
    x = np.arange(len(conds))
    for delta in (False, True):
        ms_draw = [m for m in models if not (delta and m[0] == pair)]
        fig, axes = plt.subplots(2, len(labels), figsize=(4.3 * len(labels), 7.6), squeeze=False)
        for col, h in enumerate(labels):
            r = cmp["horizons"][h]
            ms = r["methods"]
            for row, key in enumerate(("auroc", "auprc")):
                ax = axes[row, col]
                if delta:
                    ax.axhline(0, color=MUTED, lw=1.5, ls="--", label="파형만 = 0")
                else:
                    ref = ms[reference]
                    ax.axhspan(*ref[f"{key}_ci"], color=ORANGE, alpha=0.12, lw=0)
                    ax.axhline(ref[key], color=ORANGE, lw=1.5, ls="--", label=ref_label)
                    if key == "auprc":
                        ax.axhline(r["prevalence"], color=MUTED, lw=1, ls=":", label="발생률")
                width = 0.22
                for i, (prefix, label, color) in enumerate(ms_draw):
                    dx = (i - (len(ms_draw) - 1) / 2) * width
                    k = f"d_{key}_vs_{pair}" if delta else key
                    pts = [(j, ms[f"{prefix}:{c}"]) for j, (c, _) in enumerate(conds)
                           if f"{prefix}:{c}" in ms and k in ms[f"{prefix}:{c}"]]
                    if not pts:
                        continue
                    xs = np.array([j for j, _ in pts]) + dx
                    ys = np.array([v[k] for _, v in pts])
                    ci = np.array([v[f"{k}_ci" if not delta else f"d_{key}_ci_vs_{pair}"]
                                   for _, v in pts])
                    ax.errorbar(xs, ys, yerr=[ys - ci[:, 0], ci[:, 1] - ys], fmt="o", ms=5.5,
                                color=color, capsize=2.5, lw=1.4, mec=SURFACE, mew=1.0,
                                label=label)
                ax.set_xticks(x, [lab for _, lab in conds], fontsize=7.5)
                ax.grid(axis="x", visible=False)
                pos = r.get("n_cases_pos")
                ax.set_title(f"{ARR_NAME.get(h, h)} · {'Δ' if delta else ''}{key.upper()}\n"
                             f"(양성 {r['prevalence'] * r['n']:.0f}분"
                             + (f", {pos}건" if pos else "") + ")", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{key.upper()} 차이 (vs 파형만, 같은 조건)" if delta
                                  else key.upper())
        handles, labs = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labs, loc="lower center", ncol=len(labs), frameon=False,
                   bbox_to_anchor=(0.5, -0.03))
        title = ("부정맥 — 수치를 더했을 때 파형만 모델 대비 차이 "
                 "(5겹 CV, paired patient bootstrap 95% CI)" if delta else
                 "부정맥 — 파형 조건별 성능 (5겹 CV 통합 test, 95% CI; 주황 띠 = 수치만)")
        fig.suptitle(title, color=INK, fontsize=13, y=1.01)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(out / f"{name}_{'delta' if delta else 'abs'}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--finetune", help="finetune_ioh.py result.json (pre-trained encoder)")
    ap.add_argument("--scratch", help="finetune_ioh.py result.json (--scratch)")
    ap.add_argument("--compare", help="compare_ioh.py output (paired differences)")
    ap.add_argument("--fusion", nargs=3, action="append", default=[],
                    metavar=("LABEL", "RESULT_JSON", "KEY"),
                    help="result.json files to overlay (finetune_ioh / finetune_fusion)")
    ap.add_argument("--paired-out", default="ft_paired.png")
    ap.add_argument("--robust", help="compare_ioh output with <model>:<condition> scores")
    ap.add_argument("--robust-extra", help="second compare_ioh output merged into --robust")
    ap.add_argument("--robust-models", nargs="+", default=None,
                    help="PREFIX=LABEL entries for --robust (default: the dropout comparison)")
    ap.add_argument("--robust-conds", nargs="+", default=None,
                    help="COND=LABEL entries (\\n in LABEL = line break)")
    ap.add_argument("--robust-delta", action="store_true",
                    help="also draw robust_delta.png (paired difference to numeric_ft)")
    ap.add_argument("--case", nargs=2, action="append", default=[], metavar=("TASK", "JSON"),
                    help="compare_ioh output of a case-level task (repeatable)")
    ap.add_argument("--task-delta", nargs=5, action="append", default=[],
                    metavar=("LABEL", "JSON", "HORIZON", "CROSS", "RESIDUAL"),
                    help="compare_ioh output per task for task_delta.png (repeatable)")
    ap.add_argument("--arr", action="store_true",
                    help="JSON is a pooled arrhythmia CV comparison: draw arr_abs / arr_delta")
    ap.add_argument("--arr-gbm", action="store_true",
                    help="JSON is the pooled wave_features.py comparison: draw arr_wf_abs / "
                         "arr_wf_delta")
    ap.add_argument("--compare-methods", nargs="+", default=[],
                    help="NAME=LABEL entries of --compare to draw, in order")
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup()
    if args.arr_gbm:
        arr_figs(d, out, [("wf", "파형 특징만 (GBM)", BLUE),
                          ("wfnum", "파형 특징 + 수치 (GBM)", "#1d4f91")],
                 reference="hand_gbm", pair="wf", ref_label="수치만 (GBM)", name="arr_wf")
        print(f"figures in {out}")
        return
    if args.arr:
        arr_figs(d, out, [("wave", "파형만 (CARMEN)", BLUE),
                          ("fusion", "파형 + 수치 (cross-attention)", "#6250d6"),
                          ("gatef", "파형 + 수치 (잔차, 수치 고정)", "#1d4f91")])
        print(f"figures in {out}")
        return
    if args.task_delta:
        task_delta_fig([(lab.replace("\\n", "\n"), json.loads(Path(pth).read_text()), h, c, r)
                        for lab, pth, h, c, r in args.task_delta], out,
                       [("eap", "ABP\nECG\nPPG"), ("a", "ABP"), ("ep", "ECG\nPPG"),
                        ("none", "파형\n없음")])
    if args.case:
        case_fig({t: json.loads(Path(pth).read_text()) for t, pth in args.case}, out, [
            ("hypo_minutes", "저혈압 누적 시간 (단일 지표)", AQUA),
            ("hand_gbm", "수작업 특징 + GBM", MUTED),
            ("numeric_ft", "수치만 (미세조정)", ORANGE),
            ("fusion:eap", "cross-attn 융합 (ABP+ECG+PPG)", BLUE),
            ("fusion:ep", "cross-attn 융합 (동맥관 없음)", "#6250d6"),
            ("fusion:none", "cross-attn 융합, 파형 없음", INK2),
            ("gatef:eap", "잔차 융합 (ABP+ECG+PPG)", "#1d4f91"),
            ("gatef:ep", "잔차 융합 (동맥관 없음)", "#9b8ce8"),
            ("gatef:none", "잔차 융합, 파형 없음", "#5c5b57"),
        ])
    if args.case or args.task_delta:
        print(f"figures in {out}")
        return
    if args.robust and args.robust_models:
        cols = [BLUE, "#6250d6", ORANGE, MUTED]
        # conditions are categories: markers only, no connecting lines
        models = [(kv.split("=", 1)[0], kv.split("=", 1)[1], cols[i % len(cols)], "none")
                  for i, kv in enumerate(args.robust_models)]
        conds = [(c, lab.replace("\\n", "\n")) for c, lab in
                 (kv.split("=", 1) for kv in args.robust_conds)] if args.robust_conds else None
        cmp = json.loads(Path(args.robust).read_text())
        robust_fig(cmp, out, models, conds=conds)
        if args.robust_delta:
            robust_fig(cmp, out, models, conds=conds, delta=True, name="robust_delta.png")
        print(f"figures in {out}")
        return
    if args.robust:
        cmp = json.loads(Path(args.robust).read_text())
        conds = None
        if args.robust_extra:  # e.g. re-encoded no-arterial-line scores: "<run>:noabp_true"
            extra = json.loads(Path(args.robust_extra).read_text())
            rename = {"fu_fusion": "fu_fusion_eval", "fu_late": "fu_late_eval"}
            for h, r in extra["horizons"].items():
                for k, v in r["methods"].items():
                    if ":" in k:
                        run, c = k.split(":", 1)
                        cmp["horizons"][h]["methods"][f"{rename.get(run, run)}:{c}"] = v
            conds = [("full", "전체 (ABP+ECG+PPG)"),
                     ("noabp_true", "동맥관 없음\n(ECG+PPG 재인코딩)"), ("no_wave", "파형 없음")]
        robust_fig(cmp, out, [
            ("fu_fusion_md", "cross-attention + modality dropout", BLUE, "-"),
            ("fu_late_md", "단순 결합 + modality dropout", ORANGE, "-"),
            ("fu_fusion_eval", "cross-attention (dropout 없음)", BLUE, ":"),
            ("fu_late_eval", "단순 결합 (dropout 없음)", ORANGE, ":"),
        ], conds=conds)
        print(f"figures in {out}")
        return
    if args.fusion:
        colors = [BLUE, "#6250d6", ORANGE, MUTED, "#e87ba4"]
        runs = [(lab, json.loads(Path(pth).read_text()), key, colors[i % len(colors)])
                for i, (lab, pth, key) in enumerate(args.fusion)]
        fusion_figs(d, runs, out)
        if args.compare:
            cols = [BLUE, "#6250d6", ORANGE, MUTED, AQUA]
            ms = [(kv.split("=", 1)[0], kv.split("=", 1)[1], cols[i % len(cols)])
                  for i, kv in enumerate(args.compare_methods)]
            paired_fig(json.loads(Path(args.compare).read_text()), out, ms, args.paired_out)
        print(f"figures in {out}")
        return
    if args.finetune:
        ft = json.loads(Path(args.finetune).read_text())
        sc = json.loads(Path(args.scratch).read_text()) if args.scratch else None
        finetune_figs(d, ft, sc, out)
        if args.compare:
            colors = [BLUE, "#6250d6", MUTED, AQUA]
            ms = [(kv.split("=", 1)[0], kv.split("=", 1)[1], colors[i % len(colors)])
                  for i, kv in enumerate(args.compare_methods)]
            paired_fig(json.loads(Path(args.compare).read_text()), out, ms)
        print(f"figures in {out}")
        return
    forecast_examples(d, out)
    risk_timelines(d, out)
    roc(d, out)
    lead_error(d, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
