"""
기업 고유 리스크 vs 시장(금리) 요인 분리 분석
================================================

목적
----
"스프레드가 벌어진 게 금리 때문이 아니라 이 기업만의 문제였다"를
숫자와 그래프로 보여준다.

입력
----
    target_df    : 분석대상 기업의 일별 데이터 (BAS_DD, CLSPRC_YD 포함)
    peer_df      : 동일등급·유사만기 대조군(여러 종목)의 일별 데이터, 같은 컬럼 구조
                    (여러 종목이면 ISU_NM으로 구분되어 있어야 날짜별 평균을 낼 수 있음)
    baseline_start/end : "평온했던 시기" 구간 — 이 구간의 괴리 변동폭을 기준(표준편차)으로 삼는다

핵심 산출물
-----------
    1) divergence 시계열 (target − peer 평균)
    2) baseline 대비 z-score (오늘의 괴리가 평상시 대비 몇 표준편차인지)
    3) target vs peer 수익률을 겹친 라인차트 (PNG)
    4) "z-score가 2를 처음 넘은 날짜" — 이걸 실제 신용등급 하향일과 비교하면 선행일수(Lead Time)가 나옴

사용법
------
    pip install pandas matplotlib --break-system-packages

    python analyze_divergence.py \
        --target jrglobal_validation.csv \
        --peers peer_bonds.csv \
        --baseline-start 2026-01-05 --baseline-end 2026-02-27 \
        --rating-change-date 2026-04-20
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 파일로 저장되게
import matplotlib.pyplot as plt


def load_daily_series(path: str, value_col: str = "CLSPRC_YD") -> pd.DataFrame:
    """CSV를 읽어 날짜별 대표값(여러 종목이면 평균)으로 축약한다."""
    df = pd.read_csv(path)
    df["BAS_DD"] = pd.to_datetime(df["BAS_DD"], format="%Y%m%d", errors="coerce")
    df["BAS_DD"] = df["BAS_DD"].fillna(pd.to_datetime(df["BAS_DD"], errors="coerce"))
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    daily = df.groupby("BAS_DD")[value_col].mean().reset_index()
    daily = daily.rename(columns={value_col: "yield"})
    return daily.sort_values("BAS_DD")


def compute_divergence(target: pd.DataFrame, peer: pd.DataFrame) -> pd.DataFrame:
    """날짜를 맞춰 target − peer 괴리를 계산한다."""
    merged = pd.merge(target, peer, on="BAS_DD", suffixes=("_target", "_peer"))
    merged["divergence"] = merged["yield_target"] - merged["yield_peer"]
    return merged


def compute_zscore(merged: pd.DataFrame, baseline_start: str, baseline_end: str) -> pd.DataFrame:
    """베이스라인 구간의 divergence 평균·표준편차를 기준으로 z-score를 계산한다."""
    baseline = merged[(merged["BAS_DD"] >= baseline_start) & (merged["BAS_DD"] <= baseline_end)]
    if baseline.empty or baseline["divergence"].std() == 0 or pd.isna(baseline["divergence"].std()):
        raise ValueError(
            "베이스라인 구간의 데이터가 없거나 변동폭이 0입니다. "
            "baseline-start/end를 실제로 데이터가 있는 안정적인 기간으로 지정했는지 확인하세요."
        )
    base_mean = baseline["divergence"].mean()
    base_std = baseline["divergence"].std()
    merged["zscore"] = (merged["divergence"] - base_mean) / base_std
    print(f"[베이스라인] {baseline_start} ~ {baseline_end}")
    print(f"  괴리 평균: {base_mean:.4f}%p, 표준편차: {base_std:.4f}%p")
    return merged


def find_first_breakout(merged: pd.DataFrame, z_threshold: float = 2.0) -> pd.Timestamp:
    """z-score가 임계값을 처음 넘은 날짜를 찾는다."""
    breakout = merged[merged["zscore"].abs() >= z_threshold]
    if breakout.empty:
        return None
    return breakout.iloc[0]["BAS_DD"]


def plot_comparison(merged: pd.DataFrame, out_path: str, rating_change_date: str = None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(merged["BAS_DD"], merged["yield_target"], label="분석대상 기업 수익률", color="crimson")
    axes[0].plot(merged["BAS_DD"], merged["yield_peer"], label="동일등급 대조군 평균 수익률", color="steelblue")
    axes[0].set_ylabel("수익률 (%)")
    axes[0].legend()
    axes[0].set_title("분석대상 vs 대조군 수익률 추이")

    axes[1].plot(merged["BAS_DD"], merged["zscore"], color="darkorange")
    axes[1].axhline(2, color="gray", linestyle="--", linewidth=1)
    axes[1].axhline(-2, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("괴리 z-score")
    axes[1].set_title("베이스라인 대비 괴리 정도 (±2 = 이례적 구간)")

    if rating_change_date:
        rcd = pd.to_datetime(rating_change_date)
        for ax in axes:
            ax.axvline(rcd, color="black", linestyle=":", linewidth=1.5)
        axes[0].text(rcd, axes[0].get_ylim()[1], " 신용등급 하향일", rotation=90, va="top", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[저장 완료] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="분석대상 기업 CSV (BAS_DD, CLSPRC_YD 포함)")
    parser.add_argument("--peers", required=True, help="대조군 CSV (BAS_DD, CLSPRC_YD 포함, 여러 종목 가능)")
    parser.add_argument("--baseline-start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--baseline-end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--rating-change-date", default=None, help="YYYY-MM-DD (그래프에 표시용, 선택)")
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--out", default="divergence_analysis.png")
    args = parser.parse_args()

    target = load_daily_series(args.target)
    peer = load_daily_series(args.peers)

    merged = compute_divergence(target, peer)
    merged = compute_zscore(merged, args.baseline_start, args.baseline_end)

    breakout_date = find_first_breakout(merged, args.z_threshold)
    if breakout_date is not None:
        print(f"\n[핵심 결과] 괴리가 |z|≥{args.z_threshold}를 처음 넘은 날짜: {breakout_date.date()}")
        if args.rating_change_date:
            lead_days = (pd.to_datetime(args.rating_change_date) - breakout_date).days
            print(f"신용등급 하향일({args.rating_change_date}) 대비 {lead_days}일 선행")
    else:
        print(f"\n분석 기간 내 |z|≥{args.z_threshold}를 넘은 날이 없습니다. "
              f"임계값을 낮추거나(--z-threshold 1.5) 베이스라인 구간을 재검토하세요.")

    plot_comparison(merged, args.out, args.rating_change_date)
    merged.to_csv(args.out.replace(".png", ".csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
