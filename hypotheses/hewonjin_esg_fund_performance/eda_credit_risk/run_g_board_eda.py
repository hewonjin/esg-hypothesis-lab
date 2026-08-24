"""
G(지배구조)등급 - 이사회 구성(사외이사 비율) 상관관계 분석
==================================================================
collect_6_board.py로 로컬 수집한 board_composition_RAW.csv를
merged_esg_credit_REAL_with_sector.csv(E·S 분석과 동일 표본)와 corp_code로
합쳐서 다음을 검증한다.

  1. 지배구조(G)등급이 실제로 이사회 독립성(사외이사 비율)을 반영하는가?
     - G등급 ↔ outside_ratio Spearman 상관 (+ 규모 통제 편상관)
  2. 이사회 독립성(사외이사 비율) 자체가 신용위험과 상관이 있는가?
     - outside_ratio ↔ risk_score / distress, 규모 통제 편상관 + 로지스틱회귀
     - 정상군 vs 위험징후 이상군 사외이사비율 비교(Mann-Whitney)
  3. (참고) 이사회 규모(drctr_co), 기업 규모(log_capital)와 G등급의 관계

데이터 정제: outside_ratio가 [0,1] 범위를 벗어나는 행(사외이사수>이사총수, 공시 오류로 추정)은
  분석에서 제외하고 콘솔에 플래그만 남긴다 — 조용히 버리지 않는다.

입력:
  eda_credit_risk/merged_esg_credit_REAL_with_sector.csv
  eda_credit_risk/board_composition_RAW.csv   (로컬에서 collect_6_board.py로 생성)
출력:
  eda_credit_risk/chart5_G_board_REAL.png       (G등급별 평균 사외이사비율)
  eda_credit_risk/chart6_board_by_distress_REAL.png  (신용상태별 사외이사비율)
  eda_credit_risk/merged_esg_credit_board_REAL.csv    (병합된 최종 분석 표본)
  콘솔에 통계 결과 출력
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, mannwhitneyu
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

MERGED_PATH = 'eda_credit_risk/merged_esg_credit_REAL_with_sector.csv'
BOARD_PATH = 'eda_credit_risk/board_composition_RAW.csv'
OUT_DIR = 'eda_credit_risk'
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

GRADE_ORDER = {'D': 1, 'C': 2, 'B': 3, 'B+': 4, 'A': 5, 'A+': 6}
GRADE_SEQ = ['D', 'C', 'B', 'B+', 'A', 'A+']


def partial_corr(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    m = pd.read_csv(MERGED_PATH)
    b = pd.read_csv(BOARD_PATH)

    bad = b[(b['outside_ratio'] > 1) | (b['outside_ratio'] < 0)]
    if len(bad):
        print("=== 이상치 플래그 (분석에서 제외) ===")
        print(bad[['corp_code', 'corp_name', 'drctr_co', 'otcmp_drctr_co', 'outside_ratio', 'rcept_no']])
        print()
    b = b[(b['outside_ratio'] <= 1) & (b['outside_ratio'] >= 0)]

    df = m.merge(b[['corp_code', 'drctr_co', 'otcmp_drctr_co', 'outside_ratio']],
                 on='corp_code', how='inner')
    print(f"병합 결과: ESG/신용 표본 {len(m)}개사 중 이사회 데이터 매칭 {len(df)}개사")

    df = df[df['지배구조'] != '등급없음'].copy()
    df['지배구조_ord'] = df['지배구조'].map(GRADE_ORDER)
    df['distress'] = (df['stage'] >= 1).astype(int)
    df['log_capital'] = np.log(df['capital_total'].clip(lower=1))
    df = df.dropna(subset=['outside_ratio', '지배구조_ord', 'log_capital'])
    print(f"결측 제외 후 최종 분석 표본: {len(df)}개사\n")

    # 1) G등급 vs 사외이사비율
    rho, p = spearmanr(df['지배구조_ord'], df['outside_ratio'])
    print(f"[1] G등급 ↔ 사외이사비율 Spearman: rho={rho:.4f}, p={p:.4g}")
    pr1 = partial_corr(df['지배구조_ord'], df['outside_ratio'], df['log_capital'])
    print(f"    규모 통제 편상관: r={pr1:.4f}")

    # 2) 사외이사비율 vs 신용위험
    rho2, p2 = spearmanr(df['outside_ratio'], df['risk_score'])
    print(f"\n[2] 사외이사비율 ↔ risk_score Spearman: rho={rho2:.4f}, p={p2:.4g}")
    pr2 = partial_corr(df['outside_ratio'], df['risk_score'], df['log_capital'])
    print(f"    규모 통제 편상관: r={pr2:.4f}")

    X = sm.add_constant(df[['outside_ratio', 'log_capital']])
    logit = sm.Logit(df['distress'], X).fit(disp=0)
    print(logit.summary())

    # 3) 참고 지표
    rho3, p3 = spearmanr(df['지배구조_ord'], df['drctr_co'])
    print(f"\n[3] G등급 ↔ 이사회 규모(drctr_co) 참고: rho={rho3:.4f}, p={p3:.4g}")
    rho4, p4 = spearmanr(df['지배구조_ord'], df['log_capital'])
    print(f"[4] G등급 ↔ 기업 규모 참고: rho={rho4:.4f}, p={p4:.4g}")

    # 5) 정상군 vs 위험군 비교
    g_normal = df[df['distress'] == 0]['outside_ratio']
    g_distress = df[df['distress'] == 1]['outside_ratio']
    u, pu = mannwhitneyu(g_normal, g_distress)
    print(f"\n[5] 정상군(n={len(g_normal)}) 평균={g_normal.mean():.3f} vs "
          f"위험군(n={len(g_distress)}) 평균={g_distress.mean():.3f}, Mann-Whitney p={pu:.4g}")

    df.to_csv(f'{OUT_DIR}/merged_esg_credit_board_REAL.csv', index=False)

    # ---- charts ----
    fp = fm.FontProperties(fname=FONT_PATH)
    fp_bold = fm.FontProperties(fname=FONT_PATH, weight='bold')

    means = df.groupby('지배구조')['outside_ratio'].mean().reindex(GRADE_SEQ) * 100
    ns = df.groupby('지배구조')['outside_ratio'].count().reindex(GRADE_SEQ)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(means.index, means.values, color='#2a78d6', width=0.6)
    for i, (v, n) in enumerate(zip(means.values, ns.values)):
        ax.text(i, v + 1.5, f'{v:.1f}%\n(n={n})', ha='center', fontproperties=fp, fontsize=9)
    ax.set_ylabel('평균 사외이사 비율 (%)', fontproperties=fp)
    ax.set_xlabel('지배구조(G)등급', fontproperties=fp)
    ax.set_title(f'G등급별 평균 사외이사 비율 (n={len(df)})', fontproperties=fp_bold, fontsize=13)
    ax.set_ylim(0, max(means.values) * 1.25)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart5_G_board_REAL.png', dpi=150, bbox_inches='tight')
    plt.close()

    grp = df.copy()
    grp['group'] = grp['distress'].map({0: '정상', 1: '위험징후 이상'})
    means2 = grp.groupby('group')['outside_ratio'].mean().reindex(['정상', '위험징후 이상']) * 100
    ns2 = grp.groupby('group')['outside_ratio'].count().reindex(['정상', '위험징후 이상'])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(means2.index, means2.values, color=['#2a78d6', '#eb6834'], width=0.5)
    for i, (v, n) in enumerate(zip(means2.values, ns2.values)):
        ax.text(i, v + 1, f'{v:.1f}%\n(n={n})', ha='center', fontproperties=fp, fontsize=10)
    ax.set_ylabel('평균 사외이사 비율 (%)', fontproperties=fp)
    ax.set_title('신용상태별 평균 사외이사 비율', fontproperties=fp_bold, fontsize=13)
    ax.set_ylim(0, max(means2.values) * 1.3)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart6_board_by_distress_REAL.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n차트 저장: {OUT_DIR}/chart5_G_board_REAL.png, {OUT_DIR}/chart6_board_by_distress_REAL.png")


if __name__ == '__main__':
    main()
