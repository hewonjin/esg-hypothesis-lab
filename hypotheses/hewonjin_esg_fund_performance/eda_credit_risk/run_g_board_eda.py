"""
G(지배구조)등급 - 이사회 구성(사외이사 비율) 상관관계 분석
==================================================================
collect_board_composition.py로 로컬 수집한 board_composition_RAW.csv를
merged_esg_credit_REAL_with_sector.csv(E·S 분석과 동일 표본)와 corp_code로
합쳐서 다음을 검증한다.

  1. 지배구조(G)등급이 실제로 이사회 독립성(사외이사 비율)을 반영하는가?
     - G등급 ↔ outside_ratio Spearman 상관
  2. 이사회 독립성(사외이사 비율) 자체가 신용위험과 상관이 있는가?
     - outside_ratio ↔ risk_score / distress, 규모 통제 편상관 + 로지스틱회귀
  3. (참고) 이사회 규모(drctr_co) 자체도 규모 대리변수일 수 있어 함께 확인

입력:
  eda_credit_risk/merged_esg_credit_REAL_with_sector.csv
  eda_credit_risk/board_composition_RAW.csv   (로컬에서 collect_board_composition.py로 생성)
출력:
  eda_credit_risk/chart5_G_board_REAL.png
  콘솔에 통계 결과 출력
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
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


def partial_corr(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    m = pd.read_csv(MERGED_PATH)
    b = pd.read_csv(BOARD_PATH)
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

    # 2) 사외이사비율 vs 신용위험
    rho2, p2 = spearmanr(df['outside_ratio'], df['risk_score'])
    print(f"[2] 사외이사비율 ↔ risk_score Spearman: rho={rho2:.4f}, p={p2:.4g}")
    pr = partial_corr(df['outside_ratio'], df['risk_score'], df['log_capital'])
    print(f"    규모 통제 편상관: r={pr:.4f}")

    X = sm.add_constant(df[['outside_ratio', 'log_capital']])
    logit = sm.Logit(df['distress'], X).fit(disp=0)
    print(logit.summary())

    # 3) G등급 vs 이사회 규모(참고)
    rho3, p3 = spearmanr(df['지배구조_ord'], df['drctr_co'])
    print(f"\n[3] G등급 ↔ 이사회 규모(drctr_co) Spearman(참고): rho={rho3:.4f}, p={p3:.4g}")

    # ---- chart ----
    fp = fm.FontProperties(fname=FONT_PATH)
    fp_bold = fm.FontProperties(fname=FONT_PATH, weight='bold')
    order = ['D', 'C', 'B', 'B+', 'A', 'A+']
    means = df.groupby('지배구조')['outside_ratio'].mean().reindex(order) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(means.index, means.values, color='#2a78d6')
    ax.set_ylabel('평균 사외이사 비율 (%)', fontproperties=fp)
    ax.set_xlabel('지배구조(G)등급', fontproperties=fp)
    ax.set_title('G등급별 평균 사외이사 비율', fontproperties=fp_bold, fontsize=13)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart5_G_board_REAL.png', dpi=150, bbox_inches='tight')
    print(f"\n차트 저장: {OUT_DIR}/chart5_G_board_REAL.png")


if __name__ == '__main__':
    main()
