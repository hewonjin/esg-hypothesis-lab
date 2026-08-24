"""
ESG등급 - 신용위험 재무적 타당성 EDA 파이프라인
=================================================
실제 데이터 준비 완료 후 곧바로 재실행하기 위해 만들어둔 스크립트.

입력:
  - CREDIT_PANEL_PATH: 혜원님이 LIVE_COLLECT=True로 로컬에서 재생성한
    outputs/credit_stage_panel.csv (실제 DART 기반, 합성 아님)
  - KCGS_PATH: 공용 드라이브 '0. 공통' 폴더의 KCGS 평가등급 xlsx (이미 실제 데이터)

사용법:
  python3 run_esg_credit_eda.py <credit_stage_panel.csv 경로>

주의:
  - credit_stage_panel.csv가 진짜 LIVE_COLLECT=True 산출물인지 반드시 확인 후 실행할 것.
  - HD현대미포·HD현대인프라코어처럼 사명변경/계열편입으로 corp_code가 바뀐 대기업이
    '국세청 폐업(stage=3)'으로 잘못 잡히는 사례가 있었는지 재확인 필요
    (실제 데이터에서도 재현되면 개별 DART 조회로 검증할 것).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

KCGS_PATH = '/home/claude/data/KCGS_평가등급_2026.xlsx'
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
OUT_DIR = '/home/claude/analysis'

GRADE_ORDER = {'D': 1, 'C': 2, 'B': 3, 'B+': 4, 'A': 5, 'A+': 6}
GRADE_SEQ = ['D', 'C', 'B', 'B+', 'A', 'A+']
STAGE_LABELS = {0: '정상', 1: '위험징후', 2: '신용등급하락', 3: '부도·회생·파산'}
STATUS_COLORS = {0: '#0ca30c', 1: '#fab219', 2: '#ec835a', 3: '#d03b3b'}
BLUE_STEPS = {'D': '#86b6ef', 'C': '#6da7ec', 'B': '#5598e7',
              'B+': '#3987e5', 'A': '#2a78d6', 'A+': '#1c5cab'}


def load_and_merge(credit_panel_path):
    panel = pd.read_csv(credit_panel_path, dtype=str)
    panel['stock_code'] = panel['stock_code'].str.zfill(6)
    panel['stage'] = panel['stage'].astype(int)
    panel['risk_score'] = panel['risk_score'].astype(float)
    if 'capital_total' in panel.columns:
        panel['capital_total'] = pd.to_numeric(panel['capital_total'], errors='coerce')
    if '부채비율' in panel.columns:
        panel['부채비율'] = pd.to_numeric(panel['부채비율'], errors='coerce')

    kcgs = pd.read_excel(KCGS_PATH)
    kcgs['기업코드'] = kcgs['기업코드'].astype(str).str.zfill(6)

    merged = panel.merge(kcgs, left_on='stock_code', right_on='기업코드', how='inner')
    unmatched = kcgs[~kcgs['기업코드'].isin(panel['stock_code'])]
    print(f"panel n={len(panel)}, kcgs n={len(kcgs)}, merged n={len(merged)}, "
          f"unmatched(kcgs)={len(unmatched)}")
    return merged


def run_stats(merged):
    sub = merged[merged['ESG등급'] != '등급없음'].copy()
    for col in ['ESG등급', '환경', '사회', '지배구조']:
        sub[col + '_ord'] = sub[col].map(GRADE_ORDER)

    results = {}
    for col in ['ESG등급_ord', '환경_ord', '사회_ord']:
        r, p = stats.spearmanr(sub[col], sub['risk_score'])
        results[f'{col}_vs_risk_score'] = (r, p, sub[col].notna().sum())

    gov = sub[sub['지배구조'] != '등급없음'].copy()
    r, p = stats.spearmanr(gov['지배구조_ord'], gov['risk_score'])
    results['지배구조_ord_vs_risk_score'] = (r, p, len(gov))

    groups = [g['risk_score'].values for _, g in sub.groupby('ESG등급')]
    h, p = stats.kruskal(*groups)
    results['kruskal_risk_by_grade'] = (h, p, len(sub))

    sub['distress'] = (sub['stage'] >= 1).astype(int)
    sub['tier'] = np.where(sub['ESG등급_ord'] >= 5, '우량군(A이상)', '비우량군(B+이하)')
    ct = pd.crosstab(sub['tier'], sub['distress'])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    results['chi2_tier_distress'] = (chi2, p, ct)

    if 'capital_total' in sub.columns:
        size = sub.dropna(subset=['capital_total']).copy()
        size['log_capital'] = np.log(size['capital_total'].clip(lower=1))
        r, p = stats.spearmanr(size['ESG등급_ord'], size['log_capital'])
        results['esg_vs_size'] = (r, p, len(size))

    severe = sub[sub['stage'].isin([2, 3])]
    results['severe_events'] = severe[['기업명', 'ESG등급', 'stage', 'stage_reason']] \
        if 'stage_reason' in severe.columns else severe[['기업명', 'ESG등급', 'stage']]

    return sub, results


def make_charts(sub, out_dir=OUT_DIR):
    kfont = fm.FontProperties(fname=FONT_PATH)
    kfont_b = fm.FontProperties(fname=FONT_PATH, weight='bold')

    # Chart 1: 100% stacked bar — stage share by ESG grade
    ct = pd.crosstab(sub['ESG등급'], sub['stage'], normalize='index').reindex(GRADE_SEQ) * 100
    counts = sub['ESG등급'].value_counts().reindex(GRADE_SEQ)

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    left = np.zeros(len(GRADE_SEQ))
    for stg in [0, 1, 2, 3]:
        vals = ct[stg].values if stg in ct.columns else np.zeros(len(GRADE_SEQ))
        ax.barh(GRADE_SEQ, vals, left=left, color=STATUS_COLORS[stg],
                 label=STAGE_LABELS[stg], height=0.62)
        left += vals
    for i, g in enumerate(GRADE_SEQ):
        ax.text(101.5, i, f"n={counts[g]}", va='center', ha='left', fontsize=9,
                 color='#52514e', fontproperties=kfont)
    ax.set_yticklabels(GRADE_SEQ, fontproperties=kfont, fontsize=11)
    ax.set_xlim(0, 118)
    ax.set_xlabel('비중 (%)', color='#52514e', fontsize=10, fontproperties=kfont)
    ax.set_title('ESG 등급별 신용위험 단계 분포', fontsize=12.5, color='#0b0b0b',
                  pad=14, loc='left', fontproperties=kfont_b)
    for label in ax.get_xticklabels():
        label.set_fontproperties(kfont)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.grid(axis='x', color='#e1e0d9', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4,
              frameon=False, fontsize=9.5, prop=kfont)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/chart1_stage_by_grade.png', dpi=160, facecolor='#fcfcfb')
    plt.close()

    # Chart 2: firm size by ESG grade (confound check)
    if 'capital_total' in sub.columns:
        size = sub.dropna(subset=['capital_total']).copy()
        size['log_capital'] = np.log(size['capital_total'].clip(lower=1))
        r, _ = stats.spearmanr(
            size['ESG등급'].map(GRADE_ORDER), size['log_capital'])

        fig, ax = plt.subplots(figsize=(8.5, 5), facecolor='#fcfcfb')
        ax.set_facecolor('#fcfcfb')
        data = [size[size['ESG등급'] == g]['log_capital'].values for g in GRADE_SEQ]
        bp = ax.boxplot(data, positions=range(len(GRADE_SEQ)), widths=0.55,
                         patch_artist=True,
                         medianprops=dict(color='#0b0b0b', linewidth=1.6),
                         whiskerprops=dict(color='#898781'),
                         capprops=dict(color='#898781'),
                         flierprops=dict(markerfacecolor='#c3c2b7',
                                          markeredgecolor='none', markersize=3, alpha=0.5))
        for patch, g in zip(bp['boxes'], GRADE_SEQ):
            patch.set_facecolor(BLUE_STEPS[g])
            patch.set_edgecolor('none')
        ax.set_xticks(range(len(GRADE_SEQ)))
        ax.set_xticklabels(GRADE_SEQ, fontproperties=kfont, fontsize=11)
        ax.set_ylabel('log(자본총계)', color='#52514e', fontsize=10, fontproperties=kfont)
        ax.set_title(f'ESG 등급별 기업 규모 분포 — 등급-규모 교란 확인 (Spearman r={r:.2f})',
                      fontsize=12.5, color='#0b0b0b', pad=14, loc='left', fontproperties=kfont_b)
        for label in ax.get_yticklabels():
            label.set_fontproperties(kfont)
        ax.tick_params(colors='#52514e', labelsize=10)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#c3c2b7')
        ax.spines['bottom'].set_color('#c3c2b7')
        ax.grid(axis='y', color='#e1e0d9', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/chart2_size_by_grade.png', dpi=160, facecolor='#fcfcfb')
        plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python3 run_esg_credit_eda.py <credit_stage_panel.csv 경로>")
        sys.exit(1)
    merged = load_and_merge(sys.argv[1])
    merged.to_csv(f'{OUT_DIR}/merged_esg_credit_REAL.csv', index=False)
    sub, results = run_stats(merged)
    for k, v in results.items():
        print(k, ':', v)
    make_charts(sub)
    print("완료 — 차트와 merged_esg_credit_REAL.csv 저장됨")
