"""
E(환경)/S(사회) 등급 - 신용위험 상관관계 세부 분석 + 업종별 이질성 분석
=========================================================================
run_esg_credit_eda.py로 만든 merged_esg_credit_REAL.csv를 입력으로 받아
E, S 등급을 신용위험과 더 자세히 들여다보는 후속 분석.

핵심 산출:
  1. E/S 등급별 신용위험 단계 분포 (crosstab, 100% stacked)
  2. 규모(자본총계) 통제 편상관계수 (E, S, G 각각 vs risk_score)
  3. E, S, G를 동시에 넣은 로지스틱회귀 (상호 통제 시 어느 축이 살아남는가)
  4. 업종별(KSIC 2단위 중분류) E/S 편상관계수 비교

주의 - induty_code 파싱:
  이 필드는 고정 5자리가 아니라 기업별로 자릿수가 다른 KSIC 코드 원본 그대로
  들어있다(선행 0 소실 아님, 표기 자체가 3~5자리 혼재). 업종 대분류를 얻으려면
  zfill로 맞추지 말고 문자열 그대로 앞 2자리를 잘라야 한다 :
      str(int(code))[:2]
  이걸 몰라서 처음엔 전 표본이 '기타'로 잘못 분류되는 버그가 있었다(수정됨).
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

IN_PATH = '/home/claude/analysis/merged_esg_credit_REAL.csv'
OUT_DIR = '/home/claude/analysis'
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

GRADE_ORDER = {'D': 1, 'C': 2, 'B': 3, 'B+': 4, 'A': 5, 'A+': 6}
GRADE_SEQ = ['D', 'C', 'B', 'B+', 'A', 'A+']
STAGE_LABELS = {0: '정상', 1: '위험징후', 2: '신용등급하락', 3: '부도·회생·파산'}
STATUS_COLORS = {0: '#0ca30c', 1: '#fab219', 2: '#ec835a', 3: '#d03b3b'}


def sector(code):
    """KSIC 코드 앞 2자리(중분류)로 업종 대분류 버킷 분류. zfill 쓰지 말 것 (위 주의 참고)."""
    d2 = int(str(int(code))[:2])
    if 10 <= d2 <= 34:
        return '제조업'
    if d2 in (41, 42):
        return '건설업'
    if 45 <= d2 <= 47:
        return '도소매'
    if 58 <= d2 <= 63:
        return '정보통신'
    if 64 <= d2 <= 66:
        return '금융보험'
    if 70 <= d2 <= 75:
        return '전문과학기술'
    return '기타'


def partial_corr(x, y, z):
    """순위 기반 편상관계수: x, y의 상관에서 z의 영향을 통제."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    df = pd.read_csv(IN_PATH)
    sub = df[df['ESG등급'] != '등급없음'].copy()
    for c in ['ESG등급', '환경', '사회', '지배구조']:
        sub[c + '_ord'] = sub[c].map(GRADE_ORDER)
    sub['distress'] = (sub['stage'] >= 1).astype(int)
    sub['log_capital'] = np.log(sub['capital_total'].clip(lower=1))
    sub['sector'] = sub['induty_code'].apply(sector)

    print("=== 업종 분포 ===")
    print(sub['sector'].value_counts(), "\n")

    for col in ['환경', '사회']:
        tmp = sub[sub[col] != '등급없음']
        ct = (pd.crosstab(tmp[col], tmp['distress'], normalize='index') * 100).round(1)
        print(f"=== {col} 등급별 distress 비율 (n={len(tmp)}) ===")
        print(ct, "\n")

    g2 = sub.dropna(subset=['환경_ord', '사회_ord', '지배구조_ord', 'risk_score', 'log_capital'])
    print("=== 전체 표본 편상관 (규모 통제, vs risk_score) ===")
    for col in ['환경_ord', '사회_ord', '지배구조_ord']:
        pr = partial_corr(g2[col], g2['risk_score'], g2['log_capital'])
        print(f"  {col}: r={pr:.4f}")
    print()

    print("=== E/S/G 동시 투입 로지스틱회귀 (distress ~ E+S+G+size) ===")
    X = sm.add_constant(g2[['환경_ord', '사회_ord', '지배구조_ord', 'log_capital']])
    m = sm.Logit(g2['distress'], X).fit(disp=0)
    print(m.summary(), "\n")

    print("=== 업종별 E/S 편상관 (n>=30) ===")
    rows = []
    for sec, g in sub.groupby('sector'):
        gg = g.dropna(subset=['환경_ord', '사회_ord', 'risk_score', 'log_capital'])
        if len(gg) < 30:
            continue
        pe = partial_corr(gg['환경_ord'], gg['risk_score'], gg['log_capital'])
        ps = partial_corr(gg['사회_ord'], gg['risk_score'], gg['log_capital'])
        rows.append((sec, len(gg), pe, ps))
    res = pd.DataFrame(rows, columns=['sector', 'n', '환경', '사회']).sort_values('n', ascending=False)
    print(res, "\n")

    sub.to_csv(f'{OUT_DIR}/merged_esg_credit_REAL_with_sector.csv', index=False)

    # ---- charts ----
    fp = fm.FontProperties(fname=FONT_PATH)
    fp_bold = fm.FontProperties(fname=FONT_PATH, weight='bold')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, col, title in zip(axes, ['환경', '사회'], ['환경(E)등급', '사회(S)등급']):
        tmp = sub[sub[col] != '등급없음']
        ct = pd.crosstab(tmp[col], tmp['stage'], normalize='index').reindex(GRADE_SEQ) * 100
        bottom = np.zeros(len(ct))
        for stage in [0, 1, 2, 3]:
            if stage not in ct.columns:
                continue
            vals = ct[stage].values
            ax.bar(ct.index, vals, bottom=bottom, color=STATUS_COLORS[stage],
                   label=STAGE_LABELS[stage], width=0.65)
            bottom += vals
        ax.set_title(f'{title} × 신용위험 단계 (n={len(tmp)})', fontproperties=fp_bold, fontsize=13)
        ax.set_xlabel(title, fontproperties=fp)
        for lbl in ax.get_xticklabels():
            lbl.set_fontproperties(fp)
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(fp)
    axes[0].set_ylabel('비율 (%)', fontproperties=fp)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, prop=fp, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart3_ES_stage_REAL.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(res))
    w = 0.35
    ax.bar(x - w / 2, res['환경'], width=w, color='#2a78d6', label='환경(E)')
    ax.bar(x + w / 2, res['사회'], width=w, color='#eb6834', label='사회(S)')
    ax.axhline(0, color='#888888', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(res['sector'], res['n'])], fontproperties=fp)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp)
    ax.set_ylabel('편상관계수 (규모 통제, vs risk_score)', fontproperties=fp)
    ax.set_title('업종별 E/S 등급 - 신용위험 편상관 비교', fontproperties=fp_bold, fontsize=13)
    ax.legend(prop=fp)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/chart4_ES_by_industry_REAL.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("차트 저장 완료: chart3_ES_stage_REAL.png, chart4_ES_by_industry_REAL.png")


if __name__ == '__main__':
    main()
