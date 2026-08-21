"""
H4: ESG 등급 강등(하락) 이벤트 스터디
=====================================

목적
----
"ESG 등급이 하락한 기업들의 채권 스프레드가, 등급 발표 이전에 이미 움직였는지(선행)
아니면 발표 이후에 움직였는지(후행)"를 여러 기업의 강등 이벤트를 모아 평균 내어 확인한다.

부도 사례처럼 희귀한 이벤트가 아니라 '등급 하락'이라는 훨씬 빈번한 이벤트를 쓰기 때문에,
표본 크기 문제(멘토 피드백) 없이 통계적으로 의미 있는 결과를 낼 수 있다.

설계
----
- 이벤트 시점(day 0) = 각 기업의 ESG 등급 하락 발표일
- '이벤트 상대일(event-relative day)' 축으로 정렬: day -30 ~ day +30 등
- 강등된 기업들의 평균 스프레드 궤적(treatment) vs 등급이 안 바뀐 기업들의 같은 기간
  평균 스프레드 궤적(control)을 나란히 그려서 비교
- treatment가 day 0 이전부터 control과 벌어지기 시작하면 '선행', day 0 이후부터
  벌어지면 '후행'

입력 파일 형식
--------------
downgrade_events.csv (팀이 KCGS 다운로드 파일에서 만들어야 하는 파일):
    company_name,event_date
    OO건설,2025-04-15
    OO리츠,2025-04-15
    ...
  (앞서 확인이 필요하다고 말씀드린 것처럼, KCGS가 전체 기업을 한 날짜에 일괄 발표한다면
   event_date는 모든 행에서 동일한 값이어도 된다.)

control_companies.csv (등급이 하락하지 않은 대조군 기업 목록):
    company_name
    OO전자
    OO화학
    ...

사용법
------
    pip install pandas matplotlib requests --break-system-packages

    python analyze_esg_downgrade_event_study.py \
        --events downgrade_events.csv \
        --controls control_companies.csv \
        --window-before 30 --window-after 30 \
        --krx-auth-key 실제_KRX_인증키
"""

import argparse
import time
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fetch_krx_bond_daily(bas_dd: str, auth_key: str) -> pd.DataFrame:
    """data_pipeline.py의 운영 경로 호출과 동일한 로직."""
    url = "https://data-dbg.krx.co.kr/svc/apis/bon/bnd_bydd_trd"
    headers = {"AUTH_KEY": auth_key}
    params = {"basDd": bas_dd}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        records = data.get("OutBlock_1", data.get("outBlock", []))
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
    except requests.exceptions.RequestException as e:
        print(f"[KRX] {bas_dd} 호출 실패: {e}")
        return pd.DataFrame()


def collect_company_window(company_name: str, center_date: pd.Timestamp,
                            window_before: int, window_after: int, auth_key: str,
                            cache: dict) -> pd.DataFrame:
    """
    특정 기업의 특정 이벤트일 전후 영업일 구간의 CLSPRC_YD를 모은다.
    cache: {날짜문자열: DataFrame} 형태로 하루치 KRX 전체 데이터를 재사용해
           같은 날짜를 여러 기업/이벤트에서 중복 호출하지 않도록 한다.
    """
    dates = pd.bdate_range(center_date - pd.Timedelta(days=window_before * 2),
                            center_date + pd.Timedelta(days=window_after * 2))
    rows = []
    for d in dates:
        bas_dd = d.strftime("%Y%m%d")
        if bas_dd not in cache:
            cache[bas_dd] = fetch_krx_bond_daily(bas_dd, auth_key)
            time.sleep(0.3)
        day_df = cache[bas_dd]
        if day_df.empty:
            continue
        matched = day_df[day_df["ISU_NM"].str.contains(company_name, na=False)]
        if not matched.empty:
            yd = pd.to_numeric(matched["CLSPRC_YD"], errors="coerce").mean()
            event_rel_day = (d - center_date).days
            rows.append({"company": company_name, "date": d, "event_rel_day": event_rel_day, "yield": yd})
    return pd.DataFrame(rows)


def build_event_time_panel(entities: pd.DataFrame, is_treatment: bool,
                            window_before: int, window_after: int, auth_key: str,
                            cache: dict) -> pd.DataFrame:
    """여러 기업의 이벤트 상대일 패널을 하나로 합친다."""
    all_frames = []
    for _, row in entities.iterrows():
        company = row["company_name"]
        event_date = pd.to_datetime(row["event_date"])
        panel = collect_company_window(company, event_date, window_before, window_after, auth_key, cache)
        panel["is_treatment"] = is_treatment
        all_frames.append(panel)
        print(f"  {'[강등]' if is_treatment else '[대조군]'} {company}: {len(panel)}개 관측치 수집")
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def plot_event_study(panel: pd.DataFrame, out_path: str):
    """이벤트 상대일 기준으로 강등군 vs 대조군 평균 수익률 궤적을 그린다."""
    agg = panel.groupby(["event_rel_day", "is_treatment"])["yield"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for treat_flag, label, color in [(True, "ESG 등급 강등 기업 (평균)", "crimson"),
                                       (False, "대조군 (평균)", "steelblue")]:
        sub = agg[agg["is_treatment"] == treat_flag].sort_values("event_rel_day")
        ax.plot(sub["event_rel_day"], sub["yield"], label=label, color=color)

    ax.axvline(0, color="black", linestyle=":", linewidth=1.5)
    ax.text(0, ax.get_ylim()[1], " 등급 발표일(day 0)", rotation=90, va="top", fontsize=8)
    ax.set_xlabel("이벤트 상대일 (등급 발표일 기준 영업일 차이)")
    ax.set_ylabel("평균 수익률 (%)")
    ax.set_title("ESG 등급 강등 이벤트 스터디: 강등군 vs 대조군")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[저장 완료] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, help="company_name,event_date 컬럼을 가진 CSV (등급 강등 기업)")
    parser.add_argument("--controls", required=True, help="company_name 컬럼을 가진 CSV (대조군)")
    parser.add_argument("--window-before", type=int, default=30)
    parser.add_argument("--window-after", type=int, default=30)
    parser.add_argument("--krx-auth-key", required=True)
    parser.add_argument("--out", default="esg_downgrade_event_study.png")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    controls_raw = pd.read_csv(args.controls)

    # 대조군에는 이벤트일이 없으므로, 강등군 이벤트일들의 평균(또는 가장 흔한 날짜)을
    # 기준일로 부여해 같은 캘린더 구간을 비교한다. 강등군 이벤트일이 전부 같다면(일괄 발표)
    # 그 날짜를 그대로 쓰면 된다.
    common_event_date = pd.to_datetime(events["event_date"]).mode()[0]
    controls = controls_raw.copy()
    controls["event_date"] = common_event_date.strftime("%Y-%m-%d")

    cache = {}
    print("[강등군 수집 시작]")
    treat_panel = build_event_time_panel(events, True, args.window_before, args.window_after, args.krx_auth_key, cache)
    print("[대조군 수집 시작]")
    control_panel = build_event_time_panel(controls, False, args.window_before, args.window_after, args.krx_auth_key, cache)

    if treat_panel.empty or control_panel.empty:
        print("데이터가 비어 있습니다. 종목명 표기, 이벤트일, API 키를 확인하세요.")
        return

    panel = pd.concat([treat_panel, control_panel], ignore_index=True)
    panel.to_csv(args.out.replace(".png", "_panel.csv"), index=False, encoding="utf-8-sig")
    plot_event_study(panel, args.out)

    # 선행/후행 판단을 위한 간단한 요약: day 0 이전 구간과 이후 구간에서
    # 강등군-대조군 평균 차이(스프레드 프록시)가 각각 얼마나 벌어지는지 출력
    agg = panel.groupby(["event_rel_day", "is_treatment"])["yield"].mean().unstack("is_treatment")
    agg["gap"] = agg[True] - agg[False]
    pre = agg[agg.index < 0]["gap"].mean()
    post = agg[agg.index >= 0]["gap"].mean()
    print(f"\n[요약] 발표일 이전(day<0) 평균 격차: {pre:.4f}%p")
    print(f"[요약] 발표일 이후(day>=0) 평균 격차: {post:.4f}%p")
    if abs(pre) > abs(post) * 0.5:
        print("→ 발표 이전에 이미 상당한 격차가 존재 — 선행(시장이 먼저 반응) 가능성")
    else:
        print("→ 발표 이전 격차가 미미하고 이후에 커짐 — 후행(등급 발표가 새 정보) 가능성")


if __name__ == "__main__":
    main()
