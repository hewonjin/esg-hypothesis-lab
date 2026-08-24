"""
G(지배구조)등급 - 이사회 구성 데이터 수집 스크립트
=====================================================
목적: KCGS 지배구조(G)등급과 실제 이사회 구성(사외이사 비율)의 상관관계를
      분석하기 위해, DART "독립(사외)이사 및 그 변동현황" API로 회사별
      이사 총원·사외이사 수를 수집한다.

*** 반드시 로컬(LIVE_COLLECT 환경)에서 실행 ***
클라우드 세션에서는 opendart.fss.or.kr로 아웃바운드 연결이 막혀 있어
(다른 축과 동일한 문제) 이 스크립트를 클라우드에서 실행할 수 없음을
확인했다. 로컬 PC에서 실행해서 나온 CSV를 다시 올려주면 이어서
run_g_board_eda.py로 G등급 상관분석까지 진행함.

API: 독립(사외)이사 및 그 변동현황 (OpenDART DS002)
  https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS002&apiId=2020012
  https://opendart.fss.or.kr/api/outcmpnyDrctrNdChangeSttus.json
  파라미터: crtfc_key, corp_code(8자리), bsns_year(4자리), reprt_code
    reprt_code: 11013=1분기, 11012=반기, 11014=3분기, 11011=사업보고서(연간)
  응답 필드: drctr_co(이사 총수), otcmp_drctr_co(사외이사 수) 등
  → outside_ratio = otcmp_drctr_co / drctr_co 로 '이사회 독립성' 지표를 구성.

준비물
------
    pip install requests pandas --break-system-packages
    환경변수 DART_API_KEY (없으면 기존 파이프라인과 동일한 기본키 재사용)

입력: eda_credit_risk/merged_esg_credit_REAL_with_sector.csv
       (E·S 분석에 쓴 것과 같은 1,025개사 표본 — corp_code, corp_name, 지배구조등급 포함)
출력: eda_credit_risk/board_composition_RAW.csv
       (corp_code, corp_name, bsns_year, drctr_co, otcmp_drctr_co, outside_ratio, stlm_dt)

사용법:
    python3 collect_board_composition.py [--year 2025]
"""
import argparse
import os
import time
import logging

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONFIG = {
    "DART_API_KEY": os.environ.get("DART_API_KEY", "4c056faa738bcb25aa545b05a5b1d69137a71f39"),
    "REQUEST_INTERVAL_SEC": 0.3,
}

INPUT_PATH = "eda_credit_risk/merged_esg_credit_REAL_with_sector.csv"
OUTPUT_PATH = "eda_credit_risk/board_composition_RAW.csv"

BOARD_URL = "https://opendart.fss.or.kr/api/outcmpnyDrctrNdChangeSttus.json"


def fetch_board(corp_code8: str, bsns_year: str, reprt_code: str = "11011"):
    params = {
        "crtfc_key": CONFIG["DART_API_KEY"],
        "corp_code": corp_code8,
        "bsns_year": bsns_year,
        "reprt_code": reprt_code,
    }
    try:
        resp = requests.get(BOARD_URL, params=params, timeout=10)
        data = resp.json()
    except Exception as e:
        log.error(f"[이사회현황] {corp_code8} 호출 실패: {e}")
        return None

    if data.get("status") != "000":
        log.warning(f"[이사회현황] {corp_code8} {bsns_year}: {data.get('message')} (status={data.get('status')})")
        return None

    rows = data.get("list", [])
    if not rows:
        return None
    # 여러 행(정정 등)이 올 수 있으므로 가장 마지막(최신) 행 사용
    r = rows[-1]
    return {
        "corp_code": corp_code8,
        "bsns_year": bsns_year,
        "drctr_co": r.get("drctr_co"),
        "otcmp_drctr_co": r.get("otcmp_drctr_co"),
        "stlm_dt": r.get("stlm_dt"),
        "rcept_no": r.get("rcept_no"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2025", help="사업연도 (기본 2025 = KCGS 2025 평가등급과 동일 시점)")
    ap.add_argument("--reprt_code", default="11011", help="보고서 코드 (기본 11011=사업보고서)")
    args = ap.parse_args()

    companies = pd.read_csv(INPUT_PATH)
    companies = companies[["corp_code", "corp_name", "지배구조", "ESG등급"]].drop_duplicates("corp_code")
    companies["corp_code8"] = companies["corp_code"].astype(str).str.zfill(8)
    log.info(f"수집 대상 {len(companies)}개사 (연도={args.year})")

    results = []
    for i, row in companies.iterrows():
        rec = fetch_board(row["corp_code8"], args.year, args.reprt_code)
        if rec is not None:
            rec["corp_name"] = row["corp_name"]
            results.append(rec)
        if (i + 1) % 50 == 0:
            log.info(f"진행: {i+1}/{len(companies)} (수집 성공 {len(results)}건)")
        time.sleep(CONFIG["REQUEST_INTERVAL_SEC"])

    out = pd.DataFrame(results)
    if len(out) == 0:
        log.error("수집된 데이터가 0건입니다. API 키/연도/네트워크 상태를 확인하세요.")
        return
    out["drctr_co"] = pd.to_numeric(out["drctr_co"], errors="coerce")
    out["otcmp_drctr_co"] = pd.to_numeric(out["otcmp_drctr_co"], errors="coerce")
    out["outside_ratio"] = out["otcmp_drctr_co"] / out["drctr_co"]

    out.to_csv(OUTPUT_PATH, index=False)
    log.info(f"완료: {len(out)}/{len(companies)}개사 수집 -> {OUTPUT_PATH}")
    log.info(f"outside_ratio 결측 {out['outside_ratio'].isna().sum()}건 (drctr_co=0 또는 데이터 없음)")


if __name__ == "__main__":
    main()
