# -*- coding: utf-8 -*-
"""
[수집 6] G(지배구조)등급 - 이사회 구성(사외이사 비율)
========================================================
목적: KCGS 지배구조(G)등급이 실제 이사회 구성(사외이사 비율)을 반영하는지,
      그리고 이사회 독립성 자체가 신용위험과 관계있는지 검증하기 위한 수집.

무엇을 얻는가
-------------
  collected/board_composition_RAW.csv
    corp_code, corp_name, bsns_year, drctr_co(이사 총수),
    otcmp_drctr_co(사외이사 수), outside_ratio(=사외이사수/이사총수), stlm_dt

API: 독립(사외)이사 및 그 변동현황 (OpenDART DS002)
  https://opendart.fss.or.kr/api/outcmpnyDrctrNdChangeSttus.json
  파라미터: crtfc_key, corp_code(8자리), bsns_year(4자리), reprt_code
    reprt_code: 11013=1분기, 11012=반기, 11014=3분기, 11011=사업보고서(연간, 기본값)

사전 준비
---------
    pip install requests pandas
    DART API 키 (기존 collect_2_dart_pit.py와 동일한 키를 기본값으로 재사용)

입력
----
    esg_credit_sample.csv (이 스크립트와 같은 폴더에 둘 것)
      - E/S 신용위험 분석에 쓴 것과 동일한 1,025개사 표본
      - corp_code, corp_name, 지배구조, ESG등급 컬럼 포함

사용법
------
    python collect_6_board.py                  # 기본: 2025 사업보고서
    python collect_6_board.py --year 2024
"""
import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ============ 설정 ============
DART_API_KEY = os.environ.get("DART_API_KEY", "4c056faa738bcb25aa545b05a5b1d69137a71f39")
PAUSE = 0.3  # 호출 간격(초) — rate limit 방지

INPUT_PATH = Path("./esg_credit_sample.csv")
OUT = Path("./collected")
OUT.mkdir(exist_ok=True)
OUTPUT_PATH = OUT / "board_composition_RAW.csv"

BOARD_URL = "https://opendart.fss.or.kr/api/outcmpnyDrctrNdChangeSttus.json"


def fetch_board(corp_code8, bsns_year, reprt_code="11011"):
    params = {
        "crtfc_key": DART_API_KEY,
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

    if not INPUT_PATH.exists():
        log.error(f"입력 파일이 없습니다: {INPUT_PATH.resolve()}  "
                  f"(esg_credit_sample.csv를 이 스크립트와 같은 폴더에 두세요)")
        return

    companies = pd.read_csv(INPUT_PATH)
    companies = companies[["corp_code", "corp_name", "지배구조", "ESG등급"]].drop_duplicates("corp_code")
    companies["corp_code8"] = companies["corp_code"].astype(str).str.zfill(8)
    log.info(f"수집 대상 {len(companies)}개사 (연도={args.year})")

    results = []
    for i, row in companies.reset_index(drop=True).iterrows():
        rec = fetch_board(row["corp_code8"], args.year, args.reprt_code)
        if rec is not None:
            rec["corp_name"] = row["corp_name"]
            results.append(rec)
        if (i + 1) % 50 == 0:
            log.info(f"진행: {i+1}/{len(companies)} (수집 성공 {len(results)}건)")
        time.sleep(PAUSE)

    out = pd.DataFrame(results)
    if len(out) == 0:
        log.error("수집된 데이터가 0건입니다. API 키/연도/네트워크 상태를 확인하세요.")
        return
    out["drctr_co"] = pd.to_numeric(out["drctr_co"], errors="coerce")
    out["otcmp_drctr_co"] = pd.to_numeric(out["otcmp_drctr_co"], errors="coerce")
    out["outside_ratio"] = out["otcmp_drctr_co"] / out["drctr_co"]

    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    log.info(f"완료: {len(out)}/{len(companies)}개사 수집 -> {OUTPUT_PATH}")
    log.info(f"outside_ratio 결측 {out['outside_ratio'].isna().sum()}건 (drctr_co=0 또는 데이터 없음)")


if __name__ == "__main__":
    main()
