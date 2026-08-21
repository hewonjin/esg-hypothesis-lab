"""
채권 발행기업 신용경색 조기경보 시스템 — 데이터 파이프라인 초안
====================================================================

목적
----
5개 데이터 축(A~E)을 일 단위 프레임으로 결합해 종목×일자 단위의
통합 데이터셋을 만든다.

    A축 : KRX Open API — 일반채권시장 일별매매정보 (bnd_bydd_trd)
          거래량(ACC_TRDVOL), 거래대금(ACC_TRDVAL)
    B축 : 같은 API의 CLSPRC_YD(종가수익률) — KOFIA BIS 등급평균과 결합해 스프레드 산출
    C축 : 공공데이터포털 — 단기금융증권(CD/CP/ABCP/전자단기사채) 발행정보
    D축 : DART Open API — 재무제표 (분기 단위, as-of forward-fill로 일 단위 프레임에 결합)
    E축 : DART Open API — 주요사항보고서 (부도/회생절차개시신청/해산사유) → 1차 라벨

사용 전 준비물
--------------
    pip install requests pandas --break-system-packages

    환경변수 또는 CONFIG 섹션에 아래 키를 채워 넣는다.
        KRX_AUTH_KEY   : KRX Open API 인증키
        DART_API_KEY   : OpenDART API 키
        DATA_GO_KR_KEY : 공공데이터포털 서비스키(단기금융증권 발행정보)

주의
----
- KRX/공공데이터포털의 정확한 엔드포인트 URL과 파라미터명은 실제 발급받은
  API 문서 화면의 값으로 반드시 교체해야 한다. 아래 URL은 공개된 API 구조
  패턴을 따른 초안이며, 팀이 문서에서 확인한 정확한 값으로 갱신해야 한다.
- DART corp_code는 https://opendart.fss.or.kr 의 고유번호 파일(corpCode.xml)을
  최초 1회 내려받아 종목명↔corp_code 매핑 테이블을 만들어 두고 재사용한다.
"""

import os
import time
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 0. 설정
# ----------------------------------------------------------------------

CONFIG = {
    "KRX_AUTH_KEY": os.environ.get("KRX_AUTH_KEY", "여기에_KRX_인증키"),
    "DART_API_KEY": os.environ.get("DART_API_KEY", "여기에_DART_인증키"),
    "DATA_GO_KR_KEY": os.environ.get("DATA_GO_KR_KEY", "여기에_공공데이터포털_서비스키"),
    # 표본기업: {종목명 또는 종목코드: DART corp_code} 형태로 팀이 채워 넣는다.
    "SAMPLE_ISSUERS": {
        "제이알글로벌리츠": "01415892",
    },
    "START_DATE": "2024-01-01",
    "END_DATE": "2026-08-19",
    "REQUEST_INTERVAL_SEC": 0.3,  # API 호출 간 최소 간격 (rate limit 방지)
}


# ----------------------------------------------------------------------
# A·B축 : KRX Open API — 일반채권시장 일별매매정보(bnd_bydd_trd)
# ----------------------------------------------------------------------

def fetch_krx_bond_daily(bas_dd: str, use_sample: bool = False, host_override: str = None) -> pd.DataFrame:
    """
    ...(위 설명 동일)...
    host_override: "data-dbg.krx.co.kr" 대신 다른 호스트(예: "openapi.krx.co.kr")로 시도해볼 때 지정.
    """
    KRX_SAMPLE_KEY = "74D1B99DFBF345BBA3FB4476510A4BED4C78D13A"

    host = host_override or "data-dbg.krx.co.kr"

    if use_sample:
        url = f"https://{host}/svc/sample/apis/bon/bnd_bydd_trd"
        auth_key = KRX_SAMPLE_KEY
    else:
        url = f"https://{host}/svc/apis/bon/bnd_bydd_trd"
        auth_key = CONFIG["KRX_AUTH_KEY"]

    headers = {"AUTH_KEY": auth_key}
    params = {"basDd": bas_dd}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        records = data.get("OutBlock_1", data.get("outBlock", []))
        if not records:
            log.warning(f"[KRX] {bas_dd}: 응답이 비어 있음 (휴장일이거나, sample 모드라면 지원되는 날짜가 아닐 수 있음)")
            return pd.DataFrame()
        df = pd.DataFrame(records)
        return df
    except requests.exceptions.RequestException as e:
        log.error(f"[KRX] {bas_dd} 호출 실패: {e}")
        return pd.DataFrame()


def collect_krx_range(start_date: str, end_date: str) -> pd.DataFrame:
    """영업일 목록을 순회하며 KRX 일별 채권 데이터를 누적 수집한다."""
    dates = pd.bdate_range(start=start_date, end=end_date)  # 평일 기준(공휴일은 응답이 비어 자동 스킵됨)
    frames = []
    for d in dates:
        bas_dd = d.strftime("%Y%m%d")
        df = fetch_krx_bond_daily(bas_dd)
        if not df.empty:
            frames.append(df)
        time.sleep(CONFIG["REQUEST_INTERVAL_SEC"])
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    # 숫자형 컬럼 변환
    numeric_cols = ["CLSPRC", "CLSPRC_YD", "OPNPRC_YD", "HGPRC_YD", "LWPRC_YD",
                     "ACC_TRDVOL", "ACC_TRDVAL"]
    for c in numeric_cols:
        if c in result.columns:
            result[c] = pd.to_numeric(result[c].astype(str).str.replace(",", ""), errors="coerce")
    result["BAS_DD"] = pd.to_datetime(result["BAS_DD"], format="%Y%m%d")
    return result


# ----------------------------------------------------------------------
# D축 : DART Open API — 재무제표 (분기 단위)
# ----------------------------------------------------------------------

def fetch_dart_financials(corp_code: str, year: int, report_code: str = "11013") -> pd.DataFrame:
    """
    단일회사 주요계정 조회. report_code: 11013(1분기) 11012(반기) 11014(3분기) 11011(사업보고서)
    """
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"
    params = {
        "crtfc_key": CONFIG["DART_API_KEY"],
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": report_code,
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get("status") != "000":
            log.warning(f"[DART 재무] {corp_code} {year} {report_code}: {data.get('message')}")
            return pd.DataFrame()
        df = pd.DataFrame(data["list"])
        df["report_date"] = pd.to_datetime(f"{year}-{_report_code_to_month(report_code)}-01")
        return df
    except requests.exceptions.RequestException as e:
        log.error(f"[DART 재무] {corp_code} 호출 실패: {e}")
        return pd.DataFrame()


def _report_code_to_month(report_code: str) -> str:
    return {"11013": "05", "11012": "08", "11014": "11", "11011": "03"}.get(report_code, "12")


# ----------------------------------------------------------------------
# E축 : DART Open API — 주요사항보고서 (부도·회생절차개시신청·해산사유) → 1차 라벨
# ----------------------------------------------------------------------

DART_EVENT_ENDPOINTS = {
    "default": "https://opendart.fss.or.kr/api/dfOcr.json",          # 채무불이행(부도) 발생
    "rehabilitation": "https://opendart.fss.or.kr/api/ctrcvsBgrq.json",  # 회생절차개시신청
    "dissolution": "https://opendart.fss.or.kr/api/dsRsOcr.json",     # 해산사유 발생
}


def fetch_dart_event(corp_code: str, event_type: str, bgn_de: str, end_de: str) -> pd.DataFrame:
    """
    event_type: "default" | "rehabilitation" | "dissolution"
    bgn_de, end_de: YYYYMMDD
    """
    url = DART_EVENT_ENDPOINTS[event_type]
    params = {
        "crtfc_key": CONFIG["DART_API_KEY"],
        "corp_code": corp_code,
        "bgn_de": bgn_de,
        "end_de": end_de,
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        if data.get("status") != "000":
            return pd.DataFrame()  # 013(조회된 데이터 없음)이 대부분 — 정상
        df = pd.DataFrame(data["list"])
        df["event_type"] = event_type
        return df
    except requests.exceptions.RequestException as e:
        log.error(f"[DART 이벤트:{event_type}] {corp_code} 호출 실패: {e}")
        return pd.DataFrame()


def build_label_table(corp_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """세 가지 이벤트를 모두 조회해 1차 라벨(지급불능=1) 테이블을 만든다."""
    bgn_de = start_date.replace("-", "")
    end_de = end_date.replace("-", "")
    frames = [fetch_dart_event(corp_code, t, bgn_de, end_de) for t in DART_EVENT_ENDPOINTS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["corp_code", "event_date", "event_type", "label"])
    events = pd.concat(frames, ignore_index=True)
    events["label"] = 1
    return events


# ----------------------------------------------------------------------
# C축 : 공공데이터포털 — 단기금융증권 발행정보 (전자단기사채 등)
# ----------------------------------------------------------------------

# 공공데이터포털 — 단기금융증권 발행정보 서비스 (확인된 Base URL)
DATA_GO_KR_BASE = "https://apis.data.go.kr/1160100/GetShorTermSecuIssuInfoService_V2"

# 9개 세부 오퍼레이션의 정확한 명칭은 각 API명(CD발행기본정보조회 등) 링크를 클릭했을 때
# 나오는 상세페이지의 "Operation명"으로 확인해서 채워 넣는다. 지금은 자리표시자.
DATA_GO_KR_OPERATIONS = {
    "cd_issu": "TODO_CD발행기본정보조회_operation명",
    "cp_issu": "TODO_단기기업어음발행기본정보조회_operation명",
    "electronic_note_issu": "getAbstbIssuBasiInfo_V2",
    "abcp_issu": "TODO_ABCP발행기본정보조회_operation명",
    "maturity_balance_by_issuer": "TODO_발행자별단기금융증권현황_만기별발행잔액조회_operation명",
    "monthly_maturity_amount": "TODO_발행자별단기금융증권현황_월별만기금액조회_operation명",
    "balance_by_issuer_type": "TODO_발행자유형별발행잔액조회_operation명",
    "balance_by_kind": "TODO_종류별발행잔액조회_operation명",
    "issue_rate": "TODO_발행금리조회_operation명",
}


def fetch_shortterm_note_info(corp_reg_no: str, bas_dd: str, operation_key: str = "electronic_note_issu") -> pd.DataFrame:
    """
    단기금융증권 발행정보 조회 (기본값: 전자단기사채발행기본정보조회).
    operation_key는 DATA_GO_KR_OPERATIONS의 키 중 하나를 지정한다.
    """
    operation = DATA_GO_KR_OPERATIONS[operation_key]
    if operation.startswith("TODO_"):
        log.warning(f"[공공데이터포털] '{operation_key}' 오퍼레이션명이 아직 채워지지 않았습니다. "
                    f"data.go.kr 상세페이지에서 정확한 Operation명을 확인해 DATA_GO_KR_OPERATIONS에 채워 넣으세요.")

    url = f"{DATA_GO_KR_BASE}/{operation}"
    params = {
        "serviceKey": CONFIG["DATA_GO_KR_KEY"],
        "basDt": bas_dd,
        "crno": corp_reg_no,  # 법인등록번호 — 파라미터명도 실제 문서 확인 후 조정
        "resultType": "json",
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        items = data.get("response", {}).get("body", {}).get("items", [])
        return pd.DataFrame(items)
    except requests.exceptions.RequestException as e:
        log.error(f"[공공데이터포털 C축] {corp_reg_no} 호출 실패: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# 결합 : as-of forward-fill 로 A·B(일 단위) + D(분기 단위) 정렬
# ----------------------------------------------------------------------

def merge_as_of(daily_df: pd.DataFrame, quarterly_df: pd.DataFrame,
                 daily_date_col: str = "BAS_DD",
                 quarterly_date_col: str = "report_date") -> pd.DataFrame:
    """
    일 단위 시장데이터(daily_df)에 분기 재무데이터(quarterly_df)를
    '해당 일자 시점에 가장 최근 발표된 재무제표 값'으로 forward-fill 결합한다.
    미래 정보 누출(look-ahead bias) 방지를 위해 반드시 merge_asof(direction="backward") 사용.
    """
    daily_sorted = daily_df.sort_values(daily_date_col).reset_index(drop=True)
    quarterly_sorted = quarterly_df.sort_values(quarterly_date_col).reset_index(drop=True)

    merged = pd.merge_asof(
        daily_sorted,
        quarterly_sorted,
        left_on=daily_date_col,
        right_on=quarterly_date_col,
        direction="backward",  # 재무데이터 발표일이 시장데이터 날짜보다 과거(또는 같은 날)인 것만 사용
    )
    return merged


def compute_spread(merged_df: pd.DataFrame, grade_avg_yield: dict) -> pd.DataFrame:
    """
    개별종목 스프레드 = CLSPRC_YD - 동일등급 평균수익률(KOFIA BIS에서 별도 수집)
    grade_avg_yield: {"BAS_DD_str": {"AAA": 3.6, "AA+": 3.9, ...}} 형태로 KOFIA BIS 수집 후 채워 넣는다.
    """
    def _lookup(row):
        dd = row["BAS_DD"].strftime("%Y%m%d")
        grade = row.get("SIC_GRD_NM") or row.get("credit_grade")  # 종목의 신용등급 컬럼명은 실제 응답 확인 후 조정
        avg = grade_avg_yield.get(dd, {}).get(grade)
        if avg is None or pd.isna(row.get("CLSPRC_YD")):
            return None
        return row["CLSPRC_YD"] - avg

    merged_df["spread"] = merged_df.apply(_lookup, axis=1)
    return merged_df


# ----------------------------------------------------------------------
# 메인 실행 예시
# ----------------------------------------------------------------------

def run_sample_pipeline():
    """표본기업 1곳을 예시로 전체 파이프라인을 시험 실행한다."""
    if not CONFIG["SAMPLE_ISSUERS"]:
        log.warning("CONFIG['SAMPLE_ISSUERS']에 표본기업(corp_code)을 먼저 채워 넣으세요.")
        return

    start, end = CONFIG["START_DATE"], CONFIG["END_DATE"]

    # 1) A·B축: 시장 전체 일별 데이터 수집 (기간이 길면 시간이 오래 걸리므로 최초엔 짧은 기간으로 테스트 권장)
    log.info("A·B축(KRX) 수집 시작")
    krx_df = collect_krx_range(start, end)
    log.info(f"KRX 수집 결과: {len(krx_df)}행")

    for issuer_name, corp_code in CONFIG["SAMPLE_ISSUERS"].items():
        log.info(f"=== {issuer_name} ({corp_code}) 처리 시작 ===")

        # 종목명 매칭 — 실제로는 종목코드-발행사 매핑 테이블을 미리 구축해 사용
        issuer_bonds = krx_df[krx_df["ISU_NM"].str.contains(issuer_name, na=False)] if not krx_df.empty else pd.DataFrame()

        # 2) D축: 분기 재무제표 수집 (연도별로 반복)
        years = range(int(start[:4]), int(end[:4]) + 1)
        fin_frames = [fetch_dart_financials(corp_code, y) for y in years]
        fin_df = pd.concat([f for f in fin_frames if not f.empty], ignore_index=True) if any(
            not f.empty for f in fin_frames) else pd.DataFrame()

        # 3) E축: 라벨 테이블
        label_df = build_label_table(corp_code, start, end)
        if not label_df.empty:
            log.info(f"[{issuer_name}] 지급불능 이벤트 {len(label_df)}건 발견")

        # 4) as-of 결합 (재무데이터가 있을 때만)
        if not issuer_bonds.empty and not fin_df.empty:
            merged = merge_as_of(issuer_bonds, fin_df)
            out_path = f"/mnt/user-data/outputs/{issuer_name}_merged.csv"
            merged.to_csv(out_path, index=False, encoding="utf-8-sig")
            log.info(f"[{issuer_name}] 결합 데이터셋 저장: {out_path}")
        else:
            log.warning(f"[{issuer_name}] 시장데이터 또는 재무데이터가 비어 있어 결합 생략")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test-krx-sample":
        # 사용법: python data_pipeline.py test-krx-sample
        print("KRX 샘플 경로 테스트 중 (basDd=20200414, 문서에 나온 예제 날짜)...")
        df = fetch_krx_bond_daily("20200414", use_sample=True)
        if df.empty:
            print("결과 없음 — 샘플 경로/키 자체에 문제가 있거나 날짜가 안 맞을 수 있습니다.")
        else:
            print(f"성공: {len(df)}행 수신")
            print(df.head())

    elif len(sys.argv) > 1 and sys.argv[1] == "test-krx-real":
        # 사용법: python data_pipeline.py test-krx-real [YYYYMMDD] [host]
        # 팀의 실제 키로 최근 영업일 하나만 시도한다. host 인자를 생략하면 data-dbg.krx.co.kr,
        # "openapi"를 넣으면 openapi.krx.co.kr로 시도한다.
        test_date = sys.argv[2] if len(sys.argv) > 2 else "20260519"
        host_arg = sys.argv[3] if len(sys.argv) > 3 else None
        host = "openapi.krx.co.kr" if host_arg == "openapi" else host_arg
        print(f"KRX 운영 경로 테스트 중 (basDd={test_date}, host={host or 'data-dbg.krx.co.kr(기본값)'})...")
        df = fetch_krx_bond_daily(test_date, use_sample=False, host_override=host)
        if df.empty:
            print("결과 없음 또는 실패 — 위 에러 로그를 확인하세요.")
        else:
            print(f"성공: {len(df)}행 수신")
            print(df.head())

    elif len(sys.argv) > 1 and sys.argv[1] == "validate-jrglobal":
        # 사용법: python data_pipeline.py validate-jrglobal
        # 제이알글로벌리츠 디폴트 구간(2026-03-02 ~ 2026-04-28)의 실제 KRX 데이터를 모아
        # 종가수익률(CLSPRC_YD) 추이를 확인한다 — 시장이 신용평가사보다 먼저 반응했는지 검증.
        print("제이알글로벌리츠 디폴트 구간 검증 시작 (2026-03-02 ~ 2026-04-28)...")
        jr_dates = pd.bdate_range("2026-03-02", "2026-04-28")
        frames = []
        for d in jr_dates:
            bas_dd = d.strftime("%Y%m%d")
            df = fetch_krx_bond_daily(bas_dd, use_sample=False)
            if not df.empty:
                jr_rows = df[df["ISU_NM"].str.contains("제이알글로벌|JR글로벌|제이알리츠", na=False, regex=True)]
                if not jr_rows.empty:
                    frames.append(jr_rows)
            time.sleep(CONFIG["REQUEST_INTERVAL_SEC"])

        if not frames:
            print("제이알글로벌리츠 관련 종목을 찾지 못했습니다. "
                  "ISU_NM 표기가 다를 수 있으니, 하루치 원본 데이터에서 종목명을 직접 확인해보세요 "
                  "(예: df['ISU_NM'].unique() 로 리츠 관련 종목명이 정확히 뭔지 확인).")
        else:
            result = pd.concat(frames, ignore_index=True)
            result["CLSPRC_YD"] = pd.to_numeric(result["CLSPRC_YD"], errors="coerce")
            result = result.sort_values("BAS_DD")
            out_path = "/mnt/user-data/outputs/jrglobal_validation.csv"
            result.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"{len(result)}행 수집, 저장 완료: {out_path}")
            print(result[["BAS_DD", "ISU_NM", "CLSPRC_YD", "ACC_TRDVOL"]].to_string(index=False))

    elif len(sys.argv) > 1 and sys.argv[1] == "validate-event":
        # 범용 이벤트 검증 명령. H5(제주항공)를 포함해 어떤 기업·이벤트에도 재사용 가능.
        # 사용법: python data_pipeline.py validate-event <종목명검색어> <시작일 YYYY-MM-DD> <종료일 YYYY-MM-DD> <출력파일명>
        # 예시(H5, 제주항공 무안공항 사고): 사고일(2024-12-29) 전후로 넉넉히 구간을 잡는다
        #   python data_pipeline.py validate-event 제주항공 2024-11-01 2025-03-01 jejuair_validation.csv
        if len(sys.argv) < 6:
            print("사용법: python data_pipeline.py validate-event <종목명검색어> <시작일> <종료일> <출력파일명>")
            print("예시: python data_pipeline.py validate-event 제주항공 2024-11-01 2025-03-01 jejuair_validation.csv")
        else:
            search_name = sys.argv[2]
            start_d, end_d, out_name = sys.argv[3], sys.argv[4], sys.argv[5]
            print(f"'{search_name}' 이벤트 검증 시작 ({start_d} ~ {end_d})...")
            dates = pd.bdate_range(start_d, end_d)
            frames = []
            for d in dates:
                bas_dd = d.strftime("%Y%m%d")
                df = fetch_krx_bond_daily(bas_dd, use_sample=False)
                if not df.empty:
                    rows = df[df["ISU_NM"].str.contains(search_name, na=False)]
                    if not rows.empty:
                        frames.append(rows)
                time.sleep(CONFIG["REQUEST_INTERVAL_SEC"])

            if not frames:
                print(f"'{search_name}' 관련 채권 종목을 찾지 못했습니다. "
                      f"해당 기업이 이 기간에 실제로 채권을 발행/거래했는지, ISU_NM 표기가 다른지 확인하세요.")
            else:
                result = pd.concat(frames, ignore_index=True)
                result["CLSPRC_YD"] = pd.to_numeric(result["CLSPRC_YD"], errors="coerce")
                result = result.sort_values("BAS_DD")
                out_path = f"/mnt/user-data/outputs/{out_name}"
                result.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"{len(result)}행 수집, 저장 완료: {out_path}")
                print(result[["BAS_DD", "ISU_NM", "CLSPRC_YD", "ACC_TRDVOL"]].to_string(index=False))

    else:
        run_sample_pipeline()
