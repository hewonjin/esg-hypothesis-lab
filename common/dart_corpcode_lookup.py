"""
DART corp_code 조회 헬퍼
=========================

DART Open API는 회사를 '고유번호(corp_code, 8자리)'로 식별한다.
이건 종목코드(ISU_CD)나 사업자등록번호와 다른, DART 자체 코드다.

전체 상장·외감기업의 corp_code 목록은 DART가 매일 갱신하는
corpCode.xml(zip) 파일로 제공하므로, 최초 1회 내려받아
'회사명 → corp_code' 매핑 테이블을 만들어두고 재사용하면 된다.

사용법
------
    pip install requests --break-system-packages

    python dart_corpcode_lookup.py --key 여기에_DART_API_KEY --search 제이알글로벌리츠

또는 파이썬에서 직접 불러써도 된다:

    from dart_corpcode_lookup import download_corpcode_table, find_corp_code

    df = download_corpcode_table(api_key)
    df.to_csv("corpcode_table.csv", index=False)  # 한 번 저장해두면 다음부턴 재다운로드 불필요

    find_corp_code(df, "제이알글로벌리츠")
"""

import io
import zipfile
import argparse
import xml.etree.ElementTree as ET

import requests
import pandas as pd


def download_corpcode_table(api_key: str) -> pd.DataFrame:
    """
    DART 고유번호 전체 목록(corpCode.xml)을 내려받아 DataFrame으로 반환한다.
    컬럼: corp_code, corp_name, stock_code, modify_date
    stock_code가 비어있으면 비상장(외감기업이지만 주식은 상장 안 된 경우)이다.
    """
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": api_key}

    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()

    # DART는 키가 잘못됐거나 요청이 실패하면 zip 대신 XML 에러 메시지를 돌려준다.
    # 여기서 미리 확인해 사람이 읽을 수 있는 에러로 바꿔준다.
    if not res.content.startswith(b"PK"):  # zip 파일은 항상 'PK'로 시작함
        try:
            err_root = ET.fromstring(res.content)
            status = err_root.findtext("status")
            message = err_root.findtext("message")
            raise RuntimeError(
                f"DART API 에러 (status={status}): {message}\n"
                f"→ status 010이면 인증키 오류(--key 값을 실제 발급받은 키로 교체했는지 확인), "
                f"020이면 사용한도 초과, 900이면 서비스 점검 중일 수 있습니다."
            )
        except ET.ParseError:
            raise RuntimeError(f"예상치 못한 응답을 받았습니다 (zip도 XML도 아님): {res.content[:200]}")

    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        xml_bytes = z.read(z.namelist()[0])

    root = ET.fromstring(xml_bytes)
    rows = []
    for item in root.findall("list"):
        rows.append({
            "corp_code": item.findtext("corp_code"),
            "corp_name": item.findtext("corp_name"),
            "stock_code": item.findtext("stock_code", "").strip(),
            "modify_date": item.findtext("modify_date"),
        })
    return pd.DataFrame(rows)


def find_corp_code(df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    """회사명에 부분일치하는 후보들을 반환한다 (정확히 하나로 안 좁혀질 수 있으므로 결과를 눈으로 확인할 것)."""
    matches = df[df["corp_name"].str.contains(company_name, na=False)]
    return matches[["corp_code", "corp_name", "stock_code"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="DART API 인증키")
    parser.add_argument("--search", required=True, help="찾을 회사명 (부분 검색 가능)")
    parser.add_argument("--cache", default="corpcode_table.csv", help="캐시 파일 경로")
    args = parser.parse_args()

    import os
    if os.path.exists(args.cache):
        print(f"[캐시 사용] {args.cache}")
        table = pd.read_csv(args.cache, dtype=str)
    else:
        print("[다운로드 중] DART 고유번호 전체 목록 (최초 1회만 오래 걸림)")
        table = download_corpcode_table(args.key)
        table.to_csv(args.cache, index=False, encoding="utf-8-sig")
        print(f"[저장 완료] {args.cache} ({len(table)}개 기업)")

    result = find_corp_code(table, args.search)
    if result.empty:
        print(f"'{args.search}'와(과) 일치하는 회사를 찾지 못했습니다.")
    else:
        print(result.to_string(index=False))
