# B2B 가맹점 정산리스크 스코어링 (b2b-settlement-risk-scoring)

B2B 가맹점을 대상으로 정산 시점의 리스크(미정산, 부도, 이상거래 등)를 사전에 예측하고 점수화하는 프로젝트입니다.

## 프로젝트 개요

- **목표**: 가맹점별 정산 관련 리스크를 스코어링하여 사전 관리 및 대응 체계를 마련
- **주요 대상**: B2B 가맹점 거래/정산 데이터
- **핵심 산출물**: 리스크 스코어 모델, 대시보드/리포트, API(선택)

## 팀원

| 이름 | 역할 | GitHub |
|------|------|--------|
| | | |
| | | |
| | | |

## 프로젝트 구조

```
b2b-settlement-risk-scoring/
├── data/
│   ├── raw/              # 원본 데이터 (git 추적 제외)
│   ├── interim/          # 전처리 중간 산출물
│   └── processed/        # 모델 학습용 최종 데이터
├── notebooks/            # EDA 및 실험용 Jupyter 노트북
├── src/
│   ├── data/              # 데이터 수집·전처리 스크립트
│   ├── features/          # 피처 엔지니어링
│   ├── models/            # 모델 학습·평가·추론
│   └── utils/             # 공통 유틸 함수
├── models/                # 학습된 모델 아티팩트 (git 추적 제외)
├── tests/                 # 단위 테스트
├── reports/               # 분석 리포트, 시각화 결과물
├── requirements.txt        # Python 의존성
├── .gitignore
└── README.md
```

## 개발 환경 설정

```bash
# 저장소 클론
git clone https://github.com/<org-or-user>/b2b-settlement-risk-scoring.git
cd b2b-settlement-risk-scoring

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 브랜치 전략

- `main`: 배포/최종 안정 버전
- `develop`: 통합 개발 브랜치
- `feature/기능명`: 개별 기능 개발 브랜치
- 작업 완료 후 `develop`으로 Pull Request 생성 → 리뷰 후 병합

## 커밋 컨벤션

```
feat: 새로운 기능 추가
fix: 버그 수정
data: 데이터 관련 작업
model: 모델 학습/튜닝 관련
docs: 문서 수정
refactor: 코드 리팩토링
test: 테스트 코드 추가/수정
chore: 빌드, 설정 등 기타 변경
```

## 라이선스

TBD
