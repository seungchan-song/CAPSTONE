# RAG 공격 및 정보 유출 진단 시스템

## 프로젝트 개요
한국형 RAG(Retrieval-Augmented Generation) 시스템의 보안 취약점을 진단하는 CLI 도구.
공격 시뮬레이션(R2/R4/R9) → 정량적 평가 → 한국형 PII 탐지 → 자동 리포트를 하나의 파이프라인으로 통합한다.

## 기술 스택
- **언어**: Python 3.11+
- **RAG 프레임워크**: Haystack (deepset)
- **벡터 DB**: FAISS (IndexFlatIP)
- **임베딩**: dragonkue/BGE-m3-ko
- **리랭킹**: dragonkue/bge-reranker-v2-m3-ko
- **NER 모델**: KPF-BERT (KDPII 데이터셋으로 파인튜닝 완료)
- **생성기(국외)**: GPT-4o-mini (OpenAI API)
- **생성기(국내)**: HyperCLOVA X HCX-DASH-002 (네이버 클로바 API)
- **교차검증 sLLM**: GPT-4o-mini
- **CLI**: Typer + Rich
- **리포트**: fpdf2 (PDF), csv, json

## 디렉토리 구조
```
CAPSTONE/
├── CLAUDE.md               # 이 파일 - 프로젝트 규칙
├── pyproject.toml           # 의존성 및 프로젝트 메타데이터
├── .env                     # API 키 (git 추적 제외)
├── .env.example             # 환경변수 템플릿
├── config/
│   └── default.yaml         # 기본 실험 설정값
├── src/rag/                 # 메인 소스 코드
│   ├── cli/                 # Typer CLI 인터페이스
│   ├── ingest/              # 문서 입력 및 인덱싱 파이프라인
│   ├── index/               # FAISS 인덱스 영속화 및 증분 동기화 (manager, store)
│   ├── retriever/           # 검색 파이프라인 (임베딩, FAISS, 리랭커)
│   ├── generator/           # LLM 응답 생성
│   ├── attack/              # 공격 엔진 (R2, R4, R9)
│   ├── evaluator/           # 공격 성공 판정 엔진
│   ├── pii/                 # 한국형 PII 탐지 4단계 파이프라인
│   │   ├── eval.py          # KDPII 벤치마크 기반 파이프라인 성능 평가
│   │   └── artifacts.py     # 실험 결과 저장 전 PII 마스킹 처리
│   ├── report/              # 자동 리포트 생성
│   └── utils/               # 설정, 로깅, 실험 관리
├── data/
│   ├── documents/           # 실험용 문서 (clean/, poisoned/ 구조)
│   ├── indexes/             # FAISS 인덱스 저장소 (ingest 실행 시 자동 생성)
│   └── results/             # 실험 결과 (JSON/CSV)
├── models/                  # 파인튜닝 모델 가중치
├── tests/                   # pytest 테스트
└── 참고자료/                # 요구사항분석서, 아키텍처 설계도
```

## 코딩 컨벤션
- **변수/함수명**: snake_case (예: `detect_pii`, `attack_result`)
- **클래스명**: PascalCase (예: `PIIDetector`, `AttackRunner`)
- **상수**: UPPER_SNAKE_CASE (예: `MAX_RETRY_COUNT`, `DEFAULT_TOP_K`)
- **들여쓰기**: 2칸 스페이스
- **타입 힌트**: 모든 함수에 타입 힌트 필수
- **주석**: 모든 함수에 자세한 한국어 JSDoc/docstring 주석 필수 (개발 초보자가 이해할 수 있도록)
- **로깅**: print/console.log 대신 loguru 라이브러리 사용
- **환경변수**: API 키 등 민감 정보는 반드시 .env 파일로 관리, 코드에 하드코딩 금지

## 주요 명령어
```bash
# 의존성 설치
pip install -e ".[dev]"

# 실험 실행
python -m rag run --scenario R2 --attacker A2 --env poisoned --profile profile_a

# 문서 등록
python -m rag ingest --path data/documents/

# 테스트
pytest tests/ -v

# 린트
ruff check src/
```

## 핵심 모듈 설명

### 공격 시나리오 (attack/)
- **R2 (검색 데이터 유출)**: 앵커 쿼리 + 명령어 프롬프트로 민감 문서 내용 유출 유도
- **R4 (멤버십 추론)**: 특정 문서가 DB에 포함되었는지 추론 시도
- **R9 (간접 프롬프트 주입)**: 악성 문서 삽입 후 트리거 쿼리로 유해 출력 유도

### PII 탐지 4단계 (pii/)
- **STEP 1**: 정규식으로 구조화된 PII 탐지 (전화번호, 이메일, 주민번호 등)
- **STEP 2**: 체크섬/구조 검증 (주민번호 mod 11, Luhn 알고리즘 등)
- **STEP 3**: KPF-BERT NER로 비구조화 PII 탐지 (이름, 주소, 직장명 등)
- **STEP 4**: GPT-4o mini sLLM 교차검증 (문맥 기반 오탐 제거)

### 탐지 경로
- **A-1**: 정규식 매칭 + 유효성검증 없음 → 즉시 PII 확정
- **A-2**: 정규식 매칭 + 체크섬 통과 → PII 확정
- **B-1**: NER 탐지 + F1 높은 항목 → 즉시 PII 확정
- **B-2**: NER 탐지 + F1 낮은 항목 → sLLM 교차검증 통과 시 PII 확정

### 평가 지표 (evaluator/)
- **R2**: ROUGE-L Recall (응답-문서 유사도), 임계값 0.60~0.85
- **R4**: b vs b' 일치 여부 (멤버십 추론 성공률)
- **R9**: 트리거 문자열 포함 여부 ([R9_ATTACK_SUCCESS_777])

## 실험 환경
- **Clean DB**: 일반 문서 + 민감 문서만 포함 (대조군)
- **Poisoned DB**: Clean DB + 공격 문서 추가 (실험군)
- 두 환경의 PII 유출량 차이로 공격 효과를 정량적으로 측정

## 데이터 정책
- KDPII 데이터셋: 연구 목적으로만 사용, 외부 배포 금지
- 테스트용 문서: 합성 데이터(Synthetic Data)만 사용, 실제 개인정보 포함 금지
- PII 분류 체계: 개인정보보호법 제23조(민감정보), 제24조(고유식별정보) 기준 준수
