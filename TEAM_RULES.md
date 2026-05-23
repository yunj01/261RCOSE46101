# 🤝 Team Writing Rules — 3인 협업 가이드

> **목적**: 세 명이 동시에 논문 작업할 때 충돌·중복·일관성 문제 방지
> **대상**: 이윤제(P1) / 김상준(P2) / 원준서(P3)
> **장소**: Overleaf + GitHub(선택) + KakaoTalk(소통)

---

## 📌 황금률 (이것만 지켜도 90% 해결)

1. **수정 전 알리고, 수정 후 알린다** (카톡 한 줄)
2. **같은 섹션을 동시에 편집하지 않는다**
3. **숫자는 정해진 source에서만 가져온다** (`results/` 폴더)
4. **TBD는 그대로 두고 진행한다** (없는 데이터로 추측 X)
5. **각자 자기 섹션의 1차 책임자, 나머지는 review만**

---

## 1. 📝 글쓰기 규칙

### 1.1 언어
- **모든 본문은 영어**로 작성
- 한국어로 먼저 쓴 후 번역 OK, **단 최종은 영어로 자연스럽게**
- 한국어 코멘트는 `% [KO] 여기 수정 필요` 형식

### 1.2 인칭 / 시제
| 상황 | 사용 | 예 |
|------|------|----|
| 우리가 한 행동 | `We propose / We evaluate` | "We propose DALR..." |
| 실험 결과 보고 | **과거형** | "F achieved 61.79\%" |
| 일반적 주장/관찰 | **현재형** | "Small models lag behind in Korean" |
| 모델 동작 묘사 | **현재형** | "The model samples N chains and votes" |

### 1.3 약어 및 이름 (절대 통일)
| 정식 | 사용 |
|------|------|
| Difficulty-Aware Language Routing | **DALR** (첫 등장 시 풀네임) |
| Cross-Lingual Self-Consistency | **XLSC** |
| Cascade XLSC | **Cascade** 또는 **Cascade XLSC** |
| Chain-of-Thought | **CoT** |
| HRM8K | 항상 **HRM8K** (Hrm8k, hrm-8k X) |
| GSM8K | 항상 **GSM8K** |
| Qwen2.5-3B-Instruct | 본문 **Qwen2.5-3B**, 표 **Qwen2.5-3B** |
| Llama-3.2-3B-Instruct | 본문 **Llama-3.2-3B** |
| EXAONE-3.5-2.4B-Instruct | 본문 **EXAONE-3.5-2.4B** |
| Setup A/B/C/D/F | 항상 **대문자** (a, b, c X) |
| F_random | 항상 `F\_random` (LaTeX에선 `\_`) |

### 1.4 숫자 표기 (반드시 통일)
- 정확도: **소수점 2자리 + %** (예: `61.79\%`)
- 차이: **소수점 2자리 + points** (예: `+2.28 points`)
- p-value: **소수점 3자리** (예: `p = 0.030`)
- 신뢰구간: `[59.1, 64.4]` 형식
- 데이터 개수: **천 단위 콤마** (예: `1{,}319`, `7{,}473`)

❌ 잘못된 예: `61.8%`, `61.79 percent`, `0.6179`
✅ 올바른 예: `61.79\%`

### 1.5 인용
- 본문 중: `\citep{wei2022cot}` → "(Wei et al., 2022)"
- 주어 사용: `\citet{wei2022cot}` → "Wei et al. (2022)"
- 인용 전 무조건 `references.bib`에 등록 확인
- **BibTeX 키 명명**: `{firstauthorlastname}{year}{shortkeyword}`
  - 예: `wei2022cot`, `wang2022selfconsistency`

---

## 2. 🔢 숫자 / 결과 관리 (제일 자주 터지는 문제)

### 2.1 Single Source of Truth
**모든 숫자는 `results/` 폴더에서만 가져온다.**

| 결과 유형 | 파일 경로 |
|----------|----------|
| Setup A~F 단일 모델 | `results/setup_*_hrm8k.json`, `results/setup_*_gsm8k.json` |
| F_random ablation | `results/setup_f_random_*.json` |
| XLSC / Cascade | `results/xlsc_*.json` |
| Bootstrap CI | `results/statistical_tests.json` |
| McNemar p-value | `results/statistical_tests.json` |

### 2.2 숫자 인용 규칙
- 본문에 숫자 쓸 때 **반드시 출처 파일을 주석으로**:
  ```latex
  F achieved \textbf{61.79\%} on HRM8K.
  % source: results/setup_f_hrm8k.json
  ```
- 표에 들어가는 숫자도 마찬가지

### 2.3 숫자 업데이트 프로토콜
1. 새 결과가 나오면 → **카톡에 알림** + 어느 파일에 있는지
2. 본문/표에서 해당 숫자 업데이트
3. 영향받는 다른 섹션도 확인 (Abstract, Conclusion 등)
4. 카톡으로 "X 결과 업데이트 완료" 알림

### 2.4 TBD 처리
- 데이터 없으면 `\note{TBD}` 그대로 둠
- 추측/임시값 절대 금지
- 데이터 나오면 즉시 교체 + 알림

---

## 3. 📂 LaTeX / Overleaf 규칙

### 3.1 파일 구조
```
template.tex          ← 메인 파일 (모든 섹션 여기)
references.bib        ← 인용 (각자 추가)
neurips_2020.sty      ← 절대 수정 금지
figures/              ← 그림 (PNG/PDF)
```

### 3.2 섹션 책임자
| 섹션 | 1차 책임 | Reviewer |
|------|---------|----------|
| Abstract | 공동 (마지막에) | 모두 |
| 1. Introduction | P1 (이윤제) | P2, P3 |
| 2. Related Work | P1 초안 → 각자 자기 method 인용 추가 | 모두 |
| 3.1 Overview | P1 | P2, P3 |
| 3.2 DALR | P1 (이윤제) | P2, P3 |
| 3.3 XLSC | P2 (김상준) | P1, P3 |
| 3.4 Cascade | P2 (김상준) | P1, P3 |
| 3.5 Baselines | P3 (원준서) | P1, P2 |
| 4. Experiments | P3 (원준서) | P1, P2 |
| 5. Results 표 | P1 + P3 | 모두 |
| 5. Results 본문 | P1 (윤제: DALR) + P2 (상준: XLSC) | 모두 |
| 6. Analysis | P1 (DALR 분석) + P2 (XLSC 분석) | P3 |
| 7. Conclusion | 공동 | 모두 |
| Appendix | 각자 자기 contribution | — |
| References.bib | 인용 추가한 사람 | — |

### 3.3 동시 편집 방지
- **편집 시작 전 카톡 알림**: "Intro 수정합니다 (10분)"
- **편집 종료 후 카톡 알림**: "Intro 수정 끝"
- Overleaf의 실시간 협업은 같은 줄에 동시 입력 시 충돌 → 미리 알림 필수

### 3.4 노트 / TODO
- 작성 중인 TODO: `\note{TBD - XLSC 1319 결과 대기}`
- 다른 사람에게 질문: `\note{[P1->P2] 여기 XLSC 설명 보강 부탁}`
- 자기 검토 필요: `\note{[P1 self-check] 인용 맞는지 확인}`
- **제출 전 모든 `\note{}` 삭제 확인**

### 3.5 표/그림 관리
- 표: `\label{tab:main}`, `\label{tab:robustness}` 식으로 명확한 라벨
- 그림: `figures/` 폴더에 저장 후 `\includegraphics{figures/pipeline.pdf}`
- 그림 파일명: `fig_pipeline.pdf`, `fig_per_difficulty.pdf` (영어, 명확)

---

## 4. 💬 소통 규칙

### 4.1 일일 체크인 (선택)
- 매일 저녁 카톡으로 한 줄:
  - 오늘 한 거 / 내일 할 거 / 막힌 거

### 4.2 즉시 알려야 하는 것
- 결과 파일 새로 생김
- 결과 숫자 변경됨 (재실행 등)
- 섹션 편집 시작/종료
- TBD 채워짐
- Reviewer 부탁

### 4.3 카톡 메시지 양식
```
[알림] Intro 수정 시작 (P1, ~15분)
[완료] XLSC Qwen 1319 결과 나옴 → results/xlsc_qwen_hrm8k.json
[질문] Related Work에 STaR 인용 넣어도 될까?
[리뷰부탁] Section 3.3 1차 완료, 리뷰 부탁 (P2)
```

---

## 5. 🔍 Review 프로세스

### 5.1 작성 → 1차 review 사이클
```
1. 작성자: 자기 섹션 1차 완료
2. 작성자: `\note{[review 요청]}` 추가 + 카톡 알림
3. Reviewer: 24시간 안에 review
   - 의견은 `\note{[리뷰: 이름] 코멘트}` 로 인라인
4. 작성자: 코멘트 반영 후 `\note{}` 삭제
```

### 5.2 Review 체크리스트
Reviewer는 다음을 확인:
- [ ] 주장과 숫자가 일치하는가?
- [ ] 인용이 정확한가?
- [ ] 약어/표기 통일 (DALR, HRM8K 등)
- [ ] 문법/스펠링 (Grammarly 등 활용)
- [ ] 우리 narrative와 일관성 있는가?
- [ ] 다른 섹션과 중복되지 않는가?
- [ ] 표/그림 참조가 올바른가? (`Table~\ref{...}`)

### 5.3 최종 review (제출 1일 전)
- 셋이 모여 처음부터 끝까지 함께 읽기
- 모든 `\note{}` 제거 확인
- Compile 에러 없는지 확인
- PDF 페이지 수 확인 (8 페이지 이내)

---

## 6. ⏱️ 일정 / 마일스톤

### 6.1 단계별 deadline
| 단계 | 내용 | Deadline |
|------|------|----------|
| Day 1 | Outline / 분담 / template 갈아엎기 | \note{TBD} |
| Day 2-3 | 각자 초안 작성 (담당 섹션) | \note{TBD} |
| Day 4 | 1차 cross-review | \note{TBD} |
| Day 5 | XLSC/Cascade 결과 반영, 표 확정 | \note{TBD} |
| Day 6 | 그림 추가, 영어 다듬기 | \note{TBD} |
| Day 7 | 최종 합본 review + 제출 | \note{TBD} |

### 6.2 Buffer 원칙
- 모든 deadline은 **실제 제출 +24시간 buffer**
- 마지막 날 = review 전용, 작성 X

---

## 7. 🚨 자주 터지는 문제 + 예방

### 문제 1: 같은 숫자가 섹션마다 다름
- 원인: 각자 다른 시점 데이터로 작성
- **예방**: 모든 숫자 출처 주석 필수 (2.2 규칙)
- **해결**: `results/` 폴더의 최신 파일로 통일

### 문제 2: Overleaf 충돌
- 원인: 동시 같은 줄 편집
- **예방**: 편집 전 카톡 알림 (3.3 규칙)
- **해결**: history에서 복구 → 5분 단위로 자동 저장됨

### 문제 3: 인용 누락 / 중복
- 원인: bib 파일 관리 안 됨
- **예방**: 인용한 사람이 즉시 bib 추가
- **해결**: 제출 전 `\cite` 검색 → bib 모두 있는지 확인

### 문제 4: TBD 잊고 제출
- 원인: 막판 정신 없음
- **예방**: 제출 전 `\note{` 전체 검색 → 0개 확인
- **명령**: Overleaf에서 Ctrl+F → `\note{` 검색

### 문제 5: 방법 설명이 일관되지 않음
- 원인: 3명이 DALR/XLSC 설명을 미묘하게 다르게
- **예방**: Section 3 (Approach) 다 쓴 후 한 명이 통합 review
- **해결**: PROJECT_CONTEXT.md의 method 정의를 기준으로 통일

### 문제 6: Abstract와 본문 불일치
- 원인: 본문 수정 후 abstract 미반영
- **예방**: Abstract는 **마지막에** 작성 (모든 결과 확정 후)
- **해결**: 최종 review에서 Abstract ↔ Results 대조

---

## 8. ✅ 제출 전 최종 체크리스트

### 컨텐츠
- [ ] 모든 `\note{...}` 제거 (Ctrl+F로 확인)
- [ ] 모든 TBD 숫자 채워짐
- [ ] Abstract와 결과 일치
- [ ] Introduction의 contribution과 Conclusion 일치
- [ ] 모든 표/그림이 본문에서 참조됨
- [ ] 모든 `\cite{}`가 `.bib`에 있음 (반대도)
- [ ] 모든 약어 첫 등장 시 풀네임 함께

### 형식
- [ ] 페이지 수 1-8 (본문, References 제외)
- [ ] 영어 grammar 체크 (Grammarly / DeepL Write)
- [ ] LaTeX warning 0개
- [ ] PDF 폰트/줄간격 정상
- [ ] 저자 정보 정확 (이름, 학번, 팀번호)

### 제출
- [ ] PDF로 export
- [ ] 팀 1명이 BlackBoard에 업로드
- [ ] 팀원 모두에게 제출 완료 알림

---

## 9. 🔧 유용한 도구

| 용도 | 도구 |
|------|------|
| Grammar 체크 | Grammarly, DeepL Write |
| Citation 찾기 | Google Scholar → "BibTeX" |
| 표 만들기 | https://www.tablesgenerator.com/ |
| 그림 그리기 | draw.io, Excalidraw, TikZ |
| 한↔영 번역 | DeepL (Google Translate보다 자연스러움) |
| LaTeX 도움 | https://en.wikibooks.org/wiki/LaTeX |
| 동의어 찾기 | https://www.thesaurus.com/ |

---

## 10. 📂 파일 위치 한눈에

```
프로젝트 루트: C:\Users\tuni1\Desktop\nlp\korean_cot_distill\

협업 문서:
  PROJECT_CONTEXT.md       ← 프로젝트 전반 컨텍스트
  PAPER_OUTLINE.md         ← 논문 outline + 분담
  TEAM_RULES.md            ← (이 파일) 협업 규칙
  paper_draft_v1.tex       ← 논문 초안

논문 작업:
  Overleaf: https://ko.overleaf.com/project/6a1194cbcf24ba9b486c4734
  
결과 데이터:
  results/setup_*.json     ← 단일 모델 결과
  results/xlsc_*.json      ← XLSC/Cascade 결과
  results/statistical_tests.json  ← Bootstrap CI + McNemar
```

---

## 💡 마지막 한마디

> **완벽한 문장보다 빠른 소통이 중요하다.**
> 막히면 5분 안에 카톡, 결정 안 되면 셋이서 10분 미팅.
> 혼자 30분 끙끙대지 말 것.

---

*Last updated: 2026-05-20*
*All rules can be updated by team agreement. Just discuss in 카톡 first.*
