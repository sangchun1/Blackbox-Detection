# Blackbox Intentional Accident Detection

블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회 참가 프로젝트


## Project Structure

```text
├── configs/                    # Training / inference configs
├── src/
│   └── blackbox_detection/
│       ├── stage1/
│       ├── stage2/
│       ├── stage3/
│       └── utils/
├── notebooks/                  # EDA and experiments
├── scripts/                    # Training / inference / submission scripts
├── submissions/                # Submission zip files
└── outputs/                    # Experiment outputs
```

## Setup

```bash
pip install -e .
```

## Workflow

`main` 브랜치에는 직접 push하지 않고, 작업 전 최신 `main`을 받아 새로운 브랜치를 생성합니다.

```bash
git checkout main
git pull origin main
git checkout -b sangchun
```

작업 후 commit 및 push합니다.

```bash
git add .
git commit -m "add example feature"
git push origin sangchun
```

GitHub에서 `main` 브랜치를 대상으로 Pull Request를 생성합니다.

다른 팀원 1명의 승인을 받은 후 merge합니다.