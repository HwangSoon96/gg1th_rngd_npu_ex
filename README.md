# gg1th_rngd_npu_ex

FuriosaAI RNGD NPU 위에서 LLM 추론·서빙·성능 측정을 해 보는 실습 정리.

## 환경

- RNGD 카드가 장착된 Linux 서버
- Python 3.12, [uv](https://docs.astral.sh/uv/)
- `furiosa-llm`, `furiosa-smi`

```bash
uv sync
uv run python check_env.py
```

각 실습 폴더의 `.env.example`(또는 `.env.sample`)을 `.env`로 복사한 뒤 조별 NPU 장치 번호와 서버 포트를 채운다.

## 구성

| 폴더 | 내용 |
|---|---|
| `day1_npu_chat_test` | Qwen3-8B / Qwen3-VL-4B NPU 채팅, 메모리 예산 계산 |
| `day2_furiosa_sdk` | Furiosa LLM SDK 직접 추론 — 첫 추론, 배치 추론, 샘플링 실험 |
| `day3_npu_api` | `furiosa-llm serve` OpenAI 호환 서버 + Python 챗봇(스트리밍·멀티턴) |
| `day4_benchmark_test` | 동시성·프리픽스 캐시 벤치마크, 결과 시각화, Streamlit 대시보드 |
| `day4_fxb_ex` | FXB(컴파일된 NPU 실행 번들) CLI — 다운로드, 호환성 검사, 서빙 |

## 모델

- `furiosa-ai/Qwen3-8B-FP8` (DAY 1~4)
- `furiosa-ai/Qwen3-VL-4B-Instruct` (DAY 1)
- `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` + `furiosa-ai/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` FXB (DAY 4 FXB 실습)
