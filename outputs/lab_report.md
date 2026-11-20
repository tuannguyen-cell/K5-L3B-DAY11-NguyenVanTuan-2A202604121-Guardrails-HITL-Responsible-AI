# Lab 11 — Auto Report

> File này **tự sinh** bởi `scripts/grade.py`. **Không** viết / sửa tay.

- Generated (UTC): `2026-11-20T10:17:52.118212+00:00`
- Framework: `openai-sdk with ADK-style plugins`
- Technical failure: **False**

## Packaging

| File | Status |
|------|--------|
| results.json | OK |
| attack_results.json | OK |
| audit_log.json | OK |
| metrics.json | OK |

## Schema (`results.json`)

- Valid: **True**
- Error: `None`

## Defense snapshot (từ `results.json`)

- Blue provider / model: `liquid` / `lfm3-1.7b`
- Safe queries blocked: `0/5`
- Attack queries blocked: `7/7`
- Edge cases blocked: `1/3`
- Rate limit blocked/sent: `2/12`

## Red Team snapshot (từ `attack_results.json`)

- Provider / model: `gemini` / `gemini-3.5-flash-lite`
- Unsafe leaks (Red): `5/5`
- Guards leaks (Red Advance): `0/5`

## Public tests

- Return code: `0`
- Technical failure: `False`

```text
........................................................................ [ 75%]
........................                                                 [100%]
96 passed in 2.91s
```

## Notes

- Artifact chấm chính: `outputs/results.json` + `outputs/attack_results.json`.
- Bonus B1/B2 do grader replay quyết định — JSON chỉ là bằng chứng.
- Không nộp `report/*.md` viết tay; dùng file này nếu cần xem tóm tắt.
