# Implementation notes

Здесь фиксируются проверенные расхождения между `PROJECT_SPEC.md` и API
pinned upstream revisions.

## M0

- CUDA, LeRobot, SmolVLA и LIBERO ещё не импортируются: это намеренно, потому
  что exact Linux runtime и upstream API проверяются в M1.
- `uv.lock` M0 покрывает только project-owned config/test toolchain. Он будет
  расширен и повторно проверен на Linux GPU VM в M1.
- Локальная машина разработки — macOS с Python 3.13. Project environment
  создаётся `uv` на Python 3.12, но это не считается доказательством CUDA/EGL
  совместимости.
