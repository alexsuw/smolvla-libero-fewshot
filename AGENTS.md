# Agent instructions

1. Полностью прочитай `PROJECT_SPEC.md` перед изменениями.
2. Реализуй только следующий незавершённый пункт из `Explicit TODO order`.
3. Не запускай paid/long GPU training без выполненных gates M0–M5.
4. После milestone запусти acceptance checks, сохрани evidence в
   `artifacts/validation/<milestone>/`, обнови `STATUS.md` и сделай небольшой
   осмысленный commit.
5. Не удаляй datasets, checkpoints, runs, videos или remote artifacts.
6. Не добавляй в Git secrets, runtime data, model weights или файлы больше
   10 MB.
7. Не используй target success для выбора hyperparameters или seen checkpoint.
8. Проверяй pinned upstream API по source/`--help`; не имитируй отсутствующий
   API. Расхождения записывай в `docs/IMPLEMENTATION_NOTES.md`.
9. Сохраняй platform-independent code: `/content`, `/mnt/vla`, usernames,
   hosts и bucket names допустимы только в user-owned environment/setup, но
   не в production Python code.
10. Перед optimizer creation fail-closed проверяй trainable parameter allowlist.
