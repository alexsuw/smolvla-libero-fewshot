# PROJECT_SPEC — Few-shot adaptation of SmolVLA on LIBERO

> Статус документа: исполняемая спецификация проекта для coding-агента Codex/Cursor.
>
> Основной язык пояснений: русский. Имена файлов, конфигурационных ключей, команд, метрик и артефактов: English.
>
> Дата фиксации постановки: 2026-08-20.

## 0. Инструкция coding-агенту

Этот файл — не обзор и не список идей. Его нужно использовать как основной технический контракт проекта. Реализуй проект по milestones и в порядке из раздела `Explicit TODO order`. После каждого milestone:

1. запусти указанные проверки;
2. сохрани evidence в `artifacts/validation/<milestone>/`;
3. обнови `STATUS.md` короткой записью: что готово, какие команды прошли, где лежат артефакты;
4. сделай небольшой осмысленный commit;
5. не переходи к дорогому GPU-этапу, пока acceptance criteria текущего milestone не выполнены.

Если фактический API pinned-версии LeRobot отличается от примеров в этом документе, не имитируй несуществующий API. Сначала проверь `--help`, исходники и конфигурационные dataclass этой pinned-версии, затем реализуй тонкий project-owned wrapper с тем же пользовательским интерфейсом, который задан здесь. Запиши расхождение в `docs/IMPLEMENTATION_NOTES.md`.

### 0.1 Неподлежащие изменению требования

- Model origin: `lerobot/smolvla_base` с зафиксированной полной revision SHA.
- Dataset origin: `nvidia/LIBERO_LeRobot_v3`.
- Seen dataset: только `libero_90`.
- Target dataset: только указанные ниже три task text из `libero_goal`.
- Few-shot budgets: `N = 5, 10, 25`; subset для каждого N — префикс одного и того же упорядоченного списка episode IDs.
- Final adaptation training seeds: `42` и `123`.
- Final evaluation: не менее `20` rollout на каждый `method × task × N × train_seed`; для zero-shot — не менее `20` rollout на target task.
- Все target-adaptation runs стартуют независимо из одного immutable `seen_checkpoint`; `N=10` не продолжает `N=5`, `N=25` не продолжает `N=10`.
- Нельзя использовать target rollouts или target success для выбора seen checkpoint, learning rate, LoRA rank, количества steps, replay ratio или иных hyperparameters.
- W&B выключен. Обязательное логирование: TensorBoard + CSV/JSONL + plain-text logs.
- Colab используется для smoke tests, а не как единственное место хранения ценных данных.
- Финальные длительные runs запускаются как CLI-процессы на Linux GPU VM под `tmux`, а не из notebook cell и не из временного IDE terminal.
- Ни один полезный checkpoint, result, manifest или failure video не должен существовать в единственном экземпляре на ephemeral disk.
- Никаких удалений datasets/checkpoints/runs из training/evaluation code. Очистка — отдельная dry-run-first команда с явным подтверждением.

### 0.2 Что не является целью

- Не использовать reinforcement learning для основной части. Обучение — imitation learning / behavior cloning по expert actions; simulator success применяется только для evaluation.
- Не строить reward model до завершения основной cost curve.
- Не оптимизировать pipeline под один Colab notebook.
- Не использовать `HuggingFaceVLA/libero` или произвольный публичный LIBERO checkpoint для финальных результатов: их provenance может включать held-out tasks. Их можно применять только в отдельном `dev_only` режиме для отладки, с большим предупреждением в manifest.
- Не загружать secrets, datasets, checkpoints, videos и runtime caches в Git.

---

## 1. Цель и научная постановка

### 1.1 Практическая цель

Построить воспроизводимый pipeline, который:

1. адаптирует `smolvla_base` к домену LIBERO на seen-части `libero_90`;
2. измеряет zero-shot transfer на трёх held-out задачах;
3. строит naive few-shot baseline на `5/10/25` demonstrations;
4. позволяет сравнить baseline с parameter-efficient/replay adaptation;
5. показывает success-vs-demonstrations cost curve с uncertainty;
6. сохраняет checkpoints, manifests, per-rollout результаты, action traces и видео ошибок;
7. запускается одним и тем же CLI-интерфейсом в Colab Free и на Linux GPU VM, меняя только paths/device config.

### 1.2 Исследовательский вопрос

Главный вопрос:

> Как уменьшить число target demonstrations, необходимое SmolVLA для достижения заданного success rate на новой language-conditioned manipulation task после domain adaptation на LIBERO-90?

Главный объект сравнения — не только максимальный success при `N=25`, а форма cost curve:

\[
N \in \{0, 5, 10, 25\} \longrightarrow \text{success rate}.
\]

Сильный результат имеет вид:

\[
S_{method}(5) \approx S_{baseline}(10)
\]

или аналогичный сдвиг кривой влево. Если success одинаков, дополнительно сравнивать trainable parameters, GPU-hours и wall-clock, но не подменять ими основную sample-efficiency метрику.

### 1.3 Предварительная гипотеза

До первых финальных target runs создать и commit-нуть `predictions.md` со следующей falsifiable hypothesis:

> Ожидается наибольший прирост success между `N=0` и `N=5`, меньший между `N=5` и `N=10` и насыщение между `N=10` и `N=25`. Parameter-efficient adaptation с seen replay должна иметь наибольшее преимущество при `N=5`, где naive target-only fine-tuning наиболее подвержен overfitting и forgetting; к `N=25` разрыв должен уменьшаться.

Commit с `predictions.md` должен предшествовать commit с любыми финальными target results.

### 1.4 Обучающий сигнал

SmolVLA получает:

\[
o_t = (I_t^{main}, I_t^{wrist}, s_t, L)
\]

и предсказывает continuous action chunk:

\[
\hat A_t = (\hat a_t, \hat a_{t+1}, \ldots, \hat a_{t+H-1}).
\]

Training target — записанный expert action chunk из demonstrations. Использовать штатный SmolVLA flow-matching/imitation objective pinned-версии LeRobot. Не заменять его MSE без отдельной ablation и не вводить reward в основной training loop.

---

## 2. Model contract

### 2.1 Базовая модель

- Hub model: `lerobot/smolvla_base`.
- Starting pinned revision: `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- Ожидаемый размер: приблизительно 450M parameters; фактические `total_parameters` и `trainable_parameters` всегда вычислять из загруженного checkpoint и писать в manifest.
- Входы:
  - main RGB camera;
  - wrist RGB camera;
  - 8D robot state;
  - exact natural-language instruction.
- Выход:
  - action chunk, где один LIBERO action имеет размерность `7`.

### 2.2 Seen-pretrain trainable scope

Primary recipe:

- freeze vision encoder;
- freeze pretrained VLM/language backbone;
- train full Action Expert;
- train state projection;
- train action input/output/time projections, если они являются отдельными модулями в pinned implementation;
- не размораживать другие VLM weights молча.

До optimizer creation код обязан:

1. вывести все trainable parameter name prefixes;
2. записать полный список в `trainable_parameters.txt`;
3. вывести total/trainable counts и percentage;
4. проверить allowlist trainable module patterns;
5. упасть с ошибкой, если trainable scope шире или уже ожидаемого.

Project config должен описывать намерение, а model adapter преобразовывать его в фактические flags pinned LeRobot:

```yaml
trainable_scope:
  freeze_vision_encoder: true
  freeze_vlm_backbone: true
  train_action_expert: true
  train_state_projection: true
  train_action_projections: true
  strict_allowlist: true
```

### 2.3 Optional seen LoRA challenger

После primary seen run допускается один challenger на том же полном `libero_90`:

- frozen base backbone;
- LoRA rank `64`, `lora_alpha=64` как starting recipe;
- starting LR `1e-3`, scheduled target LR `1e-4`;
- target modules — фактические current LeRobot defaults для SmolVLA (`q_proj`, `v_proj` LM expert и task-dependent state/action projections), но resolved список модулей обязательно сохранить;
- data, effective batch, step budget, evaluation tasks и seeds должны совпадать с primary seen run насколько возможно.

Этот run является challenger, а не обязательным условием запуска target baseline. Выбирать между primary seen checkpoint и LoRA challenger можно только по заранее указанным seen probes, stability и compute — никогда по real target tasks.

---

## 3. Dataset и точный split

### 3.1 Dataset revision

Использовать:

```text
repo_id: nvidia/LIBERO_LeRobot_v3
repo_type: dataset
revision: e5907374380b8f96511957e6ba5582be52a1e179
```

Нельзя использовать `main` в финальном run. Если эта revision недоступна, остановиться, задокументировать проблему и получить явное решение; не подменять dataset другим репозиторием.

Проверенные свойства этой revision:

| Suite | Episodes | Frames | Unique task texts | FPS |
|---|---:|---:|---:|---:|
| `libero_90` | 3,921 | 569,249 | 73 | 20 |
| `libero_goal` | 428 | 52,042 | 10 | 20 |

`libero_90` называется LIBERO-90 исторически, но данная conversion metadata содержит 73 task IDs. Использовать фактические metadata и не assert-ить число 90.

### 3.2 Фактическая схема данных

Ожидаемые поля:

```text
action                              float[7]
observation.state                   float[8]
observation.images.image            RGB 256x256
observation.images.wrist_image      RGB 256x256
timestamp
frame_index
episode_index
task_index
index
```

Формат LeRobotDataset v3:

```text
<suite>/
├── data/chunk-*/file-*.parquet
├── meta/info.json
├── meta/stats.json
├── meta/tasks.parquet
├── meta/episodes/chunk-*/file-*.parquet
└── videos/
    ├── observation.images.image/chunk-*/file-*.mp4
    └── observation.images.wrist_image/chunk-*/file-*.mp4
```

Metadata inspection должна выполняться без декодирования всех videos.

### 3.3 Exact train/test split

#### Seen training

```text
suite: libero_90
episodes: all 3,921 episodes from the pinned revision
```

До training выполнить programmatic leakage check: ни один exact target text из таблицы ниже не должен находиться среди `libero_90` task texts. На pinned revision проверенный результат — все три `false`.

#### Held-out target tasks

| Slug | Exact task text | `libero_goal` task index | Available episodes |
|---|---|---:|---:|
| `drawer_middle` | `open the middle drawer of the cabinet` | 9 | 43 |
| `bowl_stove` | `put the bowl on the stove` | 7 | 48 |
| `wine_cabinet` | `put the wine bottle on top of the cabinet` | 4 | 47 |

Task matching — только normalized exact text matching: trim outer whitespace, normalize Unicode and collapse repeated internal whitespace; не делать fuzzy/substring matching. Сохранить и raw, и normalized text.

#### Exact episode prefixes

Создать tracked файл `configs/splits/target_splits.json` со следующим содержимым. В runtime дополнительно проверить эти IDs по pinned metadata; нельзя слепо доверять файлу.

```json
{
  "schema_version": 1,
  "dataset_repo_id": "nvidia/LIBERO_LeRobot_v3",
  "dataset_revision": "e5907374380b8f96511957e6ba5582be52a1e179",
  "suite": "libero_goal",
  "selection_rule": "first N episode_index values in ascending dataset order for exact task text",
  "tasks": {
    "drawer_middle": {
      "task_text": "open the middle drawer of the cabinet",
      "task_index": 9,
      "available_count": 43,
      "episode_ids_first_25": [20, 26, 31, 42, 58, 64, 84, 90, 94, 111, 117, 118, 137, 140, 146, 162, 168, 173, 182, 187, 198, 206, 220, 232, 252]
    },
    "bowl_stove": {
      "task_text": "put the bowl on the stove",
      "task_index": 7,
      "available_count": 48,
      "episode_ids_first_25": [13, 15, 16, 22, 36, 45, 66, 76, 116, 121, 145, 151, 165, 166, 171, 178, 179, 186, 201, 219, 225, 233, 237, 239, 250]
    },
    "wine_cabinet": {
      "task_text": "put the wine bottle on top of the cabinet",
      "task_index": 4,
      "available_count": 47,
      "episode_ids_first_25": [6, 10, 17, 18, 25, 38, 40, 44, 48, 51, 53, 57, 75, 89, 91, 96, 97, 100, 101, 103, 133, 136, 149, 154, 164]
    }
  }
}
```

Для каждого task:

```text
N=5  := episode_ids_first_25[:5]
N=10 := episode_ids_first_25[:10]
N=25 := episode_ids_first_25[:25]
```

Subset nesting является обязательным invariant:

```python
assert ids_5 == ids_10[:5]
assert ids_10 == ids_25[:10]
assert len(set(ids_25)) == 25
```

### 3.4 No-target-leakage policy

Real target tasks разрешено использовать только так:

- task text известен policy во время target evaluation и adaptation;
- zero-shot использует `0` target demonstrations;
- target adaptation task X использует только первые N demonstrations этой же task X;
- wrong-language control использует только строку другой инструкции, но не её images/states/actions;
- финальная evaluation использует simulator initial states/seeds, но не expert actions;
- target success не используется для hyperparameter selection или early stopping.

Запрещено:

- добавлять любые `libero_goal` demonstrations в seen training;
- подмешивать demonstrations одной held-out task при adaptation другой;
- использовать remaining target episodes (`N+1...`) как validation set;
- смотреть target success для выбора seen checkpoint;
- подбирать LR/steps/rank/replay ratio по трем real targets;
- использовать checkpoint с неясным LIBERO provenance в финальной таблице;
- вычислять normalization statistics на всех `libero_goal` episodes вместо выбранного train subset.

Hyperparameter development выполнять на `pseudo_target` tasks, выбранных только внутри `libero_90`. Их список, selection rule и все decisions commit-нуть до real target grid.

`scripts/verify_no_leakage.py` должен завершаться non-zero exit при любом нарушении и запускаться автоматически перед `train_seen`, `train_target` и final aggregation.

---

## 4. Observation/action conventions и gripper

### 4.1 Canonical model-facing observations

После environment preprocessing policy должна видеть:

```text
observation.images.image       main / agentview camera
observation.images.wrist_image wrist / robot0_eye_in_hand camera
observation.state              8D
task                           exact instruction text
```

8D state:

```text
[eef_x, eef_y, eef_z, axis_angle_x, axis_angle_y, axis_angle_z, gripper_qpos_0, gripper_qpos_1]
```

Pinned LeRobot environment может называть wrist image `observation.images.image2`. Project-owned adapter обязан сделать mapping явным и записать его в manifest. Нельзя зависеть от неявного совпадения keys.

### 4.2 Image orientation parity

Не предполагать 180° flip по памяти или по другому dataset. Реализовать `scripts/check_observation_parity.py`, который рядом сохраняет:

- dataset main frame;
- env main frame из соответствующего initial state;
- dataset wrist frame;
- env wrist frame;
- candidate transformed variants.

Человек должен подтвердить parity один раз, а test должен зафиксировать выбранный transform. Rotation/flip допускается ровно в одном processor. Любой double flip — fatal configuration error.

### 4.3 Action convention

LIBERO action:

```text
[delta_x, delta_y, delta_z, delta_axis_angle_x, delta_axis_angle_y, delta_axis_angle_z, gripper]
```

Первые 6 channels передавать в том control mode, которому соответствует dataset/checkpoint. В final configs явно записать:

```yaml
env:
  control_mode: relative
action:
  dim: 7
```

Если inspection докажет иной mode, изменить config и документировать evidence; не угадывать.

### 4.4 Gripper conversion

В pinned NVIDIA dataset:

```text
g_dataset = 0 -> closed
g_dataset = 1 -> open
```

LIBERO environment convention в данной постановке:

```text
g_env = +1 -> close
g_env = -1 -> open
```

Обязательное преобразование:

\[
g_{env} = 1 - 2g_{dataset}.
\]

Default design:

- dataset и normalization stats остаются в исходном dataset space `[0, 1]`;
- training targets используют dataset space;
- после policy unnormalization и непосредственно перед `env.step`, gripper переводится в env space;
- expert replay использует тот же единственный postprocessor;
- optional thresholding policy output задаётся config, default `0.5`:
  - `g_dataset < 0.5 -> +1`;
  - `g_dataset >= 0.5 -> -1`.

Не переписывать raw dataset. Если позже создаётся derived env-space dataset, он должен иметь новый dataset ID, новый manifest и полностью пересчитанные statistics; смешивать его с original stats запрещено.

Unit tests:

```python
assert dataset_gripper_to_env(0.0) == 1.0
assert dataset_gripper_to_env(1.0) == -1.0
assert dataset_gripper_to_env(0.5) == 0.0  # continuous formula test
```

Отдельно test для binary runtime postprocessor. В action trace сохранять и `policy_action_dataset_space`, и `env_action`.

### 4.5 Expert replay gate

До любого обучения выполнить прямой replay нескольких expert trajectories:

```bash
python -m scripts.replay_expert \
  --config configs/platform/colab.yaml \
  --task bowl_stove \
  --episode-id 13 \
  --save-video
```

Replay должен использовать dataset actions, observation/environment mapping и gripper conversion production-кода. Нельзя писать отдельный «подогнанный» replay path.

Минимальный gate:

- по одному episode каждой target task;
- минимум три diverse seen episodes;
- expected simulator success для корректной expert trajectory — `1`;
- видео и action trace сохранены;
- gripper visibly closes/opens в правильные моменты;
- no NaN/out-of-range actions.

Если target expert replay неуспешен из-за mismatch initial state или simulator version, training запрещен до выяснения причины. Не ослаблять gate простым удалением assert.

---

## 5. Evaluation protocol

### 5.1 Главная метрика

Per rollout:

```text
success ∈ {0, 1}
```

Per cell:

```text
cell = method × task × n_demos × train_seed
success_rate = successes / n_rollouts
```

Final cell содержит минимум `20` rollouts. Для каждой proportion показывать Wilson 95% CI. Для агрегирования по tasks/seeds дополнительно применять bootstrap, который не делает отдельные frames независимыми observations. В отчёте всегда указывать denominator.

### 5.2 Fixed final evaluation seeds

Создать tracked `configs/eval/final_seeds.json` с 20 seed values. Starting set:

```json
[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009,
 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019]
```

Одинаковые `task × eval_seed` initial states использовать для baseline, challenger и language controls. Сохранить fingerprint initial observation/state, чтобы обнаружить drift.

Final evaluation:

- `hard_reset: true`;
- fixed initial states, если pinned integration это поддерживает;
- один и тот же max horizon;
- один и тот же action chunk execution horizon;
- один и тот же gripper postprocessor;
- deterministic inference where supported;
- никаких updates model weights или normalization stats.

Soft reset разрешен только для dev evaluation и помечается `protocol=dev_soft_reset`; такие результаты не входят в final tables.

### 5.3 Zero-shot

После freezing final seen checkpoint:

```text
N = 0
checkpoint = immutable seen checkpoint
target demonstrations used = none
20 rollouts per target task
```

Zero-shot считается один раз на final seen checkpoint, а не дублируется как будто это отдельный train seed. Его checkpoint hash должен совпадать во всех трех target records.

### 5.4 Language control

Для каждого target task и каждого из тех же final eval seeds запустить paired rollouts:

- `correct_instruction`: exact task text;
- `wrong_instruction`: cyclically selected other target text;
- scene, initial state, checkpoint и seed одинаковы;
- меняется только instruction string.

Default cyclic mapping:

```yaml
drawer_middle: put the bowl on the stove
bowl_stove: put the wine bottle on top of the cabinet
wine_cabinet: open the middle drawer of the cabinet
```

Сохранять:

- success;
- end-effector trajectory;
- gripper trace;
- action L2/cosine divergence after alignment;
- first-interaction object, если доступно;
- paired videos с одинаковым naming.

Нельзя интерпретировать просто низкий wrong-instruction success как достаточное доказательство language control: дополнительно сравнивать trajectory divergence и qualitative behavior.

### 5.5 Failure videos

Каждый unsuccessful final rollout обязан иметь video. Default policy:

```text
save every failure video
save first successful video per cell
save action/state trace for every rollout
```

Чтобы не хранить сотни лишних success videos, frames можно буферизовать на scratch и кодировать после определения outcome. Failure video никогда не удалять автоматически.

---

## 6. Общая вычислительная стратегия

### 6.1 Неизменяемый интерфейс между платформами

Код, configs и entry points одинаковы в Colab и VM. Меняются только platform/storage overlay:

```bash
python -m scripts.doctor --config configs/platform/colab.yaml
python -m scripts.doctor --config configs/platform/gpu_vm.yaml
```

Ни один training script не должен содержать hard-coded `/content`, `/mnt/vla`, username, host name или bucket.

### 6.2 Colab Free: только обязательные smoke tests

Colab checklist:

1. clone repo at exact Git commit;
2. run `scripts/bootstrap_colab.sh`;
3. mount Google Drive только для durable smoke artifacts либо настроить object storage;
4. download metadata + минимально нужные episodes/videos;
5. run dataset inspection;
6. launch LIBERO headless and reset/step;
7. pass observation parity check;
8. pass expert replay;
9. load `smolvla_base` at pinned revision;
10. run one forward pass and one environment step;
11. train `100` steps;
12. save checkpoint;
13. terminate process/runtime;
14. resume `100 -> 200` steps;
15. evaluate/load resulting checkpoint;
16. sync smoke manifest/log/checkpoint to durable storage.

Не запускать full `libero_90` training в бесплатном Colab. Notebook должен быть thin launcher; вся логика живет в importable modules/CLI.

### 6.3 GPU VM: final compute

Recommended minimum practical shape:

```text
Linux Ubuntu 22.04/24.04
1 NVIDIA GPU, preferably >=24 GB VRAM; 16 GB supported via small physical batch
16+ vCPU
64+ GB RAM
200+ GB persistent SSD
public IP or private VPN/SSH access
```

Exact GPU не кодировать в project assumptions. Physical batch auto-tune отдельно, но effective batch фиксировать config и gradient accumulation.

VM checklist перед арендой/дорогим run:

- все Colab smoke gates green;
- exact resume proven;
- object storage credentials and dry-run sync proven;
- disk capacity check passes;
- dataset revision cached on persistent volume;
- training command recorded in experiment plan;
- no target leakage check green.

---

## 7. Repository structure

Создать следующую структуру. Допустимы небольшие изменения имен, но ответственность модулей сохранять.

```text
smolvla-libero-fewshot/
├── AGENTS.md
├── PROJECT_SPEC.md
├── README.md
├── STATUS.md
├── predictions.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── Makefile
│
├── configs/
│   ├── revisions.lock.yaml
│   ├── data.yaml
│   ├── storage.example.yaml
│   ├── platform/
│   │   ├── colab.yaml
│   │   └── gpu_vm.yaml
│   ├── splits/
│   │   ├── target_splits.json
│   │   └── pseudo_target_splits.json
│   ├── train/
│   │   ├── smoke.yaml
│   │   ├── seen_expert.yaml
│   │   ├── seen_lora.yaml
│   │   ├── target_baseline.yaml
│   │   ├── target_lora.yaml
│   │   └── target_replay_lora.yaml
│   └── eval/
│       ├── seen_probe.yaml
│       ├── zero_shot.yaml
│       ├── language_control.yaml
│       ├── final.yaml
│       └── final_seeds.json
│
├── src/vla_fewshot/
│   ├── __init__.py
│   ├── config.py
│   ├── paths.py
│   ├── reproducibility.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── inspection.py
│   │   ├── splits.py
│   │   ├── subset.py
│   │   ├── leakage.py
│   │   └── task_text.py
│   ├── env/
│   │   ├── libero_env.py
│   │   ├── observation_adapter.py
│   │   ├── action_adapter.py
│   │   ├── gripper.py
│   │   └── replay.py
│   ├── model/
│   │   ├── smolvla.py
│   │   ├── freezing.py
│   │   └── peft.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── checkpoint.py
│   │   ├── resume.py
│   │   ├── sampler.py
│   │   └── replay_mixer.py
│   ├── evaluation/
│   │   ├── rollout.py
│   │   ├── protocol.py
│   │   ├── language_control.py
│   │   ├── video.py
│   │   └── metrics.py
│   ├── logging/
│   │   ├── csv_logger.py
│   │   ├── tensorboard.py
│   │   ├── manifest.py
│   │   └── registry.py
│   ├── storage/
│   │   ├── layout.py
│   │   ├── checksums.py
│   │   ├── sync.py
│   │   └── retention.py
│   └── reporting/
│       ├── aggregate.py
│       ├── intervals.py
│       ├── plots.py
│       └── tables.py
│
├── scripts/
│   ├── bootstrap_colab.sh
│   ├── bootstrap_vm.sh
│   ├── doctor.py
│   ├── resolve_revisions.py
│   ├── download_dataset.py
│   ├── inspect_dataset.py
│   ├── verify_split.py
│   ├── verify_no_leakage.py
│   ├── materialize_subset.py
│   ├── check_observation_parity.py
│   ├── replay_expert.py
│   ├── smoke_inference.py
│   ├── train_seen.py
│   ├── train_target.py
│   ├── eval_seen.py
│   ├── eval_target.py
│   ├── eval_language_control.py
│   ├── verify_checkpoint.py
│   ├── sync_artifacts.py
│   ├── build_registry.py
│   ├── collect_results.py
│   ├── plot_cost_curve.py
│   ├── make_report_tables.py
│   └── prune_artifacts.py
│
├── notebooks/
│   ├── colab_smoke.ipynb
│   ├── dataset_inspection.ipynb
│   ├── training_curves.ipynb
│   └── failure_analysis.ipynb
│
├── experiments/
│   ├── plan.yaml
│   ├── decisions.md
│   └── registry.csv
│
├── docs/
│   └── IMPLEMENTATION_NOTES.md
│
├── report/
│   ├── figures/
│   ├── tables/
│   ├── failure_cases.md
│   └── report.md
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
│
├── artifacts/          # gitignored; local validation evidence
└── third_party/
    └── README.md       # records pinned upstream source layout; no floating copies
```

`outputs/`, datasets, checkpoints, Hub caches and videos должны находиться за пределами Git worktree на persistent storage. В repo допускаются только маленькие derived tables/plots/report assets, если это явно решено.

---

## 8. Environment и version pinning

### 8.1 Supported runtime

Primary runtime:

```text
OS: Linux x86_64
Python: 3.12
MuJoCo rendering: EGL on headless machine
```

В shell:

```bash
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
```

На multi-GPU host дополнительно задавать `CUDA_VISIBLE_DEVICES`, но не зашивать его в code.

### 8.2 Pinning policy

Не использовать floating `main`, `latest` или unbounded dependency specifiers в final environment.

`configs/revisions.lock.yaml` должен содержать:

```yaml
schema_version: 1
snapshot_date: "2026-08-20"
python: "3.12.x"  # replace x with the exact tested patch before M1 is complete
dataset:
  repo_id: nvidia/LIBERO_LeRobot_v3
  revision: e5907374380b8f96511957e6ba5582be52a1e179
  suite_seen: libero_90
  suite_target: libero_goal
model:
  repo_id: lerobot/smolvla_base
  revision: c83c3163b8ca9b7e67c509fffd9121e66cb96205
source:
  lerobot_git: d451fe4f1f1b00a812f95aa9534389b5e42ab155
  # The pinned LeRobot extra uses the maintained hf-libero package rather than
  # installing Lifelong-Robot-Learning/LIBERO directly.
  libero_runtime_package: "hf-libero==0.1.4"
  libero_upstream_reference: 8f1084e3132a39270c3a13ebe37270a43ece2a01
runtime:
  # Starting package set from the upstream LeRobot lock at lerobot_git above.
  # Copy these exact values into the project uv.lock, then validate on target GPU.
  torch: "2.11.0+cu128"
  torchvision: "0.26.0+cu128"
  cuda_wheel_variant: cu128
  mujoco: "3.8.1"
  transformers: "5.5.4"
  peft: "0.20.0"
  accelerate: "1.14.0"
  numpy: "2.2.6"
  ffmpeg: "7.1.1"
```

`resolve_revisions.py` разрешено запускать один раз при bootstrap. После успешного full smoke значения freeze/commit и входят в Git. Последующие runs валидируют pins, но не обновляют их автоматически.

`uv.lock` — обязательный source of truth Python packages. Если pinned LeRobot удобнее ставить из source checkout, checkout должен быть at detached full SHA, а `bootstrap_*` обязан verify `git rev-parse HEAD` до install.

### 8.3 CUDA variants

Сохранять одинаковые Python package versions между Colab и VM. CUDA wheel может различаться только если driver floor платформы требует этого; такой run обязан записать полный `torch.__version__`, `torch.version.cuda`, driver и GPU в manifest. Нельзя смешивать результаты разных runtimes без отражения этого в registry.

На GPU VM предпочесть wheel/runtime, рекомендованный pinned LeRobot. На старом driver допускается явно заданный fallback index, но не silent fallback.

### 8.4 Bootstrap outputs

Каждый bootstrap генерирует:

```text
environment_manifest.json
pip_freeze.txt
system_info.txt
nvidia_smi.txt
ffmpeg_version.txt
upstream_revisions.json
```

`doctor.py` проверяет:

- Linux;
- Python version;
- exact package pins;
- GPU visibility, capability and free memory;
- headless MuJoCo reset/render;
- ffmpeg AV1 decode;
- write access to all storage roots;
- free disk space;
- object storage read/write test on a temporary test key;
- model/dataset revision availability;
- no secrets printed into logs.

### 8.5 Standard command interface

После bootstrap следующие команды должны быть одинаковыми на обеих платформах:

```bash
uv run python scripts/doctor.py --config configs/platform/gpu_vm.yaml
uv run pytest -q
uv run python scripts/inspect_dataset.py --config configs/data.yaml
uv run python scripts/replay_expert.py --task bowl_stove --episode-id 13
uv run python scripts/train_seen.py --config configs/train/seen_expert.yaml
uv run python scripts/train_target.py \
  --config configs/train/target_baseline.yaml \
  --task bowl_stove --n-demos 5 --seed 42
uv run python scripts/eval_target.py --config configs/eval/final.yaml --checkpoint <URI>
```

Не обещать, что raw upstream `lerobot-train` принимает именно эти project flags. Эти команды — project-owned stable interface.

---

## 9. Storage architecture и защита от потери данных

### 9.1 Environment variables

```bash
export VLA_PROJECT_ROOT="$PWD"
export VLA_DATA_ROOT="/mnt/vla"
export VLA_DATASETS_DIR="$VLA_DATA_ROOT/datasets"
export VLA_RUNS_DIR="$VLA_DATA_ROOT/runs"
export VLA_CHECKPOINTS_DIR="$VLA_DATA_ROOT/checkpoints"
export VLA_CACHE_DIR="$VLA_DATA_ROOT/cache"
export VLA_SCRATCH_DIR="/local_nvme/vla_scratch"
export HF_HOME="$VLA_CACHE_DIR/huggingface"
export TORCH_HOME="$VLA_CACHE_DIR/torch"
export VLA_OBJECT_URI="s3://<bucket>/smolvla-libero-fewshot"
```

Colab overlay может задавать:

```text
VLA_DATA_ROOT=/content/drive/MyDrive/vla-fewshot
VLA_SCRATCH_DIR=/content/vla_scratch
```

Если Google Drive не смонтирован, Colab run автоматически получает `ephemeral=true` и не считается завершенным, пока ценные artifacts не синхронизированы.

### 9.2 Persistent layout

```text
/mnt/vla/
├── datasets/
│   └── nvidia_LIBERO_LeRobot_v3/e590737.../
├── cache/
│   ├── huggingface/
│   └── torch/
├── runs/
│   └── <run_id>/
├── checkpoints/
│   └── <run_id>/
├── eval/
│   └── <eval_run_id>/
├── registry/
└── backups/
```

### 9.3 Три уровня сохранности

1. `Git remote`: code, configs, split files, predictions, small report assets.
2. `Persistent SSD`: active dataset/cache/runs/checkpoints/videos.
3. `Object storage`: immutable important checkpoints, manifests, final eval artifacts, failure videos and report bundle.

Optional fourth level: private Hugging Face model repo для final immutable checkpoints после верификации provenance.

### 9.4 Sync semantics

Artifact считается backed up только после:

1. atomic local checkpoint finalize;
2. SHA-256 manifest generation;
3. upload to temporary object prefix;
4. remote size/checksum verification;
5. write remote `COMPLETED.json` marker;
6. write local `backup_status.json` with verified URI/time/checksum.

Нельзя удалять local checkpoint просто после successful process exit; только отдельный retention workflow может предложить это после verified backup.

`sync_artifacts.py` default: `--dry-run`. Для upload требуется `--execute`. Никакого `sync --delete` в project scripts.

### 9.5 Secrets

- Credentials только в environment variables, cloud secret store или local ignored `.env`.
- `.env.example` содержит только names.
- Logs/manifests redact access keys/tokens.
- Никогда не писать `HF_TOKEN`, AWS secret или SSH private key в config/command history/artifact.

---

## 10. Experiment registry и run identity

### 10.1 Run ID

Canonical format:

```text
<stage>__<method>__<task-or-suite>__n<NN>__s<seed>__<UTC_TIMESTAMP>__g<GITSHA7>
```

Примеры:

```text
seen__expert__libero90__nall__s42__20260820T180000Z__ga1b2c3d
target__baseline__bowl_stove__n05__s42__20260821T120000Z__ga1b2c3d
eval__zero_shot__drawer_middle__n00__sna__20260822T100000Z__ga1b2c3d
```

Если output directory уже существует, run должен остановиться; не overwrite и не автоматически добавлять `v2`.

### 10.2 Registry architecture

Source of truth — immutable `manifest.json` внутри каждого run. `experiments/registry.csv` строится детерминированно сканированием manifests; ручное редактирование generated rows запрещено.

Tracked `experiments/plan.yaml` содержит planned grid и статусы:

```text
planned -> running -> trained -> evaluated -> backed_up -> reported
```

Status меняется только после проверки обязательных artifacts.

### 10.3 Minimum training manifest

```json
{
  "schema_version": 1,
  "run_id": "...",
  "stage": "seen|target",
  "method": "expert|baseline|lora|replay_lora",
  "status": "running|completed|failed",
  "created_at_utc": "...",
  "git_commit": "...",
  "git_dirty": false,
  "command": ["python", "..."],
  "config_resolved_path": "...",
  "dataset_repo_id": "nvidia/LIBERO_LeRobot_v3",
  "dataset_revision": "e590737...",
  "suite": "...",
  "task_slug": null,
  "task_text": null,
  "episode_ids": [],
  "n_demos": null,
  "train_seed": 42,
  "base_checkpoint_uri": "...",
  "base_checkpoint_sha256": "...",
  "model_revision": "...",
  "trainable_parameter_count": 0,
  "total_parameter_count": 0,
  "hardware": {},
  "versions": {},
  "started_at_utc": "...",
  "finished_at_utc": null,
  "final_checkpoint_uri": null,
  "failure": null
}
```

Manifest пишется сначала как `status=running`; на exception записывает sanitized traceback и `status=failed`; `completed` ставится только после финального checkpoint verification.

---

## 11. Checkpoint и exact resume

### 11.1 Содержимое checkpoint

Checkpoint обязан включать:

- model weights;
- PEFT adapter config/weights, если применимо;
- optimizer state;
- LR scheduler state;
- AMP GradScaler state;
- `global_step`, epoch, samples seen;
- gradient accumulation position;
- sampler/dataloader progress sufficient for exact continuation;
- Python, NumPy, PyTorch CPU RNG states;
- all CUDA RNG states;
- resolved training config;
- model/dataset/source revisions;
- exact episode IDs;
- normalization statistics or immutable reference + hash;
- trainable parameter list;
- Git commit/dirty state;
- metrics cursor;
- checkpoint format version.

### 11.2 Atomic save

Порядок:

```text
write step_000100.tmp-<uuid>/
fsync files where practical
write checksums.json
verify load in a fresh object/model instance
write COMPLETED.json
atomic rename -> step_000100/
atomically update latest.json pointer
```

Evaluation игнорирует checkpoint directory без `COMPLETED.json`.

### 11.3 Save frequency

Seen starting policy:

```text
save every 5,000 steps
named milestones: 10k, 20k, 40k, 60k, 80k, 100k
```

Target starting policy:

```text
save at 25%, 50%, 75%, 100% of fixed schedule
also save every 1,000 steps if the interval is smaller
```

Не удалять checkpoints автоматически. `prune_artifacts.py` отдельно показывает candidates; default dry-run; refuses to touch unbacked or referenced checkpoints.

### 11.4 Resume acceptance test

До VM training выполнить:

```text
Run A: step 0 -> 200 continuously
Run B1: step 0 -> 100, save, process exits
Run B2: fresh process loads step 100 -> 200
```

При deterministic-compatible hardware/config сравнить:

- global step;
- sample order hashes;
- optimizer/scheduler states;
- final parameter checksum or strict tolerance;
- loss values after resume.

Если exact equality невозможна из-за nondeterministic kernels/video decoding, задокументировать источник и проверить tight numerical tolerance. Просто «checkpoint загрузился» недостаточно.

CLI:

```bash
python scripts/train_seen.py \
  --config configs/train/smoke.yaml \
  --resume-from /path/to/step_000100
```

При resume запрещено менять dataset revision, split, trainable scope, optimizer, scheduler, effective batch или seed. Разрешенные overrides (`log_freq`, destination mirror) перечислить allowlist.

---

## 12. Logging без W&B

Во всех configs:

```yaml
tracking:
  wandb_enabled: false
  tensorboard_enabled: true
  csv_enabled: true
  jsonl_events_enabled: true
```

Run directory:

```text
<run_id>/
├── manifest.json
├── config.resolved.yaml
├── environment_manifest.json
├── trainable_parameters.txt
├── train.log
├── metrics.csv
├── events.jsonl
├── tensorboard/
├── checkpoints.json
└── artifacts/
```

`metrics.csv` columns минимум:

```text
wall_time_utc
elapsed_seconds
global_step
samples_seen
epoch_fraction
loss
learning_rate
grad_norm
optimizer_step_skipped
samples_per_second
data_time_seconds
step_time_seconds
gpu_memory_allocated_mb
gpu_memory_reserved_mb
```

CSV append должен быть crash-tolerant: header один раз, flush periodically, malformed trailing line repairable. TensorBoard tags используют стабильные names.

Remote TensorBoard:

```bash
# VM
tensorboard --logdir "$VLA_RUNS_DIR" --host 127.0.0.1 --port 6006

# local Mac
ssh -L 6006:127.0.0.1:6006 vla
```

Не открывать TensorBoard публично и не bind-ить `0.0.0.0` без отдельной auth layer.

---

## 13. Training recipes

### 13.1 Common optimizer contract

Starting values; окончательно зафиксировать на pseudo-target/seen development, не на real targets:

```yaml
optimizer:
  name: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-2
  betas: [0.9, 0.95]
  eps: 1.0e-8
scheduler:
  name: cosine
  warmup_steps: 1000
  min_lr: 1.0e-5
training:
  effective_batch_size: 32
  max_grad_norm: 1.0
  mixed_precision: auto
  seed: 42
```

`mixed_precision: auto` разрешает:

- BF16 на совместимой Ampere/Ada/Hopper/Blackwell GPU после smoke stability test;
- FP16 + GradScaler на T4;
- FP32 только для tiny debugging.

Resolved precision обязательно записать; silent fallback запрещен.

### 13.2 Smoke training

```yaml
dataset:
  suite: libero_90
  max_tasks: 2
  max_episodes: 10
training:
  physical_batch_size: 1
  effective_batch_size: 2
  max_steps: 200
  save_steps: [100, 200]
  num_workers: 0
evaluation:
  enabled: false
```

Acceptance:

- finite loss;
- backward + optimizer step;
- no unintended trainable VLM params;
- checkpoint save/load;
- resume 100→200;
- one post-training inference action accepted by env;
- artifacts durable.

### 13.3 Primary seen-pretrain

```yaml
stage: seen
method: expert
model:
  repo_id: lerobot/smolvla_base
  revision: c83c3163b8ca9b7e67c509fffd9121e66cb96205
dataset:
  repo_id: nvidia/LIBERO_LeRobot_v3
  revision: e5907374380b8f96511957e6ba5582be52a1e179
  suite: libero_90
  episodes: all
trainable_scope:
  freeze_vision_encoder: true
  freeze_vlm_backbone: true
  train_action_expert: true
  train_state_projection: true
  train_action_projections: true
training:
  max_steps: 100000
  physical_batch_size: auto_fit
  effective_batch_size: 32
  gradient_accumulation: auto
  seed: 42
checkpoint:
  every_steps: 5000
  milestones: [10000, 20000, 40000, 60000, 80000, 100000]
```

На 16GB T4 ожидать physical batch `1–4`; auto-fit разрешено проводить один раз до run, затем resolved physical batch/accumulation freeze. OOM retry внутри финального run запрещен: auto-fit должен закончиться до создания final run ID.

Seen checkpoint selection rule зафиксировать до target evaluation. Default:

1. исключить checkpoints с instability/NaN;
2. использовать fixed seen probe suite из diverse `libero_90` tasks;
3. выбрать earliest checkpoint within a predeclared tolerance of best seen-probe success, чтобы не платить за бессмысленные steps;
4. если rollout noise не позволяет различить — использовать final 100k checkpoint;
5. записать selected checkpoint hash в `configs/selected_seen_checkpoint.yaml` и сделать его immutable.

Target tasks не запускать даже «для интереса» до выбора/freeze seen checkpoint.

### 13.4 Optional seen LoRA challenger

```yaml
stage: seen
method: lora
peft:
  method_type: LORA
  r: 64
  lora_alpha: 64
  lora_dropout: 0.0
optimizer:
  lr: 1.0e-3
scheduler:
  min_lr: 1.0e-4
```

Сохранить adapter и merged-free load path. Final evaluation должна уметь загрузить adapter без неявного merge. Test: logits/actions до и после serialization совпадают в tolerance.

### 13.5 Naive target baseline

Primary baseline — target-only continuation из immutable seen checkpoint с тем же standard trainable scope (`Action Expert + state/action projections`), без:

- LoRA;
- seen replay;
- image/language augmentation;
- demonstrations других tasks;
- target validation episodes;
- per-task hyperparameter tuning.

Каждая комбинация запускается независимо:

```bash
python scripts/train_target.py \
  --config configs/train/target_baseline.yaml \
  --task drawer_middle \
  --n-demos 5 \
  --seed 42
```

Starting schedule выбирается на pseudo-target tasks и затем freeze. Рекомендуемый initial schedule:

```yaml
training:
  epochs: 100
  max_steps: 12000
  effective_batch_size: 32
  sample_with_replacement: true
  select_checkpoint: final
optimizer:
  lr: 1.0e-4
```

Использовать `min(100 epochs, 12,000 optimizer steps)`. Epoch определяется по frame/action-chunk samples selected episodes. Final checkpoint используется без target-success early stopping. Один fixed schedule для всех real tasks; budget-specific schedule допускается только если он заранее выбран на pseudo-target tasks и записан до real grid.

Final mandatory grid:

```text
3 tasks × 3 budgets (5,10,25) × 2 train seeds (42,123) = 18 training runs
```

### 13.6 Target LoRA ablation

После baseline реализовать чистую LoRA ablation:

- same immutable seen checkpoint;
- target-only selected episodes;
- same final evaluation protocol;
- no replay;
- one globally frozen LoRA configuration selected on pseudo-target tasks.

Starting point:

```yaml
peft:
  method_type: LORA
  r: 64
  lora_alpha: 64
optimizer:
  lr: 1.0e-3
```

### 13.7 Candidate method: Replay-LoRA

Main research candidate after baseline and LoRA ablation:

```text
75% target samples
25% seen replay samples from libero_90
+ LoRA adaptation
```

Replay pool selection и ratio выбираются только на pseudo-target tasks. Initial simple version uses deterministic stratified random seen replay; retrieval-based replay — отдельная future ablation, не часть MVP.

Batch mixer invariants:

- target fraction and seen fraction logged per optimizer step/window;
- no `libero_goal` sample in replay;
- replay RNG controlled by train seed;
- same target episode IDs as baseline;
- seen samples never alter target demonstration count N;
- normalization conventions identical.

Чтобы понять источник эффекта, minimum comparison:

```text
baseline
LoRA
Replay-LoRA
```

Не добавлять augmentation до получения этой ablation.

---

## 14. Scripts: required behavior

### 14.1 `scripts/download_dataset.py`

Responsibilities:

- require exact dataset revision;
- support metadata-only, selected suite, selected episode/video download;
- resume interrupted downloads;
- verify expected files/sizes where available;
- write `download_manifest.json`;
- never overwrite different revision directory;
- no destructive cleanup.

### 14.2 `scripts/inspect_dataset.py`

Output machine-readable JSON and readable Markdown:

- repo ID/revision/local root;
- feature schema/dtypes/shapes;
- suite episode/frame/task counts;
- FPS/video codec/resolution;
- task text → task index → ordered episode IDs;
- action/state min/max/mean/std;
- gripper unique/range distribution;
- missing/corrupt episode checks;
- target presence in seen suite;
- exact target prefix verification.

Expected outputs:

```text
artifacts/dataset_inspection/inspection.json
artifacts/dataset_inspection/inspection.md
artifacts/dataset_inspection/target_episode_ids.json
```

### 14.3 `scripts/materialize_subset.py`

Предпочтительно поддержать logical subset wrapper без копирования videos. Если upstream trainer требует physical dataset:

- создавать новый derived directory;
- hardlink/reflink immutable media где возможно;
- metadata содержит parent repo/revision and exact episode IDs;
- recompute stats only from selected train episodes;
- idempotent output hash;
- fail if output already exists with different manifest.

### 14.4 Training scripts

`train_seen.py` и `train_target.py` должны:

- resolve config and write it before model allocation;
- validate clean/recorded Git state;
- validate pins and split;
- acquire run lock;
- log trainable scope;
- support SIGTERM handler: request checkpoint, flush logs, exit non-zero/controlled;
- support exact resume;
- never evaluate real target success inside training;
- never upload implicitly unless config explicitly enables post-run sync;
- leave failed run intact for debugging.

### 14.5 Evaluation scripts

Evaluation separate from training. `eval_target.py` must:

- load immutable checkpoint by URI and verify hash;
- apply fixed protocol/seeds;
- write one JSONL record per rollout immediately;
- resume safely by skipping only already completed unique rollout keys;
- refuse duplicate conflicting records;
- save action/state traces;
- save failure videos;
- aggregate only after all expected keys exist.

### 14.6 Reporting scripts

`collect_results.py`:

- validates expected grid completeness;
- validates protocol equality across compared cells;
- excludes `dev_only`, soft-reset and incomplete runs by default;
- produces long-form and summary tables.

`plot_cost_curve.py`:

- x-axis exactly `[0, 5, 10, 25]`;
- task-level panels plus macro average;
- raw seed points visible;
- uncertainty displayed;
- no smoothed/interpolated claims beyond observed points.

---

## 15. Result schemas

### 15.1 Per-rollout JSONL

`rollouts.jsonl` record:

```json
{
  "schema_version": 1,
  "eval_run_id": "...",
  "train_run_id": "...",
  "stage": "zero_shot|target_eval|language_control",
  "method": "baseline|lora|replay_lora|seen",
  "task_slug": "bowl_stove",
  "task_text": "put the bowl on the stove",
  "suite": "libero_goal",
  "task_index": 7,
  "n_demos": 5,
  "train_seed": 42,
  "eval_seed": 1000,
  "rollout_index": 0,
  "protocol_id": "final_v1",
  "instruction_condition": "correct|wrong",
  "instruction_text_used": "...",
  "checkpoint_uri": "...",
  "checkpoint_sha256": "...",
  "dataset_revision": "e590737...",
  "training_episode_ids": [13, 15, 16, 22, 36],
  "success": 0,
  "terminated": false,
  "truncated": true,
  "episode_length": 300,
  "wall_time_seconds": 42.0,
  "initial_state_fingerprint": "sha256:...",
  "video_uri": "...",
  "trace_uri": "...",
  "failure_category": null,
  "notes": null,
  "created_at_utc": "...",
  "git_commit": "..."
}
```

Unique key:

```text
(checkpoint_sha256, task_slug, n_demos, train_seed, eval_seed, instruction_condition, protocol_id)
```

### 15.2 Long-form CSV

`results_long.csv` columns mirror JSONL scalar fields. Episode ID list serialized as compact JSON string. Never infer success from filename.

### 15.3 Summary CSV

`results_summary.csv` минимум:

```text
method
task_slug
n_demos
train_seed
n_rollouts
n_successes
success_rate
wilson_ci_low
wilson_ci_high
checkpoint_sha256
protocol_id
```

`results_macro.csv` дополнительно содержит macro average across tasks и bootstrap CI. Не pool-ить blindly tasks с разной сложностью как одну Bernoulli sample без task-level view.

---

## 16. Failure analysis

После final evaluation вручную разметить минимум три содержательно разные failure cases в `report/failure_cases.md`:

```text
failure_id
video_uri
task/method/N/seed
observed behavior
failure category
hypothesis
alternative hypothesis
discriminating experiment
relevant action/state evidence
```

Начальные категории:

- `perception_localization`;
- `language_goal_grounding`;
- `grasp_gripper_control`;
- `trajectory_control`;
- `planning_sequence`;
- `timeout_near_success`;
- `environment_or_pipeline`;
- `unknown`.

Pipeline/environment bug нельзя выдавать за model failure. Если обнаружен bug, affected eval records помечаются invalid, исправление получает новый protocol version, а affected grid перезапускается полностью на тех же seeds.

---

## 17. Remote SSH, tmux, Codex и Cursor workflow

### 17.1 Local SSH config

На локальной машине:

```ssh-config
Host vla
    HostName <VM_IP_OR_DNS>
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

Проверка:

```bash
ssh vla
```

Private key не копировать в repo или VM project directory.

### 17.2 Remote repo

```bash
mkdir -p ~/src
cd ~/src
git clone <YOUR_GIT_REMOTE> smolvla-libero-fewshot
cd smolvla-libero-fewshot
git checkout <PINNED_PROJECT_COMMIT>
```

Код приходит через Git. Runtime artifacts не возвращаются в Git случайным `git add -A`; `.gitignore` и pre-commit check должны блокировать large/binary/secrets.

### 17.3 Codex/Cursor

- Открыть remote folder `~/src/smolvla-libero-fewshot` через доступный Remote SSH workflow клиента.
- Если конкретный Codex client не предоставляет Remote SSH UI, подключиться `ssh vla` и запустить Codex CLI непосредственно в remote repo.
- Cursor использовать для ручной навигации/редактирования в том же remote filesystem.
- Coding agents могут выполнять doctor, unit/integration tests, smoke inference и короткие debug runs.
- Финальный training запускается отдельной зафиксированной командой под `tmux`; его жизненный цикл не должен зависеть от IDE connection.
- Не разрешать двум агентам одновременно редактировать experiment registry или запускать один run ID; использовать run lock.

Prompt для первого запуска coding-agent:

```text
Read PROJECT_SPEC.md completely. Implement only the next incomplete milestone from
Explicit TODO order. Do not start paid/long GPU training. Run its acceptance checks,
save validation evidence, update STATUS.md, and report the exact files and commands.
Preserve existing user changes and never delete datasets, checkpoints, or runs.
```

### 17.4 tmux

```bash
ssh vla
tmux new -s seen
cd ~/src/smolvla-libero-fewshot
source .venv/bin/activate
python scripts/train_seen.py --config configs/train/seen_expert.yaml \
  2>&1 | tee -a "$VLA_RUNS_DIR/launcher_seen.log"
```

Detach:

```text
Ctrl+B, then D
```

Return:

```bash
ssh vla
tmux attach -t seen
```

`tmux` не заменяет checkpointing. Process interruption и VM deletion остаются возможными.

---

## 18. Safety against accidental data loss

### 18.1 Defaults

- `overwrite: false` везде.
- Output directories immutable after completion.
- Atomic writes для configs/manifests/results/checkpoints.
- No recursive delete in training/evaluation/sync code.
- No object-storage `--delete` flags.
- No broad shell globs for prune targets.
- Every material prune command begins with inventory + checksum + backup check + dry run.
- Raw dataset mount/read path по возможности read-only.

### 18.2 Disk guard

Перед run оценить:

- dataset/cache size;
- expected checkpoints;
- expected videos;
- scratch requirement;
- minimum free-space reserve.

Run не стартует, если projected free space после run меньше configured reserve, default `50 GB` VM и `10 GB` Colab smoke.

### 18.3 Signals и shutdown

На `SIGTERM`/preemption:

1. stop accepting new batches;
2. finish/abort current optimizer step consistently;
3. save emergency checkpoint atomically;
4. flush CSV/TensorBoard/events;
5. update manifest `interrupted`;
6. attempt bounded-time object sync only if configured and safe;
7. exit with non-zero code indicating interruption.

На `SIGKILL` recovery использует последний completed checkpoint; `.tmp-*` dirs остаются forensic artifacts и игнорируются loaders.

### 18.4 Git safety

`.gitignore` минимум:

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
artifacts/
outputs/
runs/
checkpoints/
videos/
datasets/
cache/
*.pt
*.pth
*.safetensors
*.mp4
*.parquet
```

Добавить pre-commit check, который отклоняет files >10 MB и probable secrets, с allowlist только для осознанных маленьких report assets.

---

## 19. Milestones и acceptance criteria

### M0 — Repository skeleton and contracts

Deliverables:

- structure, `pyproject.toml`, `uv.lock`, configs schema, `Makefile`;
- `AGENTS.md`, `README.md`, `STATUS.md`;
- test harness;
- no large files/secrets guard.

Acceptance:

- `uv sync` works on Linux/Colab runtime;
- `pytest -q` starts;
- config validation catches unknown keys;
- all CLIs have `--help`;
- no training is started.

### M1 — Pinned environment and doctor

Deliverables:

- complete revisions lock;
- bootstrap scripts;
- environment manifests;
- `doctor.py`.

Acceptance:

- exact pins validated;
- GPU and MuJoCo EGL render work;
- two-camera observation returned;
- ffmpeg decodes dataset codec;
- object storage temporary-key round trip works or is explicitly skipped in Colab with durable Drive.

### M2 — Dataset inspection and exact split

Deliverables:

- metadata downloader/inspector;
- exact split JSON;
- leakage verifier;
- inspection report.

Acceptance:

- `libero_90`: 3921 episodes, 569249 frames, 73 task texts;
- `libero_goal`: 428 episodes, 52042 frames, 10 task texts;
- all three target task indices/counts match this spec;
- all first-25 IDs match;
- target presence in seen is false;
- nested prefixes validated.

### M3 — Environment adapters and expert replay

Deliverables:

- observation key adapter;
- image parity evidence;
- state adapter;
- action/gripper postprocessor;
- replay command/videos/traces.

Acceptance:

- endpoint gripper tests pass;
- no double conversion/flip;
- at least six required expert trajectories replay successfully;
- simulator success and videos saved.

### M4 — SmolVLA inference and trainable scope

Deliverables:

- pinned model loader;
- feature compatibility checks;
- freezing allowlist;
- one inference rollout/trace.

Acceptance:

- correct input keys/shapes;
- output action shape 7 after adapters;
- finite values and env accepts action;
- trainable list matches intended scope.

### M5 — Training/checkpoint/resume smoke

Deliverables:

- trainer/logger/checkpoint stack;
- 200-step smoke;
- exact resume comparison.

Acceptance:

- finite loss/backward;
- 0→200 and 0→100→200 comparison passes tolerance;
- checkpoint fresh-process load verified;
- TensorBoard/CSV/manifests valid;
- smoke artifacts backed up.

**До M5 включительно всё должно быть выполнено до paid long VM run.**

### M6 — Primary seen-pretrain

Deliverables:

- full `libero_90` training run;
- milestone checkpoints;
- seen probe eval;
- immutable selected seen checkpoint.

Acceptance:

- no target data/eval used;
- expected full dataset consumed;
- trainable scope verified;
- final/selected checkpoint checksummed and remotely backed up;
- selected checkpoint rule documented before target runs.

### M7 — Zero-shot and language control

Deliverables:

- 3×20 zero-shot rollouts;
- paired correct/wrong language control;
- per-rollout results, traces, failure videos.

Acceptance:

- exact same checkpoint hash;
- target training episode list empty for zero-shot;
- fixed seed protocol complete;
- paired initial fingerprints match.

### M8 — Baseline 5/10/25 × two seeds

Deliverables:

- 18 baseline training runs;
- final evaluation ≥20 per checkpoint/task;
- complete registry/results.

Acceptance:

- every run starts from same seen checkpoint hash;
- exact nested episode prefixes;
- seeds `42,123`;
- no target early stopping/tuning;
- all failures have video;
- complete-cell checker passes.

### M9 — LoRA and Replay-LoRA

Deliverables:

- target LoRA ablation;
- Replay-LoRA candidate;
- matched final grid as compute allows, minimum required by assignment/project decision;
- trainable/compute comparison.

Acceptance:

- hyperparameters frozen from pseudo-target development;
- target count unchanged;
- replay contains only `libero_90`;
- matched protocols/checkpoint origin;
- serialization and resume tests for adapters pass.

### M10 — Aggregation, failure analysis, report

Deliverables:

- cost curves;
- per-task/per-seed tables with intervals;
- language control figure/table;
- at least three failure analyses;
- compute table;
- reproducibility appendix;
- final artifact bundle.

Acceptance:

- expected grid complete;
- no invalid/dev run in final tables;
- plots rebuild from `results_long.csv` with one command;
- report numbers reconcile with raw JSONL;
- final bundle checksum and remote backup verified.

---

## 20. Report outputs

Обязательные generated outputs:

```text
report/figures/cost_curve_macro.pdf
report/figures/cost_curve_by_task.pdf
report/figures/language_control.pdf
report/figures/training_curves_seen.pdf
report/tables/main_results.csv
report/tables/per_seed_results.csv
report/tables/zero_shot.csv
report/tables/language_control.csv
report/tables/compute.csv
report/tables/checkpoint_provenance.csv
report/failure_cases.md
report/reproducibility.md
report/report.md
```

Главная таблица:

```text
task × method × N ∈ {0,5,10,25} -> mean success, per-seed values, CI, rollouts
```

Compute table:

```text
method
trainable_params
total_params
peak_vram_mb
training_gpu_hours
wall_time
effective_batch
steps/samples
```

`reproducibility.md` содержит exact commands, revisions, hardware, selected episode IDs, checkpoint hashes и known deviations.

### 20.1 Suggested short-report narrative

1. Problem and cost-curve hypothesis.
2. Data/split and no-leakage protocol.
3. Seen adaptation recipe.
4. Baseline vs method at `0/5/10/25`.
5. Uncertainty and seed stability.
6. Language control.
7. Three failure modes with discriminating experiments.
8. Compute/storage/reproducibility notes.
9. Prediction vs actual result, включая отрицательный результат.

---

## 21. Testing matrix

### Unit tests

- task text normalization and exact matching;
- episode prefix nesting;
- leakage set intersection;
- gripper continuous and binary conversions;
- observation/action key mapping;
- parameter allowlist;
- config validation;
- Wilson interval known cases;
- result unique key;
- checksum and manifest transitions;
- replay mixer proportions/determinism;
- prune refuses unbacked/referenced targets.

### Integration tests

- metadata-only download pinned revision;
- one LeRobotDataset sample with both cameras/state/action/text;
- LIBERO EGL reset/render/step;
- expert replay episode;
- model load/forward/action postprocess;
- checkpoint save/fresh load;
- interrupted evaluation resume without duplicates;
- object storage upload/verify test prefix.

### GPU smoke tests

- one forward/backward in chosen precision;
- gradient accumulation equivalence sanity check;
- 100→200 resume;
- one small rollout with video;
- LoRA serialize/reload.

### Final validation

```bash
make doctor
make test
make verify-split
make verify-leakage
make verify-checkpoints
make verify-final-grid
make report
make verify-report
```

---

## 22. Explicit TODO order

Coding-agent должен выполнять задачи строго в этом порядке, если `STATUS.md` не доказывает, что пункт уже завершен.

1. **Create repository skeleton.** Добавить package, configs schema, CLI stubs, tests, Git safety.
2. **Pin runtime.** Разрешить full revisions для SmolVLA, LeRobot и LIBERO; создать `uv.lock`; зафиксировать Python/PyTorch/CUDA/MuJoCo stack после smoke-compatible install.
3. **Implement platform-independent paths.** Ни одного hard-coded Colab/VM path вне platform configs.
4. **Implement `doctor.py`.** System/GPU/MuJoCo/video/storage/version checks.
5. **Implement metadata-only dataset download.** Exact revision required.
6. **Implement dataset inspection.** Schema/counts/task texts/episode IDs/statistics.
7. **Commit exact target split.** Проверить IDs из этого spec по downloaded metadata.
8. **Implement automatic no-leakage gate.** Подключить ко всем train/report commands.
9. **Implement logical/physical subset support.** Exact episode IDs and subset-local stats where needed.
10. **Implement LIBERO environment wrapper.** Headless reset/render/step, fixed seed, hard reset.
11. **Implement observation parity tool.** Зафиксировать camera keys и orientation.
12. **Implement state/action/gripper adapters.** Unit tests and action trace dual-space logging.
13. **Implement expert replay.** Пройти required six episodes and save evidence.
14. **Implement pinned SmolVLA loader.** Validate input/output features.
15. **Implement strict freezing/trainable-scope verifier.** Fail closed.
16. **Implement common logging/manifests/registry.** TensorBoard + CSV/JSONL; W&B off.
17. **Implement atomic checkpoint/resume.** Full optimizer/scheduler/RNG/sampler state.
18. **Run Colab smoke 0→100→200.** Prove fresh-process resume and durable backup.
19. **Implement eval protocol.** Fixed seeds, hard reset, JSONL resume, traces and failure videos.
20. **Implement object-storage sync/checksum.** Dry-run-first, no delete.
21. **Create and commit `predictions.md`.** До real target results.
22. **Define pseudo-target calibration.** Выбрать только tasks внутри `libero_90`; freeze hyperparameters.
23. **Run primary full seen-pretrain on VM.** Action Expert + projections, VLM frozen.
24. **Run seen probes and freeze selected checkpoint.** Backup and checksum.
25. **Optionally run seen LoRA challenger.** Не задерживать обязательный baseline без причины.
26. **Run zero-shot final evaluation.** 3 tasks × ≥20.
27. **Run paired language control.** Correct vs wrong, same seeds/states.
28. **Run baseline grid.** 3 tasks × 3 budgets × 2 train seeds.
29. **Evaluate every baseline checkpoint.** ≥20 rollouts per cell; save all failures.
30. **Run target LoRA ablation.** Same origin/data/eval protocol.
31. **Run Replay-LoRA.** Fixed replay config, only `libero_90` replay.
32. **Validate complete registry and protocols.** Exclude dev/invalid rows.
33. **Generate tables, confidence intervals and cost curves.** One deterministic command.
34. **Perform failure analysis.** Minimum three distinct failures.
35. **Build final report bundle.** Include provenance, hashes, exact commands.
36. **Verify remote backup.** Checksum final bundle and important checkpoints.

Не начинать TODO 23, пока TODO 1–22 не выполнены или не отмечены как consciously deferred с письменным rationale. Не начинать TODO 30–31 до complete baseline TODO 28–29.

---

## 23. Definition of done

Проект считается завершенным, если одновременно выполнено следующее:

- воспроизводимое pinned environment работает в Colab smoke и Linux GPU VM;
- exact split соответствует этому документу и проверяется кодом;
- expert replay проходит;
- resume доказан fresh-process test;
- final seen checkpoint не видел target data и backed up;
- zero-shot + paired language control завершены;
- baseline `5/10/25 × 2 seeds` завершен и оценен минимум на 20 rollouts per cell;
- optional/primary proposed method сравнивается при matching protocol;
- все failures имеют video и trace;
- raw results, summary tables и plots согласованы;
- no W&B dependency;
- registry содержит provenance каждого числа в report;
- datasets/checkpoints/results защищены persistent storage + verified object backup;
- final report bundle собирается одной documented command;
- никакие final claims не основаны на target hyperparameter leakage.

---

## 24. Primary technical references

Эти ссылки нужны для проверки API pinned-версии; этот spec имеет приоритет в отношении split и experiment protocol.

- [NVIDIA LIBERO LeRobot v3 dataset card](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3)
- [LeRobot LIBERO documentation](https://huggingface.co/docs/lerobot/en/libero)
- [LeRobot SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot PEFT/LoRA training](https://huggingface.co/docs/lerobot/peft_training)
- [LeRobot installation](https://huggingface.co/docs/lerobot/en/installation)
- [LeRobot source repository](https://github.com/huggingface/lerobot)
- [LIBERO source repository](https://github.com/Lifelong-Robot-Learning/LIBERO)

Если upstream documentation и pinned source расходятся, source at pinned SHA определяет фактический API; расхождение документируется, но scientific/data requirements этого файла не ослабляются.
