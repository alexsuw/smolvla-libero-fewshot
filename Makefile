UV ?= uv
PYTHON := $(UV) run python

.PHONY: help sync sync-data sync-gpu test validate-configs validate-revisions \
	doctor-static doctor check-cli check-git-safety check check-m1-static \
	verify-split verify-leakage check-m2 check-m3 check-m4 check-m5

help:
	@echo "sync              Install the locked M0 development environment"
	@echo "sync-data         Install CPU metadata tools (huggingface-hub, pyarrow)"
	@echo "sync-gpu          Install the pinned Linux cu128 runtime"
	@echo "test              Run unit and smoke tests"
	@echo "validate-configs  Strictly validate tracked YAML configs"
	@echo "validate-revisions Validate immutable pins without network access"
	@echo "doctor-static     Run CPU/static M1 checks (not hardware acceptance)"
	@echo "doctor            Run full fail-closed GPU/EGL/storage checks"
	@echo "check-cli         Verify every Python CLI supports --help"
	@echo "check-git-safety  Reject secrets and unsafe runtime artifacts"
	@echo "check             Run all M0 acceptance checks"
	@echo "check-m1-static   Run M0 plus non-GPU M1 acceptance checks"
	@echo "verify-split      Verify tracked target prefixes against metadata"
	@echo "verify-leakage    Fail if any target text appears in libero_90"
	@echo "check-m2          Run M0/M1 static checks plus M2 unit contracts"
	@echo "check-m3          Run M2 checks plus env/gripper/parity unit contracts"
	@echo "check-m4          Run M3 checks plus SmolVLA feature/allowlist contracts"
	@echo "check-m5          Run M4 checks plus checkpoint/resume smoke contracts"

sync:
	$(UV) sync --frozen

sync-data:
	$(UV) sync --frozen --extra data

sync-gpu:
	$(UV) sync --frozen --extra gpu

test:
	$(UV) run pytest -q

validate-configs:
	$(PYTHON) scripts/validate_configs.py

validate-revisions:
	$(PYTHON) scripts/resolve_revisions.py --offline

doctor-static:
	$(PYTHON) scripts/doctor.py --config configs/platform/gpu_vm.yaml \
		--profile static --offline

doctor:
	$(PYTHON) scripts/doctor.py --config configs/platform/gpu_vm.yaml \
		--profile full

check-cli:
	$(PYTHON) scripts/check_cli_help.py

check-git-safety:
	$(PYTHON) scripts/check_git_safety.py

check: validate-configs check-cli check-git-safety test

check-m1-static: check validate-revisions doctor-static

verify-split:
	$(PYTHON) scripts/verify_split.py

verify-leakage:
	$(PYTHON) scripts/verify_no_leakage.py

check-m2:
	$(PYTHON) -c "import huggingface_hub, pyarrow"
	$(MAKE) check-m1-static

check-m3:
	$(MAKE) check-m2
	$(PYTHON) scripts/check_observation_parity.py \
		--output-dir artifacts/validation/M3/parity

check-m4:
	$(MAKE) check-m3
	$(PYTHON) scripts/smoke_inference.py --config configs/train/smoke.yaml \
		--profile static --output-dir artifacts/validation/M4

check-m5:
	$(MAKE) check-m4
	$(PYTHON) scripts/train_seen.py --config configs/train/smoke.yaml \
		--profile static --protocol resume-compare \
		--output-dir artifacts/validation/M5
	$(PYTHON) scripts/verify_checkpoint.py \
		--checkpoint artifacts/validation/M5/run_a/checkpoints/step_000200
	$(PYTHON) scripts/build_registry.py \
		--runs-root artifacts/validation/M5 \
		--output artifacts/validation/M5/registry.csv
