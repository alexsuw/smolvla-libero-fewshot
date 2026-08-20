UV ?= uv
PYTHON := $(UV) run python

.PHONY: help sync sync-gpu test validate-configs validate-revisions \
	doctor-static doctor check-cli check-git-safety check check-m1-static

help:
	@echo "sync              Install the locked M0 development environment"
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

sync:
	$(UV) sync --frozen

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
