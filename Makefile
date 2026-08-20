UV ?= uv
PYTHON := $(UV) run python

.PHONY: help sync test validate-configs check-cli check-git-safety check

help:
	@echo "sync              Install the locked M0 development environment"
	@echo "test              Run unit and smoke tests"
	@echo "validate-configs  Strictly validate tracked YAML configs"
	@echo "check-cli         Verify every Python CLI supports --help"
	@echo "check-git-safety  Reject secrets and unsafe runtime artifacts"
	@echo "check             Run all M0 acceptance checks"

sync:
	$(UV) sync --frozen

test:
	$(UV) run pytest -q

validate-configs:
	$(PYTHON) scripts/validate_configs.py

check-cli:
	$(PYTHON) scripts/check_cli_help.py

check-git-safety:
	$(PYTHON) scripts/check_git_safety.py

check: validate-configs check-cli check-git-safety test
