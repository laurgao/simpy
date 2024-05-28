.PHONY: style check-style

style:
	@echo "Running isort..."
	isort src/ tests/
	@echo "Running black..."
	black src/ tests/

check-style:
	@echo "Checking isort..."
	isort --check-only src/ tests/
	@echo "Checking black..."
	black --check src/ tests/
