.PHONY: style check-style

style:
	@echo "Running isort..."
	isort src
	@echo "Running black..."
	black src

check-style:
	@echo "Checking isort..."
	isort --check-only src
	@echo "Checking black..."
	black --check src
