.DEFAULT_GOAL: run

run:
	cd src;	poetry run python model_service.py

install: pyproject.toml
	poetry install --no-root

check:
	poetry run flake8 src