.PHONY: quality style

check_dirs := ./

# Check that source code meets quality standards

quality:
	black --check --line-length 88 --target-version py36 $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs) --max-line-length 88

# Format source code automatically

style:
	black --line-length 88 --target-version py36 $(check_dirs)
	isort $(check_dirs)