#!./.venv/bin/python3.12
import subprocess
import sys


def run_step(cmd, desc):
    print(f"\n=== {desc} ===")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    run_step("uv run ruff check .", "Ruff Linting")
    run_step("uv run mypy src/", "Mypy Type Checking")
    run_step("uv run pytest", "Pytest Unit Tests")


if __name__ == "__main__":
    main()
