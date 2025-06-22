import subprocess
import sys


def run(cmd):
    """Run a shell command and exit if it fails."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def test():
    run("pytest tests/")


def lint():
    run("ruff check .")


def fmt():
    run("ruff format .")


def typecheck():
    run("mypy src/")


def all():
    test()
    lint()
    fmt()
    typecheck()


if __name__ == "__main__":
    all()
    exit(0)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["test", "lint", "fmt", "typecheck", "all"])
    args = parser.parse_args()
    globals()[args.task]()
