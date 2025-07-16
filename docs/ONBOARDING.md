# CuMind Development Environment Setup Guide

This guide walks you through setting up the CuMind development environment on **macOS** and **Windows (WSL)** using GitHub, VSCode or Cursor, and `uv` for Python dependency management.

## 1. Platform Prerequisites

CuMind requires a Linux-based development environment.

To achieve this on your machine:

### On Windows:

- Install WSL and Ubuntu by following the [official Microsoft guide](https://learn.microsoft.com/en-us/windows/wsl/install)
- Recommended: WSL 2 and Ubuntu 22.04
- After installation, open "Ubuntu" from the Start Menu to launch your WSL terminal
- Use a code editor such as [VSCode](https://code.visualstudio.com/) or [Cursor](https://cursor.so) that supports WSL.  
  If you're using VSCode, install the **Remote - WSL** extension — you'll use it to open and work with the project from inside your WSL environment after cloning the repository.
  - Additional resource: [VSCode: Developing in WSL](https://code.visualstudio.com/docs/remote/wsl)

### On macOS:

- No WSL needed — the built-in Terminal provides a compatible Unix-based environment
- Use any local code editor such as [VSCode](https://code.visualstudio.com/) or [Cursor](https://cursor.so)

## 2. Setup Python & `uv`

### 1. **Install Python**

Ensure Python 3.12 or higher is installed:

```bash
python3 --version
```

If not installed or the version is too old, install:

- #### On WSL:

  Install Python and pip:

  ```bash
  sudo apt update
  sudo apt install python3 python3-pip
  ```

- #### On macOS:

  Download and install the latest version from: [https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. **Install `uv`**

Install the Python dependency manager `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip/pipx:

```bash
pip install uv
# or
pipx install uv
```

More details: [uv documentation](https://astral.sh/uv/)

## 4. Clone Repo and Setup Git Authentication

You can clone the repo using either **SSH** or **HTTPS**.

### Option 1: SSH (Recommended)

1. Generate a new SSH key (if you don't have one already):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "your_email@example.com"
```

2. Copy your public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

3. Add it to GitHub:
   [https://github.com/settings/ssh/new](https://github.com/settings/ssh/new)

4. Test your connection:

```bash
ssh -T git@github.com
```

5. Clone the repo:

```bash
git clone git@github.com:carletonai/CuMind.git
```

> If you already have an SSH key and have it registered with GitHub, skip steps 1–3.

### Option 2: HTTPS

Clone the repo using:

```bash
git clone https://github.com/carletonai/CuMind.git
```

You’ll be prompted to enter your GitHub **username** and a **personal access token** (PAT) instead of a password.
Generate a PAT here: [https://github.com/settings/tokens](https://github.com/settings/tokens)

## 5. Set Up and Run the Project

Once the repo is cloned, navigate to it and install dependencies:

```bash
cd CuMind
uv sync
```

Test out running a specific file to make sure your setup is working:

`uv run python src/cartpole.py`

To run the full application:

`uv run python -m cumind`

## 6. Test Your Git Setup

Make sure you can push a new branch:

```bash
git checkout -b test-branch
git push origin test-branch
```

Once confirmed, please delete the test branch:

`git push origin --delete test-branch`

## You’re Ready!

You should now be able to develop and contribute to CuMind locally. If you run into issues, feel free to reach out on Discord.
