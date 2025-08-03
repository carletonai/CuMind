# Git Basics for Contributors

## Table of Contents

- [1. Branching](#1-branching)
  - [What is a Branch?](#what-is-a-branch)
  - [Common Branching Commands](#common-branching-commands)
    - [git branch](#git-branch)
    - [git checkout](#git-checkout)
    - [git reset](#git-reset)
    - [git reflog](#git-reflog)
- [2. Committing Changes](#2-committing-changes)
  - [What is a Commit?](#what-is-a-commit)
  - [Common Committing Commands](#common-committing-commands)
    - [git add](#git-add)
    - [git commit](#git-commit)
    - [git revert](#git-revert)
- [3. Merging Branches](#3-merging-branches)
  - [What is Merging?](#what-is-merging)
  - [Common Merging Commands](#common-merging-commands)
    - [git merge](#git-merge)
    - [git status](#git-status)
    - [git diff](#git-diff)
    - [git log](#git-log)
- [4. Pushing to Remote](#4-pushing-to-remote)
  - [What Does it Mean to Push?](#what-does-it-mean-to-push)
  - [Common Pushing Commands](#common-pushing-commands)
    - [git push](#git-push)
    - [git remote](#git-remote)
- [5. Step-by-Step Workflow Examples](#5-step-by-step-workflow-examples)
  - [What is a Git Workflow?](#what-is-a-git-workflow)
  - [Examples of Common Workflows](#examples-of-common-workflows)
    - [Starting a New Feature](#starting-a-new-feature)
    - [Fixing a Small Bug](#fixing-a-small-bug)
    - [Merging a Finished Feature into Main](#merging-a-finished-feature-into-main)
    - [Updating the Last Commit](#updating-the-last-commit)
    - [Syncing with Main](#syncing-with-main)
    - [Reverting a Commit](#reverting-a-commit)
- [6. Best Practices for Collaboration](#6-best-practices-for-collaboration)
  - [Why Collaboration Practices Matter](#why-collaboration-practices-matter)
  - [Examples of Good Git Habits](#examples-of-good-git-habits)
    - [Use Feature Branches](#use-feature-branches)
    - [Pull Often](#pull-often)
    - [Write Clear Commit Messages](#write-clear-commit-messages)
    - [Test Before You Commit](#test-before-you-commit)
    - [Clean Up Old Branches](#clean-up-old-branches)


## 1. Branching

### What is a branch?
A branch is like a copy of the project where you can make changes without affecting the main code.  
You use branches to work on features or fixes safely and separately.

### Common Branching Commands
This section covers Git commands used to create, switch, and manage branches:  
- `git branch`
- `git checkout`
- `git reset`
- `git reflog`

#### **`git branch`**
#### **List all local branches**
``` bash
git branch
```
Lists all local branches. The current branch is marked by an asterisk `*`.  

#### **Delete a merged branch**
``` bash
git branch -d branch-name
```
Deletes a local branch, but only if it has been merged.

#### **Force delete a branch**
``` bash
git branch -D branch-name
```
Force deletes a branch, even if it has not been merged.
**Use with caution, changes may be lost.**

#### **`git checkout`**
#### **Switch to an existing branch**
``` bash
git checkout branch-name
```
Switches from the current branch to the branch named `branch-name`.  

#### **Create and switch into new branch**
``` bash
git checkout -b new-branch
```
Creates a new branch named `new-branch` and switches to it immediately.  

#### **`git reset`**
#### **Reset working directory to the latest commit (hard reset)**
``` bash
git reset --hard HEAD
```
Discards all uncommitted changes in the current branch and resets it to the latest commit.  
**WARNING: This is destructive and cannot be undone.**

#### **Reset to a previous commit, but keep changes staged (soft reset)**
``` bash
git reset --soft HEAD~1
```
Moves `HEAD` back one commit (or to a specific hash), but keeps your changes staged so you can recommit them.  
Useful when you want to edit a previous commit without losing your previous work.

#### **General syntax**
``` bash
git reset --{soft | hard | mixed} {commit-id | HEAD~n}
```
- `--soft`: Keeps all changes staged (in the index)
- `--mixed` (default): Keeps changes but unstages them
- `--hard`: Discards all changes. 

You can use:  
- A commit hash (`a1b2c3d`), or 
- A relative reference like `HEAD~1` (1 commit before the current), `HEAD~2` (2 commits before current), etc.

#### **`git reflog`**
#### **Show recent Git history**
``` bash
git reflog
```
Shows a log of your recent Git history, including branch switches, commits and resets. Useful for recovering deleted branches or lost commits.

**Example Use:**  
If you accidentally delete a branch or run a destructive command such as `git reset --hard`, you can use `git reflog` to find the commit before things went wrong.   
First, view the reflog:
``` bash
git reflog
```
You'll see a list of actions, each with a commit ID (a short hash like a3c1f2b, for example).
Find the entry that represents the state you want to return to, then run: 
``` bash
git checkout <commit-id>
```
Replace `commit-id` with the actual hash.  
This brings your project back to that point, even if the branch no longer exists.


## 2. Committing Changes

### What is a commit?
A commit is like a snapshot of your project at a certain point.
It saves the changes you've staged (i.e., marked as ready to commit) and adds them to your branch's history.

### Common Committing Commands
This section covers Git commands used to stage changes and record commits:
- `git add`
- `git commit`
- `git revert`

#### **`git add`**
#### **Stage a specific file**
``` bash
git add filename 
```
Stages a specific file to be included in the next commit.
"Staged" means you've marked the file as ready to commit.

#### **Stage all changed files**
``` bash
git add .
```
Stages *all* changed files in the current directory.

#### **`git commit`**
#### **Create a commit with a message**
``` bash
git commit -m "Your message here"
```
Creates a new commit that saves all staged changes to your branch, along with a message describing what was changed.

**Example Use:**  
The `-m` flag lets you write a short, descriptive message. For example:
``` bash
git commit -m "Fix typo in README"
git commit -m "Remove unused image assets"
git commit -m "Implement dark mode toggle"
```  

#### **Amending a commit**  
``` bash
git commit --amend
```
Replaces your most recent commit with a new one.
Useful if you forgot to include a file or want to update the commit message.  
**Only use this if the commit has not been pushed to a remote repository.**  
If you amend a commit that’s already been pushed, you’ll need to force-push, which can overwrite history and affect teammates.

**Example Use:**  
If you forgot to include a file in your last commit:
``` bash
git add missed-file.js
git commit --amend
```
This adds the new file and replaces the previous commit with the updated one.  
Or, to correct a commit message:
``` bash
git commit --amend -m "Corrected commit message"
```

#### **`git revert`**
#### **Undo a specific commit by creating a new one**
``` bash
git revert <commit-id>
```
Creates a new commit that undoes the changes introduced by an earlier commit.  
Unlike `reset` or `amend`, it does not modify history, so it's safe on shared branches.


## 3. Merging Branches

### What is merging?  
Merging is the process of combining changes from one branch into another.  
You usually merge feature branches back into the `main` (or `master`) branch after development is done.  
Git tries to automatically integrate the changes, if the same line of code were changed in both branches, a **merge conflict** may occur and will need to be resolved.  

### Common Merging Commands
This section covers Git commands used to merge branches and resolve conflicts:
- `git merge`
- `git status`
- `git diff`
- `git log`  
*Note: Branch names like `feature-branch` are just examples. You can name your branches based on the task you're working on.*

#### **`git merge`**
#### **Merge another branch into the current one**
``` bash
git merge feature-branch
```
Merges the features from `feature-branch` into your current branch.
This is usually done after switching to the `main` branch.

**Example Use:**
``` bash
git checkout main
git pull # Explained in next section
git merge feature-branch
```
If there are no conflicts, Git will automatically complete the merge.  

#### **`git status`**
#### **Check the status of your working directory**
``` bash
git status
```
Shows which files are staged, modified, or untracked.  
Useful during a merge to see if there are conflicts that need to be resolved.  

#### **`git diff`**
#### **See the differences between changes**
``` bash
git diff
```
Shows the differences between your current changes and the last commit.

**Example Use:**
You can also compare branches or commits:
``` bash
git diff main feature-branch
```
This shows the changes in `feature-branch` compared to `main`.

#### **`git log`**
#### **View your commit history**
``` bash
git log
```
Displays a list of past commits in the current branch, including author, date, and message.

#### **Shorter, one-line summary per commit**
``` bash
git log --oneline
```
Shows each commit as a single line for a more concise history.


## 4. Pushing to Remote

### What does it mean to "push"?
In Git, pushing means sending your local commits to a remote repository (like GitHub).  
It updates the shared version of the project so others can see your changes. 
You typically push after making and committing changes locally, once you're ready to share your work or back it up.

### Common Pushing Commands
This section covers Git commands used to push changes and check your remote settings:
- `git push`
- `git remote`  

#### **`git push`**
#### **Push commits to the remote repository**
``` bash
git push
```
Sends your local commits to the remote repository (e.g., GitHub) for the currently tracked branch.  
Only works if the branch is already linked to a remote.

#### **Push and set upstream tracking**
``` bash
git push -u origin branch-name
```
Pushes the specified branch and sets it to track the remote branch.  
This allows you to run `git push` alone in the future without specifying a branch.

**Example Use:**
``` bash
git push -u origin feature-login #feature-login is an example branch
```
This pushes the `feature-login` branch to GitHub and links it to the remote.

#### **Force push (use with caution)**
``` bash
git push --force
```
Replaces the remote branch with your local version, overwriting history.  
**Only use when necessary (e.g., after an amended commit) and never on shared branches.**

#### **`git remote`**
#### **View configured remote repositories**
``` bash
git remote -v
```
Lists the remote connections linked to your local repository.  
This shows where your code will be pushed or pulled from.

**Example Output:**
``` bash
origin  https://github.com/username/repo-name.git (fetch)
origin  https://github.com/username/repo-name.git (push)
```


## 5. Step-by-Step Workflow Examples

### What is a Git workflow?
A Git workflow is a common sequence of commands developers use to make and share changes to a project.  
The examples below show how multiple Git commands work together in a real-world scenario.  
*Note: Branch names like `navbar-feature` or `bugfix/readme-link` are just examples. You can name your branches based on the task you're working on.*

### Examples of Common Workflows

#### Starting a New Feature
``` bash
git checkout -b navbar-feature
# Work on the navbar code
git add .
git commit -m "Add responsive navbar"
git push -u origin navbar-feature
```
Creates a new branch, commits your work, and pushes it to GitHub for the first time.  

#### Fixing a Small Bug
``` bash
git checkout main
git pull
git checkout -b bugfix/readme-link
# Fix broken link in README
git add README.md
git commit -m "Fix broken link in README"
git push -u origin bugfix/readme-link
```
Start from an up-to-date `main`, create a short-lived bugfix branch, and push your fix.  

#### Merging a Finished Feature into Main
``` bash
git checkout main
git pull
git merge navbar-feature
git push
```
Merges your completed feature branch (`navbar-feature`) into the `main` branch and pushes the updated branch to GitHub.  
If there are no conflicts, Git will complete the merge automatically.  

#### Updating the Last Commit
``` bash
# Forgot to include a file
git add missed-file.js
git commit --amend
git push --force
```
Updates your most recent commit with new changes.  
Because this rewrites Git history, you'll need to use `--force` when pushing.  
**Only do this if the commit hasn't been shared (i.e., pushed to a team-accessible remote).**

#### Syncing with Main
``` bash
git checkout main
git pull
```
Switches to the `main` branch and pulls the latest updates from the remote.  
This ensures your local copy is up to date before starting new work.

#### Reverting a Commit
``` bash
git log --oneline
# Find commit ID (e.g., a1b2c3d)
git revert a1b2c3d
git push
```
Creates a new commit that undoes the changes from an earlier one, without rewriting history.  
This is the safest way to undo something on a shared branch.


## 6. Best Practices for Collaboration

### Why Collaboration Practices Matter
When working with others on a shared codebase, following a few Git habits helps keep the project organized, readable and conflict-free. These practices make it easier for everyone to contribute smoothly while avoiding conflicts or accidental overwrites.

### Examples of Good Git Habits

#### Use Feature Branches
Avoid committing directly to `main`.  
Create a new branch for each new feature or fix:
``` bash
git checkout -b add-user-profile
```
This helps keep your work isolated and easier to review.

#### Pull Often
Before starting new work (or before pushing), always pull the latest changes:
``` bash
git pull
```
This pulls updates from the remote branch your current branch is tracking (usually `main` if you're on `main`).  
If you're not sure which branch you're on, run:
``` bash
git status
```
Keeping your local copy current reduces the chance of merge conflicts and ensures you're building on the latest version of the code.

#### Write Clear Commit Messages
Use short, descriptive commit messages that explain what you've changed:
``` bash
git commit -m "Fix a typo in signup form"
```
Avoid vague messages like `"update"` or `"stuff"`, they don't help teammates understand your changes.

#### Communicate Before You Push Big Changes
If you're making major changes or rewriting history (e.g. with `--force`), let your team know first.  
Misusing commands like `git push --force` can overwrite someone else's work.

#### Test Before You Commit
Make sure your code works before committing it, especially if it affects shared parts of the project.  
Broken commits slow down everyone else.

#### Clean Up Old Branches
Once a feature is merged, delete the branch to keep things tidy:
``` bash
git branch -d add-user-profile
```
For remote branches:
``` bash
git push origin --delete add-user-profile
```
