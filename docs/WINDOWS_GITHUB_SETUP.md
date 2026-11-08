# Windows + VS Code + GitHub Setup

Target local directory:

```text
C:\python_code\data_analytics_course
```

## Local environment

1. Install Python 3.15 and Git.
2. Restart Windows if long-path support was newly enabled.
3. Open `C:\python_code\data_analytics_course` in VS Code.
4. Open a PowerShell terminal.
5. Verify:

```powershell
py -3.15 --version
git --version
```

6. Create/activate the environment:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

7. Run quality checks:

```powershell
ruff check .
mypy src
pytest --cov=project2_app --cov-report=term-missing --cov-fail-under=75
```

8. Start the app:

```powershell
streamlit run app.py
```

## Git initialization and first GitHub push

```powershell
git init
git config --global user.name "Jesus Gaud"
git config --global user.email "YOUR_GITHUB_EMAIL"
git status
git add .
git status
git commit -m "Initial IS665 Project 2 model comparison application"
git branch -M main
```

Create an empty GitHub repository named `data_analytics_course` without initializing README, `.gitignore`, or license. Then:

```powershell
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/data_analytics_course.git
git remote -v
git push -u origin main
```

Verify the GitHub Actions CI workflow is green before treating the first push as complete.

## Normal branch workflow

```powershell
git checkout main
git pull origin main
git checkout -b feature/descriptive-name
```

After changes:

```powershell
git add .
git commit -m "Describe the change"
git push -u origin feature/descriptive-name
```

Open a pull request into `main` and merge only after required CI checks pass.

## Release

```powershell
git checkout main
git pull origin main
git tag -a v1.1.0 -m "IS665 Project 2 version 1.1.0"
git push origin v1.1.0
```

The release workflow validates the repo, builds a clean ZIP, and publishes a GitHub Release.
