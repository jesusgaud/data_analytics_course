@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run START_PROJECT.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m ruff check .
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m mypy src
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pytest --cov=project2_app --cov-report=term-missing
if errorlevel 1 exit /b 1

echo All quality checks passed.
pause
