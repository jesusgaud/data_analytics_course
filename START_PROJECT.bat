@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo IS 665 Project 2 - Model Comparison Lab
echo ============================================================

echo Detecting a supported Python runtime...
set "PYTHON_CMD="

where py >nul 2>nul
if %errorlevel%==0 (
    for %%V in (3.15 3.14 3.13 3.12 3.11) do (
        if not defined PYTHON_CMD (
            py -%%V --version >nul 2>nul
            if !errorlevel! == 0 set "PYTHON_CMD=py -%%V"
        )
    )
)

if not defined PYTHON_CMD (
    where python >nul 2>nul
    if %errorlevel%==0 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo.
    echo ERROR: Python 3.11 through 3.15 was not found.
    echo Recommended: install Python 3.15 from python.org, then run this file again.
    goto :error
)

echo Using: %PYTHON_CMD%
%PYTHON_CMD% --version
if errorlevel 1 goto :error

if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :error
)

echo Installing/updating required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto :error

echo Starting application...
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b 0

:error
echo.
echo Setup failed. Review the error above.
pause
exit /b 1
