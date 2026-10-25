@echo off
setlocal
if not defined VOLGA_PYTHON set "VOLGA_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined VOLGA_DEVICE set "VOLGA_DEVICE=cpu"
if not exist "%VOLGA_PYTHON%" (
    echo Create .venv and install dependencies as described in README.md.
    exit /b 2
)
"%VOLGA_PYTHON%" "%~dp0run.py" --input "%~dp0examples\input" --output "%~dp0output\examples.csv" --device "%VOLGA_DEVICE%" %*
exit /b %ERRORLEVEL%
