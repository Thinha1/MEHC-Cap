@echo off
rem Run the backend with a Python that has torch/ultralytics/fastapi (set PY to change the interpreter). Extra uvicorn arguments are passed on, e.g. run_backend.bat --port 8001
if not defined PY set "PY=python"
rem Run from the folder that contains webapp/, because the code uses relative imports
cd /d "%~dp0.."
"%PY%" -m uvicorn webapp.backend.main:app --reload --port 8000 %*
