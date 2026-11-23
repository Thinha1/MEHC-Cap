@echo off
rem Chay backend bang Python co san torch/ultralytics/fastapi (dat bien PY de doi interpreter). Them tham so uvicorn neu can, vi du: run_backend.bat --port 8001
if not defined PY set "PY=python"
rem Phai chay tu thu muc chua webapp/ vi code dung import tuong doi
cd /d "%~dp0.."
"%PY%" -m uvicorn webapp.backend.main:app --reload --port 8000 %*
