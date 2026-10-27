@echo off
REM Start the H-XAI-NIDS website (backend + frontend) using the verified Python 3.13.
start "NIDS Backend" cmd /k "cd /d %~dp0backend && py -3.13 -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
timeout /t 6 /nobreak >nul
start "NIDS Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
