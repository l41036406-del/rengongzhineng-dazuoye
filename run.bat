@echo off
setlocal

rem Always run from the folder containing this script.
cd /d "%~dp0"
title World Cup Intelligence

echo ============================================================
echo   World Cup Match Prediction and Analytics
echo ============================================================
echo.

rem If the site is already running, open it and exit.
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch { exit 1 }"
if not errorlevel 1 (
    echo The website is already running.
    start "" "http://127.0.0.1:8000"
    exit /b 0
)

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found.
    echo Install Python 3.10 or later and add it to PATH.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo ERROR: Node.js and npm were not found.
    echo Install Node.js 20 or later and add it to PATH.
    pause
    exit /b 1
)

echo [1/3] Checking Python dependencies...
python -c "import fastapi, uvicorn, pandas, sklearn, joblib" >nul 2>nul
if errorlevel 1 (
    python -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
) else (
    echo Python dependencies are ready.
)

echo.
echo [2/3] Building the React website...
pushd web
if not exist "node_modules" (
    call npm install
    if errorlevel 1 (
        popd
        goto :failed
    )
)
call npm run build
if errorlevel 1 (
    popd
    goto :failed
)
popd

echo.
echo [3/3] Starting the website...
echo URL: http://127.0.0.1:8000
echo Keep this window open. Press Ctrl+C to stop the server.
echo.

rem Open the browser after the server has had time to start.
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:8000'"

python -m uvicorn api:app --app-dir code --host 127.0.0.1 --port 8000
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo ERROR: Startup failed. Review the messages above.
pause
exit /b 1
