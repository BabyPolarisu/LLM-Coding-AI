@echo off
chcp 65001 > nul
echo ===================================================
echo   Starting Coding Assistant LLM (MiniProject)
echo ===================================================
echo.

:: ตรวจสอบการใช้งาน uv หรือ python
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Running with uv...
    uv run streamlit run app.py
) else (
    echo [INFO] Running with streamlit...
    streamlit run app.py
)

pause
