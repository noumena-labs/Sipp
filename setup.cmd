@echo off
setlocal

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

set "COGENTLM_SETUP_CHILD=1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\setup.ps1" %*
set "SETUP_EXIT=%ERRORLEVEL%"

endlocal & set "SETUP_EXIT=%SETUP_EXIT%" & set "COGENTLM_BIN=%ROOT%\.build\bin"

if "%SETUP_EXIT%"=="0" (
  if exist "%COGENTLM_BIN%\clm.cmd" (
    echo ;%PATH%; | find /I ";%COGENTLM_BIN%;" >nul
    if errorlevel 1 set "PATH=%COGENTLM_BIN%;%PATH%"
    echo.
    echo clm is active in this CMD session.
  )
)

exit /b %SETUP_EXIT%
