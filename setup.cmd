@echo off
setlocal

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

set "SIPP_SETUP_CHILD=1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\setup.ps1" %*
set "SETUP_EXIT=%ERRORLEVEL%"

endlocal & set "SETUP_EXIT=%SETUP_EXIT%" & set "SIPP_BIN=%ROOT%\.build\bin"

if "%SETUP_EXIT%"=="0" (
  if exist "%SIPP_BIN%\sipp.cmd" (
    echo ;%PATH%; | find /I ";%SIPP_BIN%;" >nul
    if errorlevel 1 set "PATH=%SIPP_BIN%;%PATH%"
    echo.
    echo sipp is active in this CMD session.
  )
)

exit /b %SETUP_EXIT%
