@echo off
REM 以脚本所在目录为基准定位项目根目录下的 .venv
powershell.exe -NoExit -ExecutionPolicy Bypass -Command "& '%~dp0..\.venv\Scripts\Activate.ps1'"
