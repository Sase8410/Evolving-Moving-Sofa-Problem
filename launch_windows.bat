@echo off
cd /d "%~dp0"
python run_v3.py
if errorlevel 1 pause
