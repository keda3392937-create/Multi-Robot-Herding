@echo off
cd /d "%~dp0"
python keyboard_remote_control.py --skip-wifi %*
if errorlevel 1 pause
