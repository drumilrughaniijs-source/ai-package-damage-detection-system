@echo off
cd /d "%~dp0"
echo Starting Package Damage Detection app... (close this window to stop it)
py -3.14 -m streamlit run app.py
pause
