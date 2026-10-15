@echo off
cd /d "%~dp0"
echo Installing Python packages...
py -3.14 -m pip install -r requirements.txt
echo.
echo Downloading the CLIP model (~600 MB, one time only)...
set HF_HUB_DISABLE_XET=1
py -3.14 -c "import real_inspector as ri; ri.load_model(); print('Real-photo AI ready.')"
py -3.14 -c "import damage_detector as dd; dd.load_models(); print('Synthetic models ready.')"
echo.
echo Setup complete. Start the app with run_app.bat
pause
