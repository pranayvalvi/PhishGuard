@echo off
echo ===================================================
echo        Starting PhishGuard Streamlit App...
echo ===================================================
echo.
echo Please wait while the application loads. A browser window should open automatically.
echo (Keep this terminal window open while using the app)
echo.

:: Run Streamlit using the virtual environment's python
.\venv\Scripts\python.exe -m streamlit run src\06_streamlit_app.py

pause
