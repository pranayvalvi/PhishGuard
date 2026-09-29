@echo off
echo ===================================================
echo        PhishGuard Live Email Listener
echo ===================================================
echo.
echo Connecting to your email account using .env credentials...
echo (Keep this terminal window open to continuously monitor your inbox)
echo.

:: Run the email listener using the virtual environment
.\venv\Scripts\python.exe src\07_email_listener.py

pause
