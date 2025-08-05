@echo off
echo Starting Advanced Image Caption Generator v2.0...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the Streamlit app
streamlit run app.py

pause 