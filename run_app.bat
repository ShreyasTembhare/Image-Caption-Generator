@echo off
echo Starting Advanced Image Caption Generator...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the Streamlit app
streamlit run app.py

pause 