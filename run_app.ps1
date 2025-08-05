# Advanced Image Caption Generator v2.0 - Launcher Script
Write-Host "🚀 Starting Advanced Image Caption Generator v2.0..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Run the Streamlit app
Write-Host "🌐 Launching Streamlit app..." -ForegroundColor Yellow
streamlit run app.py

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 