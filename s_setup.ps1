# =================================================================
#      STREAMLIT CHATBOT - SETUP & RUN SCRIPT
# =================================================================
$ErrorActionPreference = 'Stop'
Write-Host "🚀 Starting the Streamlit Chatbot Setup..."

# 1. Create/Check Virtual Environment
if (-not (Test-Path -Path ".\venv")) {
    Write-Host "   - Setting up Python virtual environment..."
    python -m venv venv
}
Write-Host "   - Activating virtual environment..."
# Note: Activation is for this script's session
& ".\venv\Scripts\Activate.ps1"

# 2. Install Dependencies
Write-Host "   - Upgrading pip..."
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip

Write-Host "   - Installing dependencies from s_requirements.txt..."
& ".\venv\Scripts\pip.exe" install -r s_requirements.txt

# --- ADDED THIS FALLBACK ---
# This makes sure streamlit is installed even if requirements file fails
Write-Host "   - Verifying streamlit installation..."
& ".\venv\Scripts\pip.exe" install streamlit
# ---------------------------

# 3. Check for Ollama Model
Write-Host "   - Checking for Ollama and Llama 3 model..."
$ollamaExists = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaExists) {
    Write-Host "❌ Error: Ollama is not installed." -ForegroundColor Red;
    exit 1
}
try {
    $ollamaList = ollama list
    $modelToUse = "llama3:8b"
    if ($ollamaList -match $modelToUse) {
        Write-Host "✅ Model '$modelToUse' is already installed." -ForegroundColor Green
    } else {
        Write-Host "⚠️  Model '$modelToUse' not found. Pulling it now..." -ForegroundColor Yellow
        ollama pull $modelToUse
    }
} catch {# =================================================================
#      STREAMLIT CHATBOT - SETUP & RUN SCRIPT
# =================================================================
$ErrorActionPreference = 'Stop'
Write-Host "🚀 Starting the Streamlit Chatbot Setup..."

# 1. Create/Check Virtual Environment
if (-not (Test-Path -Path ".\venv")) {
    Write-Host "   - Setting up Python virtual environment..."
    python -m venv venv
}
Write-Host "   - Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"

# 2. Install Dependencies
Write-Host "   - Upgrading pip..."
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip

# Check if requirements file exists before trying to install
if (Test-Path ".\streamlit\s_requirements.txt") {
    Write-Host "   - Installing dependencies from s_requirements.txt..."
    & ".\venv\Scripts\pip.exe" install -r .\streamlit\s_requirements.txt
} else {
    Write-Host "   - s_requirements.txt not found, skipping..."
}

# Verify streamlit installation
Write-Host "   - Verifying streamlit installation..."
& ".\venv\Scripts\pip.exe" install streamlit

# 3. Check for Ollama Model
Write-Host "   - Checking for Ollama and Llama 3 model..."
$ollamaExists = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaExists) {
    Write-Host "❌ Error: Ollama is not installed." -ForegroundColor Red;
    exit 1
}
try {
    $ollamaList = ollama list
    $modelToUse = "llama3:8b"
    if ($ollamaList -match $modelToUse) {
        Write-Host "✅ Model '$modelToUse' is already installed." -ForegroundColor Green
    } else {
        Write-Host "⚠️  Model '$modelToUse' not found. Pulling it now..." -ForegroundColor Yellow
        ollama pull $modelToUse
    }
} catch {
    Write-Host "⚠️  Warning: Cannot connect to Ollama." -ForegroundColor Yellow
}

Write-Host "`n✅ Streamlit Setup Complete!" -ForegroundColor Green
Write-Host "--------------------------------------------------------"
Write-Host "Launching the Streamlit application..."
Write-Host "Make sure your database is built (run database_builder.py in the data_pipeline folder)."

# FIXED: Point to the correct path of s_app.py
& ".\venv\Scripts\streamlit.exe" run .\streamlit\s_app.py
    Write-Host "⚠️  Warning: Cannot connect to Ollama." -ForegroundColor Yellow
}

Write-Host "`n✅ Streamlit Setup Complete!" -ForegroundColor Green
Write-Host "--------------------------------------------------------"
Write-Host "Launching the Streamlit application..."
Write-Host "Make sure your database is built (run database_builder.py in the data_pipeline folder)."
& ".\venv\Scripts\streamlit.exe" run s_app.py