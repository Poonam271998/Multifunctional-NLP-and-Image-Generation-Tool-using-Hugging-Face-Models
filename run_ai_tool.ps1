# -----------------------------
# Run this script from your project folder
# Make sure app.py and requirements.txt are in the same folder
# -----------------------------

# Step 1: Check Python installation
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python is not installed or not in PATH. Install Python 3.11 first." -ForegroundColor Red
    exit
}

Write-Host "✅ Python detected at $($python.Source)"

# Step 2: Allow Python through Windows Firewall
Write-Host "✅ Adding Python to firewall rules..."
$pythonPath = $python.Source

# Check if firewall rule already exists
$rule = Get-NetFirewallRule -DisplayName "Python Streamlit" -ErrorAction SilentlyContinue
if (-not $rule) {
    New-NetFirewallRule -DisplayName "Python Streamlit" -Direction Inbound -Program $pythonPath -Action Allow -Profile Domain,Private,Public
    Write-Host "Firewall rule created for Python." -ForegroundColor Green
} else {
    Write-Host "Firewall rule already exists." -ForegroundColor Yellow
}

# Step 3: Install dependencies
Write-Host "✅ Installing dependencies from requirements.txt..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Step 4: Run Streamlit app
Write-Host "🚀 Launching Streamlit app..."
streamlit run app.py
