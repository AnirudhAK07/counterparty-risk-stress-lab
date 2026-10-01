# Launch the dashboard with the project's Python environment.
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Python environment not found. Follow the setup steps in README.md.'
}

& $projectPython -m streamlit run (Join-Path $projectRoot 'app.py') --server.address 127.0.0.1
