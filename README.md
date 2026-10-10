# HUB pulse - Employee Attendance Prototype

## Requirements
- Python 3.10+
- VS Code

## Run on Windows PowerShell
```powershell
cd path\to\attendance_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install flask
python app.py
```
Open http://127.0.0.1:5000 in your browser.

The SQLite database (`attendance.db`) is created automatically the first time the app runs.
This is a local learning prototype, not a production-secured HR system. Before real use, add authentication, role permissions, CSRF protection, backups, and secure configuration.
