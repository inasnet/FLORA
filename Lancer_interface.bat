@echo off
cd /d "%~dp0"
"C:\Users\HP\anaconda3\python.exe" -m streamlit run app.py --server.address 127.0.0.1
pause
