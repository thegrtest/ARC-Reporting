@echo off
REM Publisher: TKG
set "PUBLISHER=TKG"

cd "C:\Users\Burn Plant\Desktop\Kiln"
start "" python CIPMonitor.py
sleep(10)
start "" python CIP.py
