@echo off
rem Double-click: live M16 view (training, GPU, corpus download/processing), refreshed every 10 s.
rem Close the window or press Ctrl+C to exit; it never stops anything.
setlocal
if not defined YTS_WSL_DISTRO set YTS_WSL_DISTRO=Ubuntu-24.04
for /f "usebackq delims=" %%p in (`wsl.exe -d %YTS_WSL_DISTRO% -e wslpath -a "%~dp0..\experiments\m16-own-llm\tools\m16_watch.sh"`) do set "M16_WATCH=%%p"
wsl.exe -d %YTS_WSL_DISTRO% -e watch -n 10 -t bash "%M16_WATCH%"
