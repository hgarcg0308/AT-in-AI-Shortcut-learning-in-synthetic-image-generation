@echo off
rem ======================================================================
rem  Run_all.bat
rem  Lanza el flujo completo del proyecto (limpieza, desplazamiento, DDPM,
rem  verificador, sondas, escenarios y resumen) con un doble clic o desde la consola.
rem
rem  Uso desde la consola (en la carpeta del proyecto):
rem      Run_all.bat                  -> muestra un menu
rem      Run_all.bat quick            -> prueba corta de todo el flujo
rem      Run_all.bat full             -> flujo largo (5 semillas, guidance 0, 2 y 5)
rem      Run_all.bat shift            -> solo el desplazamiento train vs test (minutos)
rem      Run_all.bat aggregate        -> solo junta los resultados ya existentes
rem      Run_all.bat simulate         -> muestra los comandos sin ejecutar nada
rem
rem  Se pueden anadir opciones de Pipeline.py al final, por ejemplo:
rem      Run_all.bat full --seeds 0 1 2 --guidances 2
rem  (para el flujo anterior, sin guidance 5:  Run_all.bat full --guidances 0 2)
rem
rem  Es REANUDABLE: si se corta, vuelve a lanzar el mismo comando y sigue
rem  por donde iba (salta lo que ya esta hecho).
rem ======================================================================

setlocal EnableExtensions
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

rem --- Nombre del entorno de conda (cambialo si el tuyo se llama distinto) ---
set "ENV_NAME=AdvancedAITopics"

rem --- Ir a la carpeta donde esta este archivo (el proyecto) ---
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem  1) Elegir el modo
rem ----------------------------------------------------------------------
set "MODE=%~1"
set "REST="
shift
:accumulate
if "%~1"=="" goto end_accumulate
set "REST=%REST% %1"
shift
goto accumulate
:end_accumulate

if not "%MODE%"=="" goto mode_chosen
echo.
echo  ===============================================
echo   Project pipeline: what do you want to do?
echo  ===============================================
echo   1) Quick test of the whole pipeline (a few minutes with GPU)
echo   2) Full pipeline (5 seeds, guidance 0/2/5; many hours, best run overnight)
echo   3) Only merge the existing results (summary)
echo   4) Simulate: show the commands without running anything
echo   5) Only the train vs test shift analysis (a few minutes)
echo   0) Exit
echo.
set /p CHOICE="Choose an option [0-5]: "
if "%CHOICE%"=="1" set "MODE=quick"
if "%CHOICE%"=="2" set "MODE=full"
if "%CHOICE%"=="3" set "MODE=aggregate"
if "%CHOICE%"=="4" set "MODE=simulate"
if "%CHOICE%"=="5" set "MODE=shift"
if "%CHOICE%"=="0" goto quit
:mode_chosen

set "ARGS="
if /i "%MODE%"=="quick"     set "ARGS=--quick"
if /i "%MODE%"=="full"      set "ARGS="
if /i "%MODE%"=="aggregate" set "ARGS=--only aggregate"
if /i "%MODE%"=="shift"     set "ARGS=--only cleaning shift"
if /i "%MODE%"=="simulate"  set "ARGS=--dry-run"
if /i not "%MODE%"=="quick" if /i not "%MODE%"=="full" if /i not "%MODE%"=="aggregate" if /i not "%MODE%"=="shift" if /i not "%MODE%"=="simulate" (
    echo Unrecognized mode: "%MODE%". Use: quick, full, shift, aggregate or simulate.
    set "ERR=1"
    goto finish
)

rem ----------------------------------------------------------------------
rem  2) Activar el entorno de conda
rem ----------------------------------------------------------------------
if /i "%CONDA_DEFAULT_ENV%"=="%ENV_NAME%" (
    echo Environment "%ENV_NAME%" is already active.
    goto env_ready
)

set "CONDA_BAT="
where conda >nul 2>nul
if %errorlevel%==0 set "CONDA_BAT=conda"
if not defined CONDA_BAT (
    for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
        if exist "%%~D\condabin\conda.bat" set "CONDA_BAT=%%~D\condabin\conda.bat"
    )
)
if not defined CONDA_BAT (
    echo Cannot find conda. Open this file from "Anaconda Prompt" with the environment
    echo "%ENV_NAME%" already activated, or change the conda path inside the .bat.
    set "ERR=1"
    goto finish
)
call "%CONDA_BAT%" activate %ENV_NAME%
if errorlevel 1 (
    echo Could not activate the environment "%ENV_NAME%".
    set "ERR=1"
    goto finish
)
:env_ready

rem ----------------------------------------------------------------------
rem  3) Comprobaciones previas
rem ----------------------------------------------------------------------
if not exist "Pipeline.py" (
    echo Cannot find Pipeline.py in %CD%. Put this .bat in the project root.
    set "ERR=1"
    goto finish
)
if not exist "data\pneumoniamnist.npz" (
    echo Cannot find data\pneumoniamnist.npz in %CD%.
    set "ERR=1"
    goto finish
)
python -c "import torch, sklearn, scipy, matplotlib" 2>nul
if errorlevel 1 (
    echo Missing libraries in the environment "%ENV_NAME%" ^(torch, scikit-learn, scipy or matplotlib^).
    set "ERR=1"
    goto finish
)

python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo  WARNING: PyTorch does not see any GPU ^(this is the CPU version^).
    echo  On CPU the full pipeline would take far longer.
    if /i "%MODE%"=="full" (
        choice /C YN /M "Continue anyway"
        if errorlevel 2 (
            set "ERR=1"
            goto finish
        )
    )
) else (
    echo GPU detected: PyTorch will use CUDA.
)

rem ----------------------------------------------------------------------
rem  4) Lanzar
rem ----------------------------------------------------------------------
echo.
echo Mode: %MODE%   Options: %ARGS%%REST%
echo Folder: %CD%
echo To avoid interruptions, disable the computer's sleep mode while it runs.
echo.
python Pipeline.py %ARGS%%REST%
set "ERR=%errorlevel%"

:finish
echo.
if "%ERR%"=="0" (
    echo Finished successfully.
    echo   Summarized results : outputs\summary\summary.md
    echo   Shift analysis     : outputs\shift\
    echo   Messages of each step: outputs\logs\
) else if defined ERR (
    echo Finished with errors ^(code %ERR%^). Check outputs\logs\ and rerun
    echo the same command: it will continue where it left off.
)
:quit
pause
endlocal
