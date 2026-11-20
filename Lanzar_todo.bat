@echo off
rem ======================================================================
rem  Lanzar_todo.bat
rem  Lanza el flujo completo del proyecto (limpieza, DDPM, verificador,
rem  sondas, escenarios y resumen) con un doble clic o desde la consola.
rem
rem  Uso desde la consola (en la carpeta del proyecto):
rem      Lanzar_todo.bat                  -> muestra un menu
rem      Lanzar_todo.bat rapido           -> prueba corta de todo el flujo
rem      Lanzar_todo.bat completo         -> flujo largo (5 semillas, guidance 0 y 2)
rem      Lanzar_todo.bat agregar          -> solo junta los resultados ya existentes
rem      Lanzar_todo.bat simular          -> muestra los comandos sin ejecutar nada
rem
rem  Se pueden anadir opciones de Pipeline.py al final, por ejemplo:
rem      Lanzar_todo.bat completo --semillas 0 1 2 --guidances 2
rem
rem  Es REANUDABLE: si se corta, vuelve a lanzar el mismo comando y sigue
rem  por donde iba (salta lo que ya esta hecho).
rem ======================================================================

setlocal EnableExtensions
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

rem --- Nombre del entorno de conda (cambialo si el tuyo se llama distinto) ---
set "ENTORNO=AdvancedAITopics"

rem --- Ir a la carpeta donde esta este archivo (el proyecto) ---
cd /d "%~dp0"

rem ----------------------------------------------------------------------
rem  1) Elegir el modo
rem ----------------------------------------------------------------------
set "MODO=%~1"
set "RESTO="
shift
:acumular
if "%~1"=="" goto fin_acumular
set "RESTO=%RESTO% %1"
shift
goto acumular
:fin_acumular

if not "%MODO%"=="" goto modo_elegido
echo.
echo  ===============================================
echo   Flujo del proyecto: que quieres hacer?
echo  ===============================================
echo   1) Prueba rapida de todo el flujo (unos minutos con GPU)
echo   2) Flujo completo (5 semillas; muchas horas, mejor de noche)
echo   3) Solo juntar los resultados ya existentes (resumen)
echo   4) Simular: ver los comandos sin ejecutar nada
echo   0) Salir
echo.
set /p OPCION="Elige una opcion [0-4]: "
if "%OPCION%"=="1" set "MODO=rapido"
if "%OPCION%"=="2" set "MODO=completo"
if "%OPCION%"=="3" set "MODO=agregar"
if "%OPCION%"=="4" set "MODO=simular"
if "%OPCION%"=="0" goto salir
:modo_elegido

set "ARGS="
if /i "%MODO%"=="rapido"   set "ARGS=--rapido"
if /i "%MODO%"=="completo" set "ARGS="
if /i "%MODO%"=="agregar"  set "ARGS=--solo agregar"
if /i "%MODO%"=="simular"  set "ARGS=--dry-run"
if /i not "%MODO%"=="rapido" if /i not "%MODO%"=="completo" if /i not "%MODO%"=="agregar" if /i not "%MODO%"=="simular" (
    echo Modo no reconocido: "%MODO%". Usa: rapido, completo, agregar o simular.
    set "ERR=1"
    goto fin
)

rem ----------------------------------------------------------------------
rem  2) Activar el entorno de conda
rem ----------------------------------------------------------------------
if /i "%CONDA_DEFAULT_ENV%"=="%ENTORNO%" (
    echo Entorno "%ENTORNO%" ya activo.
    goto entorno_listo
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
    echo No encuentro conda. Abre este archivo desde "Anaconda Prompt" con el entorno
    echo "%ENTORNO%" ya activado, o cambia la ruta de conda dentro del .bat.
    set "ERR=1"
    goto fin
)
call "%CONDA_BAT%" activate %ENTORNO%
if errorlevel 1 (
    echo No he podido activar el entorno "%ENTORNO%".
    set "ERR=1"
    goto fin
)
:entorno_listo

rem ----------------------------------------------------------------------
rem  3) Comprobaciones previas
rem ----------------------------------------------------------------------
if not exist "Pipeline.py" (
    echo No encuentro Pipeline.py en %CD%. Pon este .bat en la raiz del proyecto.
    set "ERR=1"
    goto fin
)
if not exist "data\pneumoniamnist.npz" (
    echo No encuentro data\pneumoniamnist.npz en %CD%.
    set "ERR=1"
    goto fin
)
python -c "import torch, sklearn, scipy, matplotlib" 2>nul
if errorlevel 1 (
    echo Faltan librerias en el entorno "%ENTORNO%" ^(torch, scikit-learn, scipy o matplotlib^).
    set "ERR=1"
    goto fin
)

python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo  AVISO: PyTorch no ve ninguna GPU ^(esta version es de CPU^).
    echo  Con CPU el flujo completo tardaria muchisimo mas.
    if /i "%MODO%"=="completo" (
        choice /C SN /M "Continuar igualmente"
        if errorlevel 2 (
            set "ERR=1"
            goto fin
        )
    )
) else (
    echo GPU detectada: PyTorch usara CUDA.
)

rem ----------------------------------------------------------------------
rem  4) Lanzar
rem ----------------------------------------------------------------------
echo.
echo Modo: %MODO%   Opciones: %ARGS%%RESTO%
echo Carpeta: %CD%
echo Para que no se interrumpa, desactiva la suspension del equipo mientras corra.
echo.
python Pipeline.py %ARGS%%RESTO%
set "ERR=%errorlevel%"

:fin
echo.
if "%ERR%"=="0" (
    echo Terminado correctamente.
    echo   Resultados resumidos : outputs\resumen\resumen.md
    echo   Mensajes de cada paso: outputs\logs\
) else if defined ERR (
    echo Terminado con errores ^(codigo %ERR%^). Revisa outputs\logs\ y vuelve a lanzar
    echo el mismo comando: continuara por donde se quedo.
)
:salir
pause
endlocal
