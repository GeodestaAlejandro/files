@echo off
title Geodesia Lab - Inicializador Automatico
color 0A
echo ============================================================
echo            GEODESIA LAB - AUTOCONFIGURADOR
echo ============================================================
echo.

:: 1. Verificar si Python ya esta instalado en el sistema
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python detectado correctamente en el sistema.
    set "PYTHON_CMD=python"
    goto INSTALAR_DEPENDENCIAS
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Launcher de Python (py) detectado.
    set "PYTHON_CMD=py"
    goto INSTALAR_DEPENDENCIAS
)

:: 2. Si no esta instalado, descargar e instalar Python automaticamente
echo [INFO] Python no fue detectado en este equipo.
echo [INFO] Descargando e instalando Python de forma automatica...
echo.

set "PYTHON_INSTALLER=python_installer.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

powershell -Command "Write-Host 'Descargando instalador de Python...'; (New-Object Net.WebClient).DownloadFile('%PYTHON_URL%', '%PYTHON_INSTALLER%')"

if not exist %PYTHON_INSTALLER% (
    echo [ERROR] No se pudo descargar el instalador. Verifica la conexion a internet.
    pause
    exit
)

echo [INFO] Instalando Python (configurando PATH automaticamente)...
start /wait %PYTHON_INSTALLER% /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 SimpleInstall=1

:: Eliminar el archivo ejecutable descargado
if exist %PYTHON_INSTALLER% del %PYTHON_INSTALLER%

:: Agregar la ruta instalada a la sesion actual de la consola
set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"

set "PYTHON_CMD=python"

%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [AVISO] Python se ha instalado correctamente.
    echo Por favor, cierra esta ventana y vuelve a hacer doble clic en iniciar_plataforma.bat.
    pause
    exit
)

:INSTALAR_DEPENDENCIAS
echo.
echo [INFO] Verificando e instalando librerias requeridas...
%PYTHON_CMD% -m pip install --upgrade pip >nul 2>&1
%PYTHON_CMD% -m pip install streamlit pandas numpy plotly geographiclib pyproj

echo.
echo ============================================================
echo           INICIANDO SERVIDOR GEODESIA LAB
echo ============================================================
echo.
%PYTHON_CMD% -m streamlit run app.py

pause