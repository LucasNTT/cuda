@echo off
setlocal enabledelayedexpansion

REM === User-configurable options ===
set cc=35
set deprecatedGpu=true
REM set define=MEM_TRACKER

REM === Environment setup ===
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
cd "%~dp0"

REM === Derived build options ===
set NVCC=nvcc
set GENCODE=-gencode=arch=compute_%cc%,code=\"sm_%cc%,compute_%cc%\"
set DEFINE_OPT=
if not "%define%"=="" set DEFINE_OPT=-D%define%
set DEPRECATED_OPT=
if /i "%deprecatedGpu%"=="true" set DEPRECATED_OPT=--Wno-deprecated-gpu-targets

set OPTIONS=%GENCODE% %DEFINE_OPT% %DEPRECATED_OPT% --use-local-env --keep-dir x64 --machine 64 --compile -cudart static -Xcompiler "/W3 /O2"
set OUTDIR=x64\obj

REM === Create output directory if it doesn't exist ===
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM === Initialize object file list ===
set LINK_OBJS=

echo === Compiling all .cu and .cpp files ===

for /R %%F in (*.cu *.cpp) do (
    set "SRC=%%~fF"
    set "REL=%%F"
    set "REL=!REL:%CD%\=!"
    set "REL=!REL:/=\!"

    REM === Construct object path ===
    set "OBJ=%OUTDIR%\!REL!.obj"
    set "OBJDIR=!OBJ:\=\\!"
    for %%D in ("!OBJDIR!") do if not exist "%%~dpD" mkdir "%%~dpD"

    %NVCC% %OPTIONS% "!SRC!" -o "!OBJ!"
    if errorlevel 1 (
        echo *** Compilation failed for !SRC!
        exit /b 1
    )
    set "LINK_OBJS=!LINK_OBJS! !OBJ!"
)

echo.
echo === Linking into x64\LucasNTT.exe ===
%NVCC% %DEPRECATED_OPT% -link -o x64\LucasNTT.exe %GENCODE% --machine 64 !LINK_OBJS!

if errorlevel 1 (
    echo *** Final link failed
    exit /b 1
)

echo.
echo === Build successful ===
pause
