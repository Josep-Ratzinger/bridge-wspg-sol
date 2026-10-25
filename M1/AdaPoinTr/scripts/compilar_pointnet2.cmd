@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39
if errorlevel 1 exit /b 1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "TORCH_CUDA_ARCH_LIST=8.9"
set "MAX_JOBS=2"
set "ADAPOINTR_ROOT=%~dp0.."
cd /d "%ADAPOINTR_ROOT%\instalacion\fuentes\Pointnet2_PyTorch\pointnet2_ops_lib"
"%ADAPOINTR_ROOT%\instalacion\venv\Scripts\python.exe" -m pip install --no-cache-dir --no-build-isolation --no-deps . > "%ADAPOINTR_ROOT%\instalacion\pointnet2_build.log" 2>&1
exit /b %errorlevel%
