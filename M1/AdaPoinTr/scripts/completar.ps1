param(
    [string]$Entrada = '',
    [string]$Salida = '',
    [string]$Checkpoint = ''
)
$ErrorActionPreference = 'Stop'
$moduloAda = Split-Path -Parent $PSScriptRoot
$proyectoAda = Split-Path -Parent (Split-Path -Parent $moduloAda)
$pythonAda = Join-Path $moduloAda 'instalacion\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonAda)) { throw 'No se encuentra el entorno nativo de AdaPoinTr.' }
$argumentosAda = @('-m', 'M1.AdaPoinTr', 'completar')
if ($Entrada) { $argumentosAda += @('--entrada', $Entrada) }
if ($Salida) { $argumentosAda += @('--salida', $Salida) }
if ($Checkpoint) { $argumentosAda += @('--checkpoint', $Checkpoint) }
Push-Location -LiteralPath $proyectoAda
try {
    & $pythonAda @argumentosAda
    if ($LASTEXITCODE -ne 0) { throw "Completado interrumpido (codigo $LASTEXITCODE)." }
} finally { Pop-Location }
