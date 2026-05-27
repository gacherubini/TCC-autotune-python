# Roda o pipeline inteiro de benchmarks de deteccao de pitch.
# Uso (a partir da raiz do projeto):
#   .\scripts\run_all.ps1
#   .\scripts\run_all.ps1 -Limit 3        # limita o Vocadito aos 3 primeiros audios
#   .\scripts\run_all.ps1 -SkipVocadito   # pula o benchmark do dataset

[CmdletBinding()]
param(
    [int]$Limit = 0,
    [switch]$SkipSynthetic,
    [switch]$SkipVocadito,
    [switch]$SkipPlots,
    [switch]$SkipReport,
    [switch]$SkipRealtime
)

$ErrorActionPreference = "Stop"

# Garante que estamos na raiz do projeto (pasta-pai deste script).
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Invoke-Step {
    param([string]$Label, [string[]]$ScriptArgs)
    Write-Host ""
    Write-Host ("==> " + $Label) -ForegroundColor Cyan
    & python @ScriptArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("Falha em: " + $Label + " (exit " + $LASTEXITCODE + ")")
    }
}

$start = Get-Date

if (-not $SkipSynthetic) {
    Invoke-Step "02 - Benchmark sintetico" @("scripts/02_run_synthetic_benchmark.py")
} else {
    Write-Host "(pulando 02 - sintetico)" -ForegroundColor DarkGray
}

if (-not $SkipVocadito) {
    $vocaditoArgs = @("scripts/03_run_dataset_benchmark.py")
    if ($Limit -gt 0) { $vocaditoArgs += @("--limit", "$Limit") }
    Invoke-Step "03 - Benchmark Vocadito" $vocaditoArgs
} else {
    Write-Host "(pulando 03 - Vocadito)" -ForegroundColor DarkGray
}

# 06 antes de 04/05: o relatorio (05) precisa do CSV de realtime; figuras (04)
# nao dependem mas ficam mais coerentes geradas junto.
if (-not $SkipRealtime) {
    $rtArgs = @("scripts/06_realtime_benchmark.py")
    if ($Limit -gt 0) { $rtArgs += @("--limit", "$Limit") }
    Invoke-Step "06 - Benchmark realtime" $rtArgs
} else {
    Write-Host "(pulando 06 - realtime)" -ForegroundColor DarkGray
}

if (-not $SkipPlots) {
    Invoke-Step "04 - Figuras" @("scripts/04_plot_results.py")
} else {
    Write-Host "(pulando 04 - figuras)" -ForegroundColor DarkGray
}

if (-not $SkipReport) {
    Invoke-Step "05 - Relatorio markdown" @("scripts/05_summary_report.py")
} else {
    Write-Host "(pulando 05 - relatorio)" -ForegroundColor DarkGray
}

$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host ("Concluido em {0:mm}m{0:ss}s." -f $elapsed) -ForegroundColor Green
Write-Host "Saidas em ./results/" -ForegroundColor Green
