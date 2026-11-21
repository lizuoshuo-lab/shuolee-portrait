param(
    [Parameter(Mandatory=$true)][ValidateSet('creator','post64','post','merge-unoptimized','merge')][string]$Stage,
    [ValidateSet('0_0','0_1','1_0','1_1')][string]$Chunk,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9-]{1,100}$')][string]$SupervisionRun,
    [switch]$Execute
)
$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\Administrator\Desktop\GS-SLAM-Lab'
$taskPS7 = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
if ($PSVersionTable.PSVersion.Major -ne 7 -or (Join-Path $PSHOME 'pwsh.exe') -ne $taskPS7) { throw 'Invoke through the exact verified PowerShell7 host.' }
$taskModulePathBefore = $env:PSModulePath
try {
    # This launch-host environment is process-local; children inherit only these Windows modules.
    $env:PSModulePath = ((Join-Path $PSHOME 'Modules'), (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\Modules')) -join ';'
    $taskPython = Join-Path $taskRoot '.venv-large-gs\Scripts\python.exe'
    $taskSupervisor = Join-Path $taskRoot 'scripts\large_scene_gs\supervise_small_city_hierarchy_stage_006.py'
    $taskSupervisorSHA = '3db6b7ee81d0b7df592609509f8dd565e146179d399f4ca8fbee9baf65e85cf7'
    if ((Get-FileHash -LiteralPath $taskSupervisor -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskSupervisorSHA) { throw 'Fixed006 supervisor source changed.' }
    $taskLaunch = Join-Path $taskRoot "outputs\large-scene-gs-001\official-hierarchy-stage-supervision-launch-006\$SupervisionRun"
    $taskFlags = @('--stage', $Stage, '--supervision-run', $SupervisionRun)
    if ($Chunk) { $taskFlags += @('--chunk', $Chunk) }
    # Always CPU-check first. Only the explicit -Execute switch authorizes one child.
    & $taskPython -X utf8 $taskSupervisor @taskFlags --check-dispatch
    if ($LASTEXITCODE -ne 0) { throw 'Actual006 source/lineage/input/prior-stage/resource/exclusive gate refused.' }
    if ((Get-FileHash -LiteralPath $taskSupervisor -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskSupervisorSHA) { throw 'Fixed006 supervisor changed during CPU check.' }
    if (-not $Execute) {
        [ordered]@{ status='CPU_dispatch_check_only'; GPU_started=$false; stage=$Stage; chunk=$Chunk; supervision_run=$SupervisionRun; requires_Execute_to_start=$true } | ConvertTo-Json
        return
    }
    if (Test-Path -LiteralPath $taskLaunch) { throw 'Existing stage launch directory; refusing overwrite/retry.' }
    New-Item -ItemType Directory -Path $taskLaunch | Out-Null
    Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskLaunch 'launch-source.ps1')
    $taskArguments = @('-X', 'utf8', '-u', ('"' + $taskSupervisor + '"')) + $taskFlags + @('--execute')
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLaunch 'supervisor-stdout.log') `
        -RedirectStandardError (Join-Path $taskLaunch 'supervisor-stderr.log') -PassThru
    $taskReceipt = [ordered]@{
        launched_utc=[DateTimeOffset]::UtcNow.ToString('o'); start_process_pid=$taskProcess.Id
        start_process_creation_utc=$taskProcess.StartTime.ToUniversalTime().ToString('o')
        stage=$Stage; chunk=$Chunk; supervision_run=$SupervisionRun
        python=$taskPython; supervisor=$taskSupervisor; argument_list=$taskArguments
        supervisor_source_sha256=(Get-FileHash -LiteralPath $taskSupervisor -Algorithm SHA256).Hash.ToLowerInvariant()
        launcher_source_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        verified_PS7_host=$taskPS7; child_local_PSModulePath=$env:PSModulePath
        single_stage_only=$true; automatic_retry_or_next_stage=$false
        durability_limit='Actual handles/receipts required; no survival promise after app/job closure,logout or reboot.'
    }
    $taskReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLaunch 'launch-receipt.json') -Encoding utf8
    $taskReceipt | ConvertTo-Json -Depth 5
}
finally { $env:PSModulePath = $taskModulePathBefore }
