param(
    [Parameter(Mandatory=$true)][ValidateSet('creator','post64','post','merge-unoptimized','merge')][string]$Stage,
    [ValidateSet('0_0','0_1','1_0','1_1')][string]$Chunk,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9-]{1,100}$')][string]$SupervisionRun
)
$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\Administrator\Desktop\GS-SLAM-Lab'
$taskPS7 = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
if ($PSVersionTable.PSVersion.Major -ne 7 -or (Join-Path $PSHOME 'pwsh.exe') -ne $taskPS7) { throw 'Invoke this launcher using the exact verified PowerShell7 host.' }
$taskPython = Join-Path $taskRoot '.venv-large-gs\Scripts\python.exe'
$taskSupervisor = Join-Path $taskRoot 'scripts\large_scene_gs\supervise_small_city_hierarchy_stage_002.py'
$taskLaunch = Join-Path $taskRoot "outputs\large-scene-gs-001\official-hierarchy-stage-supervision-launch\$SupervisionRun"
if (Test-Path -LiteralPath $taskLaunch) { throw 'Existing stage launch directory; refusing overwrite/retry.' }
$taskFlags = @('--stage', $Stage, '--supervision-run', $SupervisionRun)
if ($Chunk) { $taskFlags += @('--chunk', $Chunk) }
# CPU dispatch check must pass before creating a launch receipt or starting a child.
& $taskPython -X utf8 $taskSupervisor @taskFlags --check-dispatch
if ($LASTEXITCODE -ne 0) { throw 'Actual full-training/source/prior-stage/exclusive dispatch gate refused.' }
New-Item -ItemType Directory -Path $taskLaunch | Out-Null
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskLaunch 'launch-source.ps1')
$taskArguments = @('-u', ('"' + $taskSupervisor + '"')) + $taskFlags + @('--execute')
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLaunch 'supervisor-stdout.log') `
    -RedirectStandardError (Join-Path $taskLaunch 'supervisor-stderr.log') -PassThru
$taskReceipt = [ordered]@{
    launched_utc = [DateTimeOffset]::UtcNow.ToString('o')
    start_process_pid = $taskProcess.Id
    start_process_creation_utc = $taskProcess.StartTime.ToUniversalTime().ToString('o')
    stage = $Stage; chunk = $Chunk; supervision_run = $SupervisionRun
    python = $taskPython; supervisor = $taskSupervisor; argument_list = $taskArguments
    supervisor_source_sha256 = (Get-FileHash -LiteralPath $taskSupervisor -Algorithm SHA256).Hash.ToLowerInvariant()
    launcher_source_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    verified_PS7_host = $taskPS7
    single_stage_only = $true; automatic_retry_or_next_stage = $false
    durability_limit = 'Verify actual process handles and final stage receipts; no survival promise after app/job closure, logout or reboot.'
}
$taskReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLaunch 'launch-receipt.json') -Encoding utf8
$taskReceipt | ConvertTo-Json -Depth 5
