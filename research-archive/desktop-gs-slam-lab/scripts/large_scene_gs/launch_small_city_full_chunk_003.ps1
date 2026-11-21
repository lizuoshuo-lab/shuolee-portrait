param([Parameter(Mandatory=$true)][ValidateSet('0_0','0_1','1_0','1_1')][string]$Chunk)
$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\Administrator\Desktop\GS-SLAM-Lab'
$taskNames = @{ '0_0'='small-city-chunk00-i30000-r4-003'; '0_1'='small-city-chunk01-i30000-r4-003'; '1_0'='small-city-chunk10-i30000-r4-003'; '1_1'='small-city-chunk11-i30000-r4-003' }
$taskRun = $taskNames[$Chunk]
$taskPython = Join-Path $taskRoot '.venv-large-gs\Scripts\python.exe'
$taskSupervisor = Join-Path $taskRoot 'scripts\large_scene_gs\supervise_small_city_stage.py'
$taskCoarseRun = 'small-city-coarse-i30000-r4-003'
$taskCoarseDirectory = Join-Path $taskRoot "outputs\large-scene-gs-001\official-training\$taskCoarseRun"
$taskLaunchDirectory = Join-Path $taskRoot "outputs\large-scene-gs-001\official-training-supervision-launch\$taskRun"
if (Test-Path -LiteralPath $taskLaunchDirectory) { throw 'Existing launch directory; refusing overwrite.' }
$taskCoarse = Get-Content -LiteralPath (Join-Path $taskCoarseDirectory 'outcome.json') -Raw | ConvertFrom-Json
$taskCoarseConfig = Get-Content -LiteralPath (Join-Path $taskCoarseDirectory 'configuration.json') -Raw | ConvertFrom-Json
$taskInputSha = (Get-FileHash -LiteralPath (Join-Path $taskRoot 'outputs\large-scene-gs-001\official-inputs\small-city-training-002\manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskCoarse.status -ne 'completed' -or $taskCoarseConfig.requested_iterations -lt 30000 -or $taskCoarseConfig.input_manifest_sha256 -ne $taskInputSha) {
    throw 'Actual completed local coarse >=30000 with identical frozen input required.'
}
$taskOtherGpuJob = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match 'train_coarse\.py|train_single\.py|train_post\.py|evaluate_coarse\.py|evaluate_city\.py|evaluate_official_hierarchy\.py'
})
if ($taskOtherGpuJob.Count -gt 0) { throw 'An official trainer or independent evaluator is still live.' }
$taskFreeBytes = (Get-PSDrive -Name C).Free
if ($taskFreeBytes -lt 80GB) { throw 'System-drive reserve below 80 GiB.' }
New-Item -ItemType Directory -Path $taskLaunchDirectory | Out-Null
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskLaunchDirectory 'launch-source.ps1')
$taskArguments = @('-u', ('"' + $taskSupervisor + '"'), '--stage', 'chunk', '--run', $taskRun,
    '--input', 'small-city-training-002', '--chunk', $Chunk, '--coarse-run', $taskCoarseRun,
    '--iterations', '30000', '--resolution', '4', '--exposure', 'official', '--port', '6024')
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLaunchDirectory 'supervisor-stdout.log') `
    -RedirectStandardError (Join-Path $taskLaunchDirectory 'supervisor-stderr.log') -PassThru
$taskReceipt = [ordered]@{
    status = 'hidden_start_process_returned_not_yet_independently_verified'
    launched_utc = [DateTimeOffset]::UtcNow.ToString('o')
    launching_powershell_pid = $PID
    start_process_pid = $taskProcess.Id
    start_process_creation_utc = $taskProcess.StartTime.ToUniversalTime().ToString('o')
    run = $taskRun
    chunk = $Chunk
    python = $taskPython
    supervisor = $taskSupervisor
    supervisor_source_sha256 = (Get-FileHash -LiteralPath $taskSupervisor -Algorithm SHA256).Hash.ToLowerInvariant()
    launch_source_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    input_manifest_sha256 = $taskInputSha
    actual_coarse_run = $taskCoarseRun
    actual_coarse_outcome_sha256 = (Get-FileHash -LiteralPath (Join-Path $taskCoarseDirectory 'outcome.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    argument_list = $taskArguments
    working_directory = $taskRoot
    system_drive_free_bytes = $taskFreeBytes
    requested_iterations = 30000
    expected_optimizer_update_calls = 29999
    recovery_kind = 'new_full_run_from_scratch'
    checkpoint_resume = $false
    short_preflight_model_used_for_initialization = $false
    official_algorithm_modified = $false
    verification_needed = 'Parent PowerShell exit, actual trainer progress, saved final model, all shared sky fields, source/input/coarse binding, and terminal process audit.'
    durability_limit = 'No survival assertion after app/job closure, reboot, or logout.'
}
$taskReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLaunchDirectory 'launch-receipt.json') -Encoding utf8
$taskReceipt | ConvertTo-Json -Depth 5
