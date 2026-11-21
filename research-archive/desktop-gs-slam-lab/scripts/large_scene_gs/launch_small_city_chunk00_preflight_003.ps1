$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\Administrator\Desktop\GS-SLAM-Lab'
$taskPython = Join-Path $taskRoot '.venv-large-gs\Scripts\python.exe'
$taskSupervisor = Join-Path $taskRoot 'scripts\large_scene_gs\supervise_small_city_stage.py'
$taskRun = 'small-city-chunk-0-0-preflight-r4-003'
$taskCoarseRun = 'small-city-coarse-i30000-r4-003'
$taskLaunchDirectory = Join-Path $taskRoot "outputs\large-scene-gs-001\official-training-supervision-launch\$taskRun"
if (Test-Path -LiteralPath $taskLaunchDirectory) { throw 'Existing launch directory; refusing overwrite.' }
$taskCoarseDirectory = Join-Path $taskRoot "outputs\large-scene-gs-001\official-training\$taskCoarseRun"
$taskCoarseReceipt = Get-Content -LiteralPath (Join-Path $taskCoarseDirectory 'outcome.json') -Raw | ConvertFrom-Json
$taskCoarseConfiguration = Get-Content -LiteralPath (Join-Path $taskCoarseDirectory 'configuration.json') -Raw | ConvertFrom-Json
$taskInputSha = (Get-FileHash -LiteralPath (Join-Path $taskRoot 'outputs\large-scene-gs-001\official-inputs\small-city-training-002\manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskCoarseReceipt.status -ne 'completed' -or $taskCoarseConfiguration.requested_iterations -lt 30000 `
    -or $taskCoarseConfiguration.input_manifest_sha256 -ne $taskInputSha) {
    throw 'Actual completed local coarse >=30000 with the identical frozen input required.'
}
$taskOtherGpuJob = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match 'train_coarse\.py|train_single\.py|train_post\.py|evaluate_coarse\.py|evaluate_city\.py|evaluate_official_hierarchy\.py'
})
if ($taskOtherGpuJob.Count -gt 0) { throw 'An official trainer or independent evaluator is still live; wait for GPU release.' }
$taskFreeBytes = (Get-PSDrive -Name C).Free
if ($taskFreeBytes -lt 80GB) { throw 'System-drive reserve below 80 GiB.' }
New-Item -ItemType Directory -Path $taskLaunchDirectory | Out-Null
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskLaunchDirectory 'launch-source.ps1')
$taskArguments = @('-u', ('"' + $taskSupervisor + '"'), '--stage', 'chunk', '--run', $taskRun,
    '--input', 'small-city-training-002', '--chunk', '0_0', '--coarse-run', $taskCoarseRun,
    '--iterations', '64', '--resolution', '4', '--exposure', 'official', '--port', '6023')
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLaunchDirectory 'supervisor-stdout.log') `
    -RedirectStandardError (Join-Path $taskLaunchDirectory 'supervisor-stderr.log') -PassThru
$taskReceipt = [ordered]@{
    status = 'hidden_start_process_returned_not_yet_independently_verified'
    launched_utc = [DateTimeOffset]::UtcNow.ToString('o')
    launching_powershell_pid = $PID
    start_process_pid = $taskProcess.Id
    start_process_creation_utc = $taskProcess.StartTime.ToUniversalTime().ToString('o')
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
    requested_iterations = 64
    expected_optimizer_update_calls = 63
    official_algorithm_modified = $false
    scope = 'Actual short chunk initialization/optimization preflight only; not completed formal 30000 training.'
    verification_needed = 'After this PowerShell exits, verify actual supervisor/trainer process commands, then require saved-model and frozen-skybox audit.'
    durability_limit = 'No system scheduler or power settings changed. No assertion of survival after reboot, user logout, or app job closure.'
}
$taskReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLaunchDirectory 'launch-receipt.json') -Encoding utf8
$taskReceipt | ConvertTo-Json -Depth 5
