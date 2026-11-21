param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\Administrator\Desktop\GS-SLAM-Lab'
$runName = 'small-city-chunk11-observation-i30000-r4-001'
$pythonPath = Join-Path $taskRoot '.venv-large-gs\Scripts\python.exe'
$supervisorPath = Join-Path $taskRoot 'scripts\large_scene_gs\supervise_small_city_observation_001.py'
$planPath = Join-Path $taskRoot "outputs\large-scene-gs-001\official-training-observation-plans\$runName-v3\configuration.json"
$expected = [ordered]@{
    'scripts\large_scene_gs\supervise_small_city_observation_001.py' = 'bb0a8cd8f7efd2666c56022a397ecdb2b969b02c20806538c25f90c10a7ec370'
    'scripts\large_scene_gs\run_small_city_observation_001.py' = 'ebe2581d8ae8bee8fa48519c383030a0e74be0677c38b81855715dee6a16570c'
    'scripts\large_scene_gs\observe_small_city_train_single_001.py' = '59bc69b80fb50283cddd5a35c115fb3ecd9ba37bef675c74e6f1736cc02af384'
    'scripts\large_scene_gs\run_local.py' = '1fa39a1af880809d9a8cabc62f219458358acad6b661780c6f7321df33dcc2a2'
    "outputs\large-scene-gs-001\official-training-observation-plans\$runName-v3\configuration.json" = '8b51155d5300bd75c3ebef0c4544da8e2b5a393cd90909ed97430ca0544df826'
    'outputs\large-scene-gs-001\official-training-observation-plans\small-city-initial-suspect-oracle-001\expected-raw.json' = '3fc724de94b8609f443ec63e9cdb6705a2083ca4785298af57d208064899c236'
}
foreach ($relativePath in $expected.Keys) {
    $absolutePath = Join-Path $taskRoot $relativePath
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $absolutePath).Hash.ToLowerInvariant() -ne $expected[$relativePath]) {
        throw "Frozen observation source/plan/oracle changed: $relativePath"
    }
}
$cpuResultText = & $pythonPath '-X' 'utf8' $supervisorPath '--check-dispatch'
if ($LASTEXITCODE -ne 0) { throw 'Actual CPU observation dispatch gates rejected' }
$cpuResult = $cpuResultText | ConvertFrom-Json
if ($cpuResult.GPU_execution_started -ne $false -or $cpuResult.CUDA_initialized -ne $false) { throw 'CPU prelaunch receipt scope invalid' }
if (-not $Execute) {
    $cpuResult | Select-Object status,observed_utc,plan_sha256,system_drive_free_bytes,GPU_execution_started,CUDA_initialized | ConvertTo-Json -Depth 3
    return
}
$launchPath = Join-Path $taskRoot "outputs\large-scene-gs-001\official-training-observation-launch\$runName"
if (Test-Path -LiteralPath $launchPath) { throw 'Preserve existing observation launch; no retry/overwrite' }
New-Item -ItemType Directory -Path $launchPath | Out-Null
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $launchPath 'launcher-source.ps1')
Copy-Item -LiteralPath $supervisorPath -Destination (Join-Path $launchPath 'supervisor-source.py')
Copy-Item -LiteralPath $planPath -Destination (Join-Path $launchPath 'prepared-configuration.json')
$supervisorArguments = @('-u', $supervisorPath, '--execute')
$quotedArguments = @($supervisorArguments | ForEach-Object {
    if ($_.Contains('"')) { throw 'Unexpected quote in fixed process argument' }
    '"' + $_ + '"'
})
$process = Start-Process -FilePath $pythonPath -ArgumentList $quotedArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $launchPath 'supervisor-stdout.log') -RedirectStandardError (Join-Path $launchPath 'supervisor-stderr.log')
$receipt = [ordered]@{
    status = 'Hidden_observation_supervisor_launched_pending_actual_process_and_log_verification'
    launched_utc = [DateTime]::UtcNow.ToString('o')
    run = $runName
    launching_powershell_pid = $PID
    launching_powershell_creation_utc = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
    start_process_pid = $process.Id
    start_process_creation_utc = $process.StartTime.ToUniversalTime().ToString('o')
    python = $pythonPath
    supervisor_argument_list = $supervisorArguments
    actual_prelaunch_CPU_gates = $cpuResult
    prepared_plan_sha256 = $cpuResult.plan_sha256
    supervisor_source_sha256 = $expected['scripts\large_scene_gs\supervise_small_city_observation_001.py']
    launcher_source_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    source_hashes = $expected
    full_requested_iterations = 30000
    observation_only_not_formal_completion = $true
    retry_or_next_GPU_stage_performed = $false
    persistence_scope = 'Verify actual alive PID birth/argv/log after launching PowerShell exits. Hidden does not guarantee survival after app/job closure, logout or reboot.'
}
$receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $launchPath 'launch-receipt.json') -Encoding utf8
$receipt | Select-Object status,run,start_process_pid,start_process_creation_utc,launching_powershell_pid,observation_only_not_formal_completion | ConvertTo-Json -Depth 3
