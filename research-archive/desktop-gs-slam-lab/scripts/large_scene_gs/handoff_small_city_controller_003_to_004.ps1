$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\Administrator\Desktop\GS-SLAM-Lab'
$taskOldDirectory = Join-Path $taskRoot 'outputs\large-scene-gs-001\official-training-pipeline\small-city-four-chunks-003'
$taskAuditDirectory = Join-Path $taskRoot 'outputs\large-scene-gs-001\official-training-pipeline-review\small-city-four-chunks-004'
$taskAuditPath = Join-Path $taskAuditDirectory 'controller-handoff-audit.json'
if (Test-Path -LiteralPath $taskAuditPath) { throw 'Existing handoff audit; refusing overwrite.' }
$taskOldState = Get-Content -LiteralPath (Join-Path $taskOldDirectory 'state.json') -Raw | ConvertFrom-Json
if ($taskOldState.controller_pid -ne 26260 -or $taskOldState.controller_parent_pid -ne 19204) { throw 'Old controller identity differs.' }
$taskExpectedCommand = '"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe" -u "C:\Users\Administrator\Desktop\GS-SLAM-Lab\scripts\large_scene_gs\run_small_city_full_pipeline_003.py" --adopt-first-existing --execute'
$taskExpectedShimCommand = '"C:\Users\Administrator\Desktop\GS-SLAM-Lab\.venv-large-gs\Scripts\python.exe" -u "C:\Users\Administrator\Desktop\GS-SLAM-Lab\scripts\large_scene_gs\run_small_city_full_pipeline_003.py" --adopt-first-existing --execute'
$taskBefore = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @(26260,19204,32748,20472,9324,26484) } | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine)
$taskController = $taskBefore | Where-Object { $_.ProcessId -eq 26260 }
$taskShim = $taskBefore | Where-Object { $_.ProcessId -eq 19204 }
if ($null -eq $taskController -or $null -eq $taskShim -or $taskController.Name -ne 'python.exe' `
    -or $taskController.CommandLine -cne $taskExpectedCommand -or $taskShim.CommandLine -cne $taskExpectedShimCommand `
    -or $taskController.ParentProcessId -ne 19204 `
    -or $taskController.CreationDate.ToUniversalTime().ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff'Z'") -ne '2026-10-12T04:35:04.564261Z' `
    -or $taskShim.CreationDate.ToUniversalTime().ToString("yyyy-MM-dd'T'HH:mm:ss.ffffff'Z'") -ne '2026-10-12T04:35:04.525282Z') {
    throw 'Fresh exact controller PID/argv/creation-time check failed; no process stopped.'
}
$taskSource = Join-Path $taskRoot 'scripts\large_scene_gs\run_small_city_full_pipeline_003.py'
if ((Get-FileHash -LiteralPath $taskSource -Algorithm SHA256).Hash.ToLowerInvariant() -ne '9473e98adf324f40eded5723ea1d9d707a3794f4b77c8e5cc0c15ac9afe5364e') {
    throw 'Old frozen controller source changed; no process stopped.'
}
if (@($taskBefore | Where-Object { $_.ProcessId -eq 9324 -and $_.CommandLine -match 'train_single\.py' }).Count -ne 1) {
    throw 'Expected original GPU00 trainer is not live; inspect actual completed receipt before different handoff.'
}
$taskProtectedSources = @()
foreach ($taskProperty in $taskOldState.source_snapshots_sha256.PSObject.Properties) {
    $taskPath = Join-Path $taskRoot "scripts\large_scene_gs\$($taskProperty.Name)"
    $taskHash = (Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskHash -ne $taskProperty.Value) { throw 'An old source_contract tracked file changed.' }
    $taskProtectedSources += [pscustomobject]@{ path=$taskPath; before_sha256=$taskHash }
}
# Terminate only the proven CPU controller. Its GPU supervisor is an adopted
# independent process, not its child. No trainer or GPU supervisor is stopped.
Stop-Process -Id 26260 -Force
Start-Sleep -Milliseconds 500
$taskAfter = @(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in @(26260,19204,32748,20472,9324,26484) } | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine)
if (@($taskAfter | Where-Object { $_.ProcessId -in @(26260,19204) }).Count -ne 0) { throw 'Old controller/shim still present; do not launch004.' }
if (@($taskAfter | Where-Object { $_.ProcessId -in @(32748,9324) }).Count -ne 2) { throw 'GPU supervisor/trainer state changed; inspect before004 launch.' }
foreach ($taskRecord in $taskProtectedSources) {
    $taskRecord | Add-Member -NotePropertyName after_sha256 -NotePropertyValue ((Get-FileHash -LiteralPath $taskRecord.path -Algorithm SHA256).Hash.ToLowerInvariant())
    if ($taskRecord.before_sha256 -ne $taskRecord.after_sha256) { throw 'Protected003 source changed duringhandoff.' }
}
$taskAudit = [ordered]@{
    status='CPU003_controller_only_stopped_GPU00_and_supervisor_preserved'
    observed_utc=[DateTimeOffset]::UtcNow.ToString('o')
    authorization='Parent root explicitly authorized versioned004 controller handoff; no GPU restart or algorithm change.'
    stopped_exact_cpu_controller_pid=26260
    old_controller_source_sha256='9473e98adf324f40eded5723ea1d9d707a3794f4b77c8e5cc0c15ac9afe5364e'
    actual_before_processes=$taskBefore
    actual_after_processes=$taskAfter
    protected003_source_before_after=$taskProtectedSources
    original003_state_preserved=$true
    original003_running_status_is_stale_after_this_explicit_handoff=$true
    gpu_supervisor_stopped=$false
    gpu_trainer_stopped=$false
    gpu_restarted=$false
    next_action='Hidden004 adopts this exact existing00 with creation-time/descendant binding; does not relaunch00.'
}
$taskAudit | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $taskAuditPath -Encoding utf8
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskAuditDirectory 'handoff-source.ps1')
$taskAudit | ConvertTo-Json -Depth 7
