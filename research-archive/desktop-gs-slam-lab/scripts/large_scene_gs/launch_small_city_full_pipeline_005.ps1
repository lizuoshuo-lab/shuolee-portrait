$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\Administrator\Desktop\GS-SLAM-Lab'
$taskPython = Join-Path $taskRoot '.venv-large-gs\Scripts\python.exe'
$taskManager = Join-Path $taskRoot 'scripts\large_scene_gs\run_small_city_full_pipeline_005.py'
$taskLaunchDir = Join-Path $taskRoot 'outputs\large-scene-gs-001\official-training-pipeline-launch\small-city-four-chunks-005'
if (Test-Path -LiteralPath $taskLaunchDir) { throw 'Existing pipeline launch directory; refusing overwrite.' }
New-Item -ItemType Directory -Path $taskLaunchDir | Out-Null
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskLaunchDir 'launch-source.ps1')
$taskArguments = @('-u', ('"' + $taskManager + '"'), '--adopt-first-existing', '--execute')
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLaunchDir 'controller-stdout.log') `
    -RedirectStandardError (Join-Path $taskLaunchDir 'controller-stderr.log') -PassThru
$taskReceipt = [ordered]@{
    launched_utc = [DateTimeOffset]::UtcNow.ToString('o')
    launching_powershell_pid = $PID
    start_process_pid = $taskProcess.Id
    start_process_creation_utc = $taskProcess.StartTime.ToUniversalTime().ToString('o')
    manager = $taskManager
    manager_source_sha256 = (Get-FileHash -LiteralPath $taskManager -Algorithm SHA256).Hash.ToLowerInvariant()
    launch_source_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    argument_list = $taskArguments
    existing01_adopted_without_restart = $true
    completed00_preserved_without_restart = $true
    scope = 'Authorized full chunks00->01->10->11, one GPU stage at a time; no Creator/post/Merger.'
    durability_limit = 'Verify live processes after parent exit; no guarantee after app/job closure or reboot.'
}
$taskReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLaunchDir 'launch-receipt.json') -Encoding utf8
$taskReceipt | ConvertTo-Json -Depth 5
