#Requires -RunAsAdministrator
# Prepare Microsoft's Linux platform without restarting Windows automatically.
param([switch]$PlatformOnly)
$ErrorActionPreference = 'Stop'
$gsLabRoot = Split-Path -Parent $PSScriptRoot
$gsSetupDirectory = Join-Path $gsLabRoot 'outputs\fast-livo2-wsl-setup'
New-Item -ItemType Directory -Path $gsSetupDirectory -Force | Out-Null
$gsReceiptPath = Join-Path $gsSetupDirectory ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json')
Start-Transcript -LiteralPath ([IO.Path]::ChangeExtension($gsReceiptPath, '.log')) -Force | Out-Null
@{status='started_windows_platform_configuration';process_id=$PID;automatic_restart_performed=$false;ubuntu_target='D:\WSL\Ubuntu-20.04'} | ConvertTo-Json | Set-Content -LiteralPath $gsReceiptPath -Encoding utf8
$gsFeatures = @('Microsoft-Windows-Subsystem-Linux', 'VirtualMachinePlatform')
$gsChanges = @()
$gsRestartRequired = $false
foreach ($gsFeatureName in $gsFeatures) {
    $gsFeatureBefore = Get-WindowsOptionalFeature -Online -FeatureName $gsFeatureName
    if ($gsFeatureBefore.State -eq 'EnablePending') { $gsRestartRequired = $true }
    if ($gsFeatureBefore.State -notin @('Enabled', 'EnablePending')) {
        $gsEnabledFeature = Enable-WindowsOptionalFeature -Online -FeatureName $gsFeatureName -All -NoRestart
        $gsRestartRequired = $gsRestartRequired -or $gsEnabledFeature.RestartNeeded
    }
    $gsFeatureAfter = Get-WindowsOptionalFeature -Online -FeatureName $gsFeatureName
    if ($gsFeatureAfter.State -eq 'EnablePending') { $gsRestartRequired = $true }
    $gsChanges += [ordered]@{feature=$gsFeatureName; before=[string]$gsFeatureBefore.State; after=[string]$gsFeatureAfter.State}
}
$gsReceipt = [ordered]@{
    recorded_utc=(Get-Date).ToUniversalTime().ToString('o')
    features=$gsChanges
    restart_required=$gsRestartRequired
    automatic_restart_performed=$false
    ubuntu_install_attempted=$false
    ubuntu_target='D:\WSL\Ubuntu-20.04'
    status='windows_features_checked'
}
if ($gsRestartRequired) {
    $gsReceipt.status = 'restart_required_before_linux_installation'
    $gsReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $gsReceiptPath -Encoding utf8
    Write-Output 'Windows restart is required. Work is saved. This script will not restart the computer.'
    Write-Output ('After restarting, run this same script again. Receipt: ' + $gsReceiptPath)
    Stop-Transcript | Out-Null
    exit 0
}
if ($PlatformOnly) {
    $gsReceipt.status='windows_platform_ready_requires_linux_boot_probe'
    $gsReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $gsReceiptPath -Encoding utf8
    Stop-Transcript | Out-Null
    exit 0
}
# Official Microsoft installer, explicitly retain the Ubuntu version expected by ROS Noetic.
& wsl.exe --install --distribution Ubuntu-20.04 --location 'D:\WSL\Ubuntu-20.04' --version 2 --no-launch --web-download
$gsInstallCode=$LASTEXITCODE
$gsReceipt.ubuntu_install_attempted = $true
$gsReceipt.wsl_install_exit_code = $gsInstallCode
$gsReceipt.status = if ($gsInstallCode -eq 0) { 'ubuntu_install_command_completed_requires_readback' } else { 'ubuntu_install_command_failed' }
$gsReceipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $gsReceiptPath -Encoding utf8
Stop-Transcript | Out-Null
if ($gsInstallCode -ne 0) { throw ('WSL installer exit code: ' + $gsInstallCode) }
Write-Output ('Receipt: ' + $gsReceiptPath)
