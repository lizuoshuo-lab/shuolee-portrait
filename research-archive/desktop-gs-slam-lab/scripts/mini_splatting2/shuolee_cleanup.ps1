param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$planPath = 'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\cleanup-plan.json'
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $plan.ledger -PathType Leaf)) { throw 'Missing compact result ledger' }
foreach ($protected in $plan.protected_files) {
    if ((Get-FileHash -LiteralPath $protected.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $protected.sha256) { throw "Protected input mismatch: $($protected.path)" }
}
$resolvedTargets = @()
foreach ($entry in $plan.targets) {
    if (-not (Test-Path -LiteralPath $entry.path)) { continue }
    $item = Get-Item -LiteralPath $entry.path -Force
    $absolute = [System.IO.Path]::GetFullPath($item.FullName)
    $inside = $false
    foreach ($allowed in $plan.allowed_roots) {
        $boundary = [System.IO.Path]::GetFullPath($allowed).TrimEnd('\') + '\'
        if ($absolute.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)) { $inside=$true }
    }
    if (-not $inside) { throw "Out-of-scope target: $absolute" }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse target rejected: $absolute" }
    if ($item.PSIsContainer) {
        $reparse = Get-ChildItem -LiteralPath $absolute -Recurse -Force | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }
        if ($reparse) { throw "Reparse child rejected: $absolute" }
    }
    foreach ($protected in $plan.protected_files) {
        if ($protected.path.Equals($absolute,[StringComparison]::OrdinalIgnoreCase) -or $protected.path.StartsWith($absolute+'\',[StringComparison]::OrdinalIgnoreCase)) { throw "Protected dependency under target: $absolute" }
    }
    $resolvedTargets += [pscustomobject]@{Path=$absolute;Bytes=$entry.bytes;Files=$entry.files}
}
if (-not $Execute) {
    [pscustomobject]@{State='validated_dry_run';Targets=$resolvedTargets.Count;GiB=[math]::Round(($resolvedTargets|Measure-Object Bytes -Sum).Sum/1GB,3)}|ConvertTo-Json
    exit 0
}
$freeBefore=(Get-PSDrive C).Free
$removed=@()
foreach ($entry in $resolvedTargets) {
    Remove-Item -LiteralPath $entry.Path -Force -Recurse
    if (Test-Path -LiteralPath $entry.Path) { throw "Deletion not verified: $($entry.Path)" }
    $removed += $entry
}
foreach ($protected in $plan.protected_files) {
    if ((Get-FileHash -LiteralPath $protected.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $protected.sha256) { throw "Protected file changed: $($protected.path)" }
}
$receipt=[pscustomobject]@{Owner='shuolee';Completed=(Get-Date).ToString('o');Deleted=$removed;DeletedBytes=($removed|Measure-Object Bytes -Sum).Sum;FreeBefore=$freeBefore;FreeAfter=(Get-PSDrive C).Free;ProtectedHashesVerified=$true}
$receipt|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path (Split-Path -LiteralPath $planPath) 'cleanup-receipt.json') -Encoding UTF8
[pscustomobject]@{DeletedTargets=$removed.Count;DeletedGiB=[math]::Round($receipt.DeletedBytes/1GB,3);FreeGiB=[math]::Round($receipt.FreeAfter/1GB,3);ProtectedHashesVerified=$true}|ConvertTo-Json
