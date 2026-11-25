Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
Set-Location E:\Projects\major
$env:HF_HUB_OFFLINE = "1"
$L = "E:\Projects\major\work\logs\seeds.log"
function Run($a) { cmd /c "python -u -m pipeline $a >> `"$L`" 2>&1" }
Add-Content -Encoding utf8 $L "=== $(Get-Date) seeds start: grid_seeds.yaml (40 runs) ==="
Run "grid --grid pipeline/configs/grid_seeds.yaml --only videomae_ft"
Run "probe table"
Run "figures"
Add-Content -Encoding utf8 $L "=== $(Get-Date) SEEDS ALL DONE ==="
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
