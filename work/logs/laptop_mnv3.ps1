Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
Wait-Process -Id 29328 -ErrorAction SilentlyContinue
Set-Location E:\Projects\major
$env:HF_HUB_OFFLINE = "1"
$L = "E:\Projects\major\work\logs\seeds.log"
Add-Content -Encoding utf8 $L "=== $(Get-Date) laptop: MobileNetV3-TSM seed 2 (Colab 3 does seed 1) ==="
cmd /c "python -u -m pipeline grid --grid pipeline/configs/grid_seeds.yaml --only mnv3_tsm --seeds 2 >> `"$L`" 2>&1"
cmd /c "python -u -m pipeline probe table >> `"$L`" 2>&1"
Add-Content -Encoding utf8 $L "=== $(Get-Date) LAPTOP PART DONE ==="
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
