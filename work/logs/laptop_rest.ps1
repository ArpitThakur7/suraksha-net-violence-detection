Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
Wait-Process -Id 32208 -ErrorAction SilentlyContinue
Set-Location E:\Projects\major
$env:HF_HUB_OFFLINE = "1"
$L = "E:\Projects\major\work\logs\seeds.log"
Add-Content -Encoding utf8 $L "=== $(Get-Date) laptop: all remaining seed runs (MotionLite, distillation, MobileNetV3-TSM seed 1) ==="
cmd /c "python -u -m pipeline grid --grid pipeline/configs/grid_seeds.yaml >> `"$L`" 2>&1"
cmd /c "python -u -m pipeline probe table >> `"$L`" 2>&1"
cmd /c "python -u -m pipeline figures >> `"$L`" 2>&1"
cmd /c "python reports/make_slides.py >> `"$L`" 2>&1"
Add-Content -Encoding utf8 $L "=== $(Get-Date) SEEDS ALL DONE ==="
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
