Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
Wait-Process -Id 11412 -ErrorAction SilentlyContinue
Set-Location E:\Projects\major
$env:HF_HUB_OFFLINE = "1"
$L = "work\logs\stage2.log"
Add-Content $L "=== $(Get-Date) stage 2 start: tests, VideoMAE+CLIP probe, distillation, speed bench ==="
python -m pytest -q *>> $L
python -u -m pipeline probe train --backbone videomae+clip *>> $L
python -u -m pipeline grid --grid pipeline/configs/grid_distill.yaml *>> $L
python -u -m pipeline bench --speed *>> $L
python -u -m pipeline probe table *>> $L
python -u -m pipeline figures *>> $L
Add-Content $L "=== $(Get-Date) STAGE 2 ALL DONE ==="
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
