Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
Wait-Process -Id 17356 -ErrorAction SilentlyContinue
Set-Location E:\Projects\major
$env:HF_HUB_OFFLINE = "1"
$L = "work\logs\stage3.log"
Add-Content $L "=== $(Get-Date) stage 3 start: VideoMAE partial fine-tuning (5 protocols) ==="
python -u -m pipeline grid --grid pipeline/configs/grid_videomae.yaml *>> $L
python -u -m pipeline probe table *>> $L
python -u -m pipeline figures *>> $L
Add-Content $L "=== $(Get-Date) STAGE 3 ALL DONE ==="
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
