Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: keep the PC awake (screen may turn off) until the queue ends
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
Set-Location E:\Projects\major
$env:HF_HUB_OFFLINE = "1"
$L = "E:\Projects\major\work\logs\seeds.log"
Add-Content -Encoding utf8 $L "=== $(Get-Date) laptop: option B - seed 1 of MotionLite and MotionLite+distillation only (7 runs) ==="
# Retry up to 3 times: a crash (e.g. the CUDA abort at 23:40 on Nov 24) just restarts the grid,
# which skips finished runs and resumes the interrupted one from its checkpoint.
for ($try = 1; $try -le 3; $try++) {
    cmd /c "python -u -m pipeline grid --grid pipeline/configs/grid_seeds.yaml --only motionlite --seeds 1 >> `"$L`" 2>&1"
    if ($LASTEXITCODE -eq 0) { break }
    Add-Content -Encoding utf8 $L "=== $(Get-Date) grid exited with code $LASTEXITCODE (attempt $try of 3), restarting ==="
    Start-Sleep 30
}
cmd /c "python -u -m pipeline probe table >> `"$L`" 2>&1"
cmd /c "python -u -m pipeline figures >> `"$L`" 2>&1"
cmd /c "python reports/make_slides.py >> `"$L`" 2>&1"
Add-Content -Encoding utf8 $L "=== $(Get-Date) OPTION B DONE ==="
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
