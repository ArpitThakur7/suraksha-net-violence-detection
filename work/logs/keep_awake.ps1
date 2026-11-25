Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED (no display flag: screen may turn off, PC stays awake)
[W.P]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
Add-Content E:\Projects\major\work\logs\probe.log "=== $(Get-Date) keep-awake helper started (until follow-up PID 11412 exits) ==="
Wait-Process -Id 11412 -ErrorAction SilentlyContinue
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
Add-Content E:\Projects\major\work\logs\probe.log "=== $(Get-Date) keep-awake helper released ==="
