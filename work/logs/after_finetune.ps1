Wait-Process -Id 2592 -ErrorAction SilentlyContinue
Set-Location E:\Projects\major
$env:HF_HUB_DISABLE_XET = "1"
Add-Content work\logs\probe.log "=== $(Get-Date) fine-tuning finished; CLIP extraction + probes + table + figures ==="
python -u -m pipeline probe extract --backbone clip --batch 4 --workers 4 *>> work\logs\probe.log
python -u -m pipeline probe train --backbone clip *>> work\logs\probe.log
python -u -m pipeline probe table *>> work\logs\probe.log
python -u -m pipeline figures *>> work\logs\probe.log
Add-Content work\logs\probe.log "=== $(Get-Date) ALL DONE ==="
