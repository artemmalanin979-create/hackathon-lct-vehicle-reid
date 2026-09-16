[Console]::OutputEncoding=[Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"
$pay = "D:\lct-reid\payload"
$dst = "$pay\torch-2.5.1+cu118-cp312-cp312-win_amd64.whl"
if (Test-Path $dst) { Remove-Item $dst -Force }
$out = [System.IO.File]::Create($dst)
Get-ChildItem "$pay\torch.part.*" | Sort-Object Name | ForEach-Object {
  Write-Host ("склейка " + $_.Name)
  $in = [System.IO.File]::OpenRead($_.FullName)
  $in.CopyTo($out)
  $in.Close()
}
$out.Close()
Write-Host ("размер " + (Get-Item $dst).Length)
(Get-FileHash -Algorithm MD5 $dst).Hash
