[Console]::OutputEncoding=[Text.Encoding]::UTF8
$py = "D:\lct-reid\py312\python.exe"
$pay = "D:\lct-reid\payload"
& $py -m pip install --no-index --no-deps --no-warn-script-location "$pay\torch-2.5.1+cu118-cp312-cp312-win_amd64.whl" 2>&1 | Select-Object -Last 3
& $py -m pip install --no-index --no-deps --no-warn-script-location "$pay\torchvision-0.20.1+cu118-cp312-cp312-win_amd64.whl" 2>&1 | Select-Object -Last 3
