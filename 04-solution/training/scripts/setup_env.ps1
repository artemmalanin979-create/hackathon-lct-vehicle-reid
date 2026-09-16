[Console]::OutputEncoding=[Text.Encoding]::UTF8
$ErrorActionPreference = "Continue"
$py = "D:\lct-reid\py312\python.exe"
$whl = "D:\lct-reid\wheels"
$sp = "D:\lct-reid\py312\Lib\site-packages"
New-Item -ItemType Directory -Force -Path $sp | Out-Null
# wheel = zip: pip распаковывается в site-packages напрямую (интернета нет, pip сам себя ставить отказывается)
$pipwhl = (Get-ChildItem "$whl\pip-*.whl").FullName
Copy-Item $pipwhl "$whl\pip.zip" -Force
Expand-Archive -Path "$whl\pip.zip" -DestinationPath $sp -Force
& $py -m pip --version
& $py -m pip install --no-index --find-links $whl --no-warn-script-location setuptools numpy pillow filelock typing_extensions sympy mpmath networkx jinja2 MarkupSafe fsspec 2>&1 | Select-Object -Last 2
& $py -m pip list
