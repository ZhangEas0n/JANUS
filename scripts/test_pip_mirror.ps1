$proxyVars = @(
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "PIP_NO_INDEX"
)

foreach ($name in $proxyVars) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
$env:NO_PROXY = "*"
$env:no_proxy = "*"

Get-ChildItem Env: | Where-Object { $_.Name -match "proxy|PIP" } | Format-Table -AutoSize
& ".\.venv\Scripts\python.exe" -c "import urllib.request; print(urllib.request.getproxies())"
& ".\.venv\Scripts\python.exe" -m pip --isolated install --proxy= --index-url "https://pypi.tuna.tsinghua.edu.cn/simple" --trusted-host "pypi.tuna.tsinghua.edu.cn" "hydra-core==1.1.2"
exit $LASTEXITCODE
