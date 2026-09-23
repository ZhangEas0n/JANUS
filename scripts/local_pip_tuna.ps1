param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PipArgs
)

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

& ".\.venv\Scripts\python.exe" -m pip --isolated @PipArgs --index-url "https://pypi.tuna.tsinghua.edu.cn/simple" --trusted-host "pypi.tuna.tsinghua.edu.cn"
exit $LASTEXITCODE
