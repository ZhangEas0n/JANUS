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

& ".\.venv\Scripts\python.exe" -m pip --isolated @PipArgs
exit $LASTEXITCODE
