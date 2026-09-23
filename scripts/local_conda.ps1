param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CondaArgs
)

$env:CONDA_NO_PLUGINS = "true"
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:PIP_NO_INDEX -ErrorAction SilentlyContinue

& "D:\software\anaconda\Scripts\conda.exe" @CondaArgs
exit $LASTEXITCODE
