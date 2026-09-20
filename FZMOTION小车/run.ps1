[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ClientArguments
)

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $pythonCommand) {
    Write-Error "Python was not found on PATH. Install 64-bit Python 3.10 or newer."
    exit 1
}

if ($ClientArguments.Count -eq 0) {
    $ClientArguments = @("listen")
}

Push-Location $PSScriptRoot
try {
    & $pythonCommand.Source (Join-Path $PSScriptRoot "fzmotion_client.py") @ClientArguments
    $clientExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $clientExitCode
