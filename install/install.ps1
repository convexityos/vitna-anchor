# vitna-anchor's installer for Windows on x64 (gate A12). One command, in PowerShell:
#
#   irm https://github.com/convexityos/vitna-anchor/releases/latest/download/install.ps1 | iex
#
# It fetches the engine built for this machine and checks it against the
# SHA-256 the release wrote into this script; asks the engine which model of
# its catalogue fits the machine (vitna-anchor plan); fetches that model's
# files from Hugging Face at their pinned revisions, resuming any part already
# fetched; has the engine check each against its pinned SHA-256 (vitna-anchor
# verify); and starts the server, whose chat page is at http://127.0.0.1:8765/.
# Run again, it fetches only what is missing. serve.cmd, beside what it
# installs, starts the server again later.
#
#   $env:VITNA_HOME      where it installs (default: %LOCALAPPDATA%\vitna-anchor)
#   $env:VITNA_MODEL     a model of the catalogue in place of the one it chooses
#   $env:VITNA_PORT      the server's port (default: 8765)
#   $env:VITNA_NO_START  1 to stop once everything is fetched and checked
#
# For tests before a release exists: $env:VITNA_ENGINE_URL and
# $env:VITNA_ENGINE_SHA256 fetch the engine from elsewhere, as CI's install
# check does.
#
# The repository's copy is a template: the release fills in its version and
# the engine's SHA-256. It is written for Windows PowerShell 5.1, and calls
# curl.exe by its path, since there curl is a name for Invoke-WebRequest.

& {
    $ErrorActionPreference = 'Stop'
    $Version = '@VERSION@'
    $EngineSha256 = '@SHA256_WINDOWS_X64@'

    function Say([string]$Text) { Write-Host $Text }
    function Fail([string]$Text) { throw "vitna-anchor: $Text" }

    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    if (-not (Test-Path $curl)) { Fail 'this installer needs curl.exe, which Windows 10 version 1803 and later have' }
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
        Fail "there is no build for Windows on $env:PROCESSOR_ARCHITECTURE yet; build it from source: https://github.com/convexityos/vitna-anchor"
    }
    $root = if ($env:VITNA_HOME) { $env:VITNA_HOME } else { Join-Path $env:LOCALAPPDATA 'vitna-anchor' }
    $port = if ($env:VITNA_PORT) { [int]$env:VITNA_PORT } else { 8765 }
    $url = if ($env:VITNA_ENGINE_URL) { $env:VITNA_ENGINE_URL } else { "https://github.com/convexityos/vitna-anchor/releases/download/$Version/vitna-anchor-windows-x64.exe" }
    $want = if ($env:VITNA_ENGINE_SHA256) { $env:VITNA_ENGINE_SHA256 } else { $EngineSha256 }
    if (($want + $url).Contains('@')) { Fail "this is the installer's template, which a release fills in: run the one attached to a release" }

    $bin = Join-Path $root 'bin'
    $models = Join-Path $root 'models'
    New-Item -ItemType Directory -Force -Path $bin, $models | Out-Null
    $engine = Join-Path $bin 'vitna-anchor.exe'
    $utf8 = New-Object Text.UTF8Encoding $false

    # The engine, its output read as the engine writes it. PowerShell reads what
    # a program prints in the console's encoding, and the engine writes UTF-8,
    # being a UTF-8 program on Windows 10 version 1903 and later (before that it
    # writes the ANSI code page), so the two are matched while it runs: a path
    # with any letters in it, as a user's name often has, comes back whole.
    $speaks = if ([Environment]::OSVersion.Version.Build -ge 18362) { $utf8 } else { [Text.Encoding]::Default }
    function Ask([string[]]$Arguments) {
        $was = [Console]::OutputEncoding
        try {
            try { [Console]::OutputEncoding = $speaks } catch { }
            & $engine @Arguments
        } finally {
            try { [Console]::OutputEncoding = $was } catch { }
        }
    }

    # What the engine said, kept beside the install as install.sh keeps it:
    # plan.tsv and verify.tsv, in UTF-8, each line ending in a line feed.
    function Keep([string]$Name, [string[]]$Lines) {
        [IO.File]::WriteAllText((Join-Path $root $Name), (($Lines | ForEach-Object { "$_`n" }) -join ''), $utf8)
    }

    # The file, resuming what is there, unless it is already whole; what is
    # longer than the file should be is fetched anew.
    function Fetch([string]$From, [string]$To, [long]$Size) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $To) | Out-Null
        if (Test-Path $To) {
            $have = (Get-Item $To).Length
            if ($have -eq $Size) { return }
            if ($have -gt $Size) { Remove-Item -Force $To }
        }
        & $curl -fL --retry 5 --retry-delay 3 -C - -o $To $From
        if ($LASTEXITCODE -ne 0) { Fail "could not fetch $From" }
    }

    Say "Fetching the engine: $url"
    & $curl -fL --retry 5 --retry-delay 3 -o "$engine.part" $url
    if ($LASTEXITCODE -ne 0) { Fail 'could not fetch the engine' }
    $got = (Get-FileHash -Algorithm SHA256 "$engine.part").Hash.ToLowerInvariant()
    if ($got -ne $want.ToLowerInvariant()) {
        Remove-Item -Force "$engine.part"
        Fail "the engine's SHA-256 is $got, and the release says $want"
    }
    try {
        Move-Item -Force "$engine.part" $engine
    } catch {
        Fail "could not replace $engine; if the server is running, stop it and run this again"
    }

    $planArgs = @('plan', '--dir', $models)
    if ($env:VITNA_MODEL) { $planArgs += @('--model', $env:VITNA_MODEL) }
    $plan = @(Ask $planArgs)
    $planned = $LASTEXITCODE
    Keep 'plan.tsv' $plan
    if ($planned -ne 0) { Fail 'no plan for this machine' }
    $info = @{}
    $files = @()
    $serve = @()
    foreach ($line in $plan) {
        $f = $line -split "`t"
        switch ($f[0]) {
            'file' { $files += , $f }
            'serve' { $serve = $f[1..($f.Length - 1)] }
            default { $info[$f[0]] = $f[1] }
        }
    }
    $model = $info['model']
    Say ''
    Say "Model: $($info['title'])"
    Say "  $($info['note'])"
    Say "  $($info['why'])"
    Say ('  {0:N1} GB to fetch' -f ([double]$info['fetch'] / 1e9))
    Say ''

    # Each file, then the check; a file that fails it is fetched anew, once.
    foreach ($round in 1, 2) {
        foreach ($f in $files) {
            Say "Fetching $($f[2])"
            Fetch $f[1] (Join-Path $models $f[2].Replace('/', '\')) ([long]$f[3])
        }
        Say 'Checking every file against its pinned SHA-256'
        $verdicts = @(Ask @('verify', '--dir', $models, '--model', $model))
        $verified = $LASTEXITCODE
        Keep 'verify.tsv' $verdicts
        if ($verified -eq 0) { break }
        if ($round -eq 2) {
            $verdicts | ForEach-Object { Write-Host $_ }
            Fail 'the files are not the pinned files'
        }
        foreach ($line in $verdicts) {
            $v = $line -split "`t"
            if ($v[0] -eq 'bad') {
                Say "  $($v[1]): $($v[2]); fetching it anew"
                Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $models $v[1].Replace('/', '\'))
            }
        }
    }

    # A script that starts the server again. Its paths start from its own
    # folder, %~dp0, rather than being written out, so it holds nothing but
    # ASCII however the folder is named, and works still if the folder moves.
    $start = Join-Path $root 'serve.cmd'
    $quoted = ($serve | ForEach-Object { '"' + $_.Replace($models, '%~dp0models') + '"' }) -join ' '
    $text = "@echo off`r`nrem Starts the server vitna-anchor installed, on port %VITNA_PORT% or 8765.`r`nchcp 65001 >nul`r`nsetlocal`r`n" +
        "if not defined VITNA_PORT set VITNA_PORT=8765`r`n`"%~dp0bin\vitna-anchor.exe`" serve $quoted --port %VITNA_PORT%`r`n"
    [IO.File]::WriteAllText($start, $text, $utf8)
    Say ''
    Say "Installed in $root. $start starts the server."
    Say "OpenAI clients: http://127.0.0.1:$port/v1   Anthropic clients: http://127.0.0.1:$port   model: $model"
    if ($env:VITNA_NO_START -eq '1') { return }

    Say "Starting the server; its chat page is at http://127.0.0.1:$port/ (Ctrl+C stops it)"
    $opener = Start-Job -ArgumentList $port -ScriptBlock {
        param($Port)
        for ($i = 0; $i -lt 900; $i++) {
            try {
                Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$Port/v1/health" | Out-Null
                Start-Process "http://127.0.0.1:$Port/"
                return
            } catch {
                Start-Sleep -Seconds 1
            }
        }
    }
    # The server writes to the console itself, which shows its UTF-8 so set.
    $was = [Console]::OutputEncoding
    try {
        try { [Console]::OutputEncoding = $speaks } catch { }
        & $engine serve @serve --port $port
    } finally {
        try { [Console]::OutputEncoding = $was } catch { }
        Remove-Job -Force $opener -ErrorAction SilentlyContinue
    }
}
