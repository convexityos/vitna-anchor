# build.ps1 - Build script for vitna-anchor on Windows

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$IncludeDir = Join-Path $ScriptDir "include"
$SrcDir = Join-Path $ScriptDir "src"
$TargetExe = Join-Path $ScriptDir "vitna-anchor.exe"
$AliasExe = Join-Path $ScriptDir "vitna-engine.exe"

$SrcFiles = @(
    (Join-Path $SrcDir "main.c"),
    (Join-Path $SrcDir "safetensors.c"),
    (Join-Path $SrcDir "expert_store.c"),
    (Join-Path $SrcDir "router.c"),
    (Join-Path $SrcDir "kernels.c"),
    (Join-Path $SrcDir "kv_cache.c"),
    (Join-Path $SrcDir "radix_kv.c"),
    (Join-Path $SrcDir "grammar.c"),
    (Join-Path $SrcDir "speculative.c"),
    (Join-Path $SrcDir "generate.c"),
    (Join-Path $SrcDir "server.c")
)

# Detect available compiler
$Compiler = $null
$ExtraFlags = @()

if (Get-Command zig -ErrorAction SilentlyContinue) {
    $Compiler = "zig"
    $ExtraFlags = @("cc", "-target", "x86_64-windows")
} elseif (Get-Command clang -ErrorAction SilentlyContinue) {
    $Compiler = "clang"
} elseif (Test-Path "C:\Program Files\LLVM\bin\clang.exe") {
    $Compiler = "C:\Program Files\LLVM\bin\clang.exe"
} elseif (Get-Command gcc -ErrorAction SilentlyContinue) {
    $Compiler = "gcc"
}

if (-not $Compiler) {
    Write-Error "No supported C compiler found (zig, clang, or gcc)."
    exit 1
}

Write-Host "Compiling vitna-anchor using $Compiler..."
$Args = @()
$Args += $ExtraFlags
$Args += @("-O2", "-Wall", "-I$IncludeDir")
$Args += $SrcFiles
$Args += @("-o", $TargetExe, "-lws2_32")

& $Compiler $Args

if ($LASTEXITCODE -eq 0 -and (Test-Path $TargetExe)) {
    Copy-Item -Path $TargetExe -Destination $AliasExe -Force
    Write-Host "Successfully built: $TargetExe (alias: $AliasExe)"
    exit 0
} else {
    Write-Error "Compilation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

