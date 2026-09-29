# build.ps1 - Build script for vitna-anchor on Windows
#
# Compiles every file in src\ into vitna-anchor.exe with the first compiler it
# finds: $env:VITNA_CC if set (a path to clang.exe, gcc.exe or zig.exe), then
# zig, clang or gcc on PATH, then clang in the default LLVM install folder.
#
# -Tests also builds vitna-anchor-tests.exe from src\ (less main.c) and
# tests\unit_tests.c, and runs it.

param([switch]$Tests)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$IncludeDir = Join-Path $ScriptDir "include"
$SrcDir = Join-Path $ScriptDir "src"
$TargetExe = Join-Path $ScriptDir "vitna-anchor.exe"
$AliasExe = Join-Path $ScriptDir "vitna-engine.exe"
$TestsExe = Join-Path $ScriptDir "vitna-anchor-tests.exe"

$SrcFiles = @(Get-ChildItem -Path $SrcDir -Filter "*.c" | Sort-Object Name | ForEach-Object { $_.FullName })
$LibFiles = @($SrcFiles | Where-Object { (Split-Path -Leaf $_) -ne "main.c" })

# Detect available compiler
$Compiler = $null
$ExtraFlags = @()

if ($env:VITNA_CC) {
    $Compiler = $env:VITNA_CC
    if ((Split-Path -Leaf $Compiler) -like "zig*") {
        $ExtraFlags = @("cc")
    }
} elseif (Get-Command zig -ErrorAction SilentlyContinue) {
    $Compiler = "zig"
    $ExtraFlags = @("cc")
} elseif (Get-Command clang -ErrorAction SilentlyContinue) {
    $Compiler = "clang"
} elseif (Get-Command gcc -ErrorAction SilentlyContinue) {
    $Compiler = "gcc"
} elseif (Test-Path "C:\Program Files\LLVM\bin\clang.exe") {
    $Compiler = "C:\Program Files\LLVM\bin\clang.exe"
}

if (-not $Compiler) {
    Write-Error "No supported C compiler found. Set VITNA_CC, or put zig, clang or gcc on PATH."
    exit 1
}

$CommonFlags = @()
$CommonFlags += $ExtraFlags
$CommonFlags += @("-std=c11", "-O2", "-Wall", "-Wextra", "-I$IncludeDir")

Write-Host "Compiling vitna-anchor using $Compiler..."
& $Compiler ($CommonFlags + $SrcFiles + @("-o", $TargetExe, "-lws2_32"))

if ($LASTEXITCODE -ne 0 -or -not (Test-Path $TargetExe)) {
    Write-Error "Compilation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
Copy-Item -Path $TargetExe -Destination $AliasExe -Force
Write-Host "Successfully built: $TargetExe (alias: $AliasExe)"

if ($Tests) {
    $TestSource = Join-Path $ScriptDir "tests\unit_tests.c"
    & $Compiler ($CommonFlags + $LibFiles + @($TestSource, "-o", $TestsExe, "-lws2_32"))
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Compiling the unit tests failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    & $TestsExe
    exit $LASTEXITCODE
}
exit 0
