param([string]$Executable)
$ErrorActionPreference = 'Stop'
$project = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
if (-not $Executable) { $Executable = Join-Path $project 'src-tauri\target\x86_64-pc-windows-msvc\release\macro-atlas.exe' }
if (-not (Test-Path -LiteralPath $Executable)) { throw 'The compiled executable is missing.' }
$node = $env:ATLAS_WINDOWS_NODE
if (-not $node) {
  $installedNode = Get-Command node.exe -ErrorAction SilentlyContinue
  if ($installedNode) { $node = $installedNode.Source }
}
if (-not $node) {
  $node = Join-Path $env:LOCALAPPDATA 'Programs\Microsoft VS Code\Code.exe'
  if (-not (Test-Path -LiteralPath $node)) { throw 'Set ATLAS_WINDOWS_NODE to a Windows Node executable.' }
  $env:ELECTRON_RUN_AS_NODE = '1'
}
$env:NODE_UNC_HOST_ALLOWLIST = 'wsl.localhost'
$test = Join-Path $project 'tests\normal-year-screen-native.mjs'
# A pipeline makes PowerShell wait for the GUI Code.exe Node fallback and
# populate LASTEXITCODE before this wrapper returns.
& $node $test $Executable | Out-Host
exit $LASTEXITCODE
