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
$test = Join-Path $project 'tests\native.mjs'
$bootstrap = "import(require('url').pathToFileURL(process.argv[1]).href)"
$runner = New-Object System.Diagnostics.Process
$runner.StartInfo.FileName = $node
$runner.StartInfo.Arguments = '--eval "' + $bootstrap + '" "' + $test + '" "' + $Executable + '"'
$runner.StartInfo.UseShellExecute = $false
$runner.StartInfo.EnvironmentVariables['NODE_UNC_HOST_ALLOWLIST'] = 'wsl.localhost'
$runner.StartInfo.RedirectStandardOutput = $true
$runner.StartInfo.RedirectStandardError = $true
$null = $runner.Start()
$stdout = $runner.StandardOutput.ReadToEndAsync()
$stderr = $runner.StandardError.ReadToEndAsync()
if (-not $runner.WaitForExit(300000)) { $runner.Kill(); throw 'Native test exceeded 300 seconds.' }
Write-Output $stdout.GetAwaiter().GetResult()
Write-Output $stderr.GetAwaiter().GetResult()
exit $runner.ExitCode
