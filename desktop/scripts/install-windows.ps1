param([switch]$Start)
$ErrorActionPreference = 'Stop'
$project = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$binary = Join-Path $project 'src-tauri\target\x86_64-pc-windows-msvc\release\macro-atlas.exe'
if (-not (Test-Path -LiteralPath $binary)) { throw 'The compiled Macro Atlas executable is missing.' }
$folder = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'MacroAtlas\0.1.0'
New-Item -ItemType Directory -Path $folder -Force | Out-Null
$destination = Join-Path $folder 'Macro Atlas.exe'
if (Test-Path -LiteralPath $destination) {
  $installed = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
  $built = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
  if ($installed -ne $built) { throw 'A different app already occupies this version folder; use a new version folder.' }
} else { Copy-Item -LiteralPath $binary -Destination $destination }
Copy-Item -LiteralPath (Join-Path $project 'public\THIRD-PARTY-NOTICES.txt') -Destination (Join-Path $folder 'THIRD-PARTY-NOTICES.txt')
Copy-Item -LiteralPath (Join-Path $project 'README.md') -Destination (Join-Path $folder 'README.md')
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'Macro Atlas.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
if ((Test-Path -LiteralPath $shortcutPath) -and $shortcut.TargetPath -and ($shortcut.TargetPath -ne $destination)) { throw 'An unrelated Macro Atlas shortcut already exists.' }
$shortcut.TargetPath = $destination
$shortcut.WorkingDirectory = $folder
$shortcut.Description = 'Offline world map, macroeconomic charts and saved Dalio research'
$shortcut.IconLocation = "$destination,0"
$shortcut.Save()
[pscustomobject]@{ Executable = $destination; Shortcut = $shortcutPath; SHA256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash } | ConvertTo-Json
if ($Start) {
  $process = Start-Process -FilePath $destination -WorkingDirectory $folder -PassThru
  Write-Output "Macro Atlas process: $($process.Id)"
}
