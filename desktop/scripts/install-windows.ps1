param([switch]$Start)
$ErrorActionPreference = 'Stop'
$project = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$binary = Join-Path $project 'src-tauri\target\x86_64-pc-windows-msvc\release\macro-atlas.exe'
if (-not (Test-Path -LiteralPath $binary)) { throw 'The compiled Macro Atlas executable is missing.' }
$version = (Get-Content -LiteralPath (Join-Path $project 'src-tauri\tauri.conf.json') -Raw | ConvertFrom-Json).version
if ($version -notmatch '^\d+\.\d+\.\d+$') { throw 'The app version is invalid.' }
$appFolder = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'MacroAtlas'
$folder = Join-Path $appFolder $version
New-Item -ItemType Directory -Path $folder -Force | Out-Null
$destination = Join-Path $folder 'Macro Atlas.exe'
if (Test-Path -LiteralPath $destination) {
  $installed = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
  $built = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash
  if ($installed -ne $built) { throw 'A different app already occupies this version folder; use a new version folder.' }
} else { Copy-Item -LiteralPath $binary -Destination $destination }
$financialFolder = Join-Path $folder 'financial-data'
$financialSource = Join-Path $project 'financial-data'
$packs = @(Get-ChildItem -LiteralPath $financialSource -Filter '*.sqlite' -File)
if ($packs.Count -eq 0) { throw 'The financial history pack is missing.' }
New-Item -ItemType Directory -Path $financialFolder -Force | Out-Null
foreach ($pack in $packs) {
  $packHash = (Get-FileHash -LiteralPath $pack.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($pack.BaseName -cne $packHash) { throw 'Financial history checksum failed before installation.' }
  $packDestination = Join-Path $financialFolder $pack.Name
  if (Test-Path -LiteralPath $packDestination) {
    if ((Get-FileHash -LiteralPath $packDestination -Algorithm SHA256).Hash.ToLowerInvariant() -cne $packHash) { throw 'A different financial pack already occupies this version folder.' }
  } else { Copy-Item -LiteralPath $pack.FullName -Destination $packDestination }
  if ((Get-FileHash -LiteralPath $packDestination -Algorithm SHA256).Hash.ToLowerInvariant() -cne $packHash) { throw 'Installed financial history checksum failed.' }
}
Copy-Item -LiteralPath (Join-Path $project 'public\THIRD-PARTY-NOTICES.txt') -Destination (Join-Path $folder 'THIRD-PARTY-NOTICES.txt')
$readme = Get-Content -LiteralPath (Join-Path $project 'README.md') -Raw -Encoding UTF8
[IO.File]::WriteAllText((Join-Path $folder 'README.md'), $readme.Replace('(../docs/', '(docs/'))
foreach ($relative in @('docs/company-valuation-framework.md', 'docs/company-analysis-template.md', 'docs/holmen-valuation-2026-09-11.md', 'decisions/0035-company-value-and-price.md')) {
  $source = Join-Path (Split-Path $project -Parent) $relative
  $target = Join-Path $folder $relative
  New-Item -ItemType Directory -Path (Split-Path $target -Parent) -Force | Out-Null
  Copy-Item -LiteralPath $source -Destination $target
}
Copy-Item -LiteralPath (Join-Path $project 'HANDOFF.md') -Destination (Join-Path $folder 'HANDOFF.md')
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'Macro Atlas.lnk'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$ownedPath = '^' + [regex]::Escape($appFolder) + '\\\d+\.\d+\.\d+\\Macro Atlas\.exe$'
if ((Test-Path -LiteralPath $shortcutPath) -and $shortcut.TargetPath -and ($shortcut.TargetPath -notmatch $ownedPath)) { throw 'An unrelated Macro Atlas shortcut already exists.' }
$shortcut.TargetPath = $destination
$shortcut.WorkingDirectory = $folder
$shortcut.Description = 'Offline macro, sector and company observatories with saved research'
$shortcut.IconLocation = "$destination,0"
$shortcut.Save()
[pscustomobject]@{ Executable = $destination; Shortcut = $shortcutPath; SHA256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash } | ConvertTo-Json
if ($Start) {
  # Close only older Atlas windows owned by this installation.
  foreach ($old in @(Get-Process -Name 'Macro Atlas' -ErrorAction SilentlyContinue | Where-Object { $_.Path -match $ownedPath -and $_.Path -ne $destination })) {
    $null = $old.CloseMainWindow()
    if (-not $old.WaitForExit(5000)) {
      # A closed WebView2 window can leave its viewer process alive briefly.
      # Explicitly saved comparisons and preferences are already persisted.
      $old.Kill()
      if (-not $old.WaitForExit(5000)) { throw 'The previous Atlas process could not stop.' }
    }
  }
  $process = Start-Process -FilePath $destination -WorkingDirectory $folder -PassThru
  Write-Output "Macro Atlas process: $($process.Id)"
}
