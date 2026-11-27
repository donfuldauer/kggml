<#
.SYNOPSIS
  Installs what CI needs on a Windows machine (the hosted windows-* runner, or a contributor's PC), only when
  missing. Idempotent; a no-op on a prepared machine.

.DESCRIPTION
  - Visual Studio 2022 (or its Build Tools) must exist; its C++ x64 tools are required.
  - -Arm64 also requires the MSVC ARM64 build tools and the C++ Clang tools (ggml rejects cl.exe for ARM; issue #91),
    and adds them with the Visual Studio Installer when missing.
  - Ninja is installed with winget (or Chocolatey) when missing.

  Installing Visual Studio components needs administrator rights. The script runs the installer directly when it is
  already elevated (as on hosted runners); otherwise it asks Windows to elevate through the UAC prompt, and if that
  is not possible it fails and prints the one-time command to run in an elevated prompt.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/ci/setup/windows.ps1 -Arm64
#>
param([switch]$Arm64)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Test-Admin {
    $principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$elevated = Test-Admin
Write-Host "runner setup (windows): running as $identity (administrator: $elevated)"

# ---- Visual Studio ----------------------------------------------------------------------------------------------
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw 'Visual Studio 2022 (or Build Tools) is not installed: vswhere.exe is missing' }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'No Visual Studio with the C++ x64 build tools (Microsoft.VisualStudio.Component.VC.Tools.x86.x64)' }
$vsPath = @($vsPath)[0]

$components = @()
if ($Arm64) {
    $components += 'Microsoft.VisualStudio.Component.VC.Tools.ARM64'   # MSVC ARM64 libraries (CRT, STL, link)
    $components += 'Microsoft.VisualStudio.Component.VC.Llvm.Clang'     # clang, which ggml needs for ARM
}
# `vswhere -path` ignores -requires, so ask which instances have the component and compare the path.
function Test-Component([string]$id) {
    $paths = @(& $vswhere -products * -requires $id -property installationPath)
    return $paths -contains $vsPath
}
$missing = @($components | Where-Object { -not (Test-Component $_) })

if ($missing.Count -gt 0) {
    $installer = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vs_installer.exe'
    $vsArgs = @('modify', '--installPath', "`"$vsPath`"") + ($missing | ForEach-Object { @('--add', $_) }) +
        @('--quiet', '--norestart')
    $manual = "`"$installer`" $($vsArgs -join ' ')"
    Write-Host "runner setup (windows): adding $($missing -join ', ') to $vsPath"
    try {
        if ($elevated) {
            $proc = Start-Process -FilePath $installer -ArgumentList $vsArgs -Wait -PassThru -NoNewWindow
        } else {
            $proc = Start-Process -FilePath $installer -ArgumentList $vsArgs -Wait -PassThru -Verb RunAs
        }
    } catch {
        Write-Host "::error::Installing Visual Studio components needs administrator rights, and $identity cannot elevate."
        Write-Host "Run once in an elevated prompt on this machine:"
        Write-Host "  $manual"
        exit 1
    }
    # 3010: success, reboot required (the new components are usable without one).
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        throw "vs_installer exited with $($proc.ExitCode) (is Visual Studio or its installer running?). Command: $manual"
    }
    $still = @($components | Where-Object { -not (Test-Component $_) })
    if ($still.Count -gt 0) { throw "Components still missing after the installer ran: $($still -join ', ')" }
}

# ---- Ninja -------------------------------------------------------------------------------------------------------
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Ninja-build.Ninja --exact --silent --accept-package-agreements --accept-source-agreements
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        choco install ninja -y --no-progress
    } else {
        throw 'Ninja is missing and neither winget nor Chocolatey is available'
    }
    if ($LASTEXITCODE -ne 0) { throw "Installing Ninja failed ($LASTEXITCODE)" }
    # winget puts its command shims in a per-user Links directory; make later steps of this job see them.
    $links = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links'
    if ($env:GITHUB_PATH -and (Test-Path $links)) { Add-Content -Path $env:GITHUB_PATH -Value $links }
}

Write-Host "runner setup (windows): Visual Studio at $vsPath; components OK: $(if ($components) { $components -join ', ' } else { 'C++ x64' })"
