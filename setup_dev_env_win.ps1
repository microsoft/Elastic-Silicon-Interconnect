###
###  Set up a build environment on Windows
###
### Must have vcpkg (https://github.com/microsoft/vcpkg) or tell this script
### where to install it.
###     Set $env:VCPKG_ROOT to the install/clone
###  Must install Python3 w/ pip first

param (
    [Parameter(Mandatory=$false)]
    [string]$VcpkgInstallPath=$null
)

if ($null -ne $VcpkgInstallPath) {
    $env:VCPKG_ROOT = $VcpkgInstallPath
    if (-not (Test-Path "$env:VCPKG_ROOT/vcpkg.exe")) {
        git clone https://github.com/Microsoft/vcpkg.git $env:VCPKG_ROOT
        & "$env:VCPKG_ROOT/bootstrap-vcpkg.bat"
    }
    $env:VCPKG_ROOT = Resolve-Path $VcpkgInstallPath
}

if ($null -eq $env:VCPKG_ROOT) {
    Write-Error "Install VCPKG and set the VCPKG_ROOT to it's root directory"
} else {
    & "$env:VCPKG_ROOT/vcpkg.exe" install capnproto:x64-windows
    $env:PATH="$env:PATH;$env:VCPKG_ROOT/installed/x64-windows/tools/capnproto"
}

python -m pip install -U pytest
