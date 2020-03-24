###
###  Set up a build environment on Windows
###
###  Must install Python3 w/ pip first

# Change to "tools" dir
$origDir = Get-Location
mkdir "$PSScriptRoot/tools" -ErrorAction Ignore
Set-Location "$PSScriptRoot/tools"

# Install CapnProto
$capnprotoVersion = "0.7.0"
$capnprotoZip = "capnproto-c++-win32-$capnprotoVersion.zip"
if (-not (Test-Path $capnprotoZip)) {
    Invoke-WebRequest -Uri "https://capnproto.org/$capnprotoZip" -OutFile $capnprotoZip
    Expand-Archive -Path $capnprotoZip -DestinationPath .
    Move-Item "capnproto-tools-win32-$capnprotoVersion" capnproto-tools
}
$env:PATH="$env:PATH;$(Get-Location)/capnproto-tools"

# Go back to whence we came
Set-Location $origDir

# Write string for GH build env var
Write-Output "::set-env name=PATH::$env:PATH"

pip3 install -U pytest
