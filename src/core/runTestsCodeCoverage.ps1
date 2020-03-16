param(
    [Parameter(Mandatory=$true)]
    [string]$testProjectPath,
    [Parameter(Mandatory=$true)]
    [string]$testSettingsPath,
    [Parameter(Mandatory=$true)]
    [string]$testResultsFolder
)

<#
echo "Test Project Path" $testProjectPath
echo "Test Settings Path" $testSettingsPath
echo "Test Results Folder" $testResultsFolder
#>

try {

    if (-not (Test-Path $testProjectPath)) 
    {
        throw [System.IO.FileNotFoundException] "$testProjectPath not found."
    }
    if (-not (Test-Path $testSettingsPath)) 
    {
        throw [System.IO.FileNotFoundException] "$testSettingsPath not found."
    }
    if (-not (Test-Path $testResultsFolder)) 
    {
        throw [System.IO.FileNotFoundException] "$testResultsFolder not found."
    }

    dotnet test $testProjectPath --settings:$testSettingsPath --results-directory:$testResultsFolder --collect:"Code Coverage"
    $recentCoverageFile = Get-ChildItem -File -Filter *.coverage -Path $testResultsFolder -Name -Recurse | Select-Object -First 1;
    write-host 'Test Completed'  -ForegroundColor Green

    & $env:HOME\.nuget\packages\microsoft.codecoverage\16.5.0\build\netstandard1.0\CodeCoverage\CodeCoverage.exe analyze  /output:$testResultsFolder\MyTestOutput.coveragexml  $testResultsFolder'\'$recentCoverageFile
    write-host 'CoverageXML Generated'  -ForegroundColor Green

    dotnet $env:HOME\.nuget\packages\reportgenerator\4.5.2\tools\netcoreapp2.1\ReportGenerator.dll "-reports:$testResultsFolder\MyTestOutput.coveragexml" "-targetdir:$testResultsFolder\coveragereport"
    write-host 'CoverageReport Published'  -ForegroundColor Green

}
catch {

    write-host "Caught an exception:" -ForegroundColor Red
    write-host "Exception Type: $($_.Exception.GetType().FullName)" -ForegroundColor Red
    write-host "Exception Message: $($_.Exception.Message)" -ForegroundColor Red

}