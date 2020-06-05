param (
    [Parameter(Mandatory=$true)]
    $ImageTag,

    [switch]$Build=$false
)

if ($Build)
{
    docker image build -t $ImageTag .
}

$cmd = "docker container run -v $($pwd):/esi -w /esi --rm -t -i $ImageTag"
Write-Host "Running: $cmd"
Invoke-Expression $cmd

