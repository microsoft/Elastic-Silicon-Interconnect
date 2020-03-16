param (
    [Parameter(Mandatory=$true)]
    $ImageTag,

    [switch]$Build=$false
)

if ($Build)
{
    docker image build -t $ImageTag .
}

docker container run -v $PSScriptRoot\:/esi -w /esi --rm -t -i $ImageTag

