Write-Host "Starting Node1..."


if (!(Test-Path "./models")) {
    Write-Host "Models folder not found. Creating..."
    New-Item -ItemType Directory -Path "./models"
}


$img = docker images -q edge-node

if (!$img) {
    Write-Host "Docker image not found. Building..."
    docker build -t edge-node -f node/Dockerfile .
}

Write-Host "Running Node1 container..."

docker run `
 -p 5001:5001 `
 -v ${PWD}/models:/models `
 edge-node `
 python node.py `
 --port 5001 `
 --start-layer 0 `
 --end-layer 11 `
 --model-path /models/tinyllama `
 --next-node-ip 192.168.1.21 `
 --next-node-port 5002