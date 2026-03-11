Write-Host "Starting Node2..."

# Ensure models folder exists
if (!(Test-Path "./models")) {
    Write-Host "Models folder not found. Creating..."
    New-Item -ItemType Directory -Path "./models"
}

# Check Docker image
$img = docker images -q edge-node

if (!$img) {
    Write-Host "Docker image not found. Building..."
    docker build -t edge-node -f node/Dockerfile .
}

Write-Host "Running Node2 container..."

docker run `
 -p 5002:5002 `
 -v ${PWD}/models:/models `
 edge-node `
 python node.py `
 --port 5002 `
 --start-layer 12 `
 --end-layer 23 `
 --model-path /models/tinyllama