Write-Host "Starting Controller..."

# Ensure models folder exists
if (!(Test-Path "./models")) {
    Write-Host "Models folder not found. Creating..."
    New-Item -ItemType Directory -Path "./models"
}

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue

if (!$python) {
    Write-Host "Python not found. Please install Python."
    exit
}

Write-Host "Running Controller..."

python controller/controller.py `
 --node1-ip 192.168.1.20 `
 --node1-port 5001 `
 --prompt "Explain artificial intelligence in simple terms." `
 --max-tokens 50 `
 --model-path ./models/tinyllama`

# connect to next node if present
    next_node = None

    if args.next_node_ip:
        from shared.network_utils import connect_with_retry

        next_node = NextNodeClient(
            args.next_node_ip,
            args.next_node_port
        )

        # retry until the downstream node comes up
        sock = connect_with_retry(args.next_node_ip, args.next_node_port)
        next_node.sock = sock

    print("Node ready")