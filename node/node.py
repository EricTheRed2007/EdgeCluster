import sys
import os

# allow importing shared/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch

from model_partition import load_partition
from server import NodeServer, NextNodeClient


def run_layers(model, hidden_states, start_layer, end_layer, kv_cache):
    """
    Run transformer layers assigned to this node.
    Maintains KV cache for efficient decoding.
    """

    layers = model.model.layers

    new_cache = []

    for local_idx, layer_id in enumerate(range(start_layer, end_layer + 1)):

        past = kv_cache[local_idx]

        outputs = layers[layer_id](
            hidden_states,
            past_key_value=past,
            use_cache=True
        )

        hidden_states = outputs[0]

        new_cache.append(outputs[1])

    return hidden_states, new_cache


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, required=True)

    parser.add_argument("--start-layer", type=int, required=True)
    parser.add_argument("--end-layer", type=int, required=True)

    parser.add_argument("--model-path", required=True)

    parser.add_argument("--next-node-ip")
    parser.add_argument("--next-node-port", type=int)

    args = parser.parse_args()

    print("Loading model partition...")

    model = load_partition(
        args.model_path,
        args.start_layer,
        args.end_layer
    )

    device = "cpu"
    model.to(device)
    model.eval()

    # start server for controller or previous node
    server = NodeServer(args.port)
    server.start()

    # connect to next node if present
    next_node = None

    if args.next_node_ip:

        next_node = NextNodeClient(
            args.next_node_ip,
            args.next_node_port
        )

        next_node.connect()

    print("Node ready")

    # KV cache for this node
    num_layers = args.end_layer - args.start_layer + 1
    kv_cache = [None] * num_layers

    while True:
        try:
            hidden_states = server.receive_tensor()
        except ConnectionError:
            print("[NODE] upstream connection closed, exiting")
            break

        hidden_states = hidden_states.to(device)

        with torch.no_grad():

            hidden_states, new_cache = run_layers(
                model,
                hidden_states,
                args.start_layer,
                args.end_layer,
                kv_cache
            )

        kv_cache = new_cache

        # if there is another node in pipeline
        if next_node:

            next_node.send_tensor(hidden_states)

            logits = next_node.receive_tensor()

            server.send_tensor(logits)

        else:

            # final node computes logits
            logits = model.lm_head(hidden_states)

            server.send_tensor(logits)


if __name__ == "__main__":
    main()