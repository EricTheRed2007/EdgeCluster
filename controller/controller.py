import sys
import os

# allow importing shared/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import socket
import argparse
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from shared.tensor_protocol import send_tensor, recv_tensor


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def connect(ip, port):
    """
    Create TCP connection to Node1
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    return sock


def load_model_local(model_path):
    """
    Load model from disk if present.
    Otherwise download and save.
    """

    if os.path.exists(model_path):

        print("Loading model from local path:", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        )

    else:

        print("Model not found locally. Downloading...")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cpu"
        )

        os.makedirs(model_path, exist_ok=True)

        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)

        print("Model saved to:", model_path)

    return tokenizer, model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--node1-ip", required=True)
    parser.add_argument("--node1-port", type=int, default=5001)

    parser.add_argument("--prompt", type=str, default="Hello")

    parser.add_argument("--max-tokens", type=int, default=50)

    parser.add_argument("--model-path", default="./models/tinyllama")

    args = parser.parse_args()

    print("Loading tokenizer and embeddings...")

    tokenizer, model = load_model_local(args.model_path)

    embed_tokens = model.model.embed_tokens
    embed_tokens.eval()

    print(f"Connecting to Node1 at {args.node1_ip}:{args.node1_port}")

    sock = connect(args.node1_ip, args.node1_port)

    # tokenize prompt
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    generated = input_ids.clone()

    print("\nStarting generation...\n")

    start_time = time.time()

    with torch.no_grad():

        for step in range(args.max_tokens):

            # Step 0: prefill (send entire prompt)
            if step == 0:

                hidden_states = embed_tokens(generated)

            # Step 1+: decode (send only last token)
            else:

                last_token = generated[:, -1:]
                hidden_states = embed_tokens(last_token)

            # send tensor to Node1
            send_tensor(sock, hidden_states)

            # receive logits from pipeline
            logits = recv_tensor(sock)

            # select next token
            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1
            ).unsqueeze(0)

            # append token
            generated = torch.cat([generated, next_token], dim=1)

            # decode and print text
            text = tokenizer.decode(generated[0], skip_special_tokens=True)

            print(text)

    end_time = time.time()

    tokens_generated = generated.shape[1] - input_ids.shape[1]

    duration = end_time - start_time

    print("\nPerformance Metrics")
    print("-------------------")

    print("Tokens generated:", tokens_generated)

    print("Total time:", round(duration, 2), "seconds")

    if duration > 0:
        print("Tokens/sec:", round(tokens_generated / duration, 2))


if __name__ == "__main__":
    main()