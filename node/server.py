import sys
import os

# allow importing shared/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import socket

from shared.tensor_protocol import send_tensor, recv_tensor


class NodeServer:
    """
    Handles incoming tensor connections from:
    - controller (Node1)
    - previous node (Node2)
    """

    def __init__(self, port):

        self.port = port
        self.conn = None
        self.server = None

    def start(self):

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # allow quick restart of node
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server.bind(("0.0.0.0", self.port))

        self.server.listen(1)

        print(f"[NODE] Listening on port {self.port}")

        self.conn, addr = self.server.accept()

        print(f"[NODE] Connected from {addr}")

    def receive_tensor(self):

        return recv_tensor(self.conn)

    def send_tensor(self, tensor):

        send_tensor(self.conn, tensor)


class NextNodeClient:
    """
    Handles connection to the next node in the pipeline
    """

    def __init__(self, ip, port):

        self.ip = ip
        self.port = port
        self.sock = None

    def connect(self):

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print(f"[NODE] Connecting to next node {self.ip}:{self.port}")

        self.sock.connect((self.ip, self.port))

        print("[NODE] Connected to next node")

    def send_tensor(self, tensor):

        send_tensor(self.sock, tensor)

    def receive_tensor(self):

        return recv_tensor(self.sock)