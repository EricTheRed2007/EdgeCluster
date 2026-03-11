import pickle
import struct

HEADER_SIZE = 8


def recv_exact(sock, size):
    data = b""

    while len(data) < size:
        packet = sock.recv(size - len(data))

        if not packet:
            raise ConnectionError("Socket closed")

        data += packet

    return data


def send_tensor(sock, tensor):
    tensor = tensor.detach().cpu()

    payload = pickle.dumps(tensor)

    header = struct.pack(">Q", len(payload))

    sock.sendall(header + payload)


def recv_tensor(sock):
    header = recv_exact(sock, HEADER_SIZE)

    payload_size = struct.unpack(">Q", header)[0]

    payload = recv_exact(sock, payload_size)

    tensor = pickle.loads(payload)

    return tensor