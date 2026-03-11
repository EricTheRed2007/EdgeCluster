import socket
import time


def create_server_socket(port, backlog=1):
    """
    Create a TCP server socket with safe settings.
    """

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # allow immediate restart after crash
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind(("0.0.0.0", port))

    server.listen(backlog)

    return server


def accept_connection(server):
    """
    Accept an incoming connection.
    """

    conn, addr = server.accept()

    print(f"[NETWORK] Connection from {addr}")

    return conn


def connect_with_retry(ip, port, retries=20, delay=1.0):
    """
    Connect to a remote node with retries.
    Useful when nodes start at different times.
    """

    for attempt in range(retries):

        try:

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            sock.connect((ip, port))

            print(f"[NETWORK] Connected to {ip}:{port}")

            return sock

        except ConnectionRefusedError:

            print(f"[NETWORK] Connection attempt {attempt+1} failed, retrying...")

            time.sleep(delay)

    raise ConnectionError(f"Unable to connect to {ip}:{port}")


def close_socket(sock):
    """
    Safely close a socket.
    """

    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass

    sock.close()