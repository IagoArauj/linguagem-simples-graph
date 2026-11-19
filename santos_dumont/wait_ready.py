"""Bounded startup probes; importing this module does not import Ray.

Default networking resolves the allocated hostname to IPv4, never hostname -I.
If cluster DNS selects the wrong network, set SD_NETWORK_INTERFACE on all nodes
(and HEAD_IP to rank 0's IPv4), or set VLLM_HOST_IP separately on each node.
"""

import argparse
import ipaddress
import json
import os
import socket
import time
import urllib.error
import urllib.request


def node_ip(hostname, interface=None):
    if interface:
        # Linux/Santos Dumont only; keep the module importable elsewhere.
        import fcntl
        import struct

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            address = socket.inet_ntoa(
                fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", interface.encode()[:15]))[20:24]
            )
    else:
        addresses = socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM)
        address = next(
            (entry[4][0] for entry in addresses if not ipaddress.ip_address(entry[4][0]).is_loopback),
            "127.0.0.1",
        )
    parsed = ipaddress.IPv4Address(address)
    if parsed.is_loopback or parsed.is_unspecified:
        raise ValueError(f"No routable IPv4 for {hostname}; configure SD_NETWORK_INTERFACE and HEAD_IP")
    return address


def validate_layout(nodes, gpus_per_node, tp, pp):
    if min(nodes, gpus_per_node, tp, pp) <= 0:
        raise ValueError("Node, GPU, TP and PP counts must be positive")
    if tp * pp != nodes * gpus_per_node:
        raise ValueError("TP_SIZE * PP_SIZE must equal allocated nodes * GPUs per node")


def ray_ready(nodes, expected_nodes, gpus_per_node):
    alive = [node for node in nodes if node.get("Alive")]
    return (
        len(alive) == expected_nodes
        and len({node.get("NodeManagerAddress") for node in alive}) == expected_nodes
        and all(node.get("Resources", {}).get("GPU", 0) == gpus_per_node for node in alive)
    )


def model_ready(url, model, api_key, timeout=5):
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        entries = payload.get("data") if isinstance(payload, dict) else None
        return isinstance(entries, list) and any(
            isinstance(item, dict) and item.get("id") == model for item in entries
        )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def tcp_ready(host, port, timeout=5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_until(probe, timeout, interval=2, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    while clock() < deadline:
        if probe():
            return
        sleep(min(interval, max(0, deadline - clock())))
    raise TimeoutError("Startup readiness deadline exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    ip = sub.add_parser("ip")
    ip.add_argument("hostname")
    ip.add_argument("--interface", default=os.environ.get("SD_NETWORK_INTERFACE"))
    layout = sub.add_parser("layout")
    for name in ("nodes", "gpus", "tp", "pp"):
        layout.add_argument(f"--{name}", type=int, required=True)
    tcp = sub.add_parser("tcp")
    tcp.add_argument("--host", required=True)
    tcp.add_argument("--port", type=int, required=True)
    ray = sub.add_parser("ray")
    ray.add_argument("--address", required=True)
    ray.add_argument("--nodes", type=int, required=True)
    ray.add_argument("--gpus", type=int, required=True)
    http = sub.add_parser("http")
    http.add_argument("--url", required=True)
    http.add_argument("--model", required=True)
    for command in (tcp, ray, http):
        command.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if args.mode == "ip":
        print(node_ip(args.hostname, args.interface))
    elif args.mode == "layout":
        validate_layout(args.nodes, args.gpus, args.tp, args.pp)
    elif args.mode == "tcp":
        wait_until(lambda: tcp_ready(args.host, args.port), args.timeout)
    elif args.mode == "http":
        wait_until(
            lambda: model_ready(args.url, args.model, os.environ["VLLM_API_KEY"]), args.timeout
        )
    else:
        import ray

        # ray.init can itself block: node.sh also enforces an outer deadline.
        ray.init(address=args.address, logging_level="ERROR")
        try:
            wait_until(lambda: ray_ready(ray.nodes(), args.nodes, args.gpus), args.timeout)
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
