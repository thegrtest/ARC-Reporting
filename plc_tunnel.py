"""
plc_tunnel.py

Relay EtherNet/IP (Studio 5000 / RSLinx) traffic through this PC to a PLC
that sits on an isolated machine LAN. Run this on the ARC-Reporting PC,
which is dual-homed (main network + machine LAN). A remote workstation
running Studio 5000 then points RSLinx at THIS PC's main-network IP, and
this script forwards the session across to the PLC.

What this handles
-----------------
TCP 44818 -- EtherNet/IP explicit messaging. This is what Studio 5000
             uses for project upload/download, online edits, tag access,
             firmware update, and any "Who Active -> Go Online" workflow
             where you've typed the PLC's address (the tunnel IP) by hand.

What this does NOT handle
-------------------------
* RSWho / RSLinx browse. Browse relies on Ethernet broadcast and will
  not discover the PLC through a TCP forwarder. In RSLinx add a new
  EtherNet/IP driver and enter THIS PC's main-network IP as the device
  address. The session will connect across the tunnel.
* Implicit / class-1 I/O messaging (UDP 2222). That traffic is multicast
  and cannot be unicast-forwarded the way TCP can. Programming does not
  use it.

Network prerequisites
---------------------
* Windows Firewall on the ARC PC must allow inbound TCP 44818 on the
  main-network interface.
* This PC must already have working IP routes to the PLC on the machine
  LAN -- test with `ping <plc-ip>` from this PC before running the
  tunnel.
* If you want the tunnel up at boot, register this script with NSSM or
  run it from a scheduled task with "At system startup".

Usage
-----
Forward to PLC at 192.168.10.5 on all interfaces:
    python plc_tunnel.py --plc-ip 192.168.10.5

Bind only to the main-network NIC (recommended -- prevents the listener
from showing up on the machine LAN too):
    python plc_tunnel.py --plc-ip 192.168.10.5 --listen 10.0.0.42

Forward a non-default port (rare):
    python plc_tunnel.py --plc-ip 192.168.10.5 --tcp-port 44818

Shut down with Ctrl+C.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import sys
from typing import Tuple


log = logging.getLogger("plc-tunnel")


async def _pipe(
    src: asyncio.StreamReader,
    dst: asyncio.StreamWriter,
    label: str,
) -> int:
    """Copy bytes from src to dst until EOF or error. Returns bytes copied."""
    total = 0
    try:
        while True:
            chunk = await src.read(65536)
            if not chunk:
                break
            dst.write(chunk)
            await dst.drain()
            total += len(chunk)
    except (ConnectionError, asyncio.IncompleteReadError, OSError) as exc:
        log.debug("%s pipe ended: %s", label, exc)
    finally:
        if not dst.is_closing():
            try:
                dst.close()
            except Exception:
                pass
    return total


def _tune_socket(writer: asyncio.StreamWriter) -> None:
    """EtherNet/IP is latency-sensitive. Disable Nagle, enable keep-alive."""
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        pass


async def _handle_tcp_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    plc_addr: Tuple[str, int],
) -> None:
    peer = client_writer.get_extra_info("peername")
    log.info("connect %s -> %s:%d", peer, *plc_addr)

    try:
        plc_reader, plc_writer = await asyncio.open_connection(*plc_addr)
    except OSError as exc:
        log.error("cannot reach PLC %s:%d (%s)", plc_addr[0], plc_addr[1], exc)
        client_writer.close()
        return

    _tune_socket(client_writer)
    _tune_socket(plc_writer)

    sent, received = await asyncio.gather(
        _pipe(client_reader, plc_writer, f"{peer}->PLC"),
        _pipe(plc_reader, client_writer, f"PLC->{peer}"),
    )
    log.info("close   %s -- %d B -> PLC, %d B <- PLC", peer, sent, received)


async def _run_tcp(listen: str, listen_port: int, plc_addr: Tuple[str, int]) -> None:
    server = await asyncio.start_server(
        lambda r, w: _handle_tcp_client(r, w, plc_addr),
        listen,
        listen_port,
    )
    bound = ", ".join(str(sock.getsockname()) for sock in (server.sockets or []))
    log.info("listening on %s -> %s:%d", bound, *plc_addr)
    async with server:
        await server.serve_forever()


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forward EtherNet/IP (Studio 5000) through this PC to a PLC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--plc-ip", required=True, help="PLC IP on the machine LAN")
    p.add_argument(
        "--listen", default="0.0.0.0",
        help="Interface IP to bind (default: 0.0.0.0 = all). Set to this PC's "
             "main-network IP to keep the listener off the machine LAN.",
    )
    p.add_argument(
        "--tcp-port", type=int, default=44818,
        help="TCP port for EtherNet/IP explicit messaging (default: 44818)",
    )
    p.add_argument(
        "--plc-port", type=int, default=None,
        help="Override PLC-side TCP port. Defaults to --tcp-port (44818).",
    )
    p.add_argument("--quiet", action="store_true", help="Warnings only")
    return p.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> None:
    plc_port = args.plc_port if args.plc_port is not None else args.tcp_port
    await _run_tcp(args.listen, args.tcp_port, (args.plc_ip, plc_port))


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        log.info("shutdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
