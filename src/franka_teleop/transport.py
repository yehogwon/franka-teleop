"""Thin ZMQ helpers. Streams use PUB/SUB with CONFLATE: subscribers only ever see the newest
message, so a slow reader never acts on stale commands or states.

Publishers bind and subscribers connect. (A bound SUB with a connecting PUB completes the
handshake but delivers nothing with libzmq 4.3.5 here; publishers-bind is also the usual
pattern, and it means only one ft-joystick can own the command port.)"""

from __future__ import annotations

import time

import zmq

from .protocol import ProtocolError, decode, encode

_ctx: zmq.Context | None = None


def context() -> zmq.Context:
    global _ctx
    if _ctx is None:
        _ctx = zmq.Context.instance()
    return _ctx


def publisher(endpoint: str, *, bind: bool = True) -> zmq.Socket:
    # Default SNDHWM: a small HWM makes PUB drop the *newest* messages under backpressure,
    # which is the opposite of what a CONFLATE subscriber wants.
    sock = context().socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(endpoint) if bind else sock.connect(endpoint)
    return sock


def subscriber(endpoint: str, *, bind: bool = False, conflate: bool = True) -> zmq.Socket:
    sock = context().socket(zmq.SUB)
    if conflate:
        sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.bind(endpoint) if bind else sock.connect(endpoint)
    return sock


def send(sock: zmq.Socket, msg) -> bool:
    """Non-blocking send. Returns False if the message was dropped (no peer / queue full)."""
    try:
        sock.send(encode(msg), flags=zmq.NOBLOCK)
        return True
    except zmq.Again:
        return False


def recv_latest[T](sock: zmq.Socket, typ: type[T]) -> tuple[T | None, int]:
    """Drain the socket without blocking. Returns (newest valid message or None, #bad messages)."""
    latest, bad = None, 0
    while True:
        try:
            data = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return latest, bad
        try:
            latest = decode(data, typ)
        except ProtocolError:
            bad += 1


def recv_wait[T](sock: zmq.Socket, typ: type[T], timeout_s: float) -> T | None:
    """Block up to timeout_s for one valid message."""
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        if sock.poll(int(remaining * 1000) + 1, zmq.POLLIN):
            msg, _ = recv_latest(sock, typ)
            if msg is not None:
                return msg
