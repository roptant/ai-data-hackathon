"""Minimal RFC 6455 server-side framing for the event stream.

Only what the plan's API needs: the handshake, server-to-client text frames,
ping/pong and close.  Client frames are parsed so that a close or a stray
payload is handled rather than ignored, and an over-long client frame is
refused instead of buffered - a local client must not be able to grow this
process's memory.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import dataclass

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OPCODE_CONTINUATION = 0x0
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2
OPCODE_CLOSE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA

CLOSE_NORMAL = 1000
CLOSE_POLICY_VIOLATION = 1008
CLOSE_TOO_LARGE = 1009
CLOSE_INTERNAL = 1011

#: Clients send only close and pong in this protocol, so anything larger than
#: a small control frame is refused.
MAX_CLIENT_FRAME_BYTES = 4096


def accept_key(client_key: str) -> str:
    digest = hashlib.sha1((client_key.strip() + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def handshake_response(client_key: str) -> bytes:
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept_key(client_key)}\r\n"
        "\r\n"
    ).encode("ascii")


def encode_frame(payload: bytes, opcode: int = OPCODE_TEXT) -> bytes:
    """Server frames are never masked."""
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 1 << 16:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


def encode_text(text: str) -> bytes:
    return encode_frame(text.encode("utf-8"), OPCODE_TEXT)


def encode_close(code: int = CLOSE_NORMAL, reason: str = "") -> bytes:
    return encode_frame(struct.pack("!H", code) + reason.encode("utf-8")[:120], OPCODE_CLOSE)


def encode_ping(payload: bytes = b"") -> bytes:
    return encode_frame(payload[:125], OPCODE_PING)


@dataclass(frozen=True, slots=True)
class Frame:
    opcode: int
    payload: bytes
    consumed: int
    fin: bool = True

    @property
    def is_close(self) -> bool:
        return self.opcode == OPCODE_CLOSE


class FrameTooLarge(ValueError):
    """A client frame exceeded the permitted size."""


def parse_frame(buffer: bytes) -> Frame | None:
    """Parse one client frame, or ``None`` when more bytes are needed."""
    if len(buffer) < 2:
        return None
    first, second = buffer[0], buffer[1]
    fin = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    offset = 2
    if length == 126:
        if len(buffer) < offset + 2:
            return None
        length = struct.unpack("!H", buffer[offset : offset + 2])[0]
        offset += 2
    elif length == 127:
        if len(buffer) < offset + 8:
            return None
        length = struct.unpack("!Q", buffer[offset : offset + 8])[0]
        offset += 8
    if length > MAX_CLIENT_FRAME_BYTES:
        raise FrameTooLarge(f"client frame of {length} bytes exceeds the limit")
    mask = b""
    if masked:
        if len(buffer) < offset + 4:
            return None
        mask = buffer[offset : offset + 4]
        offset += 4
    if len(buffer) < offset + length:
        return None
    payload = bytearray(buffer[offset : offset + length])
    if masked:
        for index in range(len(payload)):
            payload[index] ^= mask[index % 4]
    return Frame(opcode=opcode, payload=bytes(payload), consumed=offset + length, fin=fin)
