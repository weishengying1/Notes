'''
这是一个小小的测试， 在 vllm v1 engine 中，EngineCore 会讲
'''

import zmp
import multiprocessing
import time
from typing import (TYPE_CHECKING, Any, Callable, Generic, Literal, NamedTuple,
                    Optional, TypeVar, Union)
import psutil
# https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L2112
def make_zmq_socket(
    ctx: Union[zmq.asyncio.Context, zmq.Context],  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
) -> Union[zmq.Socket, zmq.asyncio.Socket]:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)  # 0.5GB in bytes
    else:
        buf_size = -1  # Use system default buffer size

    if socket_type == zmq.constants.PULL:
        socket.setsockopt(zmq.constants.RCVHWM, 0)
        socket.setsockopt(zmq.constants.RCVBUF, buf_size)
        socket.connect(path)
    elif socket_type == zmq.constants.PUSH:
        socket.setsockopt(zmq.constants.SNDHWM, 0)
        socket.setsockopt(zmq.constants.SNDBUF, buf_size)
        socket.bind(path)
    else:
        raise ValueError(f"Unknown Socket Type: {socket_type}")

    return socket

@contextlib.contextmanager
def zmq_socket_ctx(path: str, socket_type: Any) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(ctx, path, socket_type)

    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=0)


https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py#L394
def EngineCore():
     Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Reuse send buffer.
        buffer = bytearray()

        with zmq_socket_ctx(output_path, zmq.constants.PUSH) as socket:
            while True:
                outputs = self.output_queue.get()
                encoder.encode_into(outputs, buffer)
                socket.send_multipart((buffer, ), copy=False)

def Worker():
