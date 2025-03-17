from msgpack import msgpack

class MsgpackEncoder:
    """Encoder with custom torch tensor serialization."""

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=custom_enc_hook)

    def encode(self, obj: Any) -> bytes:
        return self.encoder.encode(obj)

    def encode_into(self, obj: Any, buf: bytearray) -> None:
        self.encoder.encode_into(obj, buf)


def custom_enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        # NOTE(rob): it is fastest to use numpy + pickle
        # when serializing torch tensors.
        # https://gist.github.com/tlrmchlsmth/8067f1b24a82b6e2f90450e7764fa103 # noqa: E501
        return msgpack.Ext(CUSTOM_TYPE_TENSOR, pickle.dumps(obj.numpy()))

    return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj))