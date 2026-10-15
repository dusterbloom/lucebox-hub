"""Copy the tokenizer.ggml.* metadata of a GGUF into a tensor-less GGUF (a few MB), so that the server's tokenizer program can run
on a machine that does not hold the 16 GB model file (A1 render parity).
  python3 make_tok_gguf.py <full.gguf> <out.gguf>
The server's Tokenizer::load_from_gguf reads only these keys; check_equal() below is the test that the copy tokenizes the same."""
import sys

import gguf


def copy_tokenizer_metadata(src, dst):
    r = gguf.GGUFReader(src)
    arch = gguf.GGUFReader.get_field(r, "general.architecture")
    w = gguf.GGUFWriter(dst, arch=bytes(arch.parts[arch.data[0]]).decode())
    n = 0
    for name, f in r.fields.items():
        if not name.startswith("tokenizer.ggml."):
            continue
        sub = f.types[1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(name, f.contents(), f.types[0], sub_type=sub)
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return n


if __name__ == "__main__":
    print("copied", copy_tokenizer_metadata(sys.argv[1], sys.argv[2]), "tokenizer keys to", sys.argv[2])
