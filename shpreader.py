"""Reader for .shp press-recorder files (header + 256 digital + N analog channels, fixed-size records)."""
import numpy as np, struct

REC_NAME_START = 0x37c   # first channel name
REC_STEP = 0x100

def read_shp(path):
    with open(path, 'rb') as fh:
        d = fh.read()
    n_dig = struct.unpack_from('<I', d, 0x10)[0]
    n_ana = struct.unpack_from('<I', d, 0x2c)[0]
    names = []
    for i in range(n_dig + n_ana):
        o = REC_NAME_START + i * REC_STEP
        names.append(d[o:o + 0x84].split(b'\0')[0].decode('latin1').strip())
    start = REC_NAME_START - 0x7c + (n_dig + n_ana) * REC_STEP + 0x7c
    R = 4 + n_dig // 8 + 4 * n_ana
    n = (len(d) - start) // R
    a = np.frombuffer(d, dtype=np.uint8, count=n * R, offset=start).reshape(n, R)
    idx = a[:, :4].copy().view('<u4').ravel()
    bits = np.unpackbits(a[:, 4:4 + n_dig // 8], axis=1, bitorder='little').astype(bool)
    ana = a[:, 4 + n_dig // 8:].copy().view('<f4')
    return dict(index=idx, dig=bits, ana=ana, dig_names=names[:n_dig], ana_names=names[n_dig:], header=d[:0x38])
