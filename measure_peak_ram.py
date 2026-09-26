"""Peak RAM of a CPU FP32 scoring pass, measured in-process.

CPU peak RAM is not instrumented anywhere in the supplied harness
(``common.device_metrics()`` only reports CUDA allocation) and psutil is not a
dependency, so read PeakWorkingSetSize straight from the Windows API.

    python measure_peak_ram.py <checkpoint.pt> [split]
"""
import ctypes
import ctypes.wintypes as wt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))

import torch                                            # noqa: E402
from common import load_data, make_model, setup         # noqa: E402
from evaluate import score                              # noqa: E402


class _Counters(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def peak_working_set():
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    # GetCurrentProcess() returns the pseudo-handle (HANDLE)-1. Left to ctypes'
    # default 32-bit int restype it is truncated, the call fails and every field
    # reads back zero, so declare the handle width and check the result.
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
    return counters.PeakWorkingSetSize


def main():
    checkpoint = Path(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else 'validation'
    device, precision = setup('cpu', 'fp32', 8)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    result = score(model, *load_data()[split], device, precision)
    print(f'{checkpoint.parent.name:22s} split={split:10s} bpb={result["bpb"]:.4f}  '
          f'score_seconds={result["seconds"]:6.2f}  peak_RAM={peak_working_set() / 2**30:.3f} GiB')


if __name__ == '__main__':
    main()
