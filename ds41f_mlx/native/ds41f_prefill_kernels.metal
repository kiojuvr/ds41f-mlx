// Minimal Metal source placeholders for the DwarfStar-authoritative prefill path.
// These kernels are not wired into correctness yet.  They define the first
// native carry/copy ownership vocabulary without changing model math.

#include <metal_stdlib>
using namespace metal;

kernel void ds41f_copy_i32(device const int *src [[buffer(0)]],
                           device int *dst [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
    dst[gid] = src[gid];
}

kernel void ds41f_copy_u8(device const uchar *src [[buffer(0)]],
                          device uchar *dst [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    dst[gid] = src[gid];
}

kernel void ds41f_zero_u8(device uchar *dst [[buffer(0)]],
                          uint gid [[thread_position_in_grid]]) {
    dst[gid] = 0;
}
