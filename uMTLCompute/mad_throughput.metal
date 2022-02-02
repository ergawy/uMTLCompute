#include <metal_stdlib>
using namespace metal;

[[kernel]] void mad_throughput(const device float4* inputA [[buffer(0)]],
                               const device float4* inputB [[buffer(1)]],
                               device float4* outputO [[buffer(2)]],
                               const uint x [[thread_position_in_grid]]) {
    float4 a = inputA[x];
    float4 b = inputB[x];
    float4 c(1.f, 1.f, 1.f, 1.f);

    for (int i = 0; i < 100000; ++i) {
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
        c = a * c + b;
    }

    outputO[x] = c;
}
