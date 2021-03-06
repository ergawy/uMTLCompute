// TODO Compare the IR of both regular loop and unrolled cases.

#include <metal_stdlib>

using namespace metal;

// TODO Parameterize these constants:
constant uint TILE_M = 2;
constant uint TILE_N = 128;
constant uint TILE_K = 4;
constant uint WG_X = 32;
constant uint WG_Y = 2;

// TODO Pass these as function constants:
constant uint M = 1024;
constant uint N = 1024;
constant uint K = 1024;

constant uint strideA = K;
constant uint strideB = N;
constant uint strideC = N;

constant uint C_ROWS = TILE_M / WG_Y;
constant uint C_COLS = TILE_N / (4 * WG_X);

uint coordToOffset(uint i, uint j, uint stride) { return (stride * i + j); }

#define USE_UNROLL
#include "matmul_tiled_fp32.h"

[[kernel]] void matmul_tiled(const device float4 *inputA [[buffer(0)]],
                             const device float4 *inputB [[buffer(1)]],
                             device float4 *outputO [[buffer(2)]],
                             const uint2 gId [[threadgroup_position_in_grid]],
                             const uint2 laneId
                             [[thread_position_in_threadgroup]]) {
  // Holds the tile result.
  array<array<float4, C_COLS>, C_ROWS> C;
  array<array<float4, C_COLS>, TILE_K> B;

#ifdef USE_UNROLL
  OuterLambda ol(C);
  unroll<C_ROWS>::call(ol);

  for (uint k = 0; k < K; k += TILE_K) {
    OuterLambda2 ol2(inputB, gId, laneId, B, k);
    unroll<C_COLS>::call(ol2);

    OuterLambda3 ol3(inputA, B, C, k, gId, laneId);
    unroll<C_ROWS>::call(ol3);
  }

  OuterLambda4 ol4(C, outputO, gId, laneId);
  unroll<C_ROWS>::call(ol4);
#else
  for (uint i = 0; i < C_ROWS; ++i) {
    for (uint j = 0; j < C_COLS; ++j) {
      C[i][j] = float4(0.f, 0.f, 0.f, 0.f);
    }
  }

  for (uint k = 0; k < K; k += TILE_K) {
    for (uint j = 0; j < C_COLS; ++j) {
      for (uint i = 0; i < TILE_K; ++i) {
        uint gj = gId.x * (TILE_N / 4) + laneId.x + j * WG_X;
        uint gk = k + i;
        B[i][j] = inputB[coordToOffset(gk, gj, strideB / 4)];
      }
    }

    for (uint i = 0; i < C_ROWS; ++i) {
      uint gi = gId.y * TILE_M + laneId.y + i * WG_Y;
      uint gk = k / 4;
      for (uint kk = 0; kk < TILE_K / 4; ++kk) {
        float4 A = inputA[coordToOffset(gi, gk + kk, strideA / 4)];
        for (uint j = 0; j < C_COLS; ++j) {
          C[i][j] += float4(A.x, A.x, A.x, A.x) * B[0 + 4 * kk][j];
          C[i][j] += float4(A.y, A.y, A.y, A.y) * B[1 + 4 * kk][j];
          C[i][j] += float4(A.z, A.z, A.z, A.z) * B[2 + 4 * kk][j];
          C[i][j] += float4(A.w, A.w, A.w, A.w) * B[3 + 4 * kk][j];
        }
      }
    }
  }

  for (uint i = 0; i < C_ROWS; ++i) {
    for (uint j = 0; j < C_COLS; ++j) {
      uint gi = gId.y * TILE_M + laneId.y + i * WG_Y;
      uint gj = gId.x * (TILE_N / 4) + laneId.x + j * WG_X;
      outputO[gi * strideC / 4 + gj] = C[i][j];
    }
  }
#endif
}
