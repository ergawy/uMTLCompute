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

template <unsigned N>
struct unroll {
  template <class Ptr, class F>
  static void call(Ptr a, F f) {
    f(N - 1, a);
    unroll<N - 1>::call(a, f);
  }
};

template <>
struct unroll<0u> {
  template <class Ptr, class F>
  static void call(Ptr a, F f) {}
};

void init1(uint idx, thread float4 *a) { a[idx] = float4(0.f, 0.f, 0.f, 0.f); }

template <class Ptr, unsigned COLS>
void init0(uint idx, Ptr aa) {
  unroll<COLS>::call(aa[idx], init1);
}

[[kernel]] void matmul_tiled(const device float4 *inputA [[buffer(0)]],
                             const device float4 *inputB [[buffer(1)]],
                             device float4 *outputO [[buffer(2)]],
                             const uint2 gId [[threadgroup_position_in_grid]],
                             const uint2 laneId
                             [[thread_position_in_threadgroup]]) {
  // Holds the tile result.
  float4 C[C_ROWS][C_COLS];
  float4 B[TILE_K][C_COLS];

  unroll<C_ROWS>::call(C, init0<thread float4(*)[C_COLS], C_COLS>);

  // TODO Unroll all the applicable loops below.
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
}
