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

template <unsigned N>
struct unroll {
  template <class L>
  inline static void call(thread L &l) {
    l(N - 1);
    unroll<N - 1>::call(l);
  }
};

template <>
struct unroll<0u> {
  template <class L>
  inline static void call(thread L &l) {}
};

#define USE_UNROLL

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
  /*
   MSL doesn't support lambdas, so the following structs emulate that. They
   basically translate to the following C++ code.

   unroll<C_ROWS>::call([&C](uint idx) {
     unroll<C_COLS>::call([&C[idx]](uint idx2) {
       C[idx][idx2] = float4(0.f, 0.f, 0.f, 0.f);
     });
   });
   */

  /*
   unroll<C_ROWS>::call(
     LOOP_BODY(C, {
       unroll<C_COLS>::call(
         LOOP_BODY(C[idx], {
           _cRowRef[idx] = float4(0.f, 0.f, 0.f, 0.f);
         }
       )
     })
   );
   */
  struct InnerLambda {
    thread array<float4, C_COLS> &_cRowRef;
    InnerLambda(thread array<float4, C_COLS> &cRowRef) : _cRowRef(cRowRef) {}

    inline void operator()(uint idx) {
      _cRowRef[idx] = float4(0.f, 0.f, 0.f, 0.f);
    }
  };

  struct OuterLambda {
    thread array<array<float4, C_COLS>, C_ROWS> &_cRef;
    OuterLambda(thread array<array<float4, C_COLS>, C_ROWS> &cRef)
        : _cRef(cRef) {}

    inline void operator()(uint idx) {
      InnerLambda il(_cRef[idx]);
      unroll<C_COLS>::call(il);
    }
  };

  OuterLambda ol(C);
  unroll<C_ROWS>::call(ol);
#else
  for (uint i = 0; i < C_ROWS; ++i) {
    for (uint j = 0; j < C_COLS; ++j) {
      C[i][j] = float4(0.f, 0.f, 0.f, 0.f);
    }
  }
#endif

#ifdef USE_UNROLL
  // TODO Unroll all the applicable loops below.
  for (uint k = 0; k < K; k += TILE_K) {
    struct InnerLambda2 {
      const device float4 *_inputB;
      uint2 _gId;
      uint2 _laneId;

      thread array<array<float4, C_COLS>, TILE_K> &_bRef;
      uint _k;
      uint _j;

      InnerLambda2(const device float4 *inputB, uint2 gId, uint2 laneId,
                   thread array<array<float4, C_COLS>, TILE_K> &bRef, uint k,
                   uint j)
          : _inputB(inputB),
            _gId(gId),
            _laneId(laneId),
            _bRef(bRef),
            _k(k),
            _j(j) {}

      inline void operator()(uint idx) {
        uint gj = _gId.x * (TILE_N / 4) + _laneId.x + _j * WG_X;
        uint gk = _k + idx;
        _bRef[idx][_j] = _inputB[coordToOffset(gk, gj, strideB / 4)];
      }
    };

    struct OuterLambda2 {
      const device float4 *_inputB;
      uint2 _gId;
      uint2 _laneId;

      thread array<array<float4, C_COLS>, TILE_K> &_bRef;
      uint _k;

      OuterLambda2(const device float4 *inputB, uint2 gId, uint2 laneId,
                   thread array<array<float4, C_COLS>, TILE_K> &bRef, uint k)
          : _inputB(inputB), _gId(gId), _laneId(laneId), _bRef(bRef), _k(k) {}

      inline void operator()(uint idx) {
        InnerLambda2 il2(_inputB, _gId, _laneId, _bRef, _k, idx);
        unroll<TILE_K>::call(il2);
      }
    };

    OuterLambda2 ol2(inputB, gId, laneId, B, k);
    unroll<C_COLS>::call(ol2);

    for (uint i = 0; i < C_ROWS; ++i) {
      uint gi = gId.y * TILE_M + laneId.y + i * WG_Y;
      uint gk = k / 4;
      for (uint kk = 0; kk < TILE_K / 4; ++kk) {
        float4 A = inputA[coordToOffset(gi, gk + kk, strideA / 4)];

        struct InnerLambda3 {
          thread float4 &_aRef;
          thread array<array<float4, C_COLS>, TILE_K> &_bRef;
          thread array<array<float4, C_COLS>, C_ROWS> &_cRef;

          uint _i;
          uint _kk;

          InnerLambda3(thread float4 &aRef,
                       thread array<array<float4, C_COLS>, TILE_K> &bRef,
                       thread array<array<float4, C_COLS>, C_ROWS> &cRef,
                       uint i, uint kk)
              : _aRef(aRef), _bRef(bRef), _cRef(cRef), _i(i), _kk(kk) {}

          void operator()(uint idx) {
            _cRef[_i][idx] += float4(_aRef.x, _aRef.x, _aRef.x, _aRef.x) *
                              _bRef[0 + 4 * _kk][idx];
            _cRef[_i][idx] += float4(_aRef.y, _aRef.y, _aRef.y, _aRef.y) *
                              _bRef[1 + 4 * _kk][idx];
            _cRef[_i][idx] += float4(_aRef.z, _aRef.z, _aRef.z, _aRef.z) *
                              _bRef[2 + 4 * _kk][idx];
            _cRef[_i][idx] += float4(_aRef.w, _aRef.w, _aRef.w, _aRef.w) *
                              _bRef[3 + 4 * _kk][idx];
          }
        };

        InnerLambda3 il3(A, B, C, i, kk);
        unroll<C_COLS>::call(il3);
      }
    }
  }
#else
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
#endif

  for (uint i = 0; i < C_ROWS; ++i) {
    for (uint j = 0; j < C_COLS; ++j) {
      uint gi = gId.y * TILE_M + laneId.y + i * WG_Y;
      uint gj = gId.x * (TILE_N / 4) + laneId.x + j * WG_X;
      outputO[gi * strideC / 4 + gj] = C[i][j];
    }
  }
}
