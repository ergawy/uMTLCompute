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

struct InnerLambda3 {
  thread float4 &_aRef;
  thread array<array<float4, C_COLS>, TILE_K> &_bRef;
  thread array<array<float4, C_COLS>, C_ROWS> &_cRef;

  uint _i;
  uint _kk;

  InnerLambda3(thread float4 &aRef,
               thread array<array<float4, C_COLS>, TILE_K> &bRef,
               thread array<array<float4, C_COLS>, C_ROWS> &cRef, uint i,
               uint kk)
      : _aRef(aRef), _bRef(bRef), _cRef(cRef), _i(i), _kk(kk) {}

  void operator()(uint idx) {
    _cRef[_i][idx] +=
        float4(_aRef.x, _aRef.x, _aRef.x, _aRef.x) * _bRef[0 + 4 * _kk][idx];
    _cRef[_i][idx] +=
        float4(_aRef.y, _aRef.y, _aRef.y, _aRef.y) * _bRef[1 + 4 * _kk][idx];
    _cRef[_i][idx] +=
        float4(_aRef.z, _aRef.z, _aRef.z, _aRef.z) * _bRef[2 + 4 * _kk][idx];
    _cRef[_i][idx] +=
        float4(_aRef.w, _aRef.w, _aRef.w, _aRef.w) * _bRef[3 + 4 * _kk][idx];
  }
};

struct MiddleLambda {
  const device float4 *_inputA;

  uint _gi;
  uint _gk;

  thread array<array<float4, C_COLS>, TILE_K> &_bRef;
  thread array<array<float4, C_COLS>, C_ROWS> &_cRef;

  uint _i;

  MiddleLambda(const device float4 *inputA, uint gi, uint gk,
               thread array<array<float4, C_COLS>, TILE_K> &bRef,
               thread array<array<float4, C_COLS>, C_ROWS> &cRef, uint i)
      : _inputA(inputA), _gi(gi), _gk(gk), _bRef(bRef), _cRef(cRef), _i(i) {}

  void operator()(uint idx) {
    float4 A = _inputA[coordToOffset(_gi, _gk + idx, strideA / 4)];
    InnerLambda3 il3(A, _bRef, _cRef, _i, idx);
    unroll<C_COLS>::call(il3);
  }
};

struct OuterLambda3 {
  const device float4 *_inputA;

  thread array<array<float4, C_COLS>, TILE_K> &_bRef;
  thread array<array<float4, C_COLS>, C_ROWS> &_cRef;

  uint _k;

  uint2 _gId;
  uint2 _laneId;

  OuterLambda3(const device float4 *inputA,
               thread array<array<float4, C_COLS>, TILE_K> &bRef,
               thread array<array<float4, C_COLS>, C_ROWS> &cRef, uint k,
               uint2 gId, uint2 laneId)
      : _inputA(inputA),
        _bRef(bRef),
        _cRef(cRef),
        _k(k),
        _gId(gId),
        _laneId(laneId) {}

  void operator()(uint idx) {
    uint gi = _gId.y * TILE_M + _laneId.y + idx * WG_Y;
    uint gk = _k / 4;

    MiddleLambda ml(_inputA, gi, gk, _bRef, _cRef, idx);
    unroll<TILE_K / 4>::call(ml);
  }
};

struct InnerLambda4 {
  const thread array<array<float4, C_COLS>, C_ROWS> &_cRef;
  device float4 *_outputO;

  uint2 _gId;
  uint2 _laneId;

  uint _i;

  InnerLambda4(const thread array<array<float4, C_COLS>, C_ROWS> &cRef,
               device float4 *outputO, uint2 gId, uint2 laneId, uint i)
      : _cRef(cRef), _outputO(outputO), _gId(gId), _laneId(laneId), _i(i) {}

  void operator()(uint idx) {
    uint gi = _gId.y * TILE_M + _laneId.y + _i * WG_Y;
    uint gj = _gId.x * (TILE_N / 4) + _laneId.x + idx * WG_X;
    _outputO[gi * strideC / 4 + gj] = _cRef[_i][idx];
  }
};

struct OuterLambda4 {
  const thread array<array<float4, C_COLS>, C_ROWS> &_cRef;
  device float4 *_outputO;

  uint2 _gId;
  uint2 _laneId;

  OuterLambda4(const thread array<array<float4, C_COLS>, C_ROWS> &cRef,
               device float4 *outputO, uint2 gId, uint2 laneId)
      : _cRef(cRef), _outputO(outputO), _gId(gId), _laneId(laneId) {}

  void operator()(uint idx) {
    InnerLambda4 il4(_cRef, _outputO, _gId, _laneId, idx);
    unroll<C_COLS>::call(il4);
  }
};

