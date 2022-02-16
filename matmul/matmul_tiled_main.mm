#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <sstream>

typedef float (^SrcBufferValueGenerator)(NSUInteger, NSUInteger);
typedef void (^SrcBufferSetter)(id<MTLBuffer>);

void setManagedBufferViaHost(id<MTLBuffer> buffer, NSUInteger buffer_size_in_bytes,
                             SrcBufferSetter setter) {
  setter(buffer);
  [buffer didModifyRange:NSMakeRange(0, buffer_size_in_bytes)];
}

void getManagedBufferFromDevice(id<MTLCommandQueue> cmd_q, id<MTLBuffer> buffer) {
  id<MTLCommandBuffer> cmd_buffer = [cmd_q commandBuffer];
  id<MTLBlitCommandEncoder> sync_dst_encoder = [cmd_buffer blitCommandEncoder];
  [sync_dst_encoder synchronizeResource:buffer];
  [sync_dst_encoder endEncoding];

  [cmd_buffer commit];
  [cmd_buffer waitUntilCompleted];
}

MTLSize operator/(MTLSize a, MTLSize b) {
  if ((a.width % b.width != 0) || (a.height % b.height != 0) || (a.depth % b.depth != 0)) {
    NSLog(@"b doesn't divide a evenly. a={%lu, %lu, %lu}, b={%lu, %lu, %lu}",
          (unsigned long)a.width, (unsigned long)a.height, (unsigned long)a.depth,
          (unsigned long)b.width, (unsigned long)b.height, (unsigned long)b.depth);
  }

  return MTLSize{a.width / b.width, a.height / b.height, a.depth / b.depth};
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device newDefaultLibrary];
    NSString *mtl_func_name = @"matmul_tiled";
    id<MTLFunction> mtl_func = [library newFunctionWithName:mtl_func_name];

    NSError *error = nil;
    id<MTLComputePipelineState> compute_pipeline_state =
        [device newComputePipelineStateWithFunction:mtl_func error:&error];

    NSUInteger M = 1024;
    NSUInteger N = 1024;
    NSUInteger K = 1024;

    NSUInteger tileM = 2;
    NSUInteger tileN = 128;

    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/
    NSUInteger src0_size = M * K * sizeof(float);
    NSUInteger src1_size = K * N * sizeof(float);
    NSUInteger dst_size = M * N * sizeof(float);

    id<MTLBuffer> src0_buffer = [device newBufferWithLength:src0_size
                                                    options:MTLResourceStorageModeManaged];
    id<MTLBuffer> src1_buffer = [device newBufferWithLength:src1_size
                                                    options:MTLResourceStorageModeManaged];

    id<MTLBuffer> dst_buffer = [device newBufferWithLength:dst_size
                                                   options:MTLResourceStorageModeManaged];
    //===-------------------------------------------------------------------===/
    // Set source buffer data
    //===-------------------------------------------------------------------===/
    SrcBufferValueGenerator getSrc0 = ^(NSUInteger i, NSUInteger j) {
      float v = ((float)((i + j * K) % 5) - 1.f) / 2.f;
      return v;
    };

    SrcBufferValueGenerator getSrc1 = ^(NSUInteger i, NSUInteger j) {
      float v = ((float)((i + j * N) % 7) - 1.f) / 2.f;
      return v;
    };

    setManagedBufferViaHost(src0_buffer, src0_size, ^(id<MTLBuffer> buffer) {
      float *float_buffer = (float *)buffer.contents;
      for (NSUInteger i = 0; i < M; ++i) {
        for (NSUInteger j = 0; j < K; ++j) {
          float_buffer[j + i * K] = getSrc0(i, j);
        }
      }
    });

    setManagedBufferViaHost(src1_buffer, src1_size, ^(id<MTLBuffer> buffer) {
      float *float_buffer = (float *)buffer.contents;
      for (NSUInteger i = 0; i < K; ++i) {
        for (NSUInteger j = 0; j < N; ++j) {
          float_buffer[j + i * N] = getSrc1(i, j);
        }
      }
    });

    //===-------------------------------------------------------------------===/
    // Dispatch
    //===-------------------------------------------------------------------===/

    id<MTLCommandQueue> cmd_q = [device newCommandQueue];

    id<MTLCommandBuffer> dispatch_cmd_buffer = [cmd_q commandBuffer];

    id<MTLComputeCommandEncoder> dispatch_cmd_encoder = [dispatch_cmd_buffer computeCommandEncoder];
    [dispatch_cmd_encoder setComputePipelineState:compute_pipeline_state];

    [dispatch_cmd_encoder setBuffer:src0_buffer offset:0 atIndex:0];
    [dispatch_cmd_encoder setBuffer:src1_buffer offset:0 atIndex:1];
    [dispatch_cmd_encoder setBuffer:dst_buffer offset:0 atIndex:2];

    MTLSize thread_groups_per_grid{N / tileN, M / tileM, 1};

    NSUInteger wg_size_x = 32;
    NSUInteger wg_size_y = 2;
    MTLSize threads_per_group{wg_size_x, wg_size_y, 1};

    [dispatch_cmd_encoder dispatchThreadgroups:thread_groups_per_grid
                         threadsPerThreadgroup:threads_per_group];
    [dispatch_cmd_encoder endEncoding];

    [dispatch_cmd_buffer commit];
    [dispatch_cmd_buffer waitUntilCompleted];

    //===-------------------------------------------------------------------===/
    // Verify destination buffer data
    //===-------------------------------------------------------------------===/
    getManagedBufferFromDevice(cmd_q, dst_buffer);
    float *dst_buffer_contents = (float *)dst_buffer.contents;

    for (NSUInteger i = 0; i < M; ++i) {
      for (NSUInteger j = 0; j < N; ++j) {
        float acc = 0.f;
        for (NSUInteger k = 0; k < K; ++k) {
          acc += getSrc0(i, k) * getSrc1(k, j);
        }
        float gpu_value = dst_buffer_contents[j + i * N];

        if (gpu_value != acc) {
          NSLog(@"destination buffer element (%lu,%lu) has incorrect value: expected to be %f but "
                @"found %f",
                (unsigned long)i, (unsigned long)j, acc, gpu_value);
          return 1;
        }
      }
    }

    //===-------------------------------------------------------------------===/
    // Benchmarking
    //===-------------------------------------------------------------------===/

    double elapsed_seconds = 0;
    NSUInteger iters = 356;

    std::stringstream oss;

    for (NSUInteger iter = 0; iter < iters; ++iter) {
      id<MTLCommandBuffer> bm_cmd_buffer = [cmd_q commandBuffer];
      id<MTLComputeCommandEncoder> bm_cmd_encoder = [bm_cmd_buffer computeCommandEncoder];
      [bm_cmd_encoder setComputePipelineState:compute_pipeline_state];

      [bm_cmd_encoder setBuffer:src0_buffer offset:0 atIndex:0];
      [bm_cmd_encoder setBuffer:src1_buffer offset:0 atIndex:1];
      [bm_cmd_encoder setBuffer:dst_buffer offset:0 atIndex:2];

      [bm_cmd_encoder dispatchThreadgroups:thread_groups_per_grid
                     threadsPerThreadgroup:threads_per_group];
      [bm_cmd_encoder endEncoding];

      auto start_time = std::chrono::high_resolution_clock::now();
      [bm_cmd_buffer commit];
      [bm_cmd_buffer waitUntilCompleted];
      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_seconds_ =
          std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
      oss << elapsed_seconds_.count() << ", ";
      elapsed_seconds += elapsed_seconds_.count();
    }

    NSLog(@"%s", oss.str().c_str());
    elapsed_seconds /= iters;
    double num_operation = double(N) * double(M) * double(K) * 2.;
    NSLog(@"Time: %luus", (unsigned long)(elapsed_seconds * 1e6));
    NSLog(@"Iterations: %lu", (unsigned long)iters);
    NSLog(@"FLOps: %fG/s", (num_operation / (1e9 * elapsed_seconds)));
  }

  return 0;
}
