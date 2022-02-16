#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <objc/NSObjCRuntime.h>
#include <chrono>

#include <cmath>

typedef float (^SrcBufferValueGenerator)(NSUInteger);
typedef void (^SrcBufferSetter)(id<MTLBuffer>);

// NOTE: Corresponds to SetDeviceBufferViaStagingBuffer in uVkCompute.
void setManagedBufferViaHost(id<MTLBuffer> buffer, NSUInteger buffer_size_in_bytes,
                             SrcBufferSetter setter) {
  setter(buffer);
  [buffer didModifyRange:NSMakeRange(0, buffer_size_in_bytes)];
}

// NOTE: Corresponds to GetDeviceBufferViaStagingBuffer in uVkCompute.
void getManagedBufferFromDevice(id<MTLCommandQueue> cmd_q, id<MTLBuffer> buffer) {
  id<MTLCommandBuffer> cmd_buffer = [cmd_q commandBuffer];
  id<MTLBlitCommandEncoder> sync_dst_encoder = [cmd_buffer blitCommandEncoder];
  [sync_dst_encoder synchronizeResource:buffer];
  [sync_dst_encoder endEncoding];

  [cmd_buffer commit];
  [cmd_buffer waitUntilCompleted];
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    // TODO Proper error handling.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> library = [device newDefaultLibrary];
    NSString *mtl_func_name = @"mad_throughput";
    id<MTLFunction> mtl_func = [library newFunctionWithName:mtl_func_name];

    NSError *error = nil;
    id<MTLComputePipelineState> compute_pipeline_state =
        [device newComputePipelineStateWithFunction:mtl_func error:&error];

    id<MTLCommandQueue> cmd_q = [device newCommandQueue];

    NSUInteger num_element = 1024 * 1024;
    NSUInteger src0_size = num_element * sizeof(float);
    NSUInteger src1_size = num_element * sizeof(float);
    NSUInteger dst_size = num_element * sizeof(float);

    // NOTE: Storage modes chosen below are probably specific to macOS. For differences between
    // macOS and iOS, see:
    // https://developer.apple.com/documentation/metal/setting_resource_storage_modes?language=objc.

    // We need to populate src buffers using the CPU and then use them on the GPU, hence the
    // managed storage mode.
    id<MTLBuffer> src0_buffer = [device newBufferWithLength:src0_size
                                                    options:MTLResourceStorageModeManaged];
    id<MTLBuffer> src1_buffer = [device newBufferWithLength:src1_size
                                                    options:MTLResourceStorageModeManaged];

    // We need to read the dst buffer on the CPU to verify the results, hence the managed
    // storage mode.
    id<MTLBuffer> dst_buffer = [device newBufferWithLength:dst_size
                                                   options:MTLResourceStorageModeManaged];

    SrcBufferValueGenerator getSrc0 = ^(NSUInteger i) {
      float v = (float)((i % 9) + 1) * 0.1f;
      return v;
    };

    SrcBufferValueGenerator getSrc1 = ^(NSUInteger i) {
      float v = (float)((i % 5) + 1) * 1.f;
      return v;
    };

    setManagedBufferViaHost(src0_buffer, src0_size, ^(id<MTLBuffer> buffer) {
      float *float_buffer = (float *)buffer.contents;
      for (NSUInteger i = 0; i < num_element; ++i) {
        float_buffer[i] = getSrc0(i);
      }
    });

    setManagedBufferViaHost(src1_buffer, src1_size, ^(id<MTLBuffer> buffer) {
      float *float_buffer = (float *)buffer.contents;
      for (NSUInteger i = 0; i < num_element; ++i) {
        float_buffer[i] = getSrc1(i);
      }
    });

    //===-------------------------------------------------------------------===/
    // Dispatch
    //===-------------------------------------------------------------------===/

    id<MTLCommandBuffer> cmd_buffer = [cmd_q commandBuffer];

    id<MTLComputeCommandEncoder> cmd_encoder = [cmd_buffer computeCommandEncoder];
    [cmd_encoder setComputePipelineState:compute_pipeline_state];

    [cmd_encoder setBuffer:src0_buffer offset:0 atIndex:0];
    [cmd_encoder setBuffer:src1_buffer offset:0 atIndex:1];
    [cmd_encoder setBuffer:dst_buffer offset:0 atIndex:2];

    const MTLSize max_threads_per_group = {64, 1, 1};
    // NOTE: In uVkCompute, the compuation is: num_element / (4 * 16). Do not know why the * 16.
    // For sake of experiment, tried to remove this part and got:
    // [mvk-error] VK_ERROR_DEVICE_LOST: Command buffer 0x7fd63a80a600 "vkQueueSubmit
    // CommandBuffer on Queue 0-0" execution failed (code 2): Caused GPU Timeout Error (IOAF
    // code 2) mad_throughput_main.cc:170: check error: INTERNAL: VK_ERROR_DEVICE_LOST
    const NSUInteger total_num_threads = num_element / (4);
    const NSUInteger thread_groups_per_grid = total_num_threads / max_threads_per_group.width;
    MTLSize tgpg;
    tgpg.width = thread_groups_per_grid;
    tgpg.height = tgpg.depth = 1;

    MTLSize tptg;
    tptg.width = max_threads_per_group.width;
    tptg.height = tptg.depth = 1;
    [cmd_encoder dispatchThreadgroups:tgpg threadsPerThreadgroup:tptg];

    [cmd_encoder endEncoding];

    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];

    //===-------------------------------------------------------------------===/
    // Verify destination buffer data
    //===-------------------------------------------------------------------===/
    getManagedBufferFromDevice(cmd_q, dst_buffer);

    float *dst_buffer_contents = (float *)dst_buffer.contents;

    for (NSUInteger i = 0; i < num_element; ++i) {
      float limit = getSrc1(i) * (1.f / (1.f - getSrc0(i)));
      if (std::fabs(limit - dst_buffer_contents[i]) >= 0.01f) {
        NSLog(@"destination buffer element #%lu has incorrect vlaue: expected to be %f but "
              @"found %f",
              (unsigned long)i, limit, dst_buffer_contents[i]);
        return 1;
      }
    }

    //===-------------------------------------------------------------------===/
    // Benchmarking
    //===-------------------------------------------------------------------===/

    id<MTLCommandBuffer> bm_cmd_buffer = [cmd_q commandBuffer];

    id<MTLComputeCommandEncoder> bm_cmd_encoder = [bm_cmd_buffer computeCommandEncoder];
    [bm_cmd_encoder setComputePipelineState:compute_pipeline_state];

    // Vulkan is nicer here since we can simply rebind descriptor set.
    [bm_cmd_encoder setBuffer:src0_buffer offset:0 atIndex:0];
    [bm_cmd_encoder setBuffer:src1_buffer offset:0 atIndex:1];
    [bm_cmd_encoder setBuffer:dst_buffer offset:0 atIndex:2];
    [bm_cmd_encoder dispatchThreadgroups:tgpg threadsPerThreadgroup:tptg];
    [bm_cmd_encoder endEncoding];

    auto start_time = std::chrono::high_resolution_clock::now();
    [bm_cmd_buffer commit];
    [bm_cmd_buffer waitUntilCompleted];
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);

    int loop_count = 100000;
    double num_operation = double(num_element) * 2. * 10. * double(loop_count);

    //===-------------------------------------------------------------------===/
    // Print Results
    //===-------------------------------------------------------------------===/

    NSLog(@"Device: %@", [device name]);
    NSLog(@"Benchmark: mad_throughput_f32");
    NSLog(@"Num of elements: %lu", num_element);
    NSLog(@"Loop count: %d", loop_count);
    NSLog(@"Time: %luus", (unsigned long)(elapsed_seconds.count() * 1e6));
    NSLog(@"Iterations: 1");
    NSLog(@"FLOps: %fT/s", (num_operation / (1e12 * elapsed_seconds.count())));
  }
  return 0;
}
