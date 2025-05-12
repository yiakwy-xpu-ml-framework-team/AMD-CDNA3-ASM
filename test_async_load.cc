#include <hip/hip_runtime.h>

#include <ck/utility/amd_buffer_addressing.hpp>

#include <iostream>
#include <iomanip>

#define WARP_SIZE warpSize

#define FULL_MASK 0xffffffff

#define M (16)
#define N (16)

#define FRAG_M 16
#define FRAG_N 16

#define BLOCK_SIZE_M (FRAG_M * 1)
#define BLOCK_SIZE_N (FRAG_N * 1)

#define ASYNC_LOAD true
#define DIRECT_STORE false // use ds_loadx128 bit pending ...

// See LLVM GFX 9 [buffer_load_dword](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX940.html), 
__device__ __inline__ void load_32bit_async(uint32_t* lds_ptr, const uint32_t* __restrict__ res_ptr/*must be dram pointer*/, uint32_t dram_offsets){
    uint32_t offsets_bytes = dram_offsets * sizeof(uint32_t) ;
    
    ck::int32x4_t src_resource = ck::make_wave_buffer_resource_with_default_range(res_ptr);

    auto const lds_ptr_sgpr =
        __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(lds_ptr)));

    asm volatile("s_mov_b32 m0, %0; \n\t"
                 "buffer_load_dword %1, %2, 0 offen offset:0 lds; \n\t" 
                 :
                 : "s"(lds_ptr_sgpr),
                   "v"( offsets_bytes ),
                   "s"( src_resource )
                 : "memory");
}

__device__ __inline__ void async_load_fence(uint32_t cnt) {
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}
__device__ __inline__ void wave_barrier() {
    asm volatile("s_barrier" : : : "memory");
}

__global__ void test_buffer_load_async(const uint32_t* __restrict__ input, uint32_t* __restrict__ output, 
                                       int m, int n) {
    using storage_t = uint32_t;

    __shared__ storage_t smem[BLOCK_SIZE_M * BLOCK_SIZE_N];

    storage_t* smem_base_ptr = &smem[0];

    memset(smem + threadIdx.x * 4, 0, 4);
    memset(output + threadIdx.x * 4, 0, 4);

    __syncthreads();

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int kElementsPerAccess = 16 / sizeof(uint32_t);
    int kNumThrPerRow = WARP_SIZE / FRAG_M;

    storage_t *input_ptr, *smem_ptr, *output_ptr;
    uint4 *input_4i_ptr, *smem_4i_ptr, *output_4i_ptr;

    {
#if ASYNC_LOAD
        // GFX 9 direct load asynchrously
        for (int i=0; i < 4; i++) {
            int s_col = threadIdx.x % BLOCK_SIZE_N;
            int s_row = i * 4 + threadIdx.x / BLOCK_SIZE_N;

            int g_col = s_col;
            int g_row = s_row;

            smem_ptr = smem_base_ptr + s_row * BLOCK_SIZE_N + s_col;

            load_32bit_async(smem_ptr, input, g_row * n + g_col);
        }
        async_load_fence(0);
        wave_barrier();

#else
        int s_row = lane_id / kNumThrPerRow;
        int s_col = lane_id % kNumThrPerRow * kElementsPerAccess;

        int g_row = s_row;
        int g_col = s_col;

        smem_ptr = smem_base_ptr + s_row * BLOCK_SIZE_N + s_col;
        input_ptr = input + g_row * n + g_col;

        input_4i_ptr = (uint4*)(input_ptr);
        smem_4i_ptr = (uint4*)(smem_ptr);

        *(smem_4i_ptr) = *(input_4i_ptr);

        output_ptr = output + g_row * n + g_col;
#endif
    } // load input

    {
#if ASYNC_LOAD 
        int s_row = lane_id / kNumThrPerRow;
        int s_col = lane_id % kNumThrPerRow * kElementsPerAccess;

        int g_row = s_row;
        int g_col = s_col;

        smem_ptr = smem_base_ptr + s_row * BLOCK_SIZE_N + s_col;
        smem_4i_ptr = (uint4*)(smem_ptr);

        output_ptr = output + g_row * n + g_col;
#endif

        output_4i_ptr = (uint4*)(output_ptr);
        *(output_4i_ptr) = *(smem_4i_ptr);
    } // store output  
}

bool assert_alleq(float* out, float* ref, int num_el) {
    for (int i = 0; i < num_el; i++) {
        if (std::abs(out[i] - ref[i]) > 1e-3) {
            printf("[x] out[%d](%f) does not meet ref[%d](%f)\n", i, out[i], i, ref[i]);
            return false;
        }
    }

    printf("[ok] out matches ref!\n");
    return true;
}

void print(float* output, int rows, int cols) {
    std::cout << "[\n";
    for (int i=0; i < rows; i++) {
        std::cout << "  [ ";
        for (int j=0; j < cols; j++) {
            std::cout << std::fixed << std::setw(4) << std::setfill(' ') << std::setprecision(2) << output[i * cols + j];
            if (j < cols-1) {
                std::cout << ", ";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "]" << std::endl; 
    }
    std::cout << "]\n" << std::endl;
}

#define hip_check(hip_call)                                                    \
{                                                                              \
    auto hip_res = hip_call;                                                   \
    if (hip_res != hipSuccess) {                                               \
      std::cerr << "Failed in HIP call: " << #hip_call \
                << " at " << __FILE__ << ":" << __LINE__ \
                << " with error: " << hipGetErrorString(hip_res) << std::endl; \
      std::abort();                                                            \
    }                                                                          \
}

int main(int argc, char * argv[]) {
    int device_id = 0;
    hipGetDevice(&device_id);
    int major = 0, minor = 0;
    hipDeviceComputeCapability(&major, &minor, device_id);

    std::cout << "[main] Mjaor: " << major << "," << "Minor: " << minor << std::endl;

    int max_smem_per_sm = 0;
    hipDeviceGetAttribute(
        &max_smem_per_sm, hipDeviceAttribute_t::hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device_id);

    std::cout << "[main] Max sems per sm : " << max_smem_per_sm << std::endl;

    int num_elements = M * N;

    float* input = (float*)malloc(sizeof(float) * num_elements);

    for (int i=0; i < num_elements; i++) {
        input[i] = i + 1.1;
    }

    float* output = (float*)malloc(sizeof(float) * num_elements);
    memset(output, 0, num_elements);

    float* output_d = nullptr;
    hipMalloc(&output_d, sizeof(float) * num_elements);

    float* input_d = nullptr;
    hipMalloc(&input_d, sizeof(float) * num_elements);

    hipMemcpy(input_d, input, sizeof(float) * num_elements, hipMemcpyHostToDevice);

    printf("[main] launching kernel ...\n");

    test_buffer_load_async<<<1, 64>>>(reinterpret_cast<uint32_t*>(input_d), reinterpret_cast<uint32_t*>(output_d), M, N);

    printf("[main] kernel launched.\n");

    hip_check(hipMemcpy(output, output_d, sizeof(float) * num_elements, hipMemcpyDeviceToHost));
    hip_check(hipDeviceSynchronize());

    printf("[main] copy output.\n");

    if (!assert_alleq(input, output, M * N)) {
        std::cout << "***** input: *****" << std::endl;
        print(input, M, N);

        std::cout << "===== output : =====" << std::endl;
        print(output, M, N);
    }

    free(input);
    free(output);

    hip_check(hipFree(input_d));
    hip_check(hipFree(output_d));

    return 0;

}