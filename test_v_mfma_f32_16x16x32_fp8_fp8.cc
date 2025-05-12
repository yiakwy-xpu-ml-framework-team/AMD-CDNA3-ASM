#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <hip/hip_fp8.h>

#include <iostream>
#include <iomanip>

using half = __half;
using float16_t = _Float16;

// amd fp8 is safely used in host side
using fp8_t = __hip_fp8_e4m3_fnuz;

#include <random>

#include <rocwmma/rocwmma.hpp>

#define M (16)
#define N (16)
#define K (32)

#define FRAG_M 16
#define FRAG_N 16
#define FRAG_K 32

#define BLOCK_SIZE_M (FRAG_M * 1)
#define BLOCK_SIZE_N (FRAG_N * 1)
#define BLOCK_SIZE_K (FRAG_K * 1)

#define USE_ASM true
#define ALU_ASYN false

// DEMO 0 : rocwmma
// DEMO 1 : __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8, demonstrate multi blocks single warp mfma
// DEMO 2 : v_mfma_f32_16x16x32_fp8_fp8, demonstrate multi blocks single warp mfma
#define DEMO 1

__device__ __inline__ void async_load_fence(uint32_t cnt) {
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}

__device__ __inline__ void __sync_warp(void) { __asm__ volatile("s_barrier" : : : "memory"); }

__global__ void v_mfma_intrinsics_test(const fp8_t* __restrict__ lhs, const fp8_t* __restrict__ rhs, float* output,
                                       unsigned int m, unsigned int n, unsigned int k,
                                       unsigned int lda, unsigned int ldb, unsigned int ldc) {
    
    int lane_id = threadIdx.x % warpSize;
    constexpr unsigned int ele_per_thread = FRAG_M * FRAG_N / warpSize;

    int c_row_base = (blockIdx.x * BLOCK_SIZE_M);
    int c_col_base = (blockIdx.y * BLOCK_SIZE_N);

    using fp8_t = __hip_fp8_e4m3_fnuz;

    __shared__ fp8_t lhs_tile[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ fp8_t rhs_tile[BLOCK_SIZE_N][BLOCK_SIZE_K];

    fp8_t* lhs_tile_base_ptr = &(lhs_tile[0][0]);
    fp8_t* rhs_tile_base_ptr = &(rhs_tile[0][0]);

    // the output fragment is stored in 4 x AccVGPRs (see CDNA3 ISA), each thread process 4 fp32 elements
    using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    uint acc_frag[4] = {0};

    floatx4 *d = reinterpret_cast<floatx4 *>(acc_frag);

    for (int i=0; i < k; i+= BLOCK_SIZE_K) {

        constexpr int thr_per_row = 4;
        {
            constexpr int vec_size = 16;
            int s_col = (lane_id % thr_per_row) * vec_size;
            int s_row = (lane_id / thr_per_row);

            int g_col = i + s_col;
            int g_row = c_row_base + s_row;

            if (g_row < m && g_col < k) {
                int4* lhs_tile_4i_ptr = (int4 *)(lhs_tile_base_ptr + s_row * BLOCK_SIZE_K + s_col);
                int4* lhs_4i_ptr = (int4 *)(lhs + g_row * lda + g_col);

                *(lhs_tile_4i_ptr) = *(lhs_4i_ptr);
            }
        } // load_tile_a

        __syncthreads();

        {
            constexpr int vec_size = 16;
            int s_col = (lane_id % thr_per_row) * vec_size;
            int s_row = (lane_id / thr_per_row);

            int g_col = i + s_col;
            int g_row = c_col_base + s_row;

            if (g_row < n && g_col < k) {
                int4* rhs_tile_4i_ptr = (int4 *)(rhs_tile_base_ptr + s_row * BLOCK_SIZE_K + s_col);
                int4* rhs_4i_ptr = (int4 *)(rhs + g_row * ldb + g_col);

                *(rhs_tile_4i_ptr) = *(rhs_4i_ptr);
            } 
        } // load_tile_b

        __syncthreads();

        fp8_t a_frag[8] = {0};
        fp8_t b_frag[8] = {0};
        fp8_t* lhs_tile_ptr = nullptr;
        fp8_t* rhs_tile_ptr = nullptr;

        {
            constexpr int vec_size = 8;
            int s_col = (lane_id / 16) * vec_size;
            int s_row = (lane_id % 16);
            lhs_tile_ptr = lhs_tile_base_ptr + s_row * BLOCK_SIZE_K + s_col;
        } //  compute smem window for compute

        {
            constexpr int vec_size = 8;
            int s_col = (lane_id / 16) * vec_size;
            int s_row = (lane_id % 16);
            rhs_tile_ptr = rhs_tile_base_ptr + s_row * BLOCK_SIZE_K + s_col;
        } // compute smem window for compute


        for (int i=0; i < 8; i++) {
            a_frag[i] = lhs_tile_ptr[i];
            b_frag[i] = rhs_tile_ptr[i];
        }

        long *a = reinterpret_cast<long *>(a_frag);
        long *b = reinterpret_cast<long *>(b_frag);

        __sync_warp();

#if !USE_ASM
        *d = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            *a,
            *b,
            *d, 0, 0, 0);
#else
        asm volatile("s_nop 1" ::);
        // 4 x outer product accumulation asynchronously
        asm volatile("v_mfma_f32_16x16x32_fp8_fp8 "
                        "%0, "
                        "%1, "
                        "%2, %3"
                        : "+v"(*d)
                        :  "v"(*a), "v"(*b), "v"(*d));
        
        __sync_warp();
#endif // USE_ASM
    }

    // Note(yiakwy) : store thread private memory back to global memory is tricky. Suppose output is still major layout , then threads configuration is (4/*x*/, 4/*j*/, 16/*y*/) :
    //
    // lane                   0     1         15     16     17         32  ...    63
    //  Row(x) \ Col (j,y) <0,0> <0,1> ... <0,15> <1, 0> <1, 1> ... <1,15> ... <4,15>
    //   0                   x  
    //   1                   x
    //   2                   x
    //   3                   x
    //
    // x = output[warp_offset + coord2Indx (x, j, y)], coord2Index : (x, j, y) -> ( y + j * 16 ) + x * 64
    for (int j=0; j < ele_per_thread; j++) {
        auto x = ( lane_id / 16 ) % 4;
        auto y = lane_id % 16; // y + j * 16 is the output data lane ID
        auto outIdx = c_row_base * ldc + c_col_base + y + (j + x * 4) * ldc;
        output[outIdx] = (*d)[j];
    }
}

__global__ void v_mfma_test(const fp8_t* __restrict__ lhs, const fp8_t* __restrict__ rhs, float* output,
                            unsigned int m, unsigned int n, unsigned int k,
                            unsigned int lda, unsigned int ldb, unsigned int ldc)
{
    int c_row_base = (blockIdx.x * BLOCK_SIZE_M);
    int c_col_base = (blockIdx.y * BLOCK_SIZE_N);

    rocwmma::fragment<rocwmma::matrix_a, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, fp8_t, rocwmma::row_major> a_frag;
    rocwmma::fragment<rocwmma::matrix_b, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, fp8_t, rocwmma::col_major> b_frag;

    rocwmma::fragment<rocwmma::accumulator, FRAG_M, FRAG_N, FRAG_K, float> acc_frag;

    rocwmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < k; i+= BLOCK_SIZE_K) {
        // each thread load 8x fp8 elements
        rocwmma::load_matrix_sync(a_frag, lhs + c_row_base * lda + i, lda);
        rocwmma::load_matrix_sync(b_frag, rhs + c_col_base * ldb + i, ldb);

        // Matrix multiply - accumulate using MFMA units
        rocwmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // output is row major
    rocwmma::store_matrix_sync(output + c_row_base * ldc + c_col_base, acc_frag, ldc, rocwmma::mem_row_major);
}

__hip_fp8_storage_t
convert_float_to_fp8(float in, /* Input val */
                    __hip_fp8_interpretation_t
                        interpret, /* interpretation of number E4M3/E5M2 */
                    __hip_saturation_t sat /* Saturation behavior */
) {
    return __hip_cvt_float_to_fp8(in, sat, interpret);
}

float convert_fp8_to_float(
    __hip_fp8_storage_t in, /* Input val */
    __hip_fp8_interpretation_t
        interpret /* interpretation of number E4M3/E5M2 */
) {
    __half hf = __hip_cvt_fp8_to_halfraw(in, interpret);
    return static_cast<float>(hf);
}

void hgemm(
    const float*  __restrict__ ptr_a/*row major*/,
    const float*  __restrict__ ptr_b/*col major*/,
    float*  ptr_c,
    int m,
    int n,
    int k,
    int lda/*stride_a_m*/,
    int ldb/*stride_b_n*/,
    int ldc/*stride_c_m*/)
{

    auto accessor = [](uint32_t row, uint32_t col, uint32_t stride_row, uint32_t stride_col) { return col * stride_col + row * stride_row; };

    for(auto i_m = 0 ; i_m < m; i_m++) {
        for(auto i_n = 0; i_n < n; i_n++) {
            float acc = 0;
            for(auto i_k = 0; i_k < k; i_k++) {
                // i_k + i_n * K
                acc += ptr_a[i_m * lda + i_k] * ptr_b[accessor(i_k, i_n, 1, K)];
            }
            ptr_c[i_m * ldc + i_n] = acc;
        }
    }
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

    std::cout << "Mjaor: " << major << "," << "Minor: " << minor << std::endl;

    int max_smem_per_sm = 0;
    hipDeviceGetAttribute(
        &max_smem_per_sm, hipDeviceAttribute_t::hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device_id);

    std::cout << "Max sems per sm : " << max_smem_per_sm << std::endl;

    hipDeviceProp_t prop;
    hip_check(hipGetDeviceProperties(&prop, 0));
    bool is_supported = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos); // gfx94x
    if(!is_supported) {
        std::cerr << "Need a gfx94x, but found: " << prop.gcnArchName << std::endl;
        std::cerr << "No device conversions are supported, only host conversions are supported." << std::endl;
        return -1;
    }

    const __hip_fp8_interpretation_t interpret = (std::string(prop.gcnArchName).find("gfx94") != std::string::npos)
                                                    ? __HIP_E4M3_FNUZ // gfx94x
                                                    : __HIP_E4M3;
    constexpr __hip_saturation_t sat = __HIP_SATFINITE;

    // tests mfma

    float* host_a, *host_b, *host_fp32_c;

    // allocate MxK, NxK two row major matrices
    host_a = (float*)malloc(M * K * sizeof(float));
    host_b = (float*)malloc(N * K * sizeof(float));
    host_fp32_c = (float*)malloc(M * N * sizeof(float));

    fp8_t *host_fp8_a, *host_fp8_b;

    host_fp8_a = (fp8_t *)malloc(M * K * sizeof(fp8_t));
    host_fp8_b = (fp8_t *)malloc(N * K * sizeof(fp8_t));

    std::random_device device;
    std::mt19937 gen(0);// gen(device());

    std::uniform_real_distribution<> dis(0.0009765625, 1.0);

    for (int i=0; i < M*K; i++ ) {
        float fval = dis(gen);
        auto fp8 = convert_float_to_fp8(fval, interpret, sat);
        auto fp32_aligned = convert_fp8_to_float(fp8, interpret);

        host_fp8_a[i] = static_cast<fp8_t>(fp32_aligned);
        host_a[i] = fp32_aligned;
    }

    for (int i=0; i < N*K; i++) {
        float fval = dis(gen);
        auto fp8 = convert_float_to_fp8(fval, interpret, sat);
        auto fp32_aligned = convert_fp8_to_float(fp8, interpret);

        host_fp8_b[i] = static_cast<fp8_t>(fp32_aligned);
        host_b[i] = fp32_aligned; 
    }

    hgemm(host_a, host_b, host_fp32_c, M, N, K, K/*stride_a_m*/, K/*stride_b_n*/, N/*stride_c_m*/);

    size_t num_ele = M * N;
    float* output = (float*)malloc(sizeof(float) * num_ele);
    
    // the amd dtype can be called in host side safely
    fp8_t *lhs_d, *rhs_d;
    hip_check(hipMalloc(&lhs_d, sizeof(fp8_t) * M * K));
    hip_check(hipMalloc(&rhs_d, sizeof(fp8_t) * N * K));
    hip_check(hipMemcpy(lhs_d, host_fp8_a, M * K * sizeof(fp8_t), hipMemcpyHostToDevice));
    hip_check(hipMemcpy(rhs_d, host_fp8_b, N * K * sizeof(fp8_t), hipMemcpyHostToDevice));

    float* output_d = nullptr;
    hip_check(hipMalloc(&output_d, sizeof(float) * num_ele));

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

    dim3 grid(CEILDIV(M, BLOCK_SIZE_M), CEILDIV(N, BLOCK_SIZE_N));
    dim3 block(64, 1);

    if (DEMO == 0) {
        v_mfma_test<<<grid, block>>>(lhs_d, rhs_d, output_d, 
                                    M, N, K, 
                                    K/*stride_a_m*/, K/*stride_b_n*/, N/*stride_c_m*/);
    } else if (DEMO == 1) {
        v_mfma_intrinsics_test<<<grid, block>>>(lhs_d, rhs_d, output_d, 
                                                M, N, K,
                                                K/*stride_a_m*/, K/*stride_b_n*/, N/*stride_c_m*/);
    } else {

    }


    hip_check(hipMemcpy(output, output_d, sizeof(float) * num_ele, hipMemcpyDeviceToHost));
    if (!assert_alleq(output, host_fp32_c, num_ele)) {

        std::cout << "***** host_a: *****" << std::endl;
        print(host_a, M, K);

        std::cout << "***** host_b: *****" << std::endl;
        print(host_b, N, K);

        std::cout << "===== host_c: =====" << std::endl;
        print(host_fp32_c, M, N);

        std::cout << "===== output : =====" << std::endl;
        print(output, M, N);

    }

    free(host_a);
    free(host_b);
    free(host_fp32_c);

    free(host_fp8_a);
    free(host_fp8_b);
    hip_check(hipFree(lhs_d));
    hip_check(hipFree(rhs_d));
    hip_check(hipFree(output_d));
    free(output);

    return 0;
}