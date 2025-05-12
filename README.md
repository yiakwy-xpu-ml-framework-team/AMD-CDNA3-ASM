## MI300X CDNA3 Instructrion

#### test_v_mfma_f32_16x16x32_fp8_fp8

This file demonstrates basic usage of AMD fp8 data type and **__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp** and ASM instruction **v_mfma_f32_16x16x32_fp8_fp8**.

It supports single warp (64 threads, 1-4-1) , multi-blocks v_mfma execution.

###### How to use

- Compilation :

    ```
    hipcc test_v_mfma_f32_16x16x32_fp8_fp8.cc -o test_v_mfma_fp8
    ```

- Execution : 

Successful exection should exepct :

![demo](assets/successful_exection.png)

- Inspect ASM : 

    ```
    hipcc -S test_v_mfma_f32_16x16x32_fp8_fp8.cc -save-temps -o test_v_mfma_fp8.S
    ```

#### test_async_load

This file demonstrates direct load of global data into shared memory asynchronously in AMD platform. This is the critical feature introduced in Ampere arch in NVIDIA GPU.

In CDNA3 (MI300X) 128 bit direct load/store synchronously, and 32 bits direct load asynchronously are supported.

With direct load feature, register used to load data will be reduced. However registers to compute per thread offset is still inevitable.

In CDNA4 (MI325X), one can do 128 bit direct load asynchronously.