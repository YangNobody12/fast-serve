//! Direct dynamic FFI bindings to `nvcuda.dll` (CUDA Driver API) and `cublas64_12.dll`.
//! Zero PyTorch / zero CuPy / zero allocator bloat — direct single-arena VRAM & CUDA Graphs.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use anyhow::{bail, Context, Result};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::ptr;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUgraphNode = *mut c_void;
pub type CUgraphConditionalHandle = u64;

pub const CU_GRAPH_NODE_TYPE_CONDITIONAL: c_uint = 13;
pub const CU_GRAPH_COND_TYPE_IF: c_uint = 0;
pub const CU_GRAPH_COND_ASSIGN_DEFAULT: c_uint = 1;

#[repr(C)]
pub struct CUDA_CONDITIONAL_NODE_PARAMS {
    pub handle: CUgraphConditionalHandle,
    pub cond_type: c_uint,
    pub size: c_uint,
    pub phGraph_out: *mut CUgraph,
    pub ctx: CUcontext,
}

#[repr(C)]
pub struct CUgraphNodeParamsCond {
    pub node_type: c_uint,
    pub reserved0: [c_int; 3],
    pub conditional: CUDA_CONDITIONAL_NODE_PARAMS,
    pub pad: [u64; 26], // Total 16 + 32 + 208 = 256 bytes (verified against CUDA 12.8 cuda.h!)
}

pub type CublasStatus = c_int;
pub type CublasHandle = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;

pub const CUBLAS_OP_N: c_int = 0;
pub const CUBLAS_OP_T: c_int = 1;
pub const CUDA_R_8I: c_int = 3;
pub const CUDA_R_32I: c_int = 10;
pub const CUDA_R_16BF: c_int = 14;
pub const CUDA_R_32F: c_int = 0;
pub const CUBLAS_COMPUTE_32F: c_int = 68;
pub const CUBLAS_COMPUTE_32I: c_int = 72;
pub const CUBLAS_COMPUTE_32F_FAST_TF32: c_int = 77;
pub const CUBLAS_GEMM_DEFAULT_TENSOR_OP: c_int = 99;
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 2;

fn check_cu(res: CUresult, op: &str) -> Result<()> {
    if res != CUDA_SUCCESS {
        bail!("CUDA Driver error in {op}: code {res}");
    }
    Ok(())
}

fn check_cublas(res: CublasStatus, op: &str) -> Result<()> {
    if res != CUBLAS_STATUS_SUCCESS {
        bail!("cuBLAS error in {op}: code {res}");
    }
    Ok(())
}

// Function pointer typedefs
type FnCuInit = unsafe extern "C" fn(c_uint) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type FnCuCtxCreate = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
type FnCuMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type FnCuMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type FnCuMemsetD8Async = unsafe extern "C" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult;
type FnCuMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type FnCuMemcpyHtoDAsync = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
type FnCuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type FnCuMemcpyDtoHAsync = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
type FnCuMemcpyDtoDAsync = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
type FnCuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult;
type FnCuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CUresult;
type FnCuStreamCreate = unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult;
type FnCuStreamSynchronize = unsafe extern "C" fn(CUstream) -> CUresult;
type FnCuStreamBeginCapture = unsafe extern "C" fn(CUstream, c_int) -> CUresult;
type FnCuStreamEndCapture = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
type FnCuGraphInstantiateWithFlags = unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult;
type FnCuGraphLaunch = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
type FnCuGraphDestroy = unsafe extern "C" fn(CUgraph) -> CUresult;
type FnCuGraphExecDestroy = unsafe extern "C" fn(CUgraphExec) -> CUresult;
type FnCuGraphCreate = unsafe extern "C" fn(*mut CUgraph, c_uint) -> CUresult;
type FnCuGraphAddChildGraphNode = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, CUgraph) -> CUresult;
type FnCuGraphConditionalHandleCreate = unsafe extern "C" fn(*mut CUgraphConditionalHandle, CUgraph, CUcontext, c_uint, c_uint) -> CUresult;
type FnCuGraphAddNode = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *mut CUgraphNodeParamsCond) -> CUresult;
type FnCuProfilerStart = unsafe extern "C" fn() -> CUresult;
type FnCuProfilerStop = unsafe extern "C" fn() -> CUresult;
type FnCuEventCreate = unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult;
type FnCuEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
type FnCuEventSynchronize = unsafe extern "C" fn(CUevent) -> CUresult;
type FnCuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnCuModuleGetFunction = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type FnCuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    c_uint, c_uint, c_uint,
    c_uint, c_uint, c_uint,
    c_uint,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;

type FnCublasCreate = unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus;
type FnCublasSetStream = unsafe extern "C" fn(CublasHandle, CUstream) -> CublasStatus;
type FnCublasGemmEx = unsafe extern "C" fn(
    CublasHandle,
    c_int, c_int,
    c_int, c_int, c_int,
    *const c_void,
    *const c_void, c_int, c_int,
    *const c_void, c_int, c_int,
    *const c_void,
    *mut c_void, c_int, c_int,
    c_int, c_int,
) -> CublasStatus;
type FnCublasGemmStridedBatchedEx = unsafe extern "C" fn(
    CublasHandle,
    c_int, c_int,
    c_int, c_int, c_int,
    *const c_void,
    *const c_void, c_int, c_int, i64,
    *const c_void, c_int, c_int, i64,
    *const c_void,
    *mut c_void, c_int, c_int, i64,
    c_int,
    c_int, c_int,
) -> CublasStatus;

pub struct CudaRuntime {
    _cuda_lib: Library,
    _cublas_lib: Library,
    pub ctx: CUcontext,
    pub stream: CUstream,
    pub cublas: CublasHandle,
    pub module: CUmodule,

    cuCtxSetCurrent: FnCuCtxSetCurrent,
    cuMemGetInfo: FnCuMemGetInfo,
    cuMemAlloc: FnCuMemAlloc,
    cuMemFree: FnCuMemFree,
    cuMemsetD8Async: FnCuMemsetD8Async,
    cuMemcpyHtoD: FnCuMemcpyHtoD,
    cuMemcpyHtoDAsync: FnCuMemcpyHtoDAsync,
    cuMemcpyDtoH: FnCuMemcpyDtoH,
    cuMemcpyDtoHAsync: FnCuMemcpyDtoHAsync,
    cuMemcpyDtoDAsync: FnCuMemcpyDtoDAsync,
    cuMemHostAlloc: FnCuMemHostAlloc,
    cuMemFreeHost: FnCuMemFreeHost,
    cuStreamSynchronize: FnCuStreamSynchronize,
    cuStreamBeginCapture: FnCuStreamBeginCapture,
    cuStreamEndCapture: FnCuStreamEndCapture,
    cuGraphInstantiateWithFlags: FnCuGraphInstantiateWithFlags,
    cuGraphLaunch: FnCuGraphLaunch,
    cuGraphDestroy: FnCuGraphDestroy,
    cuGraphExecDestroy: FnCuGraphExecDestroy,
    cuGraphCreate: FnCuGraphCreate,
    cuGraphAddChildGraphNode: FnCuGraphAddChildGraphNode,
    cuGraphConditionalHandleCreate: FnCuGraphConditionalHandleCreate,
    cuGraphAddNode: FnCuGraphAddNode,
    cuProfilerStart: FnCuProfilerStart,
    cuProfilerStop: FnCuProfilerStop,
    cuEventCreate: FnCuEventCreate,
    cuEventRecord: FnCuEventRecord,
    cuEventSynchronize: FnCuEventSynchronize,
    cuModuleGetFunction: FnCuModuleGetFunction,
    cuLaunchKernel: FnCuLaunchKernel,
    cublasGemmEx: FnCublasGemmEx,
    cublasGemmStridedBatchedEx: FnCublasGemmStridedBatchedEx,
}

// Single-threaded engine guarded by Mutex; explicit cuCtxSetCurrent ensures thread-safety across tokio threads.
unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

impl CudaRuntime {
    pub fn new(ptx_bytes: &[u8]) -> Result<Self> {
        unsafe {
            let cuda_lib = Library::new("nvcuda.dll").context("load nvcuda.dll")?;
            let cublas_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cublas64_12.dll",
                "cublas64_12.dll",
            ];
            let mut cublas_lib = None;
            for p in cublas_paths {
                if let Ok(lib) = Library::new(p) {
                    cublas_lib = Some(lib);
                    break;
                }
            }
            let cublas_lib = cublas_lib.context("load cublas64_12.dll")?;

            let cuInit: FnCuInit = *cuda_lib.get(b"cuInit\0")?;
            let cuDeviceGet: FnCuDeviceGet = *cuda_lib.get(b"cuDeviceGet\0")?;
            let cuCtxCreate: FnCuCtxCreate = *cuda_lib.get(b"cuCtxCreate_v2\0")?;
            let cuCtxSetCurrent: FnCuCtxSetCurrent = *cuda_lib.get(b"cuCtxSetCurrent\0")?;
            let cuMemGetInfo: FnCuMemGetInfo = *cuda_lib.get(b"cuMemGetInfo_v2\0")?;
            let cuMemAlloc: FnCuMemAlloc = *cuda_lib.get(b"cuMemAlloc_v2\0")?;
            let cuMemFree: FnCuMemFree = *cuda_lib.get(b"cuMemFree_v2\0")?;
            let cuMemsetD8Async: FnCuMemsetD8Async = *cuda_lib.get(b"cuMemsetD8Async\0")?;
            let cuMemcpyHtoD: FnCuMemcpyHtoD = *cuda_lib.get(b"cuMemcpyHtoD_v2\0")?;
            let cuMemcpyHtoDAsync: FnCuMemcpyHtoDAsync = *cuda_lib.get(b"cuMemcpyHtoDAsync_v2\0")?;
            let cuMemcpyDtoH: FnCuMemcpyDtoH = *cuda_lib.get(b"cuMemcpyDtoH_v2\0")?;
            let cuMemcpyDtoHAsync: FnCuMemcpyDtoHAsync = *cuda_lib.get(b"cuMemcpyDtoHAsync_v2\0")?;
            let cuMemcpyDtoDAsync: FnCuMemcpyDtoDAsync = *cuda_lib.get(b"cuMemcpyDtoDAsync_v2\0")?;
            let cuMemHostAlloc: FnCuMemHostAlloc = *cuda_lib.get(b"cuMemHostAlloc\0")?;
            let cuMemFreeHost: FnCuMemFreeHost = *cuda_lib.get(b"cuMemFreeHost\0")?;
            let cuStreamCreate: FnCuStreamCreate = *cuda_lib.get(b"cuStreamCreate\0")?;
            let cuStreamSynchronize: FnCuStreamSynchronize = *cuda_lib.get(b"cuStreamSynchronize\0")?;
            let cuStreamBeginCapture: FnCuStreamBeginCapture = *cuda_lib.get(b"cuStreamBeginCapture_v2\0")?;
            let cuStreamEndCapture: FnCuStreamEndCapture = *cuda_lib.get(b"cuStreamEndCapture\0")?;
            let cuGraphInstantiateWithFlags: FnCuGraphInstantiateWithFlags =
                *cuda_lib.get(b"cuGraphInstantiateWithFlags\0")?;
            let cuGraphLaunch: FnCuGraphLaunch = *cuda_lib.get(b"cuGraphLaunch\0")?;
            let cuGraphDestroy: FnCuGraphDestroy = *cuda_lib.get(b"cuGraphDestroy\0")?;
            let cuGraphExecDestroy: FnCuGraphExecDestroy = *cuda_lib.get(b"cuGraphExecDestroy\0")?;
            let cuGraphCreate: FnCuGraphCreate = *cuda_lib.get(b"cuGraphCreate\0")?;
            let cuGraphAddChildGraphNode: FnCuGraphAddChildGraphNode =
                *cuda_lib.get(b"cuGraphAddChildGraphNode\0")?;
            let cuGraphConditionalHandleCreate: FnCuGraphConditionalHandleCreate =
                *cuda_lib.get(b"cuGraphConditionalHandleCreate\0")?;
            let cuGraphAddNode: FnCuGraphAddNode = *cuda_lib.get(b"cuGraphAddNode\0")?;
            let cuProfilerStart: FnCuProfilerStart = *cuda_lib.get(b"cuProfilerStart\0")?;
            let cuProfilerStop: FnCuProfilerStop = *cuda_lib.get(b"cuProfilerStop\0")?;
            let cuEventCreate: FnCuEventCreate = *cuda_lib.get(b"cuEventCreate\0")?;
            let cuEventRecord: FnCuEventRecord = *cuda_lib.get(b"cuEventRecord\0")?;
            let cuEventSynchronize: FnCuEventSynchronize = *cuda_lib.get(b"cuEventSynchronize\0")?;
            let cuModuleLoadData: FnCuModuleLoadData = *cuda_lib.get(b"cuModuleLoadData\0")?;
            let cuModuleGetFunction: FnCuModuleGetFunction = *cuda_lib.get(b"cuModuleGetFunction\0")?;
            let cuLaunchKernel: FnCuLaunchKernel = *cuda_lib.get(b"cuLaunchKernel\0")?;

            let cublasCreate: FnCublasCreate = *cublas_lib.get(b"cublasCreate_v2\0")?;
            let cublasSetStream: FnCublasSetStream = *cublas_lib.get(b"cublasSetStream_v2\0")?;
            let cublasGemmEx: FnCublasGemmEx = *cublas_lib.get(b"cublasGemmEx\0")?;
            let cublasGemmStridedBatchedEx: FnCublasGemmStridedBatchedEx =
                *cublas_lib.get(b"cublasGemmStridedBatchedEx\0")?;

            check_cu(cuInit(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            check_cu(cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
            let mut ctx: CUcontext = ptr::null_mut();
            check_cu(cuCtxCreate(&mut ctx, 0, dev), "cuCtxCreate")?;

            let mut stream: CUstream = ptr::null_mut();
            check_cu(cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate")?;

            let mut cublas: CublasHandle = ptr::null_mut();
            check_cublas(cublasCreate(&mut cublas), "cublasCreate")?;
            check_cublas(cublasSetStream(cublas, stream), "cublasSetStream")?;

            let mut module: CUmodule = ptr::null_mut();
            if ptx_bytes.starts_with(b"\x7fELF") {
                // Native CUBIN (SASS): load directly with ZERO runtime PTX->SASS JIT compilation!
                check_cu(
                    cuModuleLoadData(&mut module, ptx_bytes.as_ptr() as *const c_void),
                    "cuModuleLoadData(cubin)",
                )?;
            } else {
                let mut ptx_null = Vec::with_capacity(ptx_bytes.len() + 1);
                ptx_null.extend_from_slice(ptx_bytes);
                ptx_null.push(0);
                check_cu(
                    cuModuleLoadData(&mut module, ptx_null.as_ptr() as *const c_void),
                    "cuModuleLoadData(ptx)",
                )?;
            }

            Ok(Self {
                _cuda_lib: cuda_lib,
                _cublas_lib: cublas_lib,
                ctx,
                stream,
                cublas,
                module,
                cuCtxSetCurrent,
                cuMemGetInfo,
                cuMemAlloc,
                cuMemFree,
                cuMemsetD8Async,
                cuMemcpyHtoD,
                cuMemcpyHtoDAsync,
                cuMemcpyDtoH,
                cuMemcpyDtoHAsync,
                cuMemcpyDtoDAsync,
                cuMemHostAlloc,
                cuMemFreeHost,
                cuStreamSynchronize,
                cuStreamBeginCapture,
                cuStreamEndCapture,
                cuGraphInstantiateWithFlags,
                cuGraphLaunch,
                cuGraphDestroy,
                cuGraphExecDestroy,
                cuGraphCreate,
                cuGraphAddChildGraphNode,
                cuGraphConditionalHandleCreate,
                cuGraphAddNode,
                cuProfilerStart,
                cuProfilerStop,
                cuEventCreate,
                cuEventRecord,
                cuEventSynchronize,
                cuModuleGetFunction,
                cuLaunchKernel,
                cublasGemmEx,
                cublasGemmStridedBatchedEx,
            })
        }
    }

    pub fn bind_context(&self) -> Result<()> {
        unsafe { check_cu((self.cuCtxSetCurrent)(self.ctx), "cuCtxSetCurrent") }
    }

    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.bind_context()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe { check_cu((self.cuMemGetInfo)(&mut free, &mut total), "cuMemGetInfo")? };
        Ok((free, total))
    }

    pub fn alloc(&self, bytes: usize) -> Result<CUdeviceptr> {
        let mut ptr: CUdeviceptr = 0;
        unsafe { check_cu((self.cuMemAlloc)(&mut ptr, bytes.max(16)), "cuMemAlloc")? };
        Ok(ptr)
    }

    pub fn free(&self, ptr: CUdeviceptr) {
        if ptr != 0 {
            unsafe { let _ = (self.cuMemFree)(ptr); }
        }
    }

    pub fn memset_zero_async(&self, dst: CUdeviceptr, bytes: usize) -> Result<()> {
        unsafe { check_cu((self.cuMemsetD8Async)(dst, 0, bytes, self.stream), "cuMemsetD8Async") }
    }

    pub fn htod_sync(&self, dst: CUdeviceptr, src: *const u8, bytes: usize) -> Result<()> {
        unsafe { check_cu((self.cuMemcpyHtoD)(dst, src as *const c_void, bytes), "cuMemcpyHtoD") }
    }

    pub fn htod_async(&self, dst: CUdeviceptr, src: *const u8, bytes: usize) -> Result<()> {
        unsafe {
            check_cu(
                (self.cuMemcpyHtoDAsync)(dst, src as *const c_void, bytes, self.stream),
                "cuMemcpyHtoDAsync",
            )
        }
    }

    pub fn dtoh_sync(&self, dst: *mut u8, src: CUdeviceptr, bytes: usize) -> Result<()> {
        unsafe { check_cu((self.cuMemcpyDtoH)(dst as *mut c_void, src, bytes), "cuMemcpyDtoH") }
    }

    pub fn dtoh_async(&self, dst: *mut u8, src: CUdeviceptr, bytes: usize) -> Result<()> {
        unsafe {
            check_cu(
                (self.cuMemcpyDtoHAsync)(dst as *mut c_void, src, bytes, self.stream),
                "cuMemcpyDtoHAsync",
            )
        }
    }

    pub fn dtod_async(&self, dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> Result<()> {
        unsafe { check_cu((self.cuMemcpyDtoDAsync)(dst, src, bytes, self.stream), "cuMemcpyDtoDAsync") }
    }

    pub fn alloc_pinned(&self, bytes: usize) -> Result<*mut u8> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe { check_cu((self.cuMemHostAlloc)(&mut ptr, bytes, 0), "cuMemHostAlloc")? };
        Ok(ptr as *mut u8)
    }

    pub fn free_pinned(&self, ptr: *mut u8) {
        if !ptr.is_null() {
            unsafe { let _ = (self.cuMemFreeHost)(ptr as *mut c_void); }
        }
    }

    pub fn free_pinned_fn(&self) -> unsafe extern "C" fn(*mut c_void) -> CUresult {
        self.cuMemFreeHost
    }

    pub fn sync(&self) -> Result<()> {
        unsafe { check_cu((self.cuStreamSynchronize)(self.stream), "cuStreamSynchronize") }
    }

    pub fn create_event(&self) -> Result<CUevent> {
        let mut ev: CUevent = ptr::null_mut();
        unsafe { check_cu((self.cuEventCreate)(&mut ev, CU_EVENT_DISABLE_TIMING), "cuEventCreate")? };
        Ok(ev)
    }

    pub fn record_event(&self, ev: CUevent) -> Result<()> {
        unsafe { check_cu((self.cuEventRecord)(ev, self.stream), "cuEventRecord") }
    }

    pub fn sync_event(&self, ev: CUevent) -> Result<()> {
        unsafe { check_cu((self.cuEventSynchronize)(ev), "cuEventSynchronize") }
    }

    pub fn get_func(&self, name: &str) -> Result<CUfunction> {
        let cname = CString::new(name)?;
        let mut func: CUfunction = ptr::null_mut();
        unsafe {
            check_cu(
                (self.cuModuleGetFunction)(&mut func, self.module, cname.as_ptr()),
                name,
            )?;
        }
        Ok(func)
    }

    pub fn launch(
        &self,
        func: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &mut [*mut c_void],
    ) -> Result<()> {
        unsafe {
            check_cu(
                (self.cuLaunchKernel)(
                    func,
                    grid.0, grid.1, grid.2,
                    block.0, block.1, block.2,
                    0,
                    self.stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }

    /// Row-major C[M, N] = A[M, K] @ B[N, K]^T in BF16 (FP32 accumulation, Tensor Cores)
    pub fn gemm_bf16_nt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
    ) -> Result<()> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            check_cublas(
                (self.cublasGemmEx)(
                    self.cublas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    n as c_int,
                    m as c_int,
                    k as c_int,
                    &alpha as *const f32 as *const c_void,
                    b as *const c_void,
                    CUDA_R_16BF,
                    k as c_int,
                    a as *const c_void,
                    CUDA_R_16BF,
                    k as c_int,
                    &beta as *const f32 as *const c_void,
                    c as *mut c_void,
                    CUDA_R_16BF,
                    n as c_int,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                ),
                "cublasGemmEx",
            )
        }
    }

    /// Phase R10: Row-major C_i32[M, N] = A_i8[M, K] @ B_i8[N, K]^T on Ampere INT8 Tensor Cores (IMMA, ~175 TOPS!)
    /// Requires M, N, K multiples of 4 and 16-byte aligned pointers.
    pub fn gemm_i8_nt(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
    ) -> Result<()> {
        let alpha: i32 = 1;
        let beta: i32 = 0;
        unsafe {
            check_cublas(
                (self.cublasGemmEx)(
                    self.cublas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    n as c_int,
                    m as c_int,
                    k as c_int,
                    &alpha as *const i32 as *const c_void,
                    b as *const c_void,
                    CUDA_R_8I,
                    k as c_int,
                    a as *const c_void,
                    CUDA_R_8I,
                    k as c_int,
                    &beta as *const i32 as *const c_void,
                    c as *mut c_void,
                    CUDA_R_32I,
                    n as c_int,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                ),
                "cublasGemmEx(INT8)",
            )
        }
    }

    /// Row-major Strided Batched FP32 GEMM on Ampere TF32 Tensor Cores:
    /// C[b] (M x N) = alpha * op(A[b]) (M x K) @ op(B[b]) (K x N) + beta * C[b]
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32_strided_batched(
        &self,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        a: CUdeviceptr,
        stride_a: i64,
        b: CUdeviceptr,
        stride_b: i64,
        beta: f32,
        c: CUdeviceptr,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let ldb = if trans_b { k } else { n };
        let lda = if trans_a { m } else { k };
        let ldc = n;
        unsafe {
            check_cublas(
                (self.cublasGemmStridedBatchedEx)(
                    self.cublas,
                    op_b,
                    op_a,
                    n as c_int,
                    m as c_int,
                    k as c_int,
                    &alpha as *const f32 as *const c_void,
                    b as *const c_void,
                    CUDA_R_32F,
                    ldb as c_int,
                    stride_b,
                    a as *const c_void,
                    CUDA_R_32F,
                    lda as c_int,
                    stride_a,
                    &beta as *const f32 as *const c_void,
                    c as *mut c_void,
                    CUDA_R_32F,
                    ldc as c_int,
                    stride_c,
                    batch_count as c_int,
                    CUBLAS_COMPUTE_32F_FAST_TF32,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                ),
                "cublasGemmStridedBatchedEx",
            )
        }
    }

    /// Row-major Strided Batched BF16 GEMM on Ampere BF16 Tensor Cores (FP32 accumulation):
    /// C[b] (M x N) = alpha * op(A[b]) (M x K) @ op(B[b]) (K x N) + beta * C[b]
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16_strided_batched(
        &self,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        a: CUdeviceptr,
        stride_a: i64,
        b: CUdeviceptr,
        stride_b: i64,
        beta: f32,
        c: CUdeviceptr,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let ldb = if trans_b { k } else { n };
        let lda = if trans_a { m } else { k };
        let ldc = n;
        unsafe {
            check_cublas(
                (self.cublasGemmStridedBatchedEx)(
                    self.cublas,
                    op_b,
                    op_a,
                    n as c_int,
                    m as c_int,
                    k as c_int,
                    &alpha as *const f32 as *const c_void,
                    b as *const c_void,
                    CUDA_R_16BF,
                    ldb as c_int,
                    stride_b,
                    a as *const c_void,
                    CUDA_R_16BF,
                    lda as c_int,
                    stride_a,
                    &beta as *const f32 as *const c_void,
                    c as *mut c_void,
                    CUDA_R_16BF,
                    ldc as c_int,
                    stride_c,
                    batch_count as c_int,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                ),
                "cublasGemmStridedBatchedEx(bf16)",
            )
        }
    }

    pub fn begin_capture(&self) -> Result<()> {
        unsafe {
            check_cu(
                (self.cuStreamBeginCapture)(self.stream, CU_STREAM_CAPTURE_MODE_GLOBAL),
                "cuStreamBeginCapture",
            )
        }
    }

    pub fn end_capture_and_instantiate(&self) -> Result<CUgraphExec> {
        let mut graph: CUgraph = ptr::null_mut();
        let mut exec: CUgraphExec = ptr::null_mut();
        unsafe {
            check_cu((self.cuStreamEndCapture)(self.stream, &mut graph), "cuStreamEndCapture")?;
            check_cu(
                (self.cuGraphInstantiateWithFlags)(&mut exec, graph, 0),
                "cuGraphInstantiateWithFlags",
            )?;
            let _ = (self.cuGraphDestroy)(graph);
        }
        Ok(exec)
    }

    pub fn end_capture_graph(&self) -> Result<CUgraph> {
        let mut graph: CUgraph = ptr::null_mut();
        unsafe {
            check_cu((self.cuStreamEndCapture)(self.stream, &mut graph), "cuStreamEndCapture")?;
        }
        Ok(graph)
    }

    pub fn create_graph(&self) -> Result<CUgraph> {
        let mut graph: CUgraph = ptr::null_mut();
        unsafe { check_cu((self.cuGraphCreate)(&mut graph, 0), "cuGraphCreate")? };
        Ok(graph)
    }

    pub fn destroy_graph(&self, graph: CUgraph) {
        if !graph.is_null() {
            unsafe { let _ = (self.cuGraphDestroy)(graph); }
        }
    }

    pub fn instantiate_graph(&self, graph: CUgraph) -> Result<CUgraphExec> {
        let mut exec: CUgraphExec = ptr::null_mut();
        unsafe {
            check_cu(
                (self.cuGraphInstantiateWithFlags)(&mut exec, graph, 0),
                "cuGraphInstantiateWithFlags",
            )?;
        }
        Ok(exec)
    }

    pub fn graph_add_child_node(
        &self,
        parent: CUgraph,
        deps: &[CUgraphNode],
        child: CUgraph,
    ) -> Result<CUgraphNode> {
        let mut node: CUgraphNode = ptr::null_mut();
        let dep_ptr = if deps.is_empty() { ptr::null() } else { deps.as_ptr() };
        unsafe {
            check_cu(
                (self.cuGraphAddChildGraphNode)(&mut node, parent, dep_ptr, deps.len(), child),
                "cuGraphAddChildGraphNode",
            )?;
        }
        Ok(node)
    }

    pub fn graph_create_cond_handle(
        &self,
        parent: CUgraph,
        default_val: u32,
    ) -> Result<CUgraphConditionalHandle> {
        let mut handle: CUgraphConditionalHandle = 0;
        unsafe {
            check_cu(
                (self.cuGraphConditionalHandleCreate)(
                    &mut handle,
                    parent,
                    self.ctx,
                    default_val,
                    CU_GRAPH_COND_ASSIGN_DEFAULT,
                ),
                "cuGraphConditionalHandleCreate",
            )?;
        }
        Ok(handle)
    }

    pub fn graph_add_if_node(
        &self,
        parent: CUgraph,
        deps: &[CUgraphNode],
        handle: CUgraphConditionalHandle,
    ) -> Result<(CUgraphNode, CUgraph)> {
        debug_assert_eq!(std::mem::size_of::<CUgraphNodeParamsCond>(), 256);
        let mut node: CUgraphNode = ptr::null_mut();
        let mut params = CUgraphNodeParamsCond {
            node_type: CU_GRAPH_NODE_TYPE_CONDITIONAL,
            reserved0: [0; 3],
            conditional: CUDA_CONDITIONAL_NODE_PARAMS {
                handle,
                cond_type: CU_GRAPH_COND_TYPE_IF,
                size: 1,
                phGraph_out: ptr::null_mut(),
                ctx: self.ctx,
            },
            pad: [0u64; 26],
        };
        let dep_ptr = if deps.is_empty() { ptr::null() } else { deps.as_ptr() };
        unsafe {
            check_cu(
                (self.cuGraphAddNode)(&mut node, parent, dep_ptr, deps.len(), &mut params),
                "cuGraphAddNode(CONDITIONAL_IF)",
            )?;
            let body_graph = *params.conditional.phGraph_out;
            Ok((node, body_graph))
        }
    }

    pub fn launch_graph(&self, exec: CUgraphExec) -> Result<()> {
        unsafe { check_cu((self.cuGraphLaunch)(exec, self.stream), "cuGraphLaunch") }
    }

    pub fn profiler_start(&self) -> Result<()> {
        unsafe { check_cu((self.cuProfilerStart)(), "cuProfilerStart") }
    }

    pub fn profiler_stop(&self) -> Result<()> {
        unsafe { check_cu((self.cuProfilerStop)(), "cuProfilerStop") }
    }
}

