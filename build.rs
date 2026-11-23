use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/qwen35.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let cubin_path = out_dir.join("qwen35.cubin");

    let nvcc_candidates = [
        env::var("NVCC_PATH").ok(),
        Some(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe".to_string()),
        Some("nvcc".to_string()),
    ];

    let nvcc = nvcc_candidates
        .into_iter()
        .flatten()
        .find(|p| p == "nvcc" || PathBuf::from(p).exists())
        .expect("nvcc not found");

    // Phase R12: Compile directly to native sm_86 CUBIN (SASS) -> eliminates ~50ms CUDA 12 lazy PTX->SASS JIT stall on first launch!
    let status = Command::new(&nvcc)
        .args([
            "-cubin",
            "-arch=sm_86",
            "-O3",
            "--std=c++17",
            "-o",
            cubin_path.to_str().unwrap(),
            "kernels/qwen35.cu",
        ])
        .status()
        .expect("failed to execute nvcc");

    assert!(status.success(), "nvcc failed to compile kernels/qwen35.cu");
}

