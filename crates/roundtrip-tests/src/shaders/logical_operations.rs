//! Roundtrip tests for logical builtin functions.
//!
//! Tests: `all`, `any` on bool vectors
//!
//! Covers Vec2b, Vec3b, Vec4b types.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// all_vec2b: Test all() on `Vec2<bool>`
///
/// Returns true only when ALL components are true
#[wgsl]
pub mod all_vec2b {
    use wgsl_rs::std::*;
    pub struct AllVec2bPack {
        #[storage(group(0), binding(0))]
        pub input: Storage<[u32; 128]>,
        #[storage(group(0), binding(1), read_write)]
        pub output: Storage<[u32; 64]>,
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u, pack: &AllVec2bPack) {
        let idx = global_id.x as usize;
        let input = get!(pack.input);

        let base = idx * 2;
        let vec_input = vec2b(input[base] != 0u32, input[base + 1] != 0u32);

        let result = all(vec_input);

        if result {
            get_mut!(pack.output)[idx] = 1u32;
        } else {
            get_mut!(pack.output)[idx] = 0u32;
        }
    }
}

/// all_vec3b: Test all() on `Vec3<bool>`
///
/// Returns true only when ALL components are true
#[wgsl]
pub mod all_vec3b {
    use wgsl_rs::std::*;
    pub struct AllVec3bPack {
        #[storage(group(0), binding(0))]
        pub input: Storage<[u32; 192]>,
        #[storage(group(0), binding(1), read_write)]
        pub output: Storage<[u32; 64]>,
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u, pack: &AllVec3bPack) {
        let idx = global_id.x as usize;
        let input = get!(pack.input);

        let base = idx * 3;
        let vec_input = vec3b(
            input[base] != 0u32,
            input[base + 1] != 0u32,
            input[base + 2] != 0u32,
        );

        let result = all(vec_input);

        if result {
            get_mut!(pack.output)[idx] = 1u32;
        } else {
            get_mut!(pack.output)[idx] = 0u32;
        }
    }
}

/// all_vec4b: Test all() on `Vec4<bool>`
///
/// Returns true only when ALL components are true
#[wgsl]
pub mod all_vec4b {
    use wgsl_rs::std::*;
    pub struct AllVec4bPack {
        #[storage(group(0), binding(0))]
        pub input: Storage<[u32; 256]>,
        #[storage(group(0), binding(1), read_write)]
        pub output: Storage<[u32; 64]>,
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u, pack: &AllVec4bPack) {
        let idx = global_id.x as usize;
        let input = get!(pack.input);

        let base = idx * 4;
        let vec_input = vec4b(
            input[base] != 0u32,
            input[base + 1] != 0u32,
            input[base + 2] != 0u32,
            input[base + 3] != 0u32,
        );

        let result = all(vec_input);

        if result {
            get_mut!(pack.output)[idx] = 1u32;
        } else {
            get_mut!(pack.output)[idx] = 0u32;
        }
    }
}

/// any_vec2b: Test any() on `Vec2<bool>`
///
/// Returns true if ANY component is true
#[wgsl]
pub mod any_vec2b {
    use wgsl_rs::std::*;
    pub struct AnyVec2bPack {
        #[storage(group(0), binding(0))]
        pub input: Storage<[u32; 128]>,
        #[storage(group(0), binding(1), read_write)]
        pub output: Storage<[u32; 64]>,
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u, pack: &AnyVec2bPack) {
        let idx = global_id.x as usize;
        let input = get!(pack.input);

        let base = idx * 2;
        let vec_input = vec2b(input[base] != 0u32, input[base + 1] != 0u32);

        let result = any(vec_input);

        if result {
            get_mut!(pack.output)[idx] = 1u32;
        } else {
            get_mut!(pack.output)[idx] = 0u32;
        }
    }
}

/// any_vec3b: Test any() on `Vec3<bool>`
///
/// Returns true if ANY component is true
#[wgsl]
pub mod any_vec3b {
    use wgsl_rs::std::*;
    pub struct AnyVec3bPack {
        #[storage(group(0), binding(0))]
        pub input: Storage<[u32; 192]>,
        #[storage(group(0), binding(1), read_write)]
        pub output: Storage<[u32; 64]>,
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u, pack: &AnyVec3bPack) {
        let idx = global_id.x as usize;
        let input = get!(pack.input);

        let base = idx * 3;
        let vec_input = vec3b(
            input[base] != 0u32,
            input[base + 1] != 0u32,
            input[base + 2] != 0u32,
        );

        let result = any(vec_input);

        if result {
            get_mut!(pack.output)[idx] = 1u32;
        } else {
            get_mut!(pack.output)[idx] = 0u32;
        }
    }
}

/// any_vec4b: Test any() on `Vec4<bool>`
///
/// Returns true if ANY component is true
#[wgsl]
pub mod any_vec4b {
    use wgsl_rs::std::*;
    pub struct AnyVec4bPack {
        #[storage(group(0), binding(0))]
        pub input: Storage<[u32; 256]>,
        #[storage(group(0), binding(1), read_write)]
        pub output: Storage<[u32; 64]>,
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u, pack: &AnyVec4bPack) {
        let idx = global_id.x as usize;
        let input = get!(pack.input);

        let base = idx * 4;
        let vec_input = vec4b(
            input[base] != 0u32,
            input[base + 1] != 0u32,
            input[base + 2] != 0u32,
            input[base + 3] != 0u32,
        );

        let result = any(vec_input);

        if result {
            get_mut!(pack.output)[idx] = 1u32;
        } else {
            get_mut!(pack.output)[idx] = 0u32;
        }
    }
}

// ============================================================================
// Input Generators
// ============================================================================

fn all_any_vec2b_inputs() -> [u32; N * 2] {
    let mut inputs = [0u32; N * 2];
    let patterns = [(0, 0), (1, 1), (1, 0), (0, 1)];
    for i in 0..N {
        let (a, b) = patterns[i % 4];
        inputs[i * 2] = a;
        inputs[i * 2 + 1] = b;
    }
    inputs
}

fn all_any_vec3b_inputs() -> [u32; N * 3] {
    let mut inputs = [0u32; N * 3];
    let patterns = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0)];
    for i in 0..N {
        let (a, b, c) = patterns[i % 4];
        inputs[i * 3] = a;
        inputs[i * 3 + 1] = b;
        inputs[i * 3 + 2] = c;
    }
    inputs
}

fn all_any_vec4b_inputs() -> [u32; N * 4] {
    let mut inputs = [0u32; N * 4];
    let patterns = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 0, 0, 0), (1, 1, 0, 0)];
    for i in 0..N {
        let (a, b, c, d) = patterns[i % 4];
        inputs[i * 4] = a;
        inputs[i * 4 + 1] = b;
        inputs[i * 4 + 2] = c;
        inputs[i * 4 + 3] = d;
    }
    inputs
}

// ============================================================================
// Test Implementation
// ============================================================================

pub struct LogicalOperationsTest;

impl RoundtripTest for LogicalOperationsTest {
    fn name(&self) -> &str {
        "logical_operations"
    }

    fn description(&self) -> &str {
        "all, any"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();

        // all_vec2b test
        {
            let inputs = all_any_vec2b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&all_vec2b::WGSL_SOURCE).unwrap();
            let gpu_output =
                harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                    device,
                    queue,
                    linkage: &mut linkage,
                    entry: "main",
                    input_data: input_bytes,
                    output_size: (N * std::mem::size_of::<u32>()) as u64,
                    workgroup_count: (1, 1, 1),
                });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);

            let pack = all_vec2b::AllVec2bPack {
                input: Storage::new(inputs),
                output: Storage::new([0u32; N]),
            };
            dispatch_workgroups(
                (1, 1, 1),
                linkage
                    .compute_entry("main")
                    .expect("main entry present")
                    .workgroup_size,
                |builtins| {
                    all_vec2b::main(builtins.global_invocation_id, &pack);
                },
            );
            let cpu_results: Vec<u32> = pack.output.get().to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("all_vec2b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "all_vec2b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // all_vec3b test
        {
            let inputs = all_any_vec3b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&all_vec3b::WGSL_SOURCE).unwrap();
            let gpu_output =
                harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                    device,
                    queue,
                    linkage: &mut linkage,
                    entry: "main",
                    input_data: input_bytes,
                    output_size: (N * std::mem::size_of::<u32>()) as u64,
                    workgroup_count: (1, 1, 1),
                });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);

            let pack = all_vec3b::AllVec3bPack {
                input: Storage::new(inputs),
                output: Storage::new([0u32; N]),
            };
            dispatch_workgroups(
                (1, 1, 1),
                linkage
                    .compute_entry("main")
                    .expect("main entry present")
                    .workgroup_size,
                |builtins| {
                    all_vec3b::main(builtins.global_invocation_id, &pack);
                },
            );
            let cpu_results: Vec<u32> = pack.output.get().to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("all_vec3b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "all_vec3b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // all_vec4b test
        {
            let inputs = all_any_vec4b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&all_vec4b::WGSL_SOURCE).unwrap();
            let gpu_output =
                harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                    device,
                    queue,
                    linkage: &mut linkage,
                    entry: "main",
                    input_data: input_bytes,
                    output_size: (N * std::mem::size_of::<u32>()) as u64,
                    workgroup_count: (1, 1, 1),
                });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);

            let pack = all_vec4b::AllVec4bPack {
                input: Storage::new(inputs),
                output: Storage::new([0u32; N]),
            };
            dispatch_workgroups(
                (1, 1, 1),
                linkage
                    .compute_entry("main")
                    .expect("main entry present")
                    .workgroup_size,
                |builtins| {
                    all_vec4b::main(builtins.global_invocation_id, &pack);
                },
            );
            let cpu_results: Vec<u32> = pack.output.get().to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("all_vec4b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "all_vec4b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // any_vec2b test
        {
            let inputs = all_any_vec2b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&any_vec2b::WGSL_SOURCE).unwrap();
            let gpu_output =
                harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                    device,
                    queue,
                    linkage: &mut linkage,
                    entry: "main",
                    input_data: input_bytes,
                    output_size: (N * std::mem::size_of::<u32>()) as u64,
                    workgroup_count: (1, 1, 1),
                });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);

            let pack = any_vec2b::AnyVec2bPack {
                input: Storage::new(inputs),
                output: Storage::new([0u32; N]),
            };
            dispatch_workgroups(
                (1, 1, 1),
                linkage
                    .compute_entry("main")
                    .expect("main entry present")
                    .workgroup_size,
                |builtins| {
                    any_vec2b::main(builtins.global_invocation_id, &pack);
                },
            );
            let cpu_results: Vec<u32> = pack.output.get().to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("any_vec2b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "any_vec2b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // any_vec3b test
        {
            let inputs = all_any_vec3b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&any_vec3b::WGSL_SOURCE).unwrap();
            let gpu_output =
                harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                    device,
                    queue,
                    linkage: &mut linkage,
                    entry: "main",
                    input_data: input_bytes,
                    output_size: (N * std::mem::size_of::<u32>()) as u64,
                    workgroup_count: (1, 1, 1),
                });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);

            let pack = any_vec3b::AnyVec3bPack {
                input: Storage::new(inputs),
                output: Storage::new([0u32; N]),
            };
            dispatch_workgroups(
                (1, 1, 1),
                linkage
                    .compute_entry("main")
                    .expect("main entry present")
                    .workgroup_size,
                |builtins| {
                    any_vec3b::main(builtins.global_invocation_id, &pack);
                },
            );
            let cpu_results: Vec<u32> = pack.output.get().to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("any_vec3b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "any_vec3b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        // any_vec4b test
        {
            let inputs = all_any_vec4b_inputs();
            let input_bytes = bytemuck::cast_slice::<u32, u8>(&inputs);

            let mut linkage =
                wgsl_rs_linkage_wgpu::analyze_wgsl_module(&any_vec4b::WGSL_SOURCE).unwrap();
            let gpu_output =
                harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                    device,
                    queue,
                    linkage: &mut linkage,
                    entry: "main",
                    input_data: input_bytes,
                    output_size: (N * std::mem::size_of::<u32>()) as u64,
                    workgroup_count: (1, 1, 1),
                });

            let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);

            let pack = any_vec4b::AnyVec4bPack {
                input: Storage::new(inputs),
                output: Storage::new([0u32; N]),
            };
            dispatch_workgroups(
                (1, 1, 1),
                linkage
                    .compute_entry("main")
                    .expect("main entry present")
                    .workgroup_size,
                |builtins| {
                    any_vec4b::main(builtins.global_invocation_id, &pack);
                },
            );
            let cpu_results: Vec<u32> = pack.output.get().to_vec();

            let labels: Vec<String> = (0..N).map(|i| format!("any_vec4b[{}]", i)).collect();
            let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();

            results.push(harness::compare_u32_results(
                "any_vec4b",
                gpu_results,
                &cpu_results,
                &label_refs,
            ));
        }

        results
    }
}
