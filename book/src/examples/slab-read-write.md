# Slab Read/Write

Demonstrates `wgsl-rs` macros for reading from and writing to u32 "slabs". The slab can be any indexable item such as an array, `RuntimeArray`, or storage pointer. The `slab_read_array!` and `slab_write_array!` macros expand to element-wise copy loops in WGSL.

## Rust Source

```rust
#[wgsl]
pub mod slab_read_write {
    //! `wgsl-rs` includes macros for reading to and from u32 "slabs".
    //!
    //! The slab can be any indexable item such as an array, RuntimeArray,
    //! storage pointer, etc.

    use wgsl_rs::std::*;

    pub struct Data {
        pub one: f32,
        pub two: u32,
        pub three_four: Vec2f,
    }

    impl Data {
        /// `Data`'s slab size.
        ///
        /// This is the number of u32 slots it occupies in a u32 slab.
        pub const SLAB_SIZE: usize = 4;

        /// Returns an array container to hold ephemeral data read from a slab.
        pub fn array_container() -> [u32; Self::SLAB_SIZE] {
            [0, 0, 0, 0]
        }

        /// Convert an array into `Data`.
        pub fn from_array(arr: [u32; Self::SLAB_SIZE]) -> Self {
            Self {
                one: bitcast_f32(arr[0]),
                two: arr[1],
                three_four: vec2f(bitcast_f32(arr[2]), bitcast_f32(arr[3])),
            }
        }

        /// Convert `Data` into an array.
        pub fn to_array(data: Self) -> [u32; Self::SLAB_SIZE] {
            [
                bitcast_u32(data.one),
                data.two,
                bitcast_u32(data.three_four.x()),
                bitcast_u32(data.three_four.y()),
            ]
        }
    }

    storage!(group(0), binding(0), read_write, SLAB: RuntimeArray<u32>);

    #[compute]
    #[workgroup_size(8)]
    pub fn slab_example(#[builtin(local_invocation_index)] local_idx: u32) {
        let index = local_idx;

        // Create our `Data` struct from extracted data from the slab
        let mut data: Data;
        {
            // Extract the u32 data from the slab
            let mut array_data = Data::array_container();
            slab_read_array!(get!(SLAB), index, array_data, Data::SLAB_SIZE);
            data = Data::from_array(array_data);
        }

        // Modify it
        data.three_four.x = 123.0;

        // Write the modified `Data` struct back to the slab
        let out_array = Data::to_array(data);
        slab_write_array!(get_mut!(SLAB), index, out_array, Data::SLAB_SIZE);
    }
}
```

## Generated WGSL

```wgsl
struct Data {
    one: f32,
    two: u32,
    three_four: vec2f
}
const Data__1SLAB_SIZE: u32 = 4;

fn Data__1array_container() -> array<u32, Data__1SLAB_SIZE> {
    return array(0, 0, 0, 0);
}

fn Data__1from_array(arr: array<u32, Data__1SLAB_SIZE>) -> Data {
    return Data(bitcast<f32>(arr[0]), arr[1], vec2f(bitcast<f32>(arr[2]), bitcast<f32>(arr[3])));
}

fn Data__1to_array(data: Data) -> array<u32, Data__1SLAB_SIZE> {
    return array(bitcast<u32>(data.one), data.two, bitcast<u32>(data.three_four.x), bitcast<u32>(data.three_four.y));
}
@group(0) @binding(0) var<storage, read_write> SLAB: array<u32>;

@compute @workgroup_size(8) fn slab_example(@builtin(local_invocation_index) local_idx: u32) {
    let index = local_idx;
    var data: Data;
    {
        var array_data = Data__1array_container();
        for (var _i: u32 = 0u; _i < Data__1SLAB_SIZE; _i++) {
            array_data[_i] = SLAB[index + _i];
        }
        data = Data__1from_array(array_data);
    }
    data.three_four.x = 123.0;
    let out_array = Data__1to_array(data);
    for (var _i: u32 = 0u; _i < Data__1SLAB_SIZE; _i++) {
        SLAB[index + _i] = out_array[_i];
    }
}
```

## Notes

- `slab_read_array!(slab, index, dest, size)` expands to a for-loop copying `size` u32 elements from `slab[index]` onward into `dest`.
- `slab_write_array!(slab, index, src, size)` does the reverse.
- `Self::SLAB_SIZE` (an associated const) is mangled to `Data__1SLAB_SIZE` in WGSL.