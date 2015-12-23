use ocl::{ Context, ProQueue, BuildOptions, SimpleDims, Envoy };
extern crate ocl;

const PRINT_DEBUG: bool = true;

fn main() {
    // Set our data set size and coefficent to arbitrary values:
    let data_set_size = 900000;
    let coeff = 5432.1;

    // Create a context with the default platform and device type (GPU):
    // * Use: `Context::new(None, Some(ocl::CL_DEVICE_TYPE_CPU))` for CPU.
    let ocl_cxt = Context::new(None, None).unwrap();

    // Create a program/queue with the first available device: 
    let mut ocl_pq = ProQue::new(&ocl_cxt, None);

    // Create a basic build configuration:
    let build_config = BuildConfig::new().kern_file("cl/kernel_file.cl");

    // Build with our configuration and check for errors:
    ocl_pq.build(build_config).expect("ocl program build");

    // Set up our work dimensions / data set size:
    let our_test_dims = SimpleDims::OneDim(data_set_size);

    // Create an envoy (an array + an OpenCL buffer) as a data source:
    let source_envoy = Envoy::scrambled(&our_test_dims, 0.0f32, 200.0, &ocl_pq.queue());

    // Create another empty one for results:
    let mut result_envoy = Envoy::new(&our_test_dims, 0.0f32, &ocl_pq.queue());

    // Create kernel:
    let kernel = ocl_pq.create_kernel("multiply_by_scalar", our_test_dims.work_size())
        .arg_env(&source_envoy)
        .arg_scl(coeff)
        .arg_env(&mut result_envoy)
    ;

    // Enqueue kernel depending on and creating no events:
    kernel.enqueue(None, None);

    // Read results:
    result_envoy.read_wait();

    // Check results and print the first 20:
    for idx in 0..data_set_size {
        // Check:
        assert_eq!(result_envoy[idx], source_envoy[idx] * coeff);

        // Print:
        if PRINT_DEBUG && (idx < 20) { 
            println!("source_envoy[idx]: {}, coeff: {}, result_envoy[idx]: {}",
            source_envoy[idx], coeff, result_envoy[idx]); 
        }
    }
}