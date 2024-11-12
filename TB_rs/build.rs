// build.rs
use std::env;

fn main() {
    // 设置你想要链接的libmpi.so的路径
    env::set_var(
        "MPI_LIB_DIR",
        "/opt/intel/oneapi/mpi/2021.3.0/lib/release/libmpi.so",
    );
}
