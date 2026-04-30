#cargo build  -r
#RUST_BACKTRACE=1 mpirun -np 8 ../../target/release/TB_rs
RUSTFLAGS="-C target-cpu=native" cargo build --release 
cp ../../target/release/TB_rs ~/.cargo/bin/
mpirun -np 8 ../../target/release/TB_rs
