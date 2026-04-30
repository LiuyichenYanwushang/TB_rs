RUSTFLAGS="-C target-cpu=native" cargo build --release
scp -r src/ Cargo* liuyichen@10.168.66.7:/public/home/liuyichen/TB_rs/TB_rs/
scp -r src/ Cargo* sfqian@10.168.66.7:/public/home/sfqian/TB_rs/TB_rs/
cd ../Rustb/
scp -r src/ Cargo* liuyichen@10.168.66.7:/public/home/liuyichen/TB_rs/Rustb/
scp -r src/ Cargo* sfqian@10.168.66.7:/public/home/sfqian/TB_rs/Rustb/

cp ../target/release/TB_rs ~/.cargo/bin/
