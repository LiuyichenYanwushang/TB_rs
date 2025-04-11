use bincode::{deserialize, serialize};
use mpi::request::WaitGuard;
use mpi::traits::*;
use std::fs::File;
use std::io::{Read, Write};
use zstd::stream::{decode_all, encode_all};
use Rustb::*;

const CHUNK_SIZE: usize = 256 * 1024 * 1024;
pub struct TB_file {
    pub seed_name: Option<String>,
    pub fermi_energy: f64,
}

impl TB_file {
    pub fn new() -> Self {
        TB_file {
            seed_name: None,
            fermi_energy: 0.0,
        }
    }
}

pub fn read_model(
    world: &impl AnyCommunicator,
    seed_name: TB_file,
    output_file: &mut Option<File>,
) -> Model {
    let size = world.size();
    let rank = world.rank();
    let model: Model;
    if rank == 0 {
        model = if seed_name.seed_name == None {
            writeln!(
                output_file.as_mut().unwrap(),
                "The seed_name is not specified"
            );
            panic!("The seed_name is not specified")
        } else {
            //开始读取数据
            writeln!(output_file.as_mut().unwrap(), "Loading the model...");
            let model = Model::from_hr("./", &seed_name.seed_name.unwrap(), seed_name.fermi_energy);
            writeln!(
                output_file.as_mut().unwrap(),
                "The tight binding model is loaded"
            );
            model
        };
        //数据读取完成, 开始向所有线程广播model
        let serialized = serialize(&model).unwrap();
        let mut compressed = Vec::new();
        let mut encoder = zstd::stream::Encoder::new(&mut compressed, 6).unwrap();
        for chunk in serialized.chunks(CHUNK_SIZE) {
            encoder.write_all(chunk).unwrap();
        }
        encoder.finish();

        // 3. 分块发送
        let mut num_chunks = (compressed.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
        world.process_at_rank(0).broadcast_into(&mut num_chunks); // 广播块数量

        for chunk in compressed.chunks(CHUNK_SIZE) {
            for dest_rank in 1..world.size() {
                world.process_at_rank(dest_rank).send(chunk);
            }
        }
    } else {
        // 1. 接收块数量
        let mut num_chunks: usize = 0;
        world.process_at_rank(0).broadcast_into(&mut num_chunks); // 广播块数量

        // 2. 分块接收
        let mut compressed = Vec::with_capacity(num_chunks * CHUNK_SIZE);
        for _ in 0..num_chunks {
            let (chunk, _) = world.process_at_rank(0).receive_vec();
            compressed.extend_from_slice(&chunk);
        }

        // 3. 流式解压
        let mut decoder = zstd::stream::Decoder::new(&compressed[..]).unwrap();
        let mut serialized = Vec::new();
        decoder.read_to_end(&mut serialized).unwrap();

        // 4. 反序列化
        model = deserialize(&serialized).unwrap();
        println!("The current process {} receives the model completed", rank);
    }
    model
}
