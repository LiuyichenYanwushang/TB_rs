use gnuplot::AutoOption::*;
use gnuplot::AxesCommon;
use gnuplot::Tick::*;
use gnuplot::{Caption, Color, Figure, Font, LineStyle, Solid};
use mpi::request::WaitGuard;
use mpi::traits::*;
use ndarray::*;
use ndarray_linalg::*;
use std::cmp;
///这个程序是类似wanniertools 的代码, 但是实现了一些输运相关的代码, 效率相对更高
///主程序主要用来实现对控制文件的读取, 控制文件默认叫做 TB.in
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use Rustb::*;

#[derive(Debug)]
struct Control {
    band_plot: bool,
    optical_conductivity: bool,
    kerr_angle: bool,
}
impl Control {
    pub fn new() -> Self {
        Control {
            band_plot: false,
            optical_conductivity: false,
            kerr_angle: false,
        }
    }
}
struct TB_file {
    seed_name: Option<String>,
    fermi_energy: f64,
}

impl TB_file {
    pub fn new() -> Self {
        TB_file {
            seed_name: None,
            fermi_energy: 0.0,
        }
    }
}

struct k_path {
    k_path_num: usize,
    k_path: Array3<f64>,
    k_path_name: Vec<[String; 2]>,
}

impl k_path {
    pub fn new() -> Self {
        k_path {
            k_path_num: 101,
            k_path: Array3::zeros((0, 2, 3)),
            k_path_name: vec![],
        }
    }

    pub fn get_k_path(file: &Vec<String>) -> Option<Self> {
        //这个是专门用来得到k_path 的
        let mut Kpath = k_path::new();
        let mut in_section = false;
        for line in file {
            if line.contains("begin kpoint_path") {
                in_section = true;
                continue;
            } else if line.contains("end kpoint_path") {
                break;
            } else if in_section {
                //开始分割字符串
                let mut string = line.trim().split_whitespace();
                let mut name = [String::new(), String::new()];
                let mut points = Array2::zeros((2, 3));
                name[0] = String::from(string.next().unwrap());
                points[[0, 0]] = string.next().unwrap().parse::<f64>().unwrap();
                points[[0, 1]] = string.next().unwrap().parse::<f64>().unwrap();
                points[[0, 2]] = string.next().unwrap().parse::<f64>().unwrap();
                name[1] = String::from(string.next().unwrap());
                points[[1, 0]] = string.next().unwrap().parse::<f64>().unwrap();
                points[[1, 1]] = string.next().unwrap().parse::<f64>().unwrap();
                points[[1, 2]] = string.next().unwrap().parse::<f64>().unwrap();
                Kpath.k_path_name.push(name);
                Kpath.k_path.push(Axis(0), points.view());
            }
        }
        for line in file {
            if line.contains("bands_num_points") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    Kpath.k_path_num = parts[1].trim().parse::<usize>().unwrap();
                }
            }
        }
        if in_section {
            return Some(Kpath);
        } else {
            return None;
        }
    }
    pub fn get_path_vec(
        &self,
        lat: &Array2<f64>,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<String>) {
        //返回(kvec,kdist,knode)
        let lat_t: Array2<f64> = lat.clone().reversed_axes();
        let k_metric = (lat.dot(&lat_t)).inv().unwrap();
        let n_knode = self.k_path.len_of(Axis(0)) + 1;
        let mut k_node = Array1::zeros(n_knode);
        let nk = self.k_path_num;

        for n in 0..n_knode - 1 {
            //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
            let dk: Array1<f64> =
                &self.k_path.slice(s![n, 1, ..]) - &self.k_path.slice(s![n, 0, ..]);
            let a: Array1<f64> = k_metric.dot(&dk);
            let dklen = dk.dot(&a).sqrt();
            k_node[[n + 1]] = k_node[[n]] + dklen;
        }

        println!("n_knode={},k_node={}", n_knode, k_node);
        let mut node_index: Vec<usize> = vec![0];
        for n in 1..n_knode - 1 {
            let frac = k_node[[n]] / k_node[[n_knode - 1]];
            let a = (frac * ((nk - 1) as f64).round()) as usize;
            node_index.push(a)
        }
        node_index.push(nk - 1);
        println!("node_index={:?}", node_index);
        let mut k_dist = Array1::<f64>::zeros(nk);
        let mut k_vec = Array2::<f64>::zeros((nk, 3));
        //k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
        k_vec.row_mut(0).assign(&self.k_path.slice(s![0, 0, ..]));
        for n in 0..n_knode - 1 {
            let n_i = node_index[n];
            let n_f = node_index[n + 1];
            let kd_i = k_node[[n]];
            let kd_f = k_node[[n + 1]];
            let k_i = self.k_path.slice(s![n, 0, ..]);
            let k_f = self.k_path.slice(s![n, 1, ..]);
            println!("n_i,n_f={},{}", n_i, n_f);
            for j in n_i..n_f + 1 {
                let frac: f64 = ((j - n_i) as f64) / ((n_f - n_i) as f64);
                k_dist[[j]] = kd_i + frac * (kd_f - kd_i);
                k_vec
                    .row_mut(j)
                    .assign(&((1.0 - frac) * k_i.to_owned() + frac * k_f.to_owned()));
            }
        }
        //开始对字符串进行操作, 得到能带路径
        let mut node_name = Vec::new();
        node_name.push(self.k_path_name[0][0].clone());
        for i in 1..n_knode - 1 {
            let name: String = if self.k_path_name[i - 1][1] != self.k_path_name[i][0] {
                format!("{}|{}", self.k_path_name[i - 1][1], self.k_path_name[i][0])
            } else {
                self.k_path_name[i - 1][1].clone()
            };
            node_name.push(name);
        }
        node_name.push(self.k_path_name[n_knode - 2][1].clone());
        println!("{:?}", node_name);

        (k_vec, k_dist, k_node, node_name)
    }
}

fn main() {
    //MPI 初始化
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    //在所有的线程上都读取TB.in
    //read the input file
    let path = "TB.in";
    let TB_file = File::open(path).expect(&format!(
        "Unable to open the file {:?}, please check if file is present",
        path
    ));
    let mut output_file = File::create("TB.out").expect("Unable to create TB.in");
    writeln!(output_file,"The copyright belongs to Yichen Liu.\n  This program uses the results of wainner90 to calculate various transport coefficients, as well as various angular states and surface states.\n Author's email: liuyichen@bit.edu.cn");
    //读取 input 结束
    let reader = BufReader::new(TB_file);
    let mut reads: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        reads.push(line.clone());
    }
    //初始化各种控制语句
    let mut seed_name = TB_file::new();
    let mut control = Control::new();

    //开始读取
    for i in reads.iter() {
        if i.contains("seed_name") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 {
                seed_name.seed_name = Some(String::from(parts[1].trim()));
                println!("seed_name: {}", parts[1].trim());
                writeln!(
                    output_file,
                    "The seed_name of tight-binding model is {}",
                    parts[1].trim()
                );
            }
        }
        if i.contains("fermi_energy") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 {
                seed_name.fermi_energy = parts[1].trim().parse::<f64>().unwrap();
                println!("seed_name: {}", parts[1].trim());
                writeln!(
                    output_file,
                    "The fermi energy of tight-binding model is {}",
                    parts[1].trim()
                );
            }
        }
        if i.contains("band_plot") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 {
                if parts[1].contains("T") || parts[1].contains("t") {
                    control.band_plot = true;
                }
            }
        }
    }
    //所有进程读取tight binding 数据
    let model = if seed_name.seed_name == None {
        writeln!(output_file, "The seed_name is not specified");
        panic!("The seed_name is not specified")
    } else {
        //开始读取数据
        writeln!(output_file, "Loading the model...");
        let model = Model::from_hr("./", &seed_name.seed_name.unwrap(), seed_name.fermi_energy);
        writeln!(output_file, "The tight binding model is loaded");
        model
    };
    //读取结束, 开始实现控制语句

    if control.band_plot {
        if rank == 0 {
            //read only on rank0
            writeln!(output_file, "start calculating the band structure");
            //开始读取 k_path 的数据
            let Kpath = k_path::get_k_path(&reads);
            let Kpath = if let Some(Kpath) = Kpath {
                Kpath
            } else {
                panic!("Error: The band_plot option is enabled, but no k_path was found. Please use 'begin kpoint_path' and 'end kpoint_path' to specify the desired band path.")
            };

            let (kvec, kdist, knode, kname) = Kpath.get_path_vec(&model.lat);
            //开始将 kvec 平均分发给各个线程进行计算
            //为了向上取整数, 所以要+size
            let mut chunk_size = (Kpath.k_path_num + size as usize - 1) / size as usize;
            let nk = Kpath.k_path_num;
            world.process_at_rank(0).broadcast_into(&mut chunk_size);
            for i in 1..size {
                let start = (i as usize) * chunk_size;
                let end = cmp::min(start + chunk_size, Kpath.k_path_num);
                let chunk: Array2<f64> = kvec.slice(s![start..end, ..]).to_owned();
                let mut chunk: Vec<f64> = chunk.into_iter().collect();
                world.process_at_rank(i).send(&chunk);
            }
            let chunk = kvec.slice(s![0..chunk_size, ..]).to_owned();
            let band = model.solve_band_all(&chunk);
            let mut data: Vec<f64> = band.into_iter().collect();
            let mut get_data = vec![0.0; chunk_size * model.nsta() * (size as usize)];
            world
                .process_at_rank(0)
                .gather_into_root(&data, &mut get_data);
            let band = Array1::from_vec(get_data)
                .into_shape((chunk_size * (size as usize), model.nsta()))
                .unwrap();
            let band = band.slice(s![0..nk, ..]).to_owned();
            //开始绘图

            let mut fg = Figure::new();
            let x: Vec<f64> = kdist.to_vec();
            let axes = fg.axes2d();
            for i in 0..model.nsta() {
                let y: Vec<f64> = band.slice(s![.., i]).to_owned().to_vec();
                axes.lines(&x, &y, &[Color("black"), LineStyle(Solid)]);
            }
            let axes = axes.set_x_range(Fix(0.0), Fix(knode[[knode.len() - 1]]));
            let label = kname;
            let mut show_ticks = Vec::new();
            for i in 0..knode.len() {
                let A = knode[[i]];
                let B = &label[i];
                show_ticks.push(Major(A, Fix(B)));
            }
            axes.set_x_ticks_custom(
                show_ticks.into_iter(),
                &[],
                &[Font("Times New Roman", 24.0)],
            );

            let knode = knode.to_vec();
            let mut pdf_name = String::new();
            pdf_name.push_str("band.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();
        } else {
            let mut chunk_size = 0;
            world.process_at_rank(0).broadcast_into(&mut chunk_size);
            let mut recv_chunk = vec![0.0; chunk_size * 3];
            world.process_at_rank(0).receive_into(&mut recv_chunk);
            let chunk = Array1::from_vec(recv_chunk)
                .into_shape((chunk_size, 3))
                .unwrap();
            let band = model.solve_band_all(&chunk);
            //kvec 得到, 开始计算能带
            //接下来我们将数据传输回 rank 0
            let mut data: Vec<f64> = band.into_iter().collect();
            world.process_at_rank(0).gather_into(&data)
        };
    }
}
