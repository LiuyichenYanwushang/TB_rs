//! K-path Generator Module
//! ===========================
//!
//! This module handles k-point path parsing and generation for band structure calculations.
//! Key functionality:
//! 1. Reads k-path definitions from input files ("kpoint_path" section)
//! 2. Constructs interpolated k-point sequences between high-symmetry points
//! 3. Calculates crystal momentum metrics using lattice basis
//! 4. Generates:
//!    - k-point vectors in reciprocal space
//!    - Distance metrics for band plotting
//!    - High-symmetry point labels for visualization
//!
//! Typical workflow:
//!   Parse input => Build KPath => Generate k-vec/distance via get_path_vec()
//!

use bincode::{deserialize, serialize};
use gnuplot::AutoOption::*;
use gnuplot::AxesCommon;
use gnuplot::Tick::*;
use gnuplot::{Caption, Color, Figure, Font, LineStyle, Solid};
use mpi::request::WaitGuard;
use mpi::traits::*;
use ndarray::*;
use ndarray_linalg::*;
use std::cmp;
use std::fs::create_dir_all;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use Rustb::phy_const::*;
use Rustb::*;

pub struct k_path {
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
    pub fn nk(&self) -> usize {
        self.k_path_num
    }
    pub fn path(&self) -> Array3<f64> {
        self.k_path.clone()
    }
    pub fn label(&self) -> Vec<[String; 2]> {
        self.k_path_name.clone()
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
            Some(Kpath)
        } else {
            None
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

        let mut node_index: Vec<usize> = vec![0];
        for n in 1..n_knode - 1 {
            let frac = k_node[[n]] / k_node[[n_knode - 1]];
            let a = (frac * ((nk - 1) as f64).round()) as usize;
            node_index.push(a)
        }
        node_index.push(nk - 1);
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
        (k_vec, k_dist, k_node, node_name)
    }
}

pub fn band_plot(
    world: &impl AnyCommunicator,
    model: &Model,
    Input_reads: &Vec<String>,
    output_file: &mut Option<File>,
) {
    let size = world.size();
    let rank = world.rank();
    if rank == 0 {
        //read only on rank0
        writeln!(
            output_file.as_mut().unwrap(),
            "start calculating the band structure"
        );
        //开始读取 k_path 的数据
        let Kpath = k_path::get_k_path(&Input_reads);
        let Kpath = if let Some(Kpath) = Kpath {
            Kpath
        } else {
            panic!("Error: The band_plot option is enabled, but no k_path was found. Please use 'begin kpoint_path' and 'end kpoint_path' to specify the desired band path.")
        };

        let (kvec, kdist, knode, kname) = Kpath.get_path_vec(&model.lat);
        //开始将 kvec 平均分发给各个线程进行计算
        let nk = Kpath.nk();
        let remainder: usize = nk % size as usize;
        let chunk_size0 = nk / size as usize;
        if chunk_size0 == 0 {
            panic!(
                "Error! the num of cpu {} is larger than your k points number {}!",
                nk, size
            );
        }
        let chunk_size = if remainder > 0 {
            chunk_size0 + 1
        } else {
            chunk_size0
        };
        let mut start = chunk_size;
        let mut end = 0;
        for i in 1..size {
            let chunk_size = if (i as usize) < remainder {
                chunk_size0 + 1
            } else {
                chunk_size0
            };
            world.process_at_rank(i).send(&chunk_size);
            end = start + chunk_size;
            let chunk: Array2<f64> = kvec.slice(s![start..end, ..]).to_owned();
            let mut chunk: Vec<f64> = chunk.into_iter().collect();
            world.process_at_rank(i).send(&chunk);
            start = end;
        }
        //分发结束
        let chunk = kvec.slice(s![0..chunk_size, ..]).to_owned();
        let mut band = model.solve_band_all(&chunk);
        //开始接受数据
        for i in 1..size {
            let mut received_size: usize = 0;
            world.process_at_rank(i).receive_into(&mut received_size);
            let mut received_data = vec![0u8; received_size];
            world
                .process_at_rank(i)
                .receive_into(&mut received_data[..]);
            // 反序列化
            let bands: Array2<f64> = deserialize(&received_data).unwrap();
            band.append(Axis(0), bands.view());
        }
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
            &[Font("Times New Roman", 18.0)],
        );
        axes.set_y_label("E-E_f (eV)", &[Font("Times New Roman", 18.0)]);

        let knode = knode.to_vec();
        let mut pdf_name = String::new();
        pdf_name.push_str("band.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
        //开始写入python的绘图代码和数据
        //先写数据

        create_dir_all("band").expect("can't creat the band ");
        let mut band_file = File::create(r"band/BAND.dat").expect("Failed to create Band.dat");
        for i in 0..kdist.len() {
            let mut S = String::new();
            S.push_str(&format!("   {:>3.8}", kdist[[i]]));
            for j in 0..model.nsta() {
                S.push_str(&format!("   {:>3.8}", band[[i, j]]));
            }
            writeln!(band_file, "{}", S);
        }
        let mut band_klabel_file =
            File::create(r"band/BAND_klabel.dat").expect("Failed to create Band_klabel.dat");
        for i in 0..knode.len() {
            writeln!(band_klabel_file, " {}    {:>3.8}", label[i], knode[i]);
        }
        //开始写入python 绘图代码
        let mut python_file =
            File::create(r"band/show_band.py").expect("Failed to create python file");
        let mut S = String::new();
        S.push_str("import numpy as np\n");
        S.push_str("import os\n");
        S.push_str("import matplotlib.pyplot as plt\n");
        S.push_str("band_data=np.loadtxt('BAND.dat')\n");
        S.push_str("fig, ax = plt.subplots()\n");
        S.push_str("knode_file=os.path.join('.','BAND_klabel.dat')\n");
        S.push_str("with open(knode_file,'r',encoding='utf-8') as f:\n");
        S.push_str("    lines=f.readlines()\n");
        S.push_str("knodes=[]\n");
        S.push_str("for i in range(len(lines)):\n");
        S.push_str("    knodes.append(str.split(lines[i]))\n");
        S.push_str("    knodes[i][1]=float(knodes[i][1])\n");
        S.push_str("    ax.axvline(x=knodes[i][1],linewidth=0.5,color='k')\n");
        S.push_str("ax.axhline(y=0,linewidth=0.5,color='g',ls='--')\n");
        S.push_str("knodes=list(map(list, zip(*knodes)))\n");
        S.push_str("ax.set_xticks(knodes[1])\n");
        S.push_str("ax.set_xlim(knodes[1][0],knodes[1][-1])\n");
        S.push_str("for i,a in enumerate(knodes[0]):\n");
        S.push_str("  if a=='GAMMA':\n");
        S.push_str("    knodes[0][i]='$\\Gamma$'\n");
        S.push_str("ax.set_xticklabels(knodes[0])\n");
        S.push_str("ax.set_ylabel(r'E-E$_f$ (eV)')\n");
        S.push_str("ax.set_xlabel(r'kpoints')\n");
        S.push_str("ax.set_ylim(-2,2)\n");
        S.push_str("ax.plot(band_data[:,0],band_data[:,1:],color='k',linewidth=0.5)\n");
        S.push_str("fig.savefig('bandstructure.pdf')");

        writeln!(python_file, "{}", S);
        writeln!(output_file.as_mut().unwrap(), "calculation finished");
    } else {
        let mut chunk_size = 0;
        world.process_at_rank(0).receive_into(&mut chunk_size);
        let mut recv_chunk = vec![0.0; chunk_size * 3];
        world.process_at_rank(0).receive_into(&mut recv_chunk);
        let chunk = Array1::from_vec(recv_chunk)
            .into_shape((chunk_size, 3))
            .unwrap();
        //kvec 得到, 开始计算能带
        let band = model.solve_band_all(&chunk);
        //接下来我们将数据传输回 rank 0
        let mut serialized_data = serialize(&band).unwrap();
        let mut data_size = serialized_data.len();
        world.process_at_rank(0).send(&mut data_size);
        world.process_at_rank(0).send(&mut serialized_data[..]);
    };
}
