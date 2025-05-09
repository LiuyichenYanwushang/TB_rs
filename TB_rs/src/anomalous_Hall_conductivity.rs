use bincode::{deserialize, serialize};
use gnuplot::AxesCommon;
use gnuplot::{Auto, Caption, Color, Figure, Fix, Font, LineStyle, Solid};
use mpi::request::WaitGuard;
use mpi::traits::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::fs::create_dir_all;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::ops::AddAssign;
use std::path::Path;
use std::time::{Duration, Instant};
use Rustb::phy_const::*;
use Rustb::Gauge;
use Rustb::Model;


/// This module calculates the anomalous Hall conductivity
/// The adopted definition is
/// $$\sigma_{\ap\bt}=\f{e^2}{\hbar V}\sum_{\bm k}\sum_{n} f_n \Og_{n,\ap\bt}$$
/// Where
///$$ \Og_{n\ap\bt}=\sum_{m=\not n}\f{-2 \text{Im} \bra{\psi_{n\bm k}}\p_\ap H\ket{\psi_{m\bm k}}\bra{\psi_{m\bm k}}\p_\bt H\ket{\psi_{n\bm k}}}{(\ve_{n\bm k}-\ve_{m\bm k})^2}$$


#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct AHC_parameter {
    ///Brillouin zone k point number
    ///Use "k_mesh" to specify
    k_mesh: [usize; 3],
    ///temperature, the unit is Kelvins
    ///Use "temperature" to specify
    T: f64,
    ///chemical potential, the unit is eV
    ///Use "chemical_potential_min" to specify
    mu_min: f64,
    ///Use "chemical_potential_max" to specify
    mu_max: f64,
    ///Use "chemical_potential_num" to specify
    mu_num: usize,
}

impl AHC_parameter {
    pub fn new() -> Self {
        AHC_parameter {
            k_mesh: [1, 1, 1],
            T: 0.0,
            mu_min: 0.0,
            mu_max: 1.0,
            mu_num: 10,
        }
    }

    ///从控制文件中读取温度
    pub fn get_T(&mut self, reads: &Vec<String>) -> bool {
        let mut T = None;
        for line in reads.iter() {
            if line.contains("temperature") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let T0 = string.next().unwrap().parse::<f64>().unwrap();
                    T = Some(T0);
                }
            }
        }
        if let Some(T) = T {
            self.T = T;
            true
        } else {
            false
        }
    }

    ///从输入文件中得到 k_mesh, 返回 true or false 表示是否指定了 k_mesh
    pub fn get_k_mesh(&mut self, reads: &Vec<String>) -> bool {
        let mut kmesh = None;
        for line in reads.iter() {
            if line.contains("k_mesh") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let nkx = string.next().unwrap().parse::<usize>().unwrap();
                    let nky = string.next().unwrap().parse::<usize>().unwrap();
                    let nkz = string.next().unwrap().parse::<usize>().unwrap();
                    kmesh = Some([nkx, nky, nkz]);
                }
            }
        }
        if let Some(kmesh) = kmesh {
            self.k_mesh = kmesh;
            true
        } else {
            false
        }
    }

    ///从控制文件中读取化学式范围
    pub fn get_mu(&mut self, reads: &Vec<String>) -> bool {
        let mut mu_min = None;
        let mut mu_max = None;
        let mut mu_num = None;
        for line in reads.iter() {
            if line.contains("chemical_potential_min") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let eta0 = string.next().unwrap().parse::<f64>().unwrap();
                    mu_min = Some(eta0);
                }
            }
            if line.contains("chemical_potential_max") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let eta0 = string.next().unwrap().parse::<f64>().unwrap();
                    mu_max = Some(eta0);
                }
            }
            if line.contains("chemical_potential_num") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let eta0 = string.next().unwrap().parse::<usize>().unwrap();
                    mu_num = Some(eta0);
                }
            }
        }
        if mu_min == None || mu_max == None || mu_num == None {
            false
        } else {
            self.mu_min = mu_min.unwrap();
            self.mu_max = mu_max.unwrap();
            self.mu_num = mu_num.unwrap();
            true
        }
    }

    pub fn nk(&self) -> usize {
        let nk = self.k_mesh[0] * self.k_mesh[1] * self.k_mesh[2];
        nk
    }
    pub fn temperature(&self) -> f64 {
        self.T
    }
    pub fn mu(&self) -> Array1<f64> {
        Array1::linspace(self.mu_min, self.mu_max, self.mu_num)
    }
    pub fn get_mesh_vec(&self) -> Array2<f64> {
        let nk = self.nk();
        let mut kvec = Array2::zeros((nk, 3));
        let mut i0 = 0;
        for i in 0..self.k_mesh[0] {
            for j in 0..self.k_mesh[1] {
                for k in 0..self.k_mesh[2] {
                    let k0 = array![
                        (i as f64) / (self.k_mesh[0] as f64),
                        (j as f64) / (self.k_mesh[1] as f64),
                        (k as f64) / (self.k_mesh[2] as f64)
                    ];
                    kvec.row_mut(i0).assign(&k0);
                    i0 += 1;
                }
            }
        }
        kvec
    }
}


pub fn anomalous_Hall_onek<S: Data<Elem = f64>>(
    model: &Model,
    k_vec: &ArrayBase<S, Ix1>,
    T: f64,
    mu: &Array1<f64>,
) -> Array2<f64> {
    let li: Complex<f64> = 1.0 * Complex::i();
    //产生速度算符A和哈密顿量
    let (mut A, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) =
        model.gen_v(k_vec, Gauge::Lattice);
    //计算本征值和本征态
    let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
        (eigvals, eigvecs)
    } else {
        todo!()
    };
    let evec_conj = evec.t();
    let evec = evec.mapv(|x| x.conj());

    //转化为本征态下的速度算符
    let A_x: Array2<Complex<f64>> = A.slice(s![0, .., ..]).to_owned();
    let A_y: Array2<Complex<f64>> = A.slice(s![1, .., ..]).to_owned();
    let A_z: Array2<Complex<f64>> = A.slice(s![2, .., ..]).to_owned();
    let _ = A;
    let A_x = A_x.dot(&evec);
    let A_y = A_y.dot(&evec);
    let A_z = A_z.dot(&evec);
    let A_x = evec_conj.dot(&A_x);
    let A_y = evec_conj.dot(&A_y);
    let A_z = evec_conj.dot(&A_z);

    let n_mu = mu.len();
    let mut berry_curvature = Array2::zeros((3, n_mu));
    let b = band.clone().into_shape((1, band.len())).unwrap(); // 将 band 转换为 1*nsta 的 2D 数组
    let UU = &b - &b.t(); // 利用广播和转置计算差值
    let UU0 = UU.mapv(|x| if x.abs() < 1e-8 { 0.0 } else { 1.0 / x.powi(2) });

    //由于反对称, 所以我们只计算xy, yz, xz
    let A_xy = &A_x * (&A_y.t());
    let A_yz = &A_y * (&A_z.t());
    let A_xz = &A_x * (&A_z.t());
    let A_xy = A_xy.mapv(|x| x.im);
    let A_yz = A_yz.mapv(|x| x.im);
    let A_xz = A_xz.mapv(|x| x.im);
    //let A_xy = (&A_xy * &UU0).sum_axis(Axis(1));
    //let A_yz = (&A_yz * &UU0).sum_axis(Axis(1));
    //let A_xz = (&A_xz * &UU0).sum_axis(Axis(1));
    let A_xy =A_xy.outer_iter().zip(UU0.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
    let A_yz =A_yz.outer_iter().zip(UU0.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
    let A_xz =A_xz.outer_iter().zip(UU0.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
    let A_xy =Array1::from_vec(A_xy);
    let A_yz =Array1::from_vec(A_yz);
    let A_xz =Array1::from_vec(A_xz);

    mu.iter()
        .zip(berry_curvature.axis_iter_mut(Axis(1)))
        .for_each(|(u, mut O)| {
            let fermi_dirac = if T == 0.0 {
                band.mapv(|x| if x > *u { 0.0 } else { 1.0 })
            } else {
                let beta = 1.0 / T / 8.617e-5;
                band.mapv(|x| ((beta * (x - u)).exp() + 1.0).recip())
            };
            O[[0]] += A_xy.dot(&fermi_dirac);
            O[[1]] += A_yz.dot(&fermi_dirac);
            O[[2]] += A_xz.dot(&fermi_dirac);
        });
    -2.0 * berry_curvature
}

pub fn Anomalous_Hall_conductivity(
    model: &Model,
    kvec: &Array2<f64>,
    ahc_parameter: AHC_parameter,
) -> Array2<f64> {
    let li: Complex<f64> = 1.0 * Complex::i();
    let T = ahc_parameter.temperature();
    let mu = ahc_parameter.mu();
    let n_mu = mu.len();
    let mut omega = Array2::zeros((3, n_mu));
    //let mut a = Instant::now();
    for (i, k) in kvec.outer_iter().enumerate() {
        let berry_curvature = anomalous_Hall_onek(&model, &k, T, &mu);
        omega.add_assign(&berry_curvature);
    }
    omega
}

pub fn anomalous_Hall_conductivity_calculate(
    world: &impl AnyCommunicator,
    model: &Model,
    Input_reads: &Vec<String>,
    output_file: &mut Option<File>,
) {
    let size = world.size();
    let rank = world.rank();

    //开始计算反常Hall 电导
    if rank == 0 {
        writeln!(
            output_file.as_mut().unwrap(),
            "start calculatiing the anomalous Hall conductivity"
        );
        println!("start calculatiing the anomalous Hall");
        let mut ahc_parameter = AHC_parameter::new();
        let have_kmesh = ahc_parameter.get_k_mesh(&Input_reads);
        if !have_kmesh {
            writeln!(
                output_file.as_mut().unwrap(),
                "Error: You mut set k_mesh for calculating anomalous Hall conductivity"
            );
        }
        let have_T = ahc_parameter.get_T(&Input_reads);
        let have_mu = ahc_parameter.get_mu(&Input_reads);

        if !(have_T) {
            writeln!(output_file.as_mut().unwrap(),"Warning: You don't specify temperature when calculate the anomalous hall conductivity, using default 0.0");
        }
        if !(have_mu) {
            writeln!(output_file.as_mut().unwrap(),"Warning: You don't specify chemistry potential when calculate the anomalous hall conductivity, using default 0.0");
        }
        let kvec = ahc_parameter.get_mesh_vec();

        //传输 ahc_parameter
        let mut serialized_data = serialize(&ahc_parameter).unwrap();
        let mut data_size = serialized_data.len();
        world.process_at_rank(0).broadcast_into(&mut data_size);
        world
            .process_at_rank(0)
            .broadcast_into(&mut serialized_data[..]);
        //分发kvec
        //这里, 我们采用尽可能地均分策略, 先求出 nk 对 size 地余数,
        //然后将余数分给排头靠前的rank
        let mut nk = ahc_parameter.nk();
        world.process_at_rank(0).broadcast_into(&mut nk);
        let remainder: usize = nk % size as usize;
        let chunk_size0 = nk / size as usize;
        if chunk_size0 == 0 {
            panic!(
                "Error! the num of cpu {} is larger than your k points number {}!",
                nk, size
            );
        }
        println!("remainder={},chunk_size0={}", remainder, chunk_size0);
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
        let mut conductivity: Array2<f64> =
            Anomalous_Hall_conductivity(&model, &chunk, ahc_parameter);

        for i in 1..size {
            let mut received_size: usize = 0;
            world.process_at_rank(i).receive_into(&mut received_size);
            let mut received_data = vec![0u8; received_size];
            world
                .process_at_rank(i)
                .receive_into(&mut received_data[..]);
            // 反序列化
            let conductivity0: Array2<f64> = deserialize(&received_data).unwrap();
            conductivity = conductivity + conductivity0;
        }
        let mu = ahc_parameter.mu();
        let n_mu = mu.len();
        let mu_min = mu[[0]];
        let mu_max = mu[[n_mu - 1]];
        conductivity =
            conductivity / (nk as f64) / model.lat.det().unwrap() * Quantum_conductivity * 1.0e8;
        println!("The ahc conductivity calculation is finished");
        writeln!(output_file.as_mut().unwrap(), "calculation finished");
        //开始写入
        writeln!(
            output_file.as_mut().unwrap(),
            "write data in ahc_conductivity_A.dat and ahc_conductivity_S.dat"
        );
        let mut AHC_file =
            File::create("ahc_conductivity.dat").expect("Unable to create ahc_conductivity.dat");
        let mut input_string = String::new();
        input_string.push_str("#Calculation results are reported in units of Ω^-1 cm^-1\n");
        input_string.push_str("#The arranged data are: mu,  omega_xy, omega_yz, omega_xz\n");
        for i in 0..conductivity.len_of(Axis(1)) {
            input_string.push_str(&format!(
                "{:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}\n",
                mu[[i]],
                conductivity[[0, i]],
                conductivity[[1, i]],
                conductivity[[2, i]]
            ));
        }
        writeln!(AHC_file, "{}", &input_string);

        //---------------------开始绘图------------------------

        for (row_idx, component) in ["xy", "yz", "xz"].iter().enumerate() {
            let mut fg = Figure::new();
            let x: Vec<f64> = mu.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = conductivity.row(row_idx).to_owned().to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(mu_min), Fix(mu_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                &format!("{{/Symbol s}}_{{{0}}} (Ω^{{-1}} cm^{{-1}})", component),
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("μ (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = format!("AHC_{}.pdf", component);
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();
        }
    } else {
        let mut received_size: usize = 0;
        world.process_at_rank(0).broadcast_into(&mut received_size);

        // 根据接收到的大小分配接收缓冲区
        let mut received_data = vec![0u8; received_size];
        world
            .process_at_rank(0)
            .broadcast_into(&mut received_data[..]);
        // 反序列化
        let ahc_parameter: AHC_parameter = deserialize(&received_data).unwrap();
        //接受 ahc_paramter 结束, 开始接受kvec
        let mut nk: usize = 0;
        world.process_at_rank(0).broadcast_into(&mut nk);
        let mut chunk_size = 0;
        world.process_at_rank(0).receive_into(&mut chunk_size);
        let mut recv_chunk = vec![0.0; chunk_size * 3];
        world.process_at_rank(0).receive_into(&mut recv_chunk);
        let chunk = Array1::from_vec(recv_chunk)
            .into_shape((chunk_size, 3))
            .unwrap();
        //接受kvec 开始计算 anomalous Hall effect
        let conductivity: Array2<f64> = Anomalous_Hall_conductivity(&model, &chunk, ahc_parameter);

        //传输 conductivity 到rank0
        let mut serialized_data = serialize(&conductivity).unwrap();
        let mut data_size = serialized_data.len();
        world.process_at_rank(0).send(&mut data_size);
        world.process_at_rank(0).send(&mut serialized_data[..]);
    }
}
