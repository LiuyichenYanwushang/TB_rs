use bincode::{deserialize, serialize};
use gnuplot::AxesCommon;
use gnuplot::Tick::*;
use gnuplot::{Auto, Caption, Color, Figure, Fix, Font, LineStyle, Solid};
use mpi::request::WaitGuard;
use mpi::traits::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs::create_dir_all;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::ops::AddAssign;
use std::path::Path;
use std::time::{Duration, Instant};
use Rustb::phy_const::*;
use Rustb::Gauge;
use Rustb::Model;
use Rustb::*;
use crate::spin_direction;
use ndarray::linalg::kron;


#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SC_parameter {
    ///Brillouin zone k point number
    ///Use "k_mesh" to specify
    k_mesh: [usize; 3],
    ///temperature, the unit is Kelvins
    ///Use "temperature" to specify
    eta: f64,
    spin: spin_direction,
    ///chemical potential, the unit is eV
    ///Use "chemical_potential_min" to specify
    mu_min: f64,
    ///Use "chemical_potential_max" to specify
    mu_max: f64,
    ///Use "chemical_potential_num" to specify
    mu_num: usize,
}

impl SC_parameter {
    pub fn new() -> Self {
        SC_parameter {
            k_mesh: [1, 1, 1],
            eta: 1e-3,
            spin: spin_direction::None,
            mu_min: 0.0,
            mu_max: 1.0,
            mu_num: 10,
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
    ///从控制文件中读取展宽
    pub fn get_eta(&mut self, reads: &Vec<String>) -> bool {
        let mut eta = None;
        for line in reads.iter() {
            if line.contains("broaden_energy") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let eta0 = string.next().unwrap().parse::<f64>().unwrap();
                    eta = Some(eta0);
                }
            }
        }
        if let Some(eta) = eta {
            self.eta = eta;
            true
        } else {
            false
        }
    }
    pub fn get_spin(&mut self, reads: &Vec<String>) -> bool {
        let mut spin = None;
        for line in reads.iter() {
            if line.contains("spin_direction") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let spin0 = string.next().unwrap().parse::<usize>().unwrap();
                    spin = Some(spin0);
                }
            }
        }
        if let Some(spin) = spin {
            self.spin = match spin {
                0 => spin_direction::None,
                1 => spin_direction::x,
                2 => spin_direction::y,
                3 => spin_direction::z,
                _ => todo!(),
            };
            true
        } else {
            self.spin = spin_direction::None;
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
    pub fn mu(&self) -> Array1<f64> {
        Array1::linspace(self.mu_min, self.mu_max, self.mu_num)
    }
    pub fn eta(&self) -> f64 {
        self.eta
    }
    pub fn spin(&self) -> spin_direction {
        self.spin
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

/// This module calculates the optical conductivity
/// The adopted definition is
/// $$\sigma_{\ap\bt}=\f{2ie^2\hbar}{V}\sum_{\bm k}\sum_{n} f_n (g_{n,\ap\bt}+\f{i}{2}\Og_{n,\ap\bt})$$
///
/// Where
/// $$\\begin{aligned}
/// \Og_{n\ap\bt}&=\sum_{m=\not n}\f{-2 \text{Im} \bra{\psi_{n\bm k}}\p_\ap H\ket{\psi_{m\bm k}}\bra{\psi_{m\bm k}}\p_\bt H\ket{\psi_{n\bm k}}}{(\ve_{n\bm k}-\ve_{m\bm k})^2}
/// \\end{aligned}
/// $$

pub fn spin_current_onek<S: Data<Elem = f64>>(
    model: &Model,
    k_vec: &ArrayBase<S, Ix1>,
    spin: spin_direction,
    eta: f64,
    mu: &Array1<f64>,
) -> Array2<f64> {
    //!给定一个 k 点, 指定 dir_1=$\alpha$, dir_2=$\beta$, T 代表温度, og= $\og$,
    //!mu=$\mu$ 为费米能级范围, spin=0,1,2,3 为$\sg_0,\sg_x,\sg_y,\sg_z$,
    //!当体系不存在自旋的时候无论如何输入spin都默认 spin=0
    //! 这个函数返回的是

    let li: Complex<f64> = 1.0 * Complex::i();
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

    let (J_x, J_y, J_z) = match spin {
        spin_direction::None => {
            let J_x: Array2<Complex<f64>> = A.slice(s![0, .., ..]).to_owned();
            let J_y: Array2<Complex<f64>> = A.slice(s![1, .., ..]).to_owned();
            let J_z: Array2<Complex<f64>> = A.slice(s![2, .., ..]).to_owned();
            (J_x, J_y, J_z)
        }
        spin_direction::x => {
            let pauli = arr2(&[
                [0.0 + 0.0 * li, 1.0 + 0.0 * li],
                [1.0 + 0.0 * li, 0.0 + 0.0 * li],
            ]) / 2.0;
            let X = kron(&pauli, &Array2::eye(model.norb()));
            let J_x: Array2<Complex<f64>> = A.slice(s![0, .., ..]).to_owned();
            let J_y: Array2<Complex<f64>> = A.slice(s![1, .., ..]).to_owned();
            let J_z: Array2<Complex<f64>> = A.slice(s![2, .., ..]).to_owned();
            let J_x = anti_comm(&X, &J_x) * (0.5 + 0.0 * li);
            let J_y = anti_comm(&X, &J_y) * (0.5 + 0.0 * li);
            let J_z = anti_comm(&X, &J_z) * (0.5 + 0.0 * li);
            (J_x, J_y, J_z)
        }
        spin_direction::y => {
            let pauli = arr2(&[
                [0.0 + 0.0 * li, 0.0 - 1.0 * li],
                [0.0 + 1.0 * li, 0.0 + 0.0 * li],
            ]) / 2.0;
            let X = kron(&pauli, &Array2::eye(model.norb()));
            let J_x: Array2<Complex<f64>> = A.slice(s![0, .., ..]).to_owned();
            let J_y: Array2<Complex<f64>> = A.slice(s![1, .., ..]).to_owned();
            let J_z: Array2<Complex<f64>> = A.slice(s![2, .., ..]).to_owned();
            let J_x = anti_comm(&X, &J_x) * (0.5 + 0.0 * li);
            let J_y = anti_comm(&X, &J_y) * (0.5 + 0.0 * li);
            let J_z = anti_comm(&X, &J_z) * (0.5 + 0.0 * li);
            (J_x, J_y, J_z)
        }
        spin_direction::z => {
            let pauli = arr2(&[
                [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                [0.0 + 0.0 * li, -1.0 + 0.0 * li],
            ]) / 2.0;
            let X = kron(&pauli, &Array2::eye(model.norb()));
            let J_x: Array2<Complex<f64>> = A.slice(s![0, .., ..]).to_owned();
            let J_y: Array2<Complex<f64>> = A.slice(s![1, .., ..]).to_owned();
            let J_z: Array2<Complex<f64>> = A.slice(s![2, .., ..]).to_owned();
            let J_x = anti_comm(&X, &J_x) * 0.5;
            let J_y = anti_comm(&X, &J_y) * 0.5;
            let J_z = anti_comm(&X, &J_z) * 0.5;
            (J_x, J_y, J_z)
        }
    };

    let A_x = A_x.dot(&evec);
    let A_y = A_y.dot(&evec);
    let A_z = A_z.dot(&evec);
    let A_x = evec_conj.dot(&A_x);
    let A_y = evec_conj.dot(&A_y);
    let A_z = evec_conj.dot(&A_z);

    let J_x = J_x.dot(&evec);
    let J_y = J_y.dot(&evec);
    let J_z = J_z.dot(&evec);
    let J_x = evec_conj.dot(&J_x);
    let J_y = evec_conj.dot(&J_y);
    let J_z = evec_conj.dot(&J_z);

    let n_mu = mu.len();
    let mut spin_current = Array2::zeros((6, n_mu));

    let A_xx = &J_x * (&A_x.t());
    let A_yy = &J_y * (&A_y.t());
    let A_zz = &J_z * (&A_z.t());
    let A_xy = &J_x * (&A_y.t());
    let A_yz = &J_y * (&A_z.t());
    let A_xz = &J_x * (&A_z.t());
    let A_xx = A_xx.mapv(|x| x.re);
    let A_yy = A_yy.mapv(|x| x.re);
    let A_zz = A_zz.mapv(|x| x.re);
    let A_xy = A_xy.mapv(|x| x.re);
    let A_yz = A_yz.mapv(|x| x.re);
    let A_xz = A_xz.mapv(|x| x.re);

    mu.iter()
        .zip(spin_current.axis_iter_mut(Axis(1)))
        .for_each(|(u, mut O)| {
            let U0 = band.map(|x| eta / ((x - u).powi(2) + eta.powi(2)));
            O[[0]] = U0.dot(&A_xx.dot(&U0));
            O[[1]] = U0.dot(&A_yy.dot(&U0));
            O[[2]] = U0.dot(&A_zz.dot(&U0));
            O[[3]] = U0.dot(&A_xy.dot(&U0));
            O[[4]] = U0.dot(&A_yz.dot(&U0));
            O[[5]] = U0.dot(&A_xz.dot(&U0));
        });
    spin_current
}

pub fn spin_current_conductivity(
    model: &Model,
    kvec: &Array2<f64>,
    sc_parameter: SC_parameter,
) -> Array2<f64> {
    let li: Complex<f64> = 1.0 * Complex::i();
    let eta = sc_parameter.eta();
    let mu = sc_parameter.mu();
    let n_mu = mu.len();
    let spin = sc_parameter.spin();
    let mut omega = Array2::zeros((6, n_mu));
    //let mut a = Instant::now();
    for (i, k) in kvec.outer_iter().enumerate() {
        let spin_current = spin_current_onek(&model, &k, spin, eta, &mu);
        omega.add_assign(&spin_current);
    }
    omega
}


pub fn spin_current_calculate(
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
                "start calculatiing the spin current conductivity"
            );
            println!("start calculatiing the spin current");
            let mut sc_parameter = SC_parameter::new();
            let have_kmesh = sc_parameter.get_k_mesh(&Input_reads);
            if !have_kmesh {
                writeln!(
                    output_file.as_mut().unwrap(),
                    "Error: You mut set k_mesh for calculating spin current conductivity"
                );
            }
            let have_eta = sc_parameter.get_eta(&Input_reads);
            let have_mu = sc_parameter.get_mu(&Input_reads);
            let have_spin_direction = sc_parameter.get_spin(&Input_reads);

            if !(have_eta) {
                writeln!(output_file.as_mut().unwrap(),"Warning: You don't specify smooth energy when calculate the anomalous hall conductivity, using default 0.001");
            }
            if !(have_mu) {
                writeln!(output_file.as_mut().unwrap(),"Warning: You don't specify chemistry potential when calculate the anomalous hall conductivity, using default 0.0");
            }
            let kvec = sc_parameter.get_mesh_vec();

            //传输 sc_parameter
            let mut serialized_data = serialize(&sc_parameter).unwrap();
            let mut data_size = serialized_data.len();
            world.process_at_rank(0).broadcast_into(&mut data_size);
            world
                .process_at_rank(0)
                .broadcast_into(&mut serialized_data[..]);
            //分发kvec
            //这里, 我们采用尽可能地均分策略, 先求出 nk 对 size 地余数,
            //然后将余数分给排头靠前的rank
            let mut nk = sc_parameter.nk();
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
                spin_current_conductivity(&model, &chunk, sc_parameter);

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
            let mu = sc_parameter.mu();
            let n_mu = mu.len();
            let mu_min = mu[[0]];
            let mu_max = mu[[n_mu - 1]];
            conductivity = conductivity / (nk as f64) / model.lat.det().unwrap()
                * Quantum_conductivity
                * 1.0e8;
            println!("The sc conductivity calculation is finished");
            writeln!(output_file.as_mut().unwrap(), "calculation finished");
            //-------------------------------开始写入-----------------------
            writeln!(
                output_file.as_mut().unwrap(),
                "write data in spin_current_conductivity.dat"
            );
            let mut SC_file = File::create("spin_current_conductivity.dat")
                .expect("Unable to create spin_current_conductivity.dat");
            let mut input_string = String::new();
            input_string.push_str("#Calculation results are reported in units of Ω^-1 cm^-1\n");
            input_string.push_str("#The arranged data are: mu,  xx,  yy,  zz,  xy,  yz,  xz\n");
            for i in 0..conductivity.len_of(Axis(1)) {
                input_string.push_str(&format!(
                    "{:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}\n",
                    mu[[i]],
                    conductivity[[0, i]],
                    conductivity[[1, i]],
                    conductivity[[2, i]],
                    conductivity[[3, i]],
                    conductivity[[4, i]],
                    conductivity[[5, i]],
                ));
            }
            writeln!(SC_file, "{}", &input_string);

            //---------------------开始绘图------------------------

            for (row_idx, component) in ["xx", "yy", "zz", "xy", "yz", "xz"].iter().enumerate() {
                let mut fg = Figure::new();
                let x: Vec<f64> = mu.to_vec();
                let axes = fg.axes2d();
                let y: Vec<f64> = conductivity.row(row_idx).to_owned().to_vec();
                axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
                let axes = axes.set_x_range(Fix(mu_min), Fix(mu_max));
                axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
                axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
                match sc_parameter.spin() {
                    spin_direction::None => {
                        axes.set_y_label(
                            &format!("{{/Symbol s}}_{{{0}}} (Ω^{{-1}} cm^{{-1}})", component),
                            &[Font("Times New Roman", 18.0)],
                        );
                    }
                    spin_direction::x => {
                        axes.set_y_label(
                            &format!("{{/Symbol s}}_{{{0}}}^x (Ω^{{-1}} cm^{{-1}})", component),
                            &[Font("Times New Roman", 18.0)],
                        );
                    }
                    spin_direction::y => {
                        axes.set_y_label(
                            &format!("{{/Symbol s}}_{{{0}}}^y (Ω^{{-1}} cm^{{-1}})", component),
                            &[Font("Times New Roman", 18.0)],
                        );
                    }
                    spin_direction::z => {
                        axes.set_y_label(
                            &format!("{{/Symbol s}}_{{{0}}}^z (Ω^{{-1}} cm^{{-1}})", component),
                            &[Font("Times New Roman", 18.0)],
                        );
                    }
                };
                axes.set_x_label("μ (eV)", &[Font("Times New Roman", 18.0)]);

                let mut pdf_name = format!("SC_{}.pdf", component);
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
            let sc_parameter: SC_parameter = deserialize(&received_data).unwrap();
            //接受 sc_paramter 结束, 开始接受kvec
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
            let conductivity: Array2<f64> = spin_current_conductivity(&model, &chunk, sc_parameter);

            //传输 conductivity 到rank0
            let mut serialized_data = serialize(&conductivity).unwrap();
            let mut data_size = serialized_data.len();
            world.process_at_rank(0).send(&mut data_size);
            world.process_at_rank(0).send(&mut serialized_data[..]);
        }
    }
