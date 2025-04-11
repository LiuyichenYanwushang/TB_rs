//! This module implements the calculation of optical conductivity for a given material model.
//!
//! The core structure `OC_parameter` manages parameters for Brillouin zone sampling (`k_mesh`),
//! energy broadening (`eta`), frequency range (`omega_min`, `omega_max`, `omega_num`),
//! temperature (`T`), and chemical potential (`mu`). These parameters are typically read from
//! an input configuration file via helper methods (e.g., `get_k_mesh`, `get_eta`).
//!
//! The `Optical_conductivity` function computes the optical conductivity tensor components
//! (e.g., σ_xx, σ_xy) across a specified frequency range. It iterates over k-points in the
//! Brillouin zone, calculates contributions from each k-point using linear algebra operations
//! (via `ndarray` and `ndarray_linalg`), and aggregates results into conductivity matrices.
//!
//! Key features:
//! - Supports Gaussian broadening for energy levels.
//! - Handles complex frequency-dependent responses using `num_complex`.
//! - Serialization/deserialization of parameters via `serde`.
//!
//! Typical usage involves configuring `OC_parameter`, passing it to `Optical_conductivity`
//! along with a material `Model` and k-point grid, to obtain the optical conductivity tensor
//! as a function of incident light frequency.

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

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct OC_parameter {
    ///Brillouin zone k point number
    ///Use "k_mesh" to specify
    k_mesh: [usize; 3],
    ///energy broadening
    ///Use "broaden_energy" to specify
    eta: f64,
    ///Incident light frequency, the unit is eV
    ///Use "frenquency_min", "frenquency_max", "frenquency_num" to specify
    omega_min: f64,
    omega_max: f64,
    omega_num: usize,
    ///temperature, the unit is Kelvins
    ///Use "temperature" to specify
    T: f64,
    ///chemical potential, the unit is eV
    ///Use "chemical_potential" to specify
    mu: f64,
}

impl OC_parameter {
    pub fn new() -> Self {
        OC_parameter {
            k_mesh: [1, 1, 1],
            eta: 1e-3,
            omega_min: 0.0,
            omega_max: 1.0,
            omega_num: 10,
            T: 0.0,
            mu: 0.0,
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

    ///从控制文件中读取化学势
    pub fn get_mu(&mut self, reads: &Vec<String>) -> bool {
        let mut mu = None;
        for line in reads.iter() {
            //后面会用到 chemical_potential_min, chemical_potential_max,chemical_potential_num
            if line.contains("chemical_potential")
                && !(line.contains("min") || line.contains("max") || line.contains("num"))
            {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let mu0 = string.next().unwrap().parse::<f64>().unwrap();
                    mu = Some(mu0);
                }
            }
        }
        if let Some(mu) = mu {
            self.mu = mu;
            true
        } else {
            false
        }
    }

    ///从控制文件中读取频率
    pub fn get_omega(&mut self, reads: &Vec<String>) -> bool {
        let mut omega_min = None;
        let mut omega_max = None;
        let mut omega_num = None;
        for line in reads.iter() {
            if line.contains("frenquency_min") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let eta0 = string.next().unwrap().parse::<f64>().unwrap();
                    omega_min = Some(eta0);
                }
            }
            if line.contains("frenquency_max") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let eta0 = string.next().unwrap().parse::<f64>().unwrap();
                    omega_max = Some(eta0);
                }
            }
            if line.contains("frenquency_num") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let mut string = parts[1].trim().split_whitespace();
                    let eta0 = string.next().unwrap().parse::<usize>().unwrap();
                    omega_num = Some(eta0);
                }
            }
        }
        if omega_min == None || omega_max == None || omega_num == None {
            false
        } else {
            self.omega_min = omega_min.unwrap();
            self.omega_max = omega_max.unwrap();
            self.omega_num = omega_num.unwrap();
            true
        }
    }
    pub fn nk(&self) -> usize {
        let nk = self.k_mesh[0] * self.k_mesh[1] * self.k_mesh[2];
        nk
    }
    pub fn eta(&self) -> f64 {
        self.eta
    }
    pub fn temperature(&self) -> f64 {
        self.T
    }
    pub fn mu(&self) -> f64 {
        self.mu
    }
    pub fn omega(&self) -> (f64, f64, usize) {
        (self.omega_min, self.omega_max, self.omega_num)
    }
    pub fn og(&self) -> Array1<f64> {
        let (omega_min, omega_max, omega_num) = self.omega();
        Array1::linspace(omega_min, omega_max, omega_num)
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
/// g_{n\ap\bt}&=\sum_{m=\not n}\f{\og-i\eta}{\ve_{n\bm k}-\ve_{m\bm k}}\f{\text{Re} \bra{\psi_{n\bm k}}\p_\ap H\ket{\psi_{m\bm k}}\bra{\psi_{m\bm k}}\p_\bt H\ket{\psi_{n\bm k}}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og-i\eta)^2}\\\\
/// \Og_{n\ap\bt}&=\sum_{m=\not n}\f{\text{Re} \bra{\psi_{n\bm k}}\p_\ap H\ket{\psi_{m\bm k}}\bra{\psi_{m\bm k}}\p_\bt H\ket{\psi_{n\bm k}}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og-i\eta)^2}
/// \\end{aligned}
/// $$

pub fn optical_geometry_onek<S: Data<Elem = f64>>(
    model: &Model,
    k_vec: &ArrayBase<S, Ix1>,
    T: f64,
    mu: f64,
    og: &Array1<f64>,
    eta: f64,
) -> (Array2<Complex<f64>>, Array2<Complex<f64>>) {
    //!给定一个 k 点, 指定 dir_1=$\alpha$, dir_2=$\beta$, T 代表温度, og= $\og$,
    //!mu=$\mu$ 为费米能级, spin=0,1,2,3 为$\sg_0,\sg_x,\sg_y,\sg_z$,
    //!当体系不存在自旋的时候无论如何输入spin都默认 spin=0
    //!eta=$\eta$ 是一个小量
    //! 这个函数返回的是
    //! $$ \sum_n f_n\Omega_{n,\ap\bt}^\gm(\bm k)=\sum_n \f{1}{e^{(\ve_{n\bm k}-\mu)/T/k_B}+1} \sum_{m=\not n}\f{J_{\ap,nm}^\gm v_{\bt,mn}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og+i\eta)^2}$$
    //! 其中 $J_\ap^\gm=\\{s_\gm,v_\ap\\}$
    let li: Complex<f64> = 1.0 * Complex::i();
    let (mut A, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) =
        model.gen_v(k_vec, Gauge::Lattice);
    let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
        (eigvals, eigvecs)
    } else {
        todo!()
    };
    let evec_conj = evec.t();
    let evec = evec.mapv(|x| x.conj());

    let A_x: Array2<Complex<f64>> = A.slice(s![0, .., ..]).to_owned();
    let A_y: Array2<Complex<f64>> = A.slice(s![1, .., ..]).to_owned();
    let A_z: Array2<Complex<f64>> = A.slice(s![2, .., ..]).to_owned();
    let A_x = A_x.dot(&evec);
    let A_y = A_y.dot(&evec);
    let A_z = A_z.dot(&evec);
    let A_x = evec_conj.dot(&A_x);
    let A_y = evec_conj.dot(&A_y);
    let A_z = evec_conj.dot(&A_z);

    //let og = og.mapv(|x| x + li * eta);
    let n_og = og.len();
    let mut omega_H = Array2::zeros((6, n_og));
    let mut omega_aH = Array2::zeros((6, n_og));
    let b = band.clone().into_shape((1, band.len())).unwrap(); // 将 band 转换为 1*nsta 的 2D 数组
    let UU = &b - &b.t(); // 利用广播和转置计算差值

    let A_xx = &A_x * (&A_x.t());
    let A_yy = &A_y * (&A_y.t());
    let A_zz = &A_z * (&A_z.t());
    let A_xy = &A_x * (&A_y.t());
    let A_yz = &A_y * (&A_z.t());
    let A_xz = &A_x * (&A_z.t());

    if T == 0.0 {
        let nocc: usize = band.fold(0, |acc, x| if *x > mu { acc } else { acc + 1 });
        let uu_1 = UU.slice(s![0..nocc, nocc..]);
        let A_xx1 = A_xx.slice(s![0..nocc, nocc..]);
        let A_yy1 = A_yy.slice(s![0..nocc, nocc..]);
        let A_zz1 = A_zz.slice(s![0..nocc, nocc..]);
        let A_xy1 = A_xy.slice(s![0..nocc, nocc..]);
        let A_yz1 = A_yz.slice(s![0..nocc, nocc..]);
        let A_xz1 = A_xz.slice(s![0..nocc, nocc..]);
        let uu_2 = UU.slice(s![nocc.., 0..nocc]);
        let A_xx2 = A_xx.slice(s![nocc.., 0..nocc]);
        let A_yy2 = A_yy.slice(s![nocc.., 0..nocc]);
        let A_zz2 = A_zz.slice(s![nocc.., 0..nocc]);
        let A_xy2 = A_xy.slice(s![nocc.., 0..nocc]);
        let A_yz2 = A_yz.slice(s![nocc.., 0..nocc]);
        let A_xz2 = A_xz.slice(s![nocc.., 0..nocc]);
        og.iter()
            .zip(
                omega_H
                    .axis_iter_mut(Axis(1))
                    .zip(omega_aH.axis_iter_mut(Axis(1))),
            )
            .for_each(|(a0, (mut oH, mut oaH))| {
                let a = (&uu_1 - *a0) / eta;
                let U0 = (-&a * &a / 2.0).mapv(|x| x.exp()) * PI / (2.0 * PI).sqrt() / eta;
                let U0 = U0 / &uu_1;
                let U1 = &uu_1.mapv(|x| (x * (x - a0 - li * eta)).finv().re);
                oH[[0]] += (&A_xx1 * &U0).sum();
                oH[[1]] += (&A_yy1 * &U0).sum();
                oH[[2]] += (&A_zz1 * &U0).sum();
                oH[[3]] += (&A_xy1 * &U0).sum();
                oH[[4]] += (&A_yz1 * &U0).sum();
                oH[[5]] += (&A_xz1 * &U0).sum();
                oaH[[0]] -= (&A_xx1 * U1).sum() * li;
                oaH[[1]] -= (&A_yy1 * U1).sum() * li;
                oaH[[2]] -= (&A_zz1 * U1).sum() * li;
                oaH[[3]] -= (&A_xy1 * U1).sum() * li;
                oaH[[4]] -= (&A_yz1 * U1).sum() * li;
                oaH[[5]] -= (&A_xz1 * U1).sum() * li;

                let a = (&uu_2 - *a0) / eta;
                let U0 = (-&a * &a / 2.0).mapv(|x| x.exp()) * PI / (2.0 * PI).sqrt() / eta;
                let U0 = -U0 / &uu_2;
                let U1 = &uu_2.mapv(|x| -(x * (x - a0 - li * eta)).finv().re);
                oH[[0]] += (&A_xx2 * &U0).sum();
                oH[[1]] += (&A_yy2 * &U0).sum();
                oH[[2]] += (&A_zz2 * &U0).sum();
                oH[[3]] += (&A_xy2 * &U0).sum();
                oH[[4]] += (&A_yz2 * &U0).sum();
                oH[[5]] += (&A_xz2 * &U0).sum();
                oaH[[0]] -= (&A_xx2 * U1).sum() * li;
                oaH[[1]] -= (&A_yy2 * U1).sum() * li;
                oaH[[2]] -= (&A_zz2 * U1).sum() * li;
                oaH[[3]] -= (&A_xy2 * U1).sum() * li;
                oaH[[4]] -= (&A_yz2 * U1).sum() * li;
                oaH[[5]] -= (&A_xz2 * U1).sum() * li;
            });
    } else {
        let beta = 1.0 / T / 8.617e-5;
        let fermi_dirac = band.mapv(|x| ((beta * (x - mu)).exp() + 1.0).recip());
        let b = fermi_dirac
            .clone()
            .into_shape((1, fermi_dirac.len()))
            .unwrap(); // 将 band 转换为 1*nsta 的 2D 数组
        let fermi = &b - &b.t(); // 利用广播和转置计算差值
        og.iter()
            .zip(
                omega_H
                    .axis_iter_mut(Axis(1))
                    .zip(omega_aH.axis_iter_mut(Axis(1))),
            )
            .for_each(|(a0, (mut oH, mut oaH))| {
                let a = (&UU - *a0) / eta;
                let U0 = (-&a * &a / 2.0).mapv(|x| x.exp()) * PI / (2.0 * PI).sqrt() / eta;
                let U0 = U0 / &UU * &fermi;
                let U1 = &fermi
                    * &UU.mapv(|x| {
                        if x.abs() > 1e-6 {
                            (x - a0 - li * eta).finv().re / x
                        } else {
                            0.0
                        }
                    });
                oH[[0]] += (&A_xx * &U0).sum();
                oH[[1]] += (&A_yy * &U0).sum();
                oH[[2]] += (&A_zz * &U0).sum();
                oH[[3]] += (&A_xy * &U0).sum();
                oH[[4]] += (&A_yz * &U0).sum();
                oH[[5]] += (&A_xz * &U0).sum();
                oaH[[0]] -= (&A_xx * &U1).sum() * li;
                oaH[[1]] -= (&A_yy * &U1).sum() * li;
                oaH[[2]] -= (&A_zz * &U1).sum() * li;
                oaH[[3]] -= (&A_xy * &U1).sum() * li;
                oaH[[4]] -= (&A_yz * &U1).sum() * li;
                oaH[[5]] -= (&A_xz * &U1).sum() * li;
            });
    }
    (omega_H, omega_aH)
}

pub fn Optical_conductivity(
    model: &Model,
    kvec: &Array2<f64>,
    optical_parameter: OC_parameter,
) -> (Array2<Complex<f64>>, Array2<Complex<f64>>) {
    let li: Complex<f64> = 1.0 * Complex::i();
    let eta = optical_parameter.eta();
    let T = optical_parameter.temperature();
    let mu = optical_parameter.mu();
    let og = optical_parameter.og();
    let (og_min, og_max, n_og) = optical_parameter.omega();
    let mut matric = Array2::zeros((6, n_og));
    let mut omega = Array2::zeros((3, n_og));
    //let mut a = Instant::now();
    for (i, k) in kvec.outer_iter().enumerate() {
        let (omega_H, omega_aH) = optical_geometry_onek(&model, &k, T, mu, &og, eta);

        let sig_xx: Array1<Complex<f64>> = &omega_H.row(0).mapv(|x| Complex::new(x.re, 0.0))
            + &omega_aH.row(0).mapv(|x| x.im * li);
        let sig_yy: Array1<Complex<f64>> = &omega_H.row(1).mapv(|x| Complex::new(x.re, 0.0))
            + &omega_aH.row(1).mapv(|x| x.im * li);
        let sig_zz: Array1<Complex<f64>> = &omega_H.row(2).mapv(|x| Complex::new(x.re, 0.0))
            + &omega_aH.row(2).mapv(|x| x.im * li);
        let sig_xy: Array1<Complex<f64>> = &omega_H.row(3).mapv(|x| Complex::new(x.re, 0.0))
            + &omega_aH.row(3).mapv(|x| x.im * li);
        let sig_yz: Array1<Complex<f64>> = &omega_H.row(4).mapv(|x| Complex::new(x.re, 0.0))
            + &omega_aH.row(4).mapv(|x| x.im * li);
        let sig_xz: Array1<Complex<f64>> = &omega_H.row(5).mapv(|x| Complex::new(x.re, 0.0))
            + &omega_aH.row(5).mapv(|x| x.im * li);
        matric.row_mut(0).add_assign(&sig_xx);
        matric.row_mut(1).add_assign(&sig_yy);
        matric.row_mut(2).add_assign(&sig_zz);
        matric.row_mut(3).add_assign(&sig_xy);
        matric.row_mut(4).add_assign(&sig_yz);
        matric.row_mut(5).add_assign(&sig_xz);

        let sig_xy: Array1<Complex<f64>> =
            &omega_aH.row(3).mapv(|x| x.re + 0.0 * li) + &omega_H.row(3).mapv(|x| x.im * li);
        let sig_yz: Array1<Complex<f64>> =
            &omega_aH.row(4).mapv(|x| x.re + 0.0 * li) + &omega_H.row(4).mapv(|x| x.im * li);
        let sig_xz: Array1<Complex<f64>> =
            &omega_aH.row(5).mapv(|x| x.re + 0.0 * li) + &omega_H.row(5).mapv(|x| x.im * li);
        omega.row_mut(0).add_assign(&sig_xy);
        omega.row_mut(1).add_assign(&sig_yz);
        omega.row_mut(2).add_assign(&sig_xz);
        //let mut b = Instant::now();
        //let average_time = b.duration_since(a).as_secs_f64() / (i as f64 + 1.0);
        //println!("average_time={}", average_time);
    }
    (matric, omega)
}

pub fn optical_conductivity_calculate(
    world: &impl AnyCommunicator,
    model: &Model,
    Input_reads: &Vec<String>,
    output_file: &mut Option<File>,
) {
    let size = world.size();
    let rank = world.rank();
    //开始计算光电导
    if rank == 0 {
        writeln!(
            output_file.as_mut().unwrap(),
            "start calculatiing the optical conductivity"
        );
        println!("start calculatiing the optical conductivity");
        let mut optical_parameter = OC_parameter::new();
        let have_kmesh = optical_parameter.get_k_mesh(&Input_reads);
        if !have_kmesh {
            writeln!(
                output_file.as_mut().unwrap(),
                "Error: You mut set k_mesh for calculating optical conductivity"
            );
        }
        let have_T = optical_parameter.get_T(&Input_reads);
        let have_mu = optical_parameter.get_mu(&Input_reads);
        let have_eta = optical_parameter.get_eta(&Input_reads);
        let have_omega = optical_parameter.get_omega(&Input_reads);
        if !(have_omega) {
            writeln!(
                output_file.as_mut().unwrap(),
                "Error: You must specify frequenccy when calculate the optical conductivity"
            );
            panic!("Error: You must specify  frequenccy when calculate the optical conductivity")
        }

        if !(have_T) {
            writeln!(output_file.as_mut().unwrap(),"Warning: You don't specify temperature when calculate the optical conductivity, using default 0.0");
        }
        if !(have_mu) {
            writeln!(output_file.as_mut().unwrap(),"Warning: You don't specify chemistry potential when calculate the optical conductivity, using default 0.0");
        }
        if !(have_eta) {
            writeln!(output_file.as_mut().unwrap(),"Warning: You don't specify broaden_energy when calculate the optical conductivity, using default 0.0");
        }
        let kvec = optical_parameter.get_mesh_vec();

        //向所有线程广播 optical_parameter
        let mut serialized_data = serialize(&optical_parameter).unwrap();
        let mut data_size = serialized_data.len();
        world.process_at_rank(0).broadcast_into(&mut data_size);
        world
            .process_at_rank(0)
            .broadcast_into(&mut serialized_data[..]);
        //分发kvec
        //这里, 我们采用尽可能地均分策略, 先求出 nk 对 size 地余数,
        //然后将余数分给排头靠前的rank
        let mut nk = optical_parameter.nk();
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
        let (mut metric, mut omega): (Array2<Complex<f64>>, Array2<Complex<f64>>) =
            Optical_conductivity(&model, &chunk, optical_parameter);

        //开始接收各个线程的数据
        for i in 1..size {
            let mut received_size: usize = 0;
            world.process_at_rank(i).receive_into(&mut received_size);
            let mut received_data = vec![0u8; received_size];
            world
                .process_at_rank(i)
                .receive_into(&mut received_data[..]);
            // 反序列化
            let metric0: Array2<Complex<f64>> = deserialize(&received_data).unwrap();
            metric = metric + metric0;

            let mut received_size: usize = 0;
            world.process_at_rank(i).receive_into(&mut received_size);
            let mut received_data = vec![0u8; received_size];
            world
                .process_at_rank(i)
                .receive_into(&mut received_data[..]);
            // 反序列化
            let omega0: Array2<Complex<f64>> = deserialize(&received_data).unwrap();
            omega = omega + omega0;
        }
        let og = optical_parameter.og();
        omega = omega / (nk as f64) / model.lat.det().unwrap() * Quantum_conductivity * 1.0e8;
        metric = metric / (nk as f64) / model.lat.det().unwrap() * Quantum_conductivity * 1.0e8;
        println!("The optical conductivity calculation is finished");
        writeln!(output_file.as_mut().unwrap(), "calculation finished");
        //开始写入
        writeln!(
            output_file.as_mut().unwrap(),
            "write data in optical_conductivity_A.dat and optical_conductivity_S.dat"
        );
        let mut metric_file =
            File::create("optical_conductivity_S.dat").expect("Unable to create TB.in");
        let mut input_string = String::new();
        input_string.push_str("#Calculation results are reported in units of Ω^-1 cm^-1\n");
        input_string.push_str("#For symmetrical results, the arranged data are: omega, Re(xx, Im(xx), Re(yy), Im(yy), Re(zz), Im(zz), Re(xy), Im(xy), Re(yz), Im(yz), Re(xz), Im(xz)\n");
        for i in 0..metric.len_of(Axis(1)) {
            input_string.push_str(&format!("{:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}\n",og[[i]],metric[[0,i]].re,metric[[0,i]].im,metric[[1,i]].re,metric[[1,i]].im,metric[[2,i]].re,metric[[2,i]].im,metric[[3,i]].re,metric[[3,i]].im,metric[[4,i]].re,metric[[4,i]].im,metric[[5,i]].re,metric[[5,i]].im));
        }
        writeln!(metric_file, "{}", &input_string);

        let mut berry_file =
            File::create("optical_conductivity_A.dat").expect("Unable to create TB.in");
        let mut input_string = String::new();

        input_string.push_str("#Calculation results are reported in units of Ω^-1 cm^-1\n");
        input_string.push_str("#For symmetrical results, the arranged data are: omega, Re(xy), Im(xy), Re(yz), Im(yz), Re(xz), Im(xz)\n");
        for i in 0..omega.len_of(Axis(1)) {
            input_string.push_str(&format!(
                "{:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}    {:>15.8}\n",
                og[[i]],
                omega[[0, i]].re,
                omega[[0, i]].im,
                omega[[1, i]].re,
                omega[[1, i]].im,
                omega[[2, i]].re,
                omega[[2, i]].im
            ));
        }
        writeln!(berry_file, "{}", &input_string);
        writeln!(output_file.as_mut().unwrap(), "writing end, now plotting");

        //---------------------开始绘图------------------------
        let (og_min, og_max, n_og) = optical_parameter.omega();

        for (row_idx, component) in ["xx", "yy", "zz", "xy", "yz", "xz"].iter().enumerate() {
            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = metric.row(row_idx).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = metric.row(row_idx).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                &format!("{{/Symbol s}}_{{{0}}}^S (Ω^{{-1}} cm^{{-1}})", component),
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = format!("sig_{}_S.pdf", component);
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();
        }

        for (row_idx, component) in ["xy", "yz", "xz"].iter().enumerate() {
            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = omega.row(row_idx).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = omega.row(row_idx).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                &format!("{{/Symbol s}}_{{{0}}}^A (Ω^{{-1}} cm^{{-1}})", component),
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = format!("sig_{}_A.pdf", component);
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();
        }
        //开始计算 Kerr angle 以及 Faraday angle
    } else {
        let mut received_size: usize = 0;
        world.process_at_rank(0).broadcast_into(&mut received_size);

        // 根据接收到的大小分配接收缓冲区
        let mut received_data = vec![0u8; received_size];
        world
            .process_at_rank(0)
            .broadcast_into(&mut received_data[..]);
        // 反序列化
        let optical_parameter: OC_parameter = deserialize(&received_data).unwrap();
        //接受 optical_paramter 结束, 开始接受kvec
        let mut nk: usize = 0;
        world.process_at_rank(0).broadcast_into(&mut nk);
        let mut chunk_size = 0;
        world.process_at_rank(0).receive_into(&mut chunk_size);
        let mut recv_chunk = vec![0.0; chunk_size * 3];
        world.process_at_rank(0).receive_into(&mut recv_chunk);
        let chunk = Array1::from_vec(recv_chunk)
            .into_shape((chunk_size, 3))
            .unwrap();
        //接受kvec 开始计算 quantum metric
        let (metric, omega): (Array2<Complex<f64>>, Array2<Complex<f64>>) =
            Optical_conductivity(&model, &chunk, optical_parameter);

        //先将 metric 序列化并传输回rank0
        let mut serialized_data = serialize(&metric).unwrap();
        let mut data_size = serialized_data.len();
        world.process_at_rank(0).send(&mut data_size);
        world.process_at_rank(0).send(&mut serialized_data[..]);

        //再传输 omega 到rank0
        let mut serialized_data = serialize(&omega).unwrap();
        let mut data_size = serialized_data.len();
        world.process_at_rank(0).send(&mut data_size);
        world.process_at_rank(0).send(&mut serialized_data[..]);
    }
}
