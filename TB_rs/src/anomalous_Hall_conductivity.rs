use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::Eigh;
use ndarray_linalg::UPLO;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::ops::AddAssign;
use std::time::{Duration, Instant};
use Rustb::Model;
use Rustb::Gauge;

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

/// This module calculates the optical conductivity
/// The adopted definition is
/// $$\sigma_{\ap\bt}=\f{2ie^2\hbar}{V}\sum_{\bm k}\sum_{n} f_n (g_{n,\ap\bt}+\f{i}{2}\Og_{n,\ap\bt})$$
///
/// Where
/// $$\\begin{aligned}
/// \Og_{n\ap\bt}&=\sum_{m=\not n}\f{-2 \text{Im} \bra{\psi_{n\bm k}}\p_\ap H\ket{\psi_{m\bm k}}\bra{\psi_{m\bm k}}\p_\bt H\ket{\psi_{n\bm k}}}{(\ve_{n\bm k}-\ve_{m\bm k})^2}
/// \\end{aligned}
/// $$

pub fn anomalous_Hall_onek<S: Data<Elem = f64>>(
    model: &Model,
    k_vec: &ArrayBase<S, Ix1>,
    T: f64,
    mu: &Array1<f64>,
) -> Array2<f64> {
    //!给定一个 k 点, 指定 dir_1=$\alpha$, dir_2=$\beta$, T 代表温度, og= $\og$,
    //!mu=$\mu$ 为费米能级范围, spin=0,1,2,3 为$\sg_0,\sg_x,\sg_y,\sg_z$,
    //!当体系不存在自旋的时候无论如何输入spin都默认 spin=0
    //! 这个函数返回的是
    //! $$ \sum_n f_n\Omega_{n,\ap\bt}^\gm(\bm k)=\sum_n \f{1}{e^{(\ve_{n\bm k}-\mu)/T/k_B}+1} \sum_{m=\not n}\f{J_{\ap,nm}^\gm v_{\bt,mn}}{(\ve_{n\bm k}-\ve_{m\bm k})^2}$$
    //! 其中 $J_\ap^\gm=\\{s_\gm,v_\ap\\}$
    let li: Complex<f64> = 1.0 * Complex::i();
    //产生速度算符A和哈密顿量
    let (mut A, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) = model.gen_v(k_vec,Gauge::Lattice);
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
    let A_xy = (&A_xy * &UU0).sum_axis(Axis(1));
    let A_yz = (&A_yz * &UU0).sum_axis(Axis(1));
    let A_xz = (&A_xz * &UU0).sum_axis(Axis(1));

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
