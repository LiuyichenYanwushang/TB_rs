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

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Op_conductivity {
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

impl Op_conductivity {
    pub fn new() -> Self {
        Op_conductivity {
            k_mesh: [1, 1, 1],
            eta: 1e-3,
            omega_min: 0.0,
            omega_max: 1.0,
            omega_num: 10,
            T: 0.0,
            mu: 0.0,
        }
    }
    ///得到 k_mesh, 返回 true or false 表示是否指定了 k_mesh
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
        let mut eta = None;
        for line in reads.iter() {
            //后面会用到 chemical_potential_min, chemical_potential_max,chemical_potential_num
            if line.contains("chemical_potential")
                && !(line.contains("min") || line.contains("max") || line.contains("num"))
            {
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

    ///从控制文件中读取
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
    let (mut A, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) = model.gen_v(k_vec);
    let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
        (eigvals, eigvecs)
    } else {
        todo!()
    };
    let evec_conj = evec.t();
    let evec = evec.mapv(|x| x.conj());

    //Zip::from(A.outer_iter_mut()) .apply(|mut a| a.assign(&evec_conj.dot(&a.dot(&evec))));

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
        let nocc: usize = band.fold(0, |acc, x| if *x > 0.0 { acc } else { acc + 1 });
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
                let U1 = &uu_1.mapv(|x| (x - a0 - li * eta).finv().re / x);
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
                let U1 = &uu_2.mapv(|x| (x - a0 - li * eta).finv().re / x);
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
                let U0 = U0 / &UU;
                let U1 = &fermi
                    * &UU.mapv(|x| {
                        if x.abs() > 1e-6 {
                            (x - a0 - li * eta).finv().re
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

                /*
                for i in 0..band.len() {
                    for j in 0..band.len() {
                        let a = (UU[[i, j]] - a0) / eta;
                        let U0 = PI * (-a * a / 2.0).exp() / (2.0 * PI).sqrt() / eta / UU[[i, j]];
                        let U1 = (UU[[i, j]] * (UU[[i, j]] - a0 - li * eta)).finv().re;
                        oH[[0]] += A_xx[[i, j]] * U0;
                        oH[[1]] += A_yy[[i, j]] * U0;
                        oH[[2]] += A_zz[[i, j]] * U0;
                        oH[[3]] += A_xy[[i, j]] * U0;
                        oH[[4]] += A_yz[[i, j]] * U0;
                        oH[[5]] += A_xz[[i, j]] * U0;
                        oaH[[0]] -= A_xx[[i, j]] * U1 * li;
                        oaH[[1]] -= A_yy[[i, j]] * U1 * li;
                        oaH[[2]] -= A_zz[[i, j]] * U1 * li;
                        oaH[[3]] -= A_xy[[i, j]] * U1 * li;
                        oaH[[4]] -= A_yz[[i, j]] * U1 * li;
                        oaH[[5]] -= A_xz[[i, j]] * U1 * li;
                    }
                }
                */
            });
    }
    (omega_H, omega_aH)
}

pub fn Optical_conductivity(
    model: &Model,
    kvec: &Array2<f64>,
    optical_parameter: Op_conductivity,
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
