use ndarray::*;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
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
            eta: 1e-8,
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
            return false;
        } else {
            self.omega_min = omega_min.unwrap();
            self.omega_max = omega_max.unwrap();
            self.omega_num = omega_num.unwrap();
            return true;
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
                    i0+=1;
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

#[inline(always)]
pub fn optical_geometry_onek<S: Data<Elem = f64>>(
    model: &Model,
    k_vec: &ArrayBase<S, Ix1>,
    T: f64,
    mu: f64,
    og: &ArrayBase<S, Ix1>,
    eta: f64,
) -> (Array2<Complex<f64>>, Array2<Complex<f64>>) {
    /// This function calculates $g_{\ap\bt}$ and $\og_{\ap\bt}$
    ///
    /// `og` represents the frequency
    ///
    /// `eta` is a small quantity
    ///
    /// `T` is the temperature
    ///
    /// `k_vec` is the k vector
    ///
    ///直接计算 xx, yy, zz, xy, yz, xz 这六个量的光电导, 分为对称和反对称部分.
    ///
    ///输出格式为 ($\sigma_{ab}^S$, $\sigma_{ab}^A), 这里 S 和 A 表示 symmetry and antisymmetry.
    ///
    ///$sigma_{ab}^S$ 是 $6\times n_\omega$
    let li: Complex<f64> = 1.0 * Complex::i();
    let (band, evec) = model.solve_onek(&k_vec);

    let mut v: Array3<Complex<f64>> = model.gen_v(k_vec);

    let evec_conj: Array2<Complex<f64>> = evec.mapv(|x| x.conj());
    let evec = evec.t();

    // Calculate the energy differences and their inverses
    let mut U0 = Array2::zeros((model.nsta(), model.nsta()));
    let mut Us = Array2::zeros((model.nsta(), model.nsta()));
    let nsta = band.len();
    for i in 0..nsta {
        for j in 0..nsta {
            let a = band[[i]] - band[[j]];
            U0[[i, j]] = Complex::new(a, 0.0);
            Us[[i, j]] = if a.abs() > 1e-6 {
                Complex::new(1.0 / a, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
    }
    let U0 = U0;
    let Us = Us;

    let fermi_dirac = if T == 0.0 {
        band.mapv(|x| if x > mu { 0.0 } else { 1.0 })
    } else {
        let beta = 1.0 / T / 8.617e-5;
        band.mapv(|x| ((beta * (x - mu)).exp() + 1.0).recip())
    };
    let fermi_dirac = fermi_dirac.mapv(|x| Complex::new(x, 0.0));

    let n_og = og.len();
    assert_eq!(
        band.len(),
        nsta,
        "this is strange for band's length is not equal to nsta"
    );

    let mut matric_n = Array2::zeros((6, n_og));
    let mut omega_n = Array2::zeros((3, n_og));

    let mut A = Array3::zeros((3, nsta, nsta));
    //transfrom the basis into bolch state
    Zip::from(A.outer_iter_mut())
        .and(v.outer_iter())
        .apply(|mut a, v| a.assign(&evec_conj.dot(&v.dot(&evec))));

    let A_xx = &A.slice(s![0, .., ..]) * &A.slice(s![0, .., ..]).t();
    let A_yy = &A.slice(s![1, .., ..]) * &A.slice(s![1, .., ..]).t();
    let A_zz = &A.slice(s![2, .., ..]) * &A.slice(s![2, .., ..]).t();
    let A_xy = &A.slice(s![0, .., ..]) * &A.slice(s![1, .., ..]).t();
    let A_yz = &A.slice(s![1, .., ..]) * &A.slice(s![2, .., ..]).t();
    let A_xz = &A.slice(s![0, .., ..]) * &A.slice(s![2, .., ..]).t();
    let re_xx: Array2<Complex<f64>> = Complex::new(2.0, 0.0) * A_xx;
    let re_yy: Array2<Complex<f64>> = Complex::new(2.0, 0.0) * A_yy;
    let re_zz: Array2<Complex<f64>> = Complex::new(2.0, 0.0) * A_zz;
    let Complex { re, im } = A_xy.view().split_complex();
    let re_xy: Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0 * x, 0.0));
    let im_xy: Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0 * x));
    let Complex { re, im } = A_yz.view().split_complex();
    let re_yz: Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0 * x, 0.0));
    let im_yz: Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0 * x));
    let Complex { re, im } = A_xz.view().split_complex();
    let re_xz: Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0 * x, 0.0));
    let im_xz: Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0 * x));

    // Calculate the matrices for each frequency
    Zip::from(omega_n.axis_iter_mut(Axis(1)))
        .and(matric_n.axis_iter_mut(Axis(1)))
        .and(og.view())
        .apply(|mut omega, mut matric, a0| {
            let li_eta = a0 + li * eta;
            let UU = U0.mapv(|x| (x * x - li_eta * li_eta).finv());
            let U1 = &UU * &Us * li_eta;
            let o_xy = im_xy
                .outer_iter()
                .zip(UU.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect();
            let o_yz = im_yz
                .outer_iter()
                .zip(UU.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect();
            let o_xz = im_xz
                .outer_iter()
                .zip(UU.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect();
            let m_xx = re_xx
                .outer_iter()
                .zip(U1.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect();
            let m_yy = re_yy
                .outer_iter()
                .zip(U1.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect();
            let m_zz = re_zz
                .outer_iter()
                .zip(U1.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect();
            let m_xy = re_xy
                .outer_iter()
                .zip(U1.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect();
            let m_yz = re_yz
                .outer_iter()
                .zip(U1.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect();
            let m_xz = re_xz
                .outer_iter()
                .zip(U1.outer_iter())
                .map(|(a, b)| a.dot(&b))
                .collect();
            let o_xy = Array1::from_vec(o_xy).dot(&fermi_dirac);
            let o_yz = Array1::from_vec(o_yz).dot(&fermi_dirac);
            let o_xz = Array1::from_vec(o_xz).dot(&fermi_dirac);
            let m_xx = Array1::from_vec(m_xx).dot(&fermi_dirac);
            let m_yy = Array1::from_vec(m_yy).dot(&fermi_dirac);
            let m_zz = Array1::from_vec(m_zz).dot(&fermi_dirac);
            let m_xy = Array1::from_vec(m_xy).dot(&fermi_dirac);
            let m_yz = Array1::from_vec(m_yz).dot(&fermi_dirac);
            let m_xz = Array1::from_vec(m_xz).dot(&fermi_dirac);
            matric[[0]] = m_xx;
            matric[[1]] = m_yy;
            matric[[2]] = m_zz;
            matric[[3]] = m_xy;
            matric[[4]] = m_yz;
            matric[[5]] = m_xz;
            omega[[0]] = o_xy;
            omega[[1]] = o_yz;
            omega[[2]] = o_xz;
        });
    (matric_n, omega_n)
}

pub fn Optical_conductivity(
    model: &Model,
    kvec: &Array2<f64>,
    optical_parameter: Op_conductivity,
) -> (Array2<Complex<f64>>, Array2<Complex<f64>>) {
    let eta = optical_parameter.eta();
    let T = optical_parameter.temperature();
    let mu = optical_parameter.mu();
    let og = optical_parameter.og();
    let (og_min, og_max, n_og) = optical_parameter.omega();
    let mut matric = Array2::zeros((6, n_og));
    let mut omega = Array2::zeros((3, n_og));
    for k in kvec.outer_iter() {
        let (matric_n, omega_n) = optical_geometry_onek(&model, &k, T, mu, &og.view(), eta);
        matric += &matric_n;
        omega += &omega_n;
    }
    (matric, omega)
}
