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

use ndarray::*;
use ndarray_linalg::*;
use std::cmp;

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
