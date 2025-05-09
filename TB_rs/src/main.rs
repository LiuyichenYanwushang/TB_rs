//! This module implements a parallel computational framework for calculating transport properties of materials using tight-binding models.
//!
//! Core Functionalities:
//! - Band structure generation along user-defined k-paths
//! - Optical conductivity calculations via Kubo formula (intra-band and inter-band contributions)
//! - Anomalous Hall conductivity and Kerr angle computations
//! - MPI-based distributed computing with load balancing
//! - Automated generation of publication-quality PDF figures
//!
//! Input/Output:
//! - Reads parameters from `TB.in` configuration file
//! - Writes results to `TB.out` with detailed computation logs
//! - Generates band structure (.pdf) and conductivity tensor data files
//!
//! Technical Highlights:
//! - MPI parallelization for k-space sampling
//! - Bincode serialization for efficient IPC
//! - Built-in crystal symmetry analysis
//! - Smart input parsing with comment filtering

pub mod anomalous_Hall_conductivity;
pub mod band_plot;
pub mod cons;
pub mod optical_conductivity;
pub mod read_model;
pub mod spin_current;
pub mod berry_curvature_dipole_conductivity;
use crate::anomalous_Hall_conductivity::*;
use crate::band_plot::k_path;
use crate::cons::spin_direction;
use crate::optical_conductivity::OC_parameter;
use crate::optical_conductivity::Optical_conductivity;
use crate::spin_current::{spin_current_conductivity, SC_parameter};
use crate::berry_curvature_dipole_conductivity::{berry_curvature_dipole_conductivity_calculate, BCD_parameter};
use bincode::{deserialize, serialize};
use gnuplot::AutoOption::*;
use gnuplot::AxesCommon;
use gnuplot::Tick::*;
use gnuplot::{Caption, Color, Figure, Font, LineStyle, Solid};
use mpi::request::WaitGuard;
use mpi::traits::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
//用来压缩数据
use crate::band_plot::band_plot;
use crate::optical_conductivity::optical_conductivity_calculate;
use crate::anomalous_Hall_conductivity::anomalous_Hall_conductivity_calculate;
use crate::spin_current::spin_current_calculate;

use crate::read_model::read_model;
use crate::read_model::TB_file;
use std::io::Read;
use Rustb::phy_const::*;
#[derive(Debug)]
struct Control {
    band_plot: bool,
    optical_conductivity: bool,
    anomalous_Hall_conductivity: bool,
    spin_current_conductivity: bool,
    berry_curvature_dipole_conductivity:bool,
}

impl Control {
    pub fn new() -> Self {
        Control {
            band_plot: false,
            optical_conductivity: false,
            anomalous_Hall_conductivity: false,
            spin_current_conductivity: false,
            berry_curvature_dipole_conductivity:false,
        }
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

    let mut output_file: Option<File> = if rank == 0 {
        let file = File::create("TB.out").expect("Failed to create TB.out");
        writeln!(&file, "The copyright belongs to Yichen Liu.\n").unwrap();
        Some(file) // Rank 0 持有 Some(File)
    } else {
        None // 其他进程为 None
    };

    //读取 input 结束
    let reader = BufReader::new(TB_file);
    let mut Input_reads: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        Input_reads.push(line.clone());
    }
    Input_reads.iter_mut().for_each(|s| {
        // 原来的处理：找到并截断到'!'、'#'、或 "//"的位置
        if let Some(pos) = [s.find('!'), s.find('#'), s.find("//")]
            .iter()
            .filter_map(|&x| x)
            .min()
        {
            s.truncate(pos);
        }
        // 现在处理分号结尾的情况
        if s.ends_with(';') {
            s.truncate(s.len() - 1);
        }
    });

    //初始化各种控制语句
    let mut seed_name = TB_file::new();
    let mut control = Control::new();

    //开始读取
    for i in Input_reads.iter() {
        if i.contains("seed_name") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 {
                seed_name.seed_name = Some(String::from(parts[1].trim()));
                if let Some(file) = &mut output_file {
                    println!("seed_name: {}", parts[1].trim());
                    writeln!(
                        file,
                        "The seed_name of tight-binding model is {}",
                        parts[1].trim()
                    )
                    .unwrap();
                }
            }
        }
        if i.contains("fermi_energy") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 {
                seed_name.fermi_energy = parts[1].trim().parse::<f64>().unwrap();
                if let Some(file) = &mut output_file {
                    println!("fermi energy: {}", parts[1].trim());
                    writeln!(
                        file,
                        "The fermi energy of tight-binding model is {}",
                        parts[1].trim()
                    )
                    .unwrap();
                }
            }
        }
        if i.contains("band_plot") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 && (parts[1].contains("T") || parts[1].contains("t")) {
                control.band_plot = true;
            }
        }
        if i.contains("optical_conductivity") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 && (parts[1].contains("T") || parts[1].contains("t")) {
                control.optical_conductivity = true;
            }
        }
        if i.contains("anomalous_Hall_conductivity") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 && (parts[1].contains("T") || parts[1].contains("t")) {
                control.anomalous_Hall_conductivity = true;
            }
        }

        if i.contains("spin_current_conductivity") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 && (parts[1].contains("T") || parts[1].contains("t")) {
                control.spin_current_conductivity = true;
            }
        }

        if i.contains("berry_curvature_dipole_conductivity") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 && (parts[1].contains("T") || parts[1].contains("t")) {
                control.berry_curvature_dipole_conductivity = true;
            }
        }
    }

    //开始读取模型文件

    let model = read_model(&world, seed_name, &mut output_file);
    world.barrier();

    if control.band_plot {
        band_plot(&world, &model, &Input_reads, &mut output_file)
    }
    world.barrier();

    if control.optical_conductivity {
        optical_conductivity_calculate(&world, &model, &Input_reads, &mut output_file)
    }

    //开始给出计算 anomalous Hall conductuvity, spin Hall conductivity conductivity.

    if control.anomalous_Hall_conductivity {
        anomalous_Hall_conductivity_calculate(&world, &model, &Input_reads, &mut output_file)
    }

    if control.spin_current_conductivity {
        spin_current_calculate(&world, &model, &Input_reads, &mut output_file)
    }


    if control.berry_curvature_dipole_conductivity {
        berry_curvature_dipole_conductivity_calculate(&world, &model, &Input_reads, &mut output_file)
    }
}
