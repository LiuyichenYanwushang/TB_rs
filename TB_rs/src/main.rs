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
pub mod spin_current;
use crate::anomalous_Hall_conductivity::*;
use crate::band_plot::k_path;
use crate::cons::spin_direction;
use crate::optical_conductivity::OC_parameter;
use crate::optical_conductivity::Optical_conductivity;
use crate::spin_current::{spin_current_conductivity, SC_parameter};
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
use Rustb::phy_const::*;
use Rustb::*;

#[derive(Debug)]
struct Control {
    band_plot: bool,
    optical_conductivity: bool,
    anomalous_Hall_conductivity: bool,
    spin_current_conductivity: bool,
}

impl Control {
    pub fn new() -> Self {
        Control {
            band_plot: false,
            optical_conductivity: false,
            anomalous_Hall_conductivity: false,
            spin_current_conductivity: false,
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
                if rank == 0 {
                    println!("seed_name: {}", parts[1].trim());
                }

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
                if rank == 0 {
                    println!("fermi energy: {}", parts[1].trim());
                }
                writeln!(
                    output_file,
                    "The fermi energy of tight-binding model is {}",
                    parts[1].trim()
                );
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
            writeln!(output_file, "calculation finished");
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

    if control.optical_conductivity {
        //开始计算光电导
        if rank == 0 {
            writeln!(output_file, "start calculatiing the optical conductivity");
            println!("start calculatiing the optical conductivity");
            let mut optical_parameter = OC_parameter::new();
            let have_kmesh = optical_parameter.get_k_mesh(&Input_reads);
            if !have_kmesh {
                writeln!(
                    output_file,
                    "Error: You mut set k_mesh for calculating optical conductivity"
                );
            }
            let have_T = optical_parameter.get_T(&Input_reads);
            let have_mu = optical_parameter.get_mu(&Input_reads);
            let have_eta = optical_parameter.get_eta(&Input_reads);
            let have_omega = optical_parameter.get_omega(&Input_reads);
            if !(have_omega) {
                writeln!(
                    output_file,
                    "Error: You must specify frequenccy when calculate the optical conductivity"
                );
                panic!(
                    "Error: You must specify  frequenccy when calculate the optical conductivity"
                )
            }

            if !(have_T) {
                writeln!(output_file,"Warning: You don't specify temperature when calculate the optical conductivity, using default 0.0");
            }
            if !(have_mu) {
                writeln!(output_file,"Warning: You don't specify chemistry potential when calculate the optical conductivity, using default 0.0");
            }
            if !(have_eta) {
                writeln!(output_file,"Warning: You don't specify broaden_energy when calculate the optical conductivity, using default 0.0");
            }
            let kvec = optical_parameter.get_mesh_vec();

            //传输 optical_parameter
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
            writeln!(output_file, "calculation finished");
            //开始写入
            writeln!(
                output_file,
                "write data in optical_conductivity_A.dat and optical_conductivity_S.dat"
            );
            let mut metric_file =
                File::create("optical_conductivity_S.dat").expect("Unable to create TB.in");
            let mut input_string = String::new();
            input_string.push_str("#Calculation results are reported in units of Ω^-1 cm^-1\n");
            input_string.push_str("#For symmetrical results, the arranged data are: omega, Re(xx, Im(xx), Re(yy), Im(yy), Re(zz), Im(zz), Re(xy), Im(xy), Re(yz), Im(yz), Re(xz), Im(xz)\n");
            for i in 0..metric.len_of(Axis(1)) {
                input_string.push_str(&format!("{:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}\n",og[[i]],metric[[0,i]].re,metric[[0,i]].im,metric[[1,i]].re,metric[[1,i]].im,metric[[2,i]].re,metric[[2,i]].im,metric[[3,i]].re,metric[[3,i]].im,metric[[4,i]].re,metric[[4,i]].im,metric[[5,i]].re,metric[[5,i]].im));
            }
            writeln!(metric_file, "{}", &input_string);

            let mut berry_file =
                File::create("optical_conductivity_A.dat").expect("Unable to create TB.in");
            let mut input_string = String::new();

            input_string.push_str("#Calculation results are reported in units of Ω^-1 cm^-1\n");
            input_string.push_str("#For symmetrical results, the arranged data are: omega, Re(xy), Im(xy), Re(yz), Im(yz), Re(xz), Im(xz)\n");
            for i in 0..omega.len_of(Axis(1)) {
                input_string.push_str(&format!(
                    "{:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}\n",
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
            writeln!(output_file, "writing end, now plotting");

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

    //开始给出计算 anomalous Hall conductuvity, spin Hall conductivity conductivity.

    if control.anomalous_Hall_conductivity {
        //开始计算反常Hall 电导
        if rank == 0 {
            writeln!(
                output_file,
                "start calculatiing the anomalous Hall conductivity"
            );
            println!("start calculatiing the anomalous Hall");
            let mut ahc_parameter = AHC_parameter::new();
            let have_kmesh = ahc_parameter.get_k_mesh(&Input_reads);
            if !have_kmesh {
                writeln!(
                    output_file,
                    "Error: You mut set k_mesh for calculating anomalous Hall conductivity"
                );
            }
            let have_T = ahc_parameter.get_T(&Input_reads);
            let have_mu = ahc_parameter.get_mu(&Input_reads);

            if !(have_T) {
                writeln!(output_file,"Warning: You don't specify temperature when calculate the anomalous hall conductivity, using default 0.0");
            }
            if !(have_mu) {
                writeln!(output_file,"Warning: You don't specify chemistry potential when calculate the anomalous hall conductivity, using default 0.0");
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
            conductivity = conductivity / (nk as f64) / model.lat.det().unwrap()
                * Quantum_conductivity
                * 1.0e8;
            println!("The ahc conductivity calculation is finished");
            writeln!(output_file, "calculation finished");
            //开始写入
            writeln!(
                output_file,
                "write data in ahc_conductivity_A.dat and ahc_conductivity_S.dat"
            );
            let mut AHC_file = File::create("ahc_conductivity.dat")
                .expect("Unable to create ahc_conductivity.dat");
            let mut input_string = String::new();
            input_string.push_str("#Calculation results are reported in units of Ω^-1 cm^-1\n");
            input_string.push_str("#The arranged data are: mu,  omega_xy, omega_yz, omega_xz\n");
            for i in 0..conductivity.len_of(Axis(1)) {
                input_string.push_str(&format!(
                    "{:11.8}    {:11.8}    {:11.8}    {:11.8}\n",
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
            let conductivity: Array2<f64> =
                Anomalous_Hall_conductivity(&model, &chunk, ahc_parameter);

            //传输 conductivity 到rank0
            let mut serialized_data = serialize(&conductivity).unwrap();
            let mut data_size = serialized_data.len();
            world.process_at_rank(0).send(&mut data_size);
            world.process_at_rank(0).send(&mut serialized_data[..]);
        }
    }

    if control.spin_current_conductivity {
        //开始计算反常Hall 电导
        if rank == 0 {
            writeln!(
                output_file,
                "start calculatiing the spin current conductivity"
            );
            println!("start calculatiing the spin current");
            let mut sc_parameter = SC_parameter::new();
            let have_kmesh = sc_parameter.get_k_mesh(&Input_reads);
            if !have_kmesh {
                writeln!(
                    output_file,
                    "Error: You mut set k_mesh for calculating spin current conductivity"
                );
            }
            let have_eta = sc_parameter.get_eta(&Input_reads);
            let have_mu = sc_parameter.get_mu(&Input_reads);
            let have_spin_direction = sc_parameter.get_spin(&Input_reads);

            if !(have_eta) {
                writeln!(output_file,"Warning: You don't specify smooth energy when calculate the anomalous hall conductivity, using default 0.001");
            }
            if !(have_mu) {
                writeln!(output_file,"Warning: You don't specify chemistry potential when calculate the anomalous hall conductivity, using default 0.0");
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
            writeln!(output_file, "calculation finished");
            //-------------------------------开始写入-----------------------
            writeln!(output_file, "write data in spin_current_conductivity.dat");
            let mut SC_file = File::create("spin_current_conductivity.dat")
                .expect("Unable to create spin_current_conductivity.dat");
            let mut input_string = String::new();
            input_string.push_str("#Calculation results are reported in units of Ω^-1 cm^-1\n");
            input_string.push_str("#The arranged data are: mu,  xx,  yy,  zz,  xy,  yz,  xz\n");
            for i in 0..conductivity.len_of(Axis(1)) {
                input_string.push_str(&format!(
                    "{:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}\n",
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
}
