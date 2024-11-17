pub mod band_plot;
pub mod optical_conductivity;
use crate::band_plot::k_path;
use crate::optical_conductivity::Op_conductivity;
use crate::optical_conductivity::Optical_conductivity;
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
use std::cmp;
///这个程序是类似wanniertools 的代码, 但是实现了一些输运相关的代码, 效率相对更高
///主程序主要用来实现对控制文件的读取, 控制文件默认叫做 TB.in
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
    kerr_angle: bool,
}
impl Control {
    pub fn new() -> Self {
        Control {
            band_plot: false,
            optical_conductivity: false,
            kerr_angle: false,
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
                    println!("seed_name: {}", parts[1].trim());
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
            if parts.len() == 2 {
                if parts[1].contains("T") || parts[1].contains("t") {
                    control.band_plot = true;
                }
            }
        }
        if i.contains("optical_conductivity") {
            let parts: Vec<&str> = i.split('=').collect();
            if parts.len() == 2 {
                if parts[1].contains("T") || parts[1].contains("t") {
                    control.optical_conductivity = true;
                }
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
            let mut optical_parameter = Op_conductivity::new();
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
            let (mut matric, mut omega): (Array2<Complex<f64>>, Array2<Complex<f64>>) =
                Optical_conductivity(&model, &chunk, optical_parameter);

            for i in 1..size {
                let mut received_size: usize = 0;
                world.process_at_rank(i).receive_into(&mut received_size);
                let mut received_data = vec![0u8; received_size];
                world
                    .process_at_rank(i)
                    .receive_into(&mut received_data[..]);
                // 反序列化
                let matric0: Array2<Complex<f64>> = deserialize(&received_data).unwrap();
                matric = matric + matric0;

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
            matric = matric / (nk as f64) / model.lat.det().unwrap() * Quantum_conductivity * 1.0e8;
            println!("The optical conductivity calculation is finished");
            writeln!(output_file, "calculation finished");
            //开始写入
            writeln!(
                output_file,
                "write data in optical_conductivity_A.dat and optical_conductivity_S.dat"
            );
            let mut matric_file =
                File::create("optical_conductivity_S.dat").expect("Unable to create TB.in");
            let mut input_string = String::new();
            input_string.push_str("#Calculation results are reported in units of Ω^-1 cm^-1\n");
            input_string.push_str("#For symmetrical results, the arranged data are: omega, Re(xx, Im(xx), Re(yy), Im(yy), Re(zz), Im(zz), Re(xy), Im(xy), Re(yz), Im(yz), Re(xz), Im(xz)\n");
            for i in 0..matric.len_of(Axis(1)) {
                input_string.push_str(&format!("{:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}    {:11.8}\n",og[[i]],matric[[0,i]].re,matric[[0,i]].im,matric[[1,i]].re,matric[[1,i]].im,matric[[2,i]].re,matric[[2,i]].im,matric[[3,i]].re,matric[[3,i]].im,matric[[4,i]].re,matric[[4,i]].im,matric[[5,i]].re,matric[[5,i]].im));
            }
            writeln!(matric_file, "{}", &input_string);

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
            //开始绘图

            let (og_min, og_max, n_og) = optical_parameter.omega();
            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = matric.row(0).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = matric.row(0).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                "{/Symbol s}_{xx}^S (Ω^{-1} cm^{-1})",
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = String::new();
            pdf_name.push_str("sig_xx_S.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();

            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = matric.row(1).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = matric.row(1).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                "{/Symbol s}_{yy}^S (Ω^{-1} cm^{-1})",
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = String::new();
            pdf_name.push_str("sig_yy_S.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();

            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = matric.row(2).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = matric.row(2).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                "{/Symbol s}_{zz}^S (Ω^{-1} cm^{-1})",
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = String::new();
            pdf_name.push_str("sig_zz_S.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();

            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = matric.row(3).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = matric.row(3).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                "{/Symbol s}_{xy}^S (Ω^{-1} cm^{-1})",
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = String::new();
            pdf_name.push_str("sig_xy_S.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();

            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = matric.row(4).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = matric.row(4).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                "{/Symbol s}_{yz}^S (Ω^{-1} cm^{-1})",
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = String::new();
            pdf_name.push_str("sig_yz_S.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();

            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = matric.row(5).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = matric.row(5).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                "{/Symbol s}_{xz}^S (Ω^{-1} cm^{-1})",
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = String::new();
            pdf_name.push_str("sig_xz_S.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();

            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = omega.row(0).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = omega.row(0).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                "{/Symbol s}_{xy}^A (Ω^{-1} cm^{-1})",
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = String::new();
            pdf_name.push_str("sig_xy_A.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();

            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = omega.row(1).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = omega.row(1).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                "{/Symbol s}_{yz}^A (Ω^{-1} cm^{-1})",
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = String::new();
            pdf_name.push_str("sig_yz_A.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();

            let mut fg = Figure::new();
            let x: Vec<f64> = og.to_vec();
            let axes = fg.axes2d();
            let y: Vec<f64> = omega.row(2).to_owned().map(|x| x.re).to_vec();
            axes.lines(&x, &y, &[Color("blue"), LineStyle(Solid)]);
            let y: Vec<f64> = omega.row(2).to_owned().map(|x| x.im).to_vec();
            axes.lines(&x, &y, &[Color("red"), LineStyle(Solid)]);
            let axes = axes.set_x_range(Fix(og_min), Fix(og_max));
            axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 18.0)]);
            axes.set_y_label(
                "{/Symbol s}_{xz}^A (Ω^{-1} cm^{-1})",
                &[Font("Times New Roman", 18.0)],
            );
            axes.set_x_label("ℏω (eV)", &[Font("Times New Roman", 18.0)]);

            let mut pdf_name = String::new();
            pdf_name.push_str("sig_xz_A.pdf");
            fg.set_terminal("pdfcairo", &pdf_name);
            fg.show();
        } else {
            let mut received_size: usize = 0;
            world.process_at_rank(0).broadcast_into(&mut received_size);

            // 根据接收到的大小分配接收缓冲区
            let mut received_data = vec![0u8; received_size];
            world
                .process_at_rank(0)
                .broadcast_into(&mut received_data[..]);
            // 反序列化
            let optical_parameter: Op_conductivity = deserialize(&received_data).unwrap();
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
            //接受kvec 开始计算 quantum matric
            let (matric, omega): (Array2<Complex<f64>>, Array2<Complex<f64>>) =
                Optical_conductivity(&model, &chunk, optical_parameter);

            //先将 matric 序列化并传输回rank0
            let mut serialized_data = serialize(&matric).unwrap();
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
}
