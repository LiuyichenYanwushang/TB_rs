use ndarray_linalg::solve::Inverse;
use pyo3::prelude::{pyclass,pymodule,PyModule,PyResult,pymethods,Py,Python,pyfunction,IntoPy,wrap_pyfunction};
use pyo3::types::PyComplex;
use numpy::{PyReadonlyArray, ToPyArray,IntoPyArray,PyArray};
use Rustb::*; // 导入 Rustb 中定义的类型和函数
use num_complex::Complex;
use ndarray::*;
///这个包是 Rustb 对python的一个绑定

// 为 tb_model 结构体添加一个生命周期参数 'a
#[pyclass]
#[derive(Clone, Debug)]
pub struct Model {
    #[pyo3(get, set)]
    pub dim_r: usize,
    #[pyo3(get, set)]
    pub spin: bool,
    #[pyo3(get, set)]
    pub lat: Py<PyArray<f64,Ix2>>,
    #[pyo3(get, set)]
    pub orb: Py<PyArray<f64,Ix2>>,
    #[pyo3(get, set)]
    pub orb_projection: Vec<OrbProj>,
    #[pyo3(get, set)]
    pub atoms: Vec<Atom>,
    #[pyo3(get, set)]
    pub ham: Py<PyArray<Complex<f64>,Ix3>>,
    #[pyo3(get, set)]
    pub hamR: Py<PyArray<isize,Ix2>>,
    #[pyo3(get, set)]
    pub rmatrix: Py<PyArray<Complex<f64>,Ix4>>,
}
#[pyclass]
#[derive(Debug, Clone)]
pub struct Atom {
    #[pyo3(get, set)]
    pub position: Py<PyArray<f64,Ix1>>,
    #[pyo3(get, set)]
    pub name: AtomType,
    #[pyo3(get, set)]
    pub atom_list: usize
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrbProj {
    s,
    px,
    py,
    pz,
    dxy,
    dyz,
    dxz,
    dz2,
    dx2y2,
    fz3,
    fxz2,
    fyz2,
    fzx2y2,
    fxyz,
    fxx23y2,
    fy3x2y2,
    sp_1,
    sp_2,
    sp2_1,
    sp2_2,
    sp2_3,
    sp3_1,
    sp3_2,
    sp3_3,
    sp3_4,
    sp3d_1,
    sp3d_2,
    sp3d_3,
    sp3d_4,
    sp3d_5,
    sp3d2_1,
    sp3d2_2,
    sp3d2_3,
    sp3d2_4,
    sp3d2_5,
    sp3d2_6,
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AtomType {
    H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn, Fr, Ra,
}

/*
#[pyfunction]
pub fn from_hr<'py>(path:&str,file_name:&str,zero_energy:f64)->PyResult<tb_model>{
    let new_tb_model =Python::with_gil(|py|{
        let new_model=Model::from_hr(path,file_name,zero_energy);
        let a=tb_model {
            dim_r: 3,
            norb: new_model.norb,
            nsta: new_model.nsta,
            natom: new_model.natom,
            spin: new_model.spin,
            // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
            lat: new_model.lat.to_pyarray(py).into(),
            orb: new_model.orb.to_pyarray(py).into(),
            atom: new_model.atom.to_pyarray(py).into(),
            atom_list: new_model.atom_list,
            ham: new_model.ham.to_pyarray(py).into(),
            hamR: new_model.hamR.to_pyarray(py).into(),
            rmatrix: new_model.rmatrix.to_pyarray(py).into(),
        };
    // 把新的 tb_model 实例用 PyResult 包装并返回给 Python
        Ok(a)
    });
    new_tb_model
}

#[pymethods]
impl tb_model{
    // 去掉多余的冒号，并且添加返回类型
    #[new]
    pub fn new<'py>(dim_r: usize, lat: PyReadonlyArray<f64,Ix2>, orb: PyReadonlyArray<f64,Ix2>, spin: bool, atom: Option<PyReadonlyArray<f64,Ix2>>, atom_list: Option<Vec<usize>>) -> PyResult<Self> {
        let new_tb_model =Python::with_gil(|py|{
        // 把 PyReadonlyArray 转换成 ndarray 的 Array2
        let lat = lat.as_array().to_owned();
        let orb = orb.as_array().to_owned();
        let new_model =match atom{
            None=>{Model::tb_model(dim_r, lat, orb, spin, None, None)},
            Some(ref array)=>{
            let atom=atom.unwrap().as_array().to_owned();
            let atom_list=atom_list.unwrap();
            Model::tb_model(dim_r, lat, orb, spin, Some(atom), Some(atom_list))
            },
        };
        // 创建一个新的 tb_model 实例，并且用 PyResult 包装它
         let a=tb_model{
            dim_r: dim_r,
            norb: new_model.norb,
            nsta: new_model.nsta,
            natom: new_model.natom,
            spin: spin,
            // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
            lat: new_model.lat.to_pyarray(py).into(),
            orb: new_model.orb.to_pyarray(py).into(),
            atom: new_model.atom.to_pyarray(py).into(),
            atom_list: new_model.atom_list,
            ham: new_model.ham.to_pyarray(py).into(),
            hamR: new_model.hamR.to_pyarray(py).into(),
            rmatrix: new_model.rmatrix.to_pyarray(py).into(),
        };
        // 把新的 tb_model 实例用 PyResult 包装并返回给 Python
        Ok(a)});
        new_tb_model
    }



    pub fn add_hop(&mut self,tmp:Py<PyComplex>,ind_i:usize,ind_j:usize,R:PyReadonlyArray<isize,Ix1>,pauli:isize){
        //!参数和 set_hop 一致, 但是 $\bra{i\bm 0}\hat H\ket{j\bm R}$+=tmp
        Python::with_gil(|py|{
            let mut new_model=unsafe{ Model{
                dim_r:self.dim_r,
                norb: self.norb,
                nsta: self.nsta,
                natom: self.natom,
                spin: self.spin,
                // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
                lat: self.lat.as_ref(py).as_array().to_owned(),
                orb: self.orb.as_ref(py).as_array().to_owned(),
                atom: self.atom.as_ref(py).as_array().to_owned(),
                atom_list: self.atom_list.clone(),
                ham: self.ham.as_ref(py).as_array().to_owned(),
                hamR: self.hamR.as_ref(py).as_array().to_owned(),
                rmatrix: self.rmatrix.as_ref(py).as_array().to_owned(),
            }};
            let tmp=tmp.into_ref(py);
            let tmp=Complex::new(tmp.real(),tmp.imag());
            new_model.add_hop(tmp,ind_i,ind_j,&R.as_array().to_owned(),pauli);
            self.ham= new_model.ham.to_pyarray(py).into();
            self.hamR= new_model.hamR.to_pyarray(py).into();
            self.rmatrix= new_model.rmatrix.to_pyarray(py).into();
        });
    }


    pub fn set_hop(&mut self,tmp:Py<PyComplex>,ind_i:usize,ind_j:usize,R:PyReadonlyArray<isize,Ix1>,pauli:isize){
        //!参数和 set_hop 一致, 但是 $\bra{i\bm 0}\hat H\ket{j\bm R}$+=tmp
        Python::with_gil(|py|{
            let mut new_model=unsafe{ Model{
                dim_r:self.dim_r,
                norb: self.norb,
                nsta: self.nsta,
                natom: self.natom,
                spin: self.spin,
                // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
                lat: self.lat.as_ref(py).as_array().to_owned(),
                orb: self.orb.as_ref(py).as_array().to_owned(),
                atom: self.atom.as_ref(py).as_array().to_owned(),
                atom_list: self.atom_list.clone(),
                ham: self.ham.as_ref(py).as_array().to_owned(),
                hamR: self.hamR.as_ref(py).as_array().to_owned(),
                rmatrix: self.rmatrix.as_ref(py).as_array().to_owned(),
            }};
            let tmp=tmp.into_ref(py);
            let tmp=Complex::new(tmp.real(),tmp.imag());
            new_model.set_hop(tmp,ind_i,ind_j,&R.as_array().to_owned(),pauli);
            self.ham= new_model.ham.to_pyarray(py).into();
            self.hamR= new_model.hamR.to_pyarray(py).into();
            self.rmatrix= new_model.rmatrix.to_pyarray(py).into();
        });
    }


    pub fn set_onsite(&mut self,onsite:PyReadonlyArray<f64,Ix1>,pauli:isize){
        //!参数和 set_hop 一致, 但是 $\bra{i\bm 0}\hat H\ket{j\bm R}$+=tmp
        Python::with_gil(|py|{
            let mut new_model=unsafe{ Model{
                dim_r:self.dim_r,
                norb: self.norb,
                nsta: self.nsta,
                natom: self.natom,
                spin: self.spin,
                // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
                lat: self.lat.as_ref(py).as_array().to_owned(),
                orb: self.orb.as_ref(py).as_array().to_owned(),
                atom: self.atom.as_ref(py).as_array().to_owned(),
                atom_list: self.atom_list.clone(),
                ham: self.ham.as_ref(py).as_array().to_owned(),
                hamR: self.hamR.as_ref(py).as_array().to_owned(),
                rmatrix: self.rmatrix.as_ref(py).as_array().to_owned(),
            }};
            let onsite=onsite.as_array().to_owned();
            new_model.set_onsite(&onsite,pauli);
            self.ham= new_model.ham.to_pyarray(py).into();
            self.hamR= new_model.hamR.to_pyarray(py).into();
            self.rmatrix= new_model.rmatrix.to_pyarray(py).into();
        });
    }
    pub fn del_hop(&mut self,ind_i:usize,ind_j:usize,R:PyReadonlyArray<isize,Ix1>,pauli:isize){
        //!参数和 set_hop 一致, 但是 $\bra{i\bm 0}\hat H\ket{j\bm R}$+=tmp
        Python::with_gil(|py|{
            let mut new_model=unsafe{ Model{
                dim_r:self.dim_r,
                norb: self.norb,
                nsta: self.nsta,
                natom: self.natom,
                spin: self.spin,
                // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
                lat: self.lat.as_ref(py).as_array().to_owned(),
                orb: self.orb.as_ref(py).as_array().to_owned(),
                atom: self.atom.as_ref(py).as_array().to_owned(),
                atom_list: self.atom_list.clone(),
                ham: self.ham.as_ref(py).as_array().to_owned(),
                hamR: self.hamR.as_ref(py).as_array().to_owned(),
                rmatrix: self.rmatrix.as_ref(py).as_array().to_owned(),
            }};
            new_model.del_hop(ind_i,ind_j,&R.as_array().to_owned(),pauli);
            self.ham= new_model.ham.to_pyarray(py).into();
            self.hamR= new_model.hamR.to_pyarray(py).into();
            self.rmatrix= new_model.rmatrix.to_pyarray(py).into();
        });
    }
    pub fn k_path(&mut self,path:PyReadonlyArray<f64,Ix2>,nk:usize)->(Py<PyArray<f64,Ix2>>,Py<PyArray<f64,Ix1>>,Py<PyArray<f64,Ix1>>){
        Python::with_gil(|py|{
            let new_model=unsafe{ Model{
                dim_r:self.dim_r,
                norb: self.norb,
                nsta: self.nsta,
                natom: self.natom,
                spin: self.spin,
                // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
                lat: self.lat.as_ref(py).as_array().to_owned(),
                orb: self.orb.as_ref(py).as_array().to_owned(),
                atom: self.atom.as_ref(py).as_array().to_owned(),
                atom_list: self.atom_list.clone(),
                ham: self.ham.as_ref(py).as_array().to_owned(),
                hamR: self.hamR.as_ref(py).as_array().to_owned(),
                rmatrix: self.rmatrix.as_ref(py).as_array().to_owned(),
            }};
            //let (kvec,kdist,knode)=new_model.k_path(&path.as_array().to_owned(),nk);
            let path=path.as_array().to_owned();
            let (kvec,kdist,knode)={

                if new_model.dim_r==0{
                    panic!("the k dimension of the model is 0, do not use k_path")
                }
                let n_node:usize=path.len_of(Axis(0));
                if new_model.dim_r != path.len_of(Axis(1)){
                    panic!("Wrong, the path's length along 1 dimension must equal to the model's dimension")
                }
                let k_metric=(new_model.lat.dot(&new_model.lat.t())).inv().unwrap();
                let mut k_node=Array1::<f64>::zeros(n_node);
                for n in 1..n_node{
                    let dk=path.row(n).to_owned()-path.slice(s![n-1,..]).to_owned();
                    let a=k_metric.dot(&dk);
                    let dklen=dk.dot(&a).sqrt();
                    k_node[[n]]=k_node[[n-1]]+dklen;
                }
                let mut node_index:Vec<usize>=vec![0];
                for n in 1..n_node-1{
                    let frac=k_node[[n]]/k_node[[n_node-1]];
                    let a=(frac*((nk-1) as f64).round()) as usize;
                    node_index.push(a)
                }
                node_index.push(nk-1);
                let mut k_dist=Array1::<f64>::zeros(nk);
                let mut k_vec=Array2::<f64>::zeros((nk,new_model.dim_r));
                k_vec.row_mut(0).assign(&path.row(0));
                for n in 1..n_node {
                    let n_i=node_index[n-1];
                    let n_f=node_index[n];
                    let kd_i=k_node[[n-1]];
                    let kd_f=k_node[[n]];
                    let k_i=path.row(n-1);
                    let k_f=path.row(n);
                    for j in n_i..n_f+1{
                        let frac:f64= ((j-n_i) as f64)/((n_f-n_i) as f64);
                        k_dist[[j]]=kd_i + frac*(kd_f-kd_i);
                        k_vec.row_mut(j).assign(&((1.0-frac)*k_i.to_owned() +frac*k_f.to_owned()));

                    }
                }
                (k_vec,k_dist,k_node)
            };
            let kvec: Py<PyArray<f64, Ix2>> = Py::from(kvec.into_pyarray(py).to_owned());
            let kdist: Py<PyArray<f64, Ix1>> = Py::from(kdist.into_pyarray(py).to_owned());
            let knode: Py<PyArray<f64, Ix1>> = Py::from(knode.into_pyarray(py).to_owned());
            (kvec,kdist,knode)
        })
    }

    pub fn gen_ham(&mut self,kvec:PyReadonlyArray<f64,Ix1>)->Py<PyArray<Complex<f64>,Ix2>>{
        Python::with_gil(|py|{
            let new_model=unsafe{ Model{
                dim_r:self.dim_r,
                norb: self.norb,
                nsta: self.nsta,
                natom: self.natom,
                spin: self.spin,
                // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
                lat: self.lat.as_ref(py).as_array().to_owned(),
                orb: self.orb.as_ref(py).as_array().to_owned(),
                atom: self.atom.as_ref(py).as_array().to_owned(),
                atom_list: self.atom_list.clone(),
                ham: self.ham.as_ref(py).as_array().to_owned(),
                hamR: self.hamR.as_ref(py).as_array().to_owned(),
                rmatrix: self.rmatrix.as_ref(py).as_array().to_owned(),
            }};
            let kvec=kvec.as_array().to_owned();
            let ham=new_model.gen_ham(&kvec);
            let ham=Py::from(ham.into_pyarray(py).to_owned());
            ham
        })
    }

    pub fn solve_onek(&self,kvec:PyReadonlyArray<f64,Ix1>)->(Py<PyArray<f64,Ix1>>,Py<PyArray<Complex<f64>,Ix2>>){
        Python::with_gil(|py|{
            let new_model=unsafe{ Model{
                dim_r:self.dim_r,
                norb: self.norb,
                nsta: self.nsta,
                natom: self.natom,
                spin: self.spin,
                // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
                lat: self.lat.as_ref(py).as_array().to_owned(),
                orb: self.orb.as_ref(py).as_array().to_owned(),
                atom: self.atom.as_ref(py).as_array().to_owned(),
                atom_list: self.atom_list.clone(),
                ham: self.ham.as_ref(py).as_array().to_owned(),
                hamR: self.hamR.as_ref(py).as_array().to_owned(),
                rmatrix: self.rmatrix.as_ref(py).as_array().to_owned(),
            }};
            let kvec=kvec.as_array().to_owned();
            let (eval,evec)=new_model.solve_onek(&kvec);
            let eval=Py::from(eval.into_pyarray(py).to_owned());
            let evec=Py::from(evec.into_pyarray(py).to_owned());
            (eval,evec)
        })
    }
    pub fn solve_all(&self,kvec:PyReadonlyArray<f64,Ix2>)->(Py<PyArray<f64,Ix2>>,Py<PyArray<Complex<f64>,Ix3>>){
        Python::with_gil(|py|{
            let new_model=unsafe{ Model{
                dim_r:self.dim_r,
                norb: self.norb,
                nsta: self.nsta,
                natom: self.natom,
                spin: self.spin,
                // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
                lat: self.lat.as_ref(py).as_array().to_owned(),
                orb: self.orb.as_ref(py).as_array().to_owned(),
                atom: self.atom.as_ref(py).as_array().to_owned(),
                atom_list: self.atom_list.clone(),
                ham: self.ham.as_ref(py).as_array().to_owned(),
                hamR: self.hamR.as_ref(py).as_array().to_owned(),
                rmatrix: self.rmatrix.as_ref(py).as_array().to_owned(),
            }};
            let kvec=kvec.as_array().to_owned();
            let (eval,evec)=new_model.solve_all(&kvec);
            let eval=Py::from(eval.into_pyarray(py).to_owned());
            let evec=Py::from(evec.into_pyarray(py).to_owned());
            (eval,evec)
        })
    }
    pub fn solve_all_parallel(&self,kvec:PyReadonlyArray<f64,Ix2>)->(Py<PyArray<f64,Ix2>>,Py<PyArray<Complex<f64>,Ix3>>){
        Python::with_gil(|py|{
            let new_model=unsafe{ Model{
                dim_r:self.dim_r,
                norb: self.norb,
                nsta: self.nsta,
                natom: self.natom,
                spin: self.spin,
                // 把 Array2 转换成 PyReadonlyArray，并且用 Py 包装它
                lat: self.lat.as_ref(py).as_array().to_owned(),
                orb: self.orb.as_ref(py).as_array().to_owned(),
                atom: self.atom.as_ref(py).as_array().to_owned(),
                atom_list: self.atom_list.clone(),
                ham: self.ham.as_ref(py).as_array().to_owned(),
                hamR: self.hamR.as_ref(py).as_array().to_owned(),
                rmatrix: self.rmatrix.as_ref(py).as_array().to_owned(),
            }};
            let kvec=kvec.as_array().to_owned();
            let (eval,evec)=new_model.solve_all_parallel(&kvec);
            let eval=Py::from(eval.into_pyarray(py).to_owned());
            let evec=Py::from(evec.into_pyarray(py).to_owned());
            (eval,evec)
        })
    }



}

#[pymodule]
fn Rustb4py(py:Python,m:&PyModule)->PyResult<()>{
    m.add_class::<tb_model>()?; 
    m.add_function(wrap_pyfunction!(from_hr,m)?)?;
    Ok(())
}
*/
