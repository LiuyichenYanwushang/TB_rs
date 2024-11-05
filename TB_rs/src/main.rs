///这个程序是类似wanniertools 的代码, 但是实现了一些输运相关的代码, 效率相对更高
///主程序主要用来实现对控制文件的读取, 控制文件默认叫做 TB.in
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
fn main() {
    let path = "TB.in";
    let TB_file = File::open(path).expect(&format!(
        "Unable to open the file {:?}, please check if file is present",
        path
    ));
}
