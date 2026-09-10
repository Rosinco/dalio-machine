use atlas_financial_store::Pack;
use std::{path::PathBuf, time::Instant};
fn main() {
    let path = PathBuf::from(std::env::args().nth(1).expect("Pass a financial pack path"));
    let start = Instant::now();
    let pack = Pack::open(&path, path.file_stem().unwrap().to_str().unwrap()).unwrap();
    println!(
        "Index verified in {:?}: {}",
        start.elapsed(),
        pack.index()["summary"]
    );
    let start = Instant::now();
    pack.check_all().unwrap();
    println!("All company records verified in {:?}", start.elapsed());
    let start = Instant::now();
    for id in ["102", "20", "58", "1429", "55268"] {
        pack.company(id).unwrap();
    }
    println!("Five company lookups: {:?}", start.elapsed());
}
