use chrono::Utc;
use clap::Parser;
use ndarray::{Array, Axis};
use spatial_population_sim::*;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    max_t: u64,

    #[arg(short, long)]
    bins: usize,

    #[arg(short, long)]
    data_path: String,

    #[arg(short, long, default_value_t = 1)]
    runs: u64,

    #[arg(short, long, num_args = 1.., value_delimiter=' ')]
    species_ids: Vec<usize>,
}

fn load_species_array(species_ids: Vec<usize>) -> Vec<Species> {
    let mut rdr = csv::Reader::from_path("data/species_params.csv").unwrap();
    let species_array = Array::from_iter(
        rdr.deserialize::<Species>()
            .map(|x| -> Species { x.unwrap() }),
    );
    species_array.select(Axis(0), &species_ids).to_vec()
}

fn create_sim_dir(data_path: String) -> PathBuf {
    if !Path::new(&data_path).exists() {
        fs::create_dir(&data_path).unwrap();
    }
    let now = Utc::now();
    let mut data_path = String::from("./data/");
    data_path.push_str(&now.format("%Y-%m-%d_%H:%M:%S").to_string());
    fs::create_dir(&data_path).unwrap();
    Path::new(&data_path).to_owned()
}

fn main() {
    let args = Args::parse();
    let species_array = load_species_array(args.species_ids);
    let sim_path = create_sim_dir(args.data_path);

    for run_number in 0..args.runs {
        let run_path = sim_path.join(format!("run_{:0>4}", run_number));
        fs::create_dir(&run_path).unwrap();
        let mut population = Population::new(species_array.clone());
        population.simulate(args.max_t as f64, run_path, args.bins);
    }
}
