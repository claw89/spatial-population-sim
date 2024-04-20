<!-- ABOUT THE PROJECT -->
## About The Project


Simulate changes to the spatial structure of a population over time. A CLI tool to run simulations and save population density heatmaps in .npy format


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

* Rust
  ```sh
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### Installation


1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Optional: Compile
    ```sh
    cargo build
    ```



<!-- USAGE EXAMPLES -->
## Usage

Example script call:
```sh
cargo run -- --max-t 20 --runs 2 --bins 32 --species-ids 1 --data-path data --param-path species_params.csv
```
Or if already compiled (Installation step 2)  
```sh
./target/debug/spatial-population-sim --max-t 20 --runs 2 --bins 32 --species-ids 1 --data-path data --param-path species_params.csv
```


## CLI Reference

- `max-t`: the total length of the simulation 
- `runs`: the number of repeats to run the simulation for
- `bins`: the number of bins for generating 2d species density heatmaps
- `species-ids`: the id numbers for species to simulate (list of space-separated integers); ids should correspond to the indexes in `param-path`
- `data-path`: the directory to store output heatmaps
- `param-path`: the path to a csv file containing species parameters