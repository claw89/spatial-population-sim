use indicatif::ProgressBar;
use itertools::Itertools;
use ndarray::{s, Array, Array3};
use ndarray_npy::write_npy;
use ndhistogram::{axis::Uniform, ndhistogram, Histogram};
use rand::prelude::*;
use rand_distr::{Normal, WeightedIndex};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone)]
pub struct SpeciesCoords {
    pub species_id: usize,
    pub x_coords: Vec<f64>,
    pub y_coords: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SpeciesHeatmap {
    pub species_id: usize,
    pub heatmap: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Checkpoint {
    pub time: f64,
    pub species_individuals: Vec<SpeciesCoords>,
    pub species_heatmap: Vec<SpeciesHeatmap>,
}

#[derive(Clone, Copy)]
enum Event {
    BIRTH,
    DEATH,
    // MOVE,
}
/// A Species object holding the parameters that individuals of this species will use
#[derive(PartialEq, Debug, Serialize, Deserialize, Clone, Copy)]
pub struct Species {
    pub id: usize,
    pub b0: f64,
    pub b1: f64,
    pub c1: f64,
    pub d0: f64,
    pub d1: f64,
    pub mbrmax: f64,
    pub mbsd: f64,
    pub mintegral: f64,
    pub move_radius_max: f64,
    pub move_std: f64,
    pub birth_radius_max: f64,
    pub birth_std: f64,
    birth_norm: Option<f64>,
    pub death_radius_max: f64,
    pub death_std: f64,
    death_norm: Option<f64>,
}

impl Species {
    /// Creates a new species; birth and death norms are calculated from the respective radius_max an std values
    pub fn derive_norms(&mut self) {
        if self.birth_std != 0.0 {
            self.birth_norm = Some(
                2.0 * self.birth_std.powi(2)
                    * PI
                    * (1.0
                        - ((-1.0 * self.birth_radius_max.powi(2))
                            / (2.0 * self.birth_std.powi(2)))
                        .exp()),
            );
        }
        if self.death_std != 0.0 {
            self.death_norm = Some(
                2.0 * self.death_std.powi(2)
                    * PI
                    * (1.0
                        - ((-1.0 * self.death_radius_max.powi(2))
                            / (2.0 * self.death_std.powi(2)))
                        .exp()),
            );
        }
    }
}

/// An individual member of the population, which belongs to a species
#[derive(Serialize, Deserialize, PartialEq, Clone)]
struct Individual {
    id: usize,
    species: Species,
    x_coord: f64,
    y_coord: f64,
    p_birth: f64,
    p_death: f64,
    p_move: f64,
    birth_neighbor_weight: f64,
    death_neighbor_weight: f64,
    birth_distances: Vec<(usize, f64)>,
    death_distances: Vec<(usize, f64)>,
}

impl Individual {
    pub fn new(id: usize, species: Species, x_coord: f64, y_coord: f64) -> Self {
        Individual {
            id,
            species,
            x_coord,
            y_coord,
            p_birth: 0.0,
            p_death: 0.0,
            p_move: 0.0,
            birth_neighbor_weight: 0.0,
            death_neighbor_weight: 0.0,
            birth_distances: vec![],
            death_distances: vec![],
        }
    }

    pub fn distance(&self, other: &Individual) -> f64 {
        // Compute the Euclidean distance between the positions of two individuals

        let inside_delta_x = (self.x_coord - other.x_coord).abs();
        let delta_x = inside_delta_x.min(1.0 - inside_delta_x);

        let inside_delta_y = (self.y_coord - other.y_coord).abs();
        let delta_y = inside_delta_y.min(1.0 - inside_delta_y);

        (delta_x.powi(2) + delta_y.powi(2)).sqrt()
    }

    fn update_distances(&mut self, distance: f64, event: Event, idx: usize) {
        match event {
            Event::BIRTH => {
                let radius = self.species.birth_radius_max;
                let var = self.species.birth_std.powi(2);
                if distance < radius && var != 0.0 {
                    self.birth_distances.push((idx, get_weight(distance, var)));
                }
            }
            Event::DEATH => {
                let radius = self.species.death_radius_max;
                let var = self.species.death_std.powi(2);
                if distance < radius && var != 0.0 {
                    self.death_distances.push((idx, get_weight(distance, var)));
                }
            }
        }
    }

    pub fn update_probabilities(&mut self) {
        // Update individual birth, death, and move probabilities

        self.p_birth = self.species.b0 + self.birth_neighbor_weight;
        self.p_death = self.species.d0 + self.death_neighbor_weight;
        self.p_move = self.species.mintegral;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Population {
    species_list: Vec<Species>,
    individuals: Vec<Individual>,
    pub size: usize,
    pub t: f64,
}

impl Population {
    pub fn new(species_list: Vec<Species>) -> Self {
        // create individuals for each species
        let mut individuals: Vec<Individual> = vec![];
        let mut idx = 0;
        let mut rng = rand::thread_rng();
        for species in species_list.clone() {
            for _ in 0..(species.c1 as usize) {
                let new_individual = Individual::new(idx, species, rng.gen(), rng.gen());
                individuals.push(new_individual);
                idx += 1;
            }
        }

        // instantiate population
        Population {
            species_list,
            individuals,
            size: idx,
            t: 0.0,
        }
    }

    pub fn compute_initial_distances(&mut self) {
        let second_individuals = &self.individuals.clone();

        for first in &mut self.individuals {
            for second in second_individuals {
                if first.id != second.id {
                    let distance = first.distance(second);
                    first.update_distances(distance, Event::BIRTH, second.id);
                    first.update_distances(distance, Event::DEATH, second.id);
                }
            }
        }
    }

    fn compute_neighbor_weights(&mut self, event: &Event) {
        match event {
            Event::BIRTH => {
                for individual in &mut self.individuals {
                    match individual.species.birth_norm {
                        Some(_) => {
                            individual.birth_neighbor_weight = (individual
                                .birth_distances
                                .iter()
                                .fold(0.0, |acc, (_, d)| acc + *d)
                                * individual.species.b1)
                                / individual.species.birth_norm.unwrap();
                        }
                        None => individual.birth_neighbor_weight = 0.0,
                    }
                }
            }
            Event::DEATH => {
                for individual in &mut self.individuals {
                    match individual.species.death_norm {
                        Some(_) => {
                            individual.death_neighbor_weight = (individual
                                .death_distances
                                .iter()
                                .fold(0.0, |acc, (_, d)| acc + *d)
                                * individual.species.d1)
                                / individual.species.death_norm.unwrap();
                        }
                        None => individual.death_neighbor_weight = 0.0,
                    }
                }
            }
        }
    }

    fn update_probabilities(&mut self) {
        // update birth, death, and move probabilities
        for individual in self.individuals.iter_mut() {
            individual.update_probabilities();
        }
    }

    fn execute_birth(&mut self, parent: Individual) {
        // create a new invidual
        let parent = parent.clone();

        // initialise child position from parent with Gaussian kernel
        let mut rng = rand::thread_rng();
        let mut child_x_coord = Normal::new(parent.x_coord, parent.species.mbsd)
            .unwrap()
            .sample(&mut rng)
            % 1.0;
        if child_x_coord < 0.0 {
            child_x_coord += 1.0;
        }
        let mut child_y_coord = Normal::new(parent.y_coord, parent.species.mbsd)
            .unwrap()
            .sample(&mut rng)
            % 1.0;
        if child_y_coord < 0.0 {
            child_y_coord += 1.0;
        }

        let max_id = self.individuals.iter().map(|x| x.id).max().unwrap();
        let mut child = Individual::new(max_id + 1, parent.species, child_x_coord, child_y_coord);

        // initialize child distances and update other individuals
        for individual in &mut self.individuals {
            let distance = child.distance(individual);
            child.update_distances(distance, Event::BIRTH, individual.id);
            child.update_distances(distance, Event::DEATH, individual.id);

            individual.update_distances(distance, Event::BIRTH, child.id);
            individual.update_distances(distance, Event::DEATH, child.id);
        }

        // add child to vector of individuals
        self.individuals.push(child);
        self.size += 1;
    }

    fn execute_death(&mut self, deceased: Individual) {
        // remove an individual from the population
        let deceased_id = self
            .individuals
            .iter()
            .position(|x| *x == deceased)
            .unwrap();
        for individual in &mut self.individuals {
            individual
                .birth_distances
                .retain(|(idx, _)| *idx != deceased_id);
            individual
                .death_distances
                .retain(|(idx, _)| *idx != deceased_id);
        }
        self.individuals.remove(deceased_id);
        self.size -= 1;
    }

    // fn execute_move<'b>(&'b mut self) {
    //     // move an individual within the population
    // }

    fn choose_event(&self) -> (Event, Individual, f64) {
        // pick the event type and individual at random from the poopulation
        let p_birth_sum = self.individuals.iter().fold(0.0, |acc, x| acc + x.p_birth);
        let p_death_sum = self.individuals.iter().fold(0.0, |acc, x| acc + x.p_death);
        let p_move_sum = self.individuals.iter().fold(0.0, |acc, x| acc + x.p_move);
        let p_total = p_birth_sum + p_death_sum + p_move_sum;

        let mut rng = rand::thread_rng();

        let choices = vec![Event::BIRTH, Event::DEATH, Event::DEATH];
        let weights = if p_total > 0.0 {
            vec![
                p_birth_sum / p_total,
                p_death_sum / p_total,
                p_move_sum / p_total,
            ]
        } else {
            vec![0.0, 0.0, 0.0]
        };
        let chosen_event = weighted_sample(&choices, &weights, &mut rng);

        let chosen_individual = match chosen_event {
            Event::BIRTH => {
                let weights = self
                    .individuals
                    .iter()
                    .map(|x| -> f64 {
                        if p_birth_sum > 0.0 {
                            x.p_birth / p_birth_sum
                        } else {
                            0.0
                        }
                    })
                    .collect();
                weighted_sample(&self.individuals.clone(), &weights, &mut rng)
            }
            Event::DEATH => {
                let weights = self
                    .individuals
                    .iter()
                    .map(|x| -> f64 {
                        if p_death_sum > 0.0 {
                            x.p_death / p_death_sum
                        } else {
                            0.0
                        }
                    })
                    .collect();
                weighted_sample(&self.individuals.clone(), &weights, &mut rng)
            } // Event::Move => {
              //     let weights = self
              //         .individuals
              //         .iter()
              //         .map(|x| x.p_move / p_move_sum)
              //         .collect();
              //     weighted_sample(&self.individuals, &weights, &mut rng)
              // }
        };

        (chosen_event, chosen_individual, p_total)
    }

    pub fn get_heatmap(&self, heatmap_bins: usize) -> Vec<SpeciesHeatmap> {
        let mut full_heatmap: Vec<SpeciesHeatmap> = vec![];
        // : ArrayBase<&f64>, Dim<[usize; &self.species_list.len()]>> = vec![];
        for species in self.species_list.iter() {
            // let heatmap = Array2::<f64>::zeros(n_bins + 2, n_bins + 2);
            let mut layer_heatmap = ndhistogram!(
                Uniform::new(heatmap_bins, 0.0, 1.0),
                Uniform::new(heatmap_bins, 0.0, 1.0)
            );
            for individual in self
                .individuals
                .iter()
                .filter(|individual| individual.species.id == species.id)
            {
                layer_heatmap.fill(&(individual.x_coord, individual.y_coord));
            }
            let heatmap: Vec<Vec<f64>> = layer_heatmap
                .values()
                .cloned()
                .collect::<Vec<f64>>()
                .chunks(heatmap_bins + 2)
                .map(|c| (c.to_vec()[1..heatmap_bins]).to_vec())
                .enumerate()
                .filter(|(idx, _)| *idx != 0 && *idx != heatmap_bins)
                .map(|(_, v)| v)
                .collect();
            full_heatmap.push(SpeciesHeatmap {
                species_id: species.id,
                heatmap,
            })
        }
        full_heatmap
    }

    fn save_heatmap(
        &self,
        species_heatmaps: Vec<SpeciesHeatmap>,
        heatmap_bins: usize,
        data_path: PathBuf,
    ) {
        let species_id_max = species_heatmaps.iter().map(|s| s.species_id).max().unwrap();
        let mut heatmap = Array3::<f64>::zeros((species_id_max + 1, heatmap_bins, heatmap_bins));
        for species_heatmap in species_heatmaps {
            let species_heatmap_array = Array::from_shape_vec(
                (heatmap_bins, heatmap_bins),
                species_heatmap.heatmap.into_iter().flatten().collect_vec(),
            )
            .unwrap();
            heatmap
                .slice_mut(s![
                    species_heatmap.species_id,
                    0..heatmap_bins,
                    0..heatmap_bins
                ])
                .assign(&species_heatmap_array);
        }
        write_npy(
            data_path.join(format!("{:0>4}.npy", (self.t.floor() as u64))),
            &heatmap,
        )
        .unwrap();
    }

    fn get_checkpoint(&self, heatmap_bins: usize) -> Checkpoint {
        let mut species_individuals = vec![] as Vec<SpeciesCoords>;
        for species in self.species_list.clone() {
            let coords: Vec<(f64, f64)> = self
                .individuals
                .iter()
                .filter(|x| x.species.id == species.id)
                .map(|x| (x.x_coord, x.y_coord))
                .collect::<Vec<(f64, f64)>>();
            let (x_coords, y_coords) = coords.into_iter().unzip();
            species_individuals.push(SpeciesCoords {
                species_id: species.id,
                x_coords,
                y_coords,
            });
        }
        Checkpoint {
            time: self.t,
            species_individuals,
            species_heatmap: self.get_heatmap(heatmap_bins),
        }
    }

    pub fn step(&mut self, heatmap_bins: usize) -> (Checkpoint, f64) {
        for event in [Event::BIRTH, Event::DEATH] {
            self.compute_neighbor_weights(&event);
        }
        self.update_probabilities();

        let (chosen_event, chosen_individual_id, p_total) = self.choose_event();
        match chosen_event {
            Event::BIRTH => self.execute_birth(chosen_individual_id),
            Event::DEATH => self.execute_death(chosen_individual_id),
            // Event::Move => self.execute_move(),
        }
        (self.get_checkpoint(heatmap_bins), p_total)
    }

    pub fn increment_time(&mut self, p_total: f64) {
        let mut rng = rand::thread_rng();
        let delta_t: f64 = (-1.0 / p_total) * (1.0 - rng.gen::<f64>()).ln();
        assert!(delta_t > 0.0);
        self.t += delta_t;
    }

    pub fn simulate(&mut self, max_t: f64, data_path: PathBuf, heatmap_bins: usize) {
        // somulate the behaviour of the population over time
        let prog = ProgressBar::new((max_t - 1.0) as u64);

        self.compute_initial_distances();
        while self.t < max_t {
            let (checkpoint, p_total) = self.step(heatmap_bins);
            self.increment_time(p_total);
            if self.t as u64 > prog.position() + 1 {
                prog.inc(1);
                self.save_heatmap(checkpoint.species_heatmap, heatmap_bins, data_path.clone());
            }
        }
    }
}

fn weighted_sample<T>(choices: &[T], weights: &Vec<f64>, rng: &mut ThreadRng) -> T
where
    T: Clone,
{
    if weights.iter().fold(0.0, |acc, w| acc + *w) > 0.0 {
        let dist = WeightedIndex::new(weights).unwrap();
        choices[dist.sample(rng)].clone()
    } else {
        choices.choose(rng).unwrap().clone()
    }
}

fn get_weight(distance: f64, var: f64) -> f64 {
    ((-1.0 * distance.powi(2)) / (2.0 * var)).exp()
}
