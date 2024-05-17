use std::f64::consts::PI;

use graphics::{color, ellipse, line, Context, Transformed, Viewport};
use kdtree::KdTree;
use opengl_graphics::GlGraphics;
use rayon::{iter, prelude::*};
extern crate nalgebra as na;

const NUM_PARTICLES: usize = 500;
const PARTICLE_RADIUS: f64 = 6.;

const GRAVITY: na::Vector2<f64> = na::Vector2::new(0., 0.);

const SMOOTHING_RADIUS: f64 = 120.;

const TIME_STEP: f64 = 5.;

const COLLISSION_DAMPING: f64 = 0.85;

const TARGET_DENSITY: f64 = 2.75;
// const TARGET_DENSITY: f64 = 0.1;
// const PRESSURE_MULTIPLIER: f64 = 100.;
const PRESSURE_MULTIPLIER: f64 = 1000.;

#[derive(Debug, Clone)]
pub struct SimulationState {
    pub particles: Vec<Particle>,
    viewport: Viewport,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle {
    pub position: na::Vector2<f64>,
    pub velocity: na::Vector2<f64>,
    mass: f64,
    density: f64,
    density_gradient: na::Vector2<f64>,
}

type ParticleKdTree<'a> = KdTree<f64, &'a Particle, [f64; 2]>;

impl SimulationState {
    pub fn new(viewport: Viewport) -> Self {
        SimulationState {
            viewport,
            particles: iter::repeat(())
                .take(NUM_PARTICLES)
                .map(|_| {
                    let x = rand::random::<f64>() * viewport.window_size[0];
                    let y = rand::random::<f64>() * viewport.window_size[1];
                    Particle {
                        position: na::Vector2::new(x, y),
                        velocity: na::Vector2::zeros(),
                        mass: 1.,
                        density: 0.,
                        density_gradient: na::Vector2::zeros(),
                    }
                })
                .collect(),
        }
    }

    pub fn update_viewport(&mut self, viewport: Viewport) -> &mut Self {
        self.viewport = viewport;
        self
    }

    pub fn draw(&self, context: Context, gl: &mut GlGraphics) {
        let circle = graphics::ellipse::circle(0., 0., PARTICLE_RADIUS);
        let smoothing_circle = graphics::ellipse::circle(0., 0., SMOOTHING_RADIUS);
        self.particles
            .to_vec()
            .into_iter()
            .zip(0..self.particles.len())
            .map(|(particle, index)| {
                let r = na::Vector4::from(color::YELLOW) * (particle.density * 1500.) as f32;
                let b = na::Vector4::from(color::BLUE) * ((1. - particle.density) * 100.) as f32;
                line(
                    color::WHITE,
                    5.,
                    [
                        0.,
                        0.,
                        particle.density_gradient[0] * 100.,
                        particle.density_gradient[1] * 100.,
                    ],
                    context.transform.trans_pos(particle.position),
                    gl,
                );
                ellipse(
                    if index == 1 {
                        graphics::color::LIME
                    } else {
                        (r + b).into()
                    },
                    circle,
                    context
                        .transform
                        .trans_pos::<[f64; 2]>(particle.position.into()),
                    gl,
                );
                if index == 1 {
                    ellipse(
                        graphics::color::alpha(0.2),
                        smoothing_circle,
                        context
                            .transform
                            .trans_pos::<[f64; 2]>(particle.position.into()),
                        gl,
                    )
                }
            })
            .collect()
    }

    pub fn update(&mut self, dt: f64) {
        let boundaries = [
            0.,
            0.,
            self.viewport.window_size[0],
            self.viewport.window_size[1],
        ];

        let mut update_buffer: Vec<Particle> = Vec::new();

        let mut kdtree: ParticleKdTree = KdTree::new(2);
        self.particles
            .iter()
            .map(|p| {
                kdtree.add(p.position.into(), &p).unwrap();
            })
            .for_each(drop);

        self.particles
            .to_vec()
            .into_par_iter()
            .update(|particle: &mut Particle| {
                let updated_velocity = particle.velocity + GRAVITY * dt * TIME_STEP;
                let particles_to_consider = kdtree
                    .within(
                        &[particle.position.x, particle.position.y],
                        SMOOTHING_RADIUS,
                        &kdtree::distance::squared_euclidean,
                    )
                    .unwrap()
                    .iter()
                    .map(|p| **p.1)
                    .collect::<Vec<Particle>>();

                let density = get_density(&particles_to_consider, &particle);

                particle.velocity = updated_velocity;
                particle.density = density;
            })
            .update(|particle: &mut Particle| {
                let particles_to_consider = kdtree
                    .within(
                        &[particle.position.x, particle.position.y],
                        SMOOTHING_RADIUS,
                        &|a, b| {
                            a.into_iter()
                                .zip(b.into_iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt()
                        },
                    )
                    .unwrap()
                    .iter()
                    .map(|p| **p.1)
                    .collect::<Vec<Particle>>();
                let pressure_force = get_pressure_force(&particles_to_consider, &particle);
                let pressure_acceleration = if particle.density > 0. {
                    pressure_force / particle.density
                } else {
                    na::Vector2::zeros()
                };
                particle.velocity += pressure_acceleration * dt * TIME_STEP;
            })
            .update(|particle| {
                particle.position += particle.velocity * dt * TIME_STEP;
                handle_boundaries(&mut particle.position, &mut particle.velocity, boundaries);
            })
            .collect_into_vec(&mut update_buffer);

        self.particles = update_buffer
            .try_into()
            .expect("Failed to convert updated particles into array");
    }
}

fn get_pressure_from_density(density: f64) -> f64 {
    let density_error = density - TARGET_DENSITY;
    let pressure = density_error * PRESSURE_MULTIPLIER;
    pressure
}

fn get_density(particles: &Vec<Particle>, particle: &Particle) -> f64 {
    particles
        .into_par_iter()
        .map(|other_particle| {
            let distance = (other_particle.position - particle.position).magnitude();
            if particle.position.eq(&other_particle.position) {
                0.
            } else {
                let influence = smoothing_kernel(distance, SMOOTHING_RADIUS);
                other_particle.mass * influence
            }
        })
        .sum()
}

fn get_pressure_force(particles: &Vec<Particle>, particle: &Particle) -> na::Vector2<f64> {
    particles
        .into_par_iter()
        .map(|other_particle| {
            let distance = (other_particle.position - particle.position).magnitude();
            if particle.position.eq(&other_particle.position) || distance.lt(&5.) {
                na::Vector2::zeros()
            } else {
                let direction = (other_particle.position - particle.position) / distance;
                let slope = smoothing_kernel_prime(distance, SMOOTHING_RADIUS);
                let shared_pressure = get_shared_pressure_force(other_particle, particle);
                -shared_pressure * direction * slope * other_particle.mass
            }
        })
        .sum()
}

fn get_shared_pressure_force(first: &Particle, second: &Particle) -> f64 {
    (get_pressure_from_density(first.density) + get_pressure_from_density(second.density)) / 2.
}

fn handle_boundaries(
    position: &mut na::Vector2<f64>,
    velocity: &mut na::Vector2<f64>,
    boundaries: [f64; 4],
) {
    if position.x > (boundaries[0] + boundaries[2] - PARTICLE_RADIUS)
        || position.x < PARTICLE_RADIUS
    {
        position.x = position.x.clamp(
            boundaries[0] + PARTICLE_RADIUS,
            boundaries[2] - PARTICLE_RADIUS,
        );
        velocity.x *= -1. * COLLISSION_DAMPING;
    }

    if (position.y < boundaries[1] + PARTICLE_RADIUS)
        || position.y > (boundaries[1] + boundaries[3] - PARTICLE_RADIUS)
    {
        position.y = position.y.clamp(
            boundaries[1] + PARTICLE_RADIUS,
            boundaries[3] - PARTICLE_RADIUS,
        );
        velocity.y *= -1. * COLLISSION_DAMPING;
    }
}

pub fn smoothing_kernel(dst: f64, radius: f64) -> f64 {
    if dst >= radius {
        0.
    } else {
        let volume = (PI * radius.powi(4)) / 6.;
        (radius - dst) * (radius - dst) / volume
    }
}

fn smoothing_kernel_prime(dst: f64, radius: f64) -> f64 {
    if dst > radius {
        0.
    } else {
        let scale = 12. / (radius.powi(4) * PI);
        (dst - radius) * scale
    }
}
