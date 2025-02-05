mod fluid_simulation;
mod ui;

extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use std::iter;

use fluid_simulation::{Particle, V2};
use glutin_window::GlutinWindow as Window;
use graphics::glyph_cache::rusttype::GlyphCache;
use graphics::Viewport;
use opengl_graphics::{GlGraphics, OpenGL, Texture, TextureSettings};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;

use crate::fluid_simulation::SimulationState;
use crate::ui::render_ui;

const NUM_PARTICLES: usize = 1000;
const FONT_PATH: &str = ".\\assets\\roboto.ttf";
const VIEWPORT_WIDTH: u32 = 1024;
const VIEWPORT_HEIGHT: u32 = 768;
pub struct App<'a> {
    gl: GlGraphics, // OpenGL drawing backend.
    simulation_state: SimulationState<'a>,
    glyphs: GlyphCache<'static, (), Texture>,
    viewport: Viewport,
}

impl App<'a> {
    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        let viewport = args.viewport().clone();

        self.viewport = viewport;

        self.gl.draw(args.viewport(), |c, gl| {
            // Clear the screen.
            clear(graphics::color::BLACK, gl);
            self.simulation_state.draw(c, gl);
            // render_ui(&mut self.glyphs, c, gl);
        });
    }

    fn update(&mut self, args: &UpdateArgs) {
        self.simulation_state
            .update_viewport(self.viewport)
            .update(args.dt);
    }
}

fn main() {
    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;
    let glyphs = GlyphCache::new(FONT_PATH, (), TextureSettings::new()).unwrap();

    // Create a Glutin window.
    let mut window: Window =
        WindowSettings::new("Fluid Simulation", [VIEWPORT_WIDTH, VIEWPORT_HEIGHT])
            .graphics_api(opengl)
            .exit_on_esc(true)
            .build()
            .unwrap();

    let default_viewport = Viewport {
        rect: [0, 0, VIEWPORT_WIDTH as i32, VIEWPORT_HEIGHT as i32],
        draw_size: [VIEWPORT_WIDTH, VIEWPORT_HEIGHT],
        window_size: [VIEWPORT_WIDTH as f64, VIEWPORT_HEIGHT as f64],
    };

    let mut particles: Vec<Particle> = iter::repeat(())
        .take(NUM_PARTICLES)
        .map(|_| {
            let x = rand::random::<f64>() * default_viewport.window_size[0];
            let y = rand::random::<f64>() * default_viewport.window_size[1];
            Particle {
                position: V2::new(x, y),
                velocity: V2::zeros(),
                predicted_position: V2::zeros(),
                mass: 1.,
                density: 0.,
                density_gradient: V2::zeros(),
            }
        })
        .collect();

    // Create a new game and run it.
    let mut app = App {
        gl: GlGraphics::new(opengl),
        viewport: default_viewport,
        simulation_state: SimulationState::new(default_viewport, particles.iter_mut().collect()),
        glyphs,
    };

    let mut events = Events::new(EventSettings::new());
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(&args);
        }

        if let Some(args) = e.update_args() {
            app.update(&args);
        }
    }
}
