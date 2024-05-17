use graphics::{glyph_cache::rusttype::GlyphCache, Context, Transformed};
use opengl_graphics::{GlGraphics, Texture};

pub fn render_ui(glyphs: &mut GlyphCache<'static, (), Texture>, c: Context, gl: &mut GlGraphics) {
    // load font
    graphics::text::Text::new_color(graphics::color::WHITE, 24)
        .draw(
            "Henlo",
            glyphs,
            &c.draw_state,
            c.transform.trans(10., 30.),
            gl,
        )
        .inspect_err(|a| println!("{:?}", a))
        .unwrap()
}
