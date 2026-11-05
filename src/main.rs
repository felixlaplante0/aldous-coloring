mod process;

use eframe::egui;
use egui::{Color32, ColorImage, Pos2, Rangef, Rect, Sense, Stroke, TextureHandle, TextureOptions, emath::OrderedFloat, pos2, remap};
use process::{DOMAIN, MAX_LEVEL, RASTER, TileId, World, level_for, render_tile, s_top};
use quick_cache::unsync::Cache;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::SplitMix64;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, mpsc};

const BACKGROUND: Color32 = Color32::from_rgb(18, 18, 22);
const PICK_RADIUS: f32 = 14.0;
const MIN_SCALE: f64 = (DOMAIN >> (MAX_LEVEL + 1)) as f64;
/// Raster tiles kept on the GPU, about 266 KiB each.
const TEXTURE_BUDGET: usize = 1024;
/// Enough to keep the cores busy, few enough that tiles still appear nearest the pointer first.
const TILES_AT_ONCE: usize = 4;

type Rendered = (u64, TileId, ColorImage);

/// Everything a tile's pixels depend on, apart from its id. The UI edits these directly.
#[derive(Clone, PartialEq)]
struct Params {
    rain: u64,
    seeds: Vec<(u64, u64)>,
    palette_offset: f64,
    density: f64,
    aa: u32,
}

/// The latest request, which is also the current picture.
struct Job {
    epoch: u64,
    params: Params,
    world: Arc<World>,
    palette: Vec<Color32>,
    tiles: Vec<TileId>,
    cancel: AtomicBool,
}

impl Job {
    fn new(epoch: u64, params: &Params, world: Arc<World>, tiles: Vec<TileId>) -> Arc<Job> {
        let palette = make_palette(params.seeds.len(), params.palette_offset);
        Arc::new(Job { epoch, params: params.clone(), world, palette, tiles, cancel: AtomicBool::new(false) })
    }
}

fn worker(jobs: mpsc::Receiver<Arc<Job>>, done: mpsc::Sender<Rendered>, ctx: egui::Context) {
    while let Ok(job) = jobs.recv() {
        let job = jobs.try_iter().last().unwrap_or(job);
        let Params { density, aa, .. } = job.params;
        for group in job.tiles.chunks(TILES_AT_ONCE) {
            group.par_iter().for_each(|&id| {
                let level = level_for(id.0, density);
                if let Some(image) = render_tile(&job.world, id, level, aa, &job.palette, &job.cancel) {
                    let _ = done.send((job.epoch, id, image));
                    ctx.request_repaint();
                }
            });
            if job.cancel.load(Relaxed) {
                break;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Layout {
    Random,
    Circle,
    Grid,
    Manual,
}

fn make_palette(k: usize, offset: f64) -> Vec<Color32> {
    (0..k)
        .map(|i| {
            let i = i as f64;
            let h = (offset + i * 0.618_033_988_75).fract() as f32;
            let s = (0.50 + 0.35 * (i * 0.381_966).fract()) as f32;
            let v = (0.70 + 0.28 * (i * 0.754_877_666).fract()) as f32;
            let [r, g, b] = egui::ecolor::rgb_from_hsv((h, s, v)).map(|t| (t * 255.0).round() as u8);
            Color32::from_rgb(r, g, b)
        })
        .collect()
}

fn random_seeds(salt: u64, k: usize) -> Vec<(u64, u64)> {
    let mut rng = SplitMix64::seed_from_u64(salt);
    (0..k).map(|_| (rng.random_range(0..DOMAIN), rng.random_range(0..DOMAIN))).collect()
}

/// Screen span of tile edges `a..b` clipped to `clip`, and the matching texture span.
/// Clipping in `f64` keeps huge stand-in tiles accurate at deep zoom.
fn clip_axis(a: f64, b: f64, clip: Rangef) -> Option<(Rangef, Rangef)> {
    let (p, q) = (a.max(clip.min as f64), b.min(clip.max as f64));
    let n = (RASTER + 2) as f64;
    let uv = |v: f64| remap(v, a..=b, 1.0 / n..=(n - 1.0) / n) as f32;
    (p < q).then(|| (Rangef::new(p as f32, q as f32), Rangef::new(uv(p), uv(q))))
}

/// The canvas `rect`, centered on `(cx, cy)`, at `scale` fixed-point units per physical pixel.
#[derive(Clone, Copy)]
struct View {
    cx: i64,
    cy: i64,
    scale: f64,
    rect: Rect,
    ppp: f32,
}

impl View {
    /// Coarsest raster zoom whose texels are no wider than a screen pixel.
    fn target_z(&self) -> u32 {
        (DOMAIN as f64 / (RASTER as f64 * self.scale)).log2().ceil().max(0.0) as u32
    }

    /// Tiles at zoom `z` under the view, plus half a tile of margin so that panning finds them ready.
    fn tiles(&self, z: u32) -> impl Iterator<Item = TileId> + use<> {
        let side = (DOMAIN >> z) as f64;
        let last = (1i64 << z) - 1;
        let size = self.rect.size() * self.ppp;
        let range = |c: i64, len: f32| {
            let half = len as f64 * 0.5 * self.scale + side * 0.5;
            let tile = |v: f64| ((v / side).floor() as i64).clamp(0, last);
            tile(c as f64 - half)..=tile(c as f64 + half)
        };
        let (xs, ys) = (range(self.cx, size.x), range(self.cy, size.y));
        ys.flat_map(move |y| xs.clone().map(move |x| (z, x, y)))
    }

    /// The center moved by pixel offset `u` at `scale` units per pixel; a zoomed-out view can span more than i64.
    fn shifted(&self, (ux, uy): (f64, f64), scale: f64) -> (i64, i64) {
        let moved = |c: i64, u: f64| c.saturating_add((u * scale).round() as i64);
        (moved(self.cx, ux), moved(self.cy, uy))
    }

    fn recenter(&mut self, u: (f64, f64), scale: f64) {
        let (x, y) = self.shifted(u, scale);
        (self.cx, self.cy) = (x.clamp(0, DOMAIN as i64), y.clamp(0, DOMAIN as i64));
    }

    /// This view magnified twice around pixel offset `u` from its center.
    fn zoomed_in(&self, u: (f64, f64)) -> View {
        let mut view = View { scale: self.scale * 0.5, ..*self };
        view.recenter(u, view.scale);
        view
    }

    fn fit_scale(&self) -> f64 {
        DOMAIN as f64 / (self.rect.size().min_elem().max(1.0) * self.ppp * 0.94) as f64
    }

    fn reset(&mut self) {
        (self.cx, self.cy) = ((DOMAIN / 2) as i64, (DOMAIN / 2) as i64);
        self.scale = self.fit_scale();
    }

    /// Offset of screen point `p` from the view center, in physical pixels.
    fn pixel_offset(&self, p: Pos2) -> (f64, f64) {
        let d = (p - self.rect.center()) * self.ppp;
        (d.x as f64, d.y as f64)
    }

    fn to_world(self, p: Pos2) -> (i64, i64) {
        self.shifted(self.pixel_offset(p), self.scale)
    }

    fn screen(&self, x: i64, y: i64) -> (f64, f64) {
        let axis = |center: f32, v: i64, c: i64| center as f64 + (v - c) as f64 / self.scale / self.ppp as f64;
        let center = self.rect.center();
        (axis(center.x, x, self.cx), axis(center.y, y, self.cy))
    }

    fn to_screen(self, x: i64, y: i64) -> Pos2 {
        let (x, y) = self.screen(x, y);
        pos2(x as f32, y as f32)
    }

    /// Zooms keeping `p` fixed, without going through world coordinates that could saturate.
    fn zoom_at(&mut self, p: Pos2, factor: f64) {
        let old_scale = self.scale;
        self.scale = (old_scale / factor).clamp(MIN_SCALE, self.fit_scale() * 4.0);
        self.recenter(self.pixel_offset(p), old_scale - self.scale);
    }

    fn pan_pixels(&mut self, dx: f64, dy: f64) {
        self.recenter((-dx, -dy), self.scale);
    }
}

struct App {
    view: View,
    layout: Layout,
    seed_salt: u64,
    params: Params,
    show_seeds: bool,

    /// The latest texture of each tile, from any epoch until its replacement arrives.
    textures: Cache<TileId, (u64, TextureHandle)>,
    jobs: mpsc::Sender<Arc<Job>>,
    done: mpsc::Receiver<Rendered>,
    job: Arc<Job>,
    missing: usize,
    dragged_seed: Option<usize>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (jobs, job_rx) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();
        let ctx = cc.egui_ctx.clone();
        std::thread::spawn(move || worker(job_rx, done_tx, ctx));
        let seed_salt = 1;
        let params = Params { rain: 2017, seeds: random_seeds(seed_salt, 6), palette_offset: 0.0, density: 1.0, aa: 2 };
        App {
            view: View { cx: 0, cy: 0, scale: 1.0, rect: Rect::ZERO, ppp: 1.0 },
            layout: Layout::Random,
            seed_salt,
            show_seeds: true,
            textures: Cache::new(TEXTURE_BUDGET),
            jobs,
            done,
            job: Job::new(0, &params, Arc::new(World::new(params.rain, params.seeds.clone())), Vec::new()),
            params,
            missing: 0,
            dragged_seed: None,
        }
    }

    fn regenerate_seeds(&mut self, k: usize) {
        let salt = self.seed_salt;
        let at = |x: f64, y: f64| ((x * DOMAIN as f64) as u64, (y * DOMAIN as f64) as u64);
        self.params.seeds = match self.layout {
            Layout::Random => random_seeds(salt, k),
            Layout::Circle => {
                let phase: f64 = SplitMix64::seed_from_u64(salt).random();
                (0..k)
                    .map(|i| std::f64::consts::TAU * (i as f64 / k as f64 + phase))
                    .map(|a| at(0.5 + 0.3 * a.cos(), 0.5 + 0.3 * a.sin()))
                    .collect()
            }
            Layout::Grid => {
                let m = (k as f64).sqrt().ceil() as usize;
                let rows = k.div_ceil(m);
                (0..k).map(|i| at(((i % m) as f64 + 0.5) / m as f64, ((i / m) as f64 + 0.5) / rows as f64)).collect()
            }
            Layout::Manual => {
                let mut seeds = std::mem::take(&mut self.params.seeds);
                seeds.truncate(k);
                seeds.extend(random_seeds(salt, k).into_iter().skip(seeds.len()));
                seeds
            }
        };
    }

    fn edit_seeds(&mut self) -> &mut Vec<(u64, u64)> {
        self.layout = Layout::Manual;
        &mut self.params.seeds
    }

    fn nearest_seed(&self, p: Pos2) -> Option<usize> {
        self.params
            .seeds
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| (i, self.view.to_screen(x as i64, y as i64).distance(p)))
            .filter(|&(_, d)| d < PICK_RADIUS)
            .min_by_key(|&(_, d)| OrderedFloat(d))
            .map(|(i, _)| i)
    }

    /// Uploads finished tiles and asks the worker for missing ones, coarsest and nearest to `anchor`
    /// first. With the view complete, it loads what a zoom at `anchor` will show.
    fn update_tiles(&mut self, ctx: &egui::Context, anchor: (f64, f64)) {
        for (epoch, id, image) in self.done.try_iter() {
            self.textures.insert(id, (epoch, ctx.load_texture("tile", image, TextureOptions::LINEAR)));
        }
        let view = self.view;
        let z = view.target_z();
        let old = self.job.clone();
        let new_epoch = self.params != old.params;
        let epoch = old.epoch + new_epoch as u64;
        let stale = |id: &TileId| self.textures.get(id).is_none_or(|t| t.0 != epoch);
        let mut want: Vec<TileId> = [0, z.saturating_sub(1), z].into_iter().flat_map(|z| view.tiles(z)).filter(stale).collect();
        want.sort_unstable();
        want.dedup();
        self.missing = want.len();
        if want.is_empty() {
            let deepest = View { scale: MIN_SCALE, ..view }.target_z();
            let zoomed_out = View { scale: view.scale * 2.0, ..view }.tiles(z.saturating_sub(1));
            let zoomed_in = view.zoomed_in(anchor).tiles(z + 1).filter(|_| z < deepest);
            want = zoomed_out.chain(zoomed_in).filter(stale).collect();
        }
        let (ax, ay) = view.shifted(anchor, view.scale);
        want.sort_by_cached_key(|&(tile_z, x, y)| {
            let side = (DOMAIN >> tile_z) as f64;
            let dist = |t: i64, a: i64| (t as f64 + 0.5 - a as f64 / side).abs();
            (tile_z, OrderedFloat(dist(x, ax).max(dist(y, ay))))
        });
        if new_epoch || want.iter().any(|id| !old.tiles.contains(id)) {
            // While a seed is dragged, the running job finishes so that the screen keeps updating.
            old.cancel.store(!new_epoch || self.dragged_seed.is_none(), Relaxed);
            let p = &self.params;
            let same_world = (p.rain, &p.seeds) == (old.params.rain, &old.params.seeds);
            let world = if same_world { old.world.clone() } else { Arc::new(World::new(p.rain, p.seeds.clone())) };
            self.job = Job::new(epoch, p, world, want);
            self.jobs.send(self.job.clone()).expect("worker runs as long as the app");
        }
    }

    /// Paints coarse to fine, with the target zoom on top. The root tile covers the whole
    /// domain, so these layers leave no gap.
    fn paint_tiles(&self, painter: &egui::Painter) {
        let z = self.view.target_z();
        let mut layers: Vec<_> = self.textures.iter().filter(|(id, _)| self.missing > 0 || id.0 == z).collect();
        layers.sort_unstable_by_key(|&(&(tile_z, ..), _)| (tile_z == z, tile_z));
        for (&(tile_z, tx, ty), (_, texture)) in layers {
            let side = (DOMAIN >> tile_z) as i64;
            let (ax, ay) = self.view.screen(tx * side, ty * side);
            let (bx, by) = self.view.screen((tx + 1) * side, (ty + 1) * side);
            let (clip_x, clip_y) = (self.view.rect.x_range(), self.view.rect.y_range());
            if let (Some((x, u)), Some((y, v))) = (clip_axis(ax, bx, clip_x), clip_axis(ay, by, clip_y)) {
                painter.image(texture.id(), Rect::from_x_y_ranges(x, y), Rect::from_x_y_ranges(u, v), Color32::WHITE);
            }
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Aldous coloring");
        ui.label("Poisson rain, n = ∞ at pixel precision");
        ui.separator();

        ui.strong("Seeds");
        let before = (self.layout, self.seed_salt, self.params.seeds.len());
        egui::ComboBox::from_label("layout").selected_text(format!("{:?}", self.layout)).show_ui(ui, |ui| {
            for layout in [Layout::Random, Layout::Circle, Layout::Grid, Layout::Manual] {
                ui.selectable_value(&mut self.layout, layout, format!("{layout:?}"));
            }
        });
        let mut k = before.2;
        ui.add(egui::Slider::new(&mut k, 1..=5000).logarithmic(true).text("count"));
        ui.horizontal(|ui| {
            if ui.button("New positions").clicked() {
                self.seed_salt = rand::random();
                if self.layout == Layout::Manual {
                    self.layout = Layout::Random;
                }
            }
            ui.checkbox(&mut self.show_seeds, "show");
        });
        if (self.layout, self.seed_salt, k) != before {
            self.regenerate_seeds(k);
        }
        if ui.button("New colors").clicked() {
            self.params.palette_offset = rand::random();
        }

        ui.separator();
        ui.strong("Rain");
        ui.horizontal(|ui| {
            ui.label("realization");
            ui.add(egui::DragValue::new(&mut self.params.rain));
            if ui.button("New").clicked() {
                self.params.rain = rand::random_range(0..1_000_000);
            }
        });

        ui.separator();
        ui.strong("Quality");
        ui.add(egui::Slider::new(&mut self.params.density, 0.25..=4.0).logarithmic(true).text("particles / texel (min)"));
        ui.add(egui::Slider::new(&mut self.params.aa, 1..=4).text("supersampling"));

        ui.separator();
        ui.strong("View");
        let z = self.view.target_z();
        let level = level_for(z, self.params.density);
        ui.label(format!("zoom ×{:.3e}", self.view.fit_scale() / self.view.scale));
        ui.label(format!("n ≈ {:.3e} rain points", s_top(level)));
        ui.label(format!("level {level} · tile zoom {z} · {} tiles missing", self.missing));
        ui.label(format!(
            "cache: {:.1} M particles · {} textures",
            self.job.world.cached_particles() as f64 / 1e6,
            self.textures.len()
        ));
        if ui.button("Reset view (R)").clicked() {
            self.view.reset();
        }

        ui.separator();
        ui.small(
            "drag: pan\nwheel / pinch: zoom\narrows, +/-: pan, zoom\n\
             Shift+click: add a seed\nright click: remove a seed\nCtrl+drag: move a seed",
        );
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let rect = ui.max_rect();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let ctx = ui.ctx().clone();
        let first_frame = self.view.rect == Rect::ZERO;
        (self.view.rect, self.view.ppp) = (rect, ctx.pixels_per_point());
        if first_frame {
            self.view.reset();
        }

        let (scroll, pinch, mods) = ctx.input(|i| (i.smooth_scroll_delta().y, i.zoom_delta(), i.modifiers));
        let pointer = resp.hover_pos().unwrap_or(rect.center());
        if resp.hovered() && (scroll != 0.0 || pinch != 1.0) {
            self.view.zoom_at(pointer, pinch as f64 * (scroll as f64 * 0.004).exp());
        }

        let press = resp.interact_pointer_pos();
        if resp.drag_started() {
            self.dragged_seed = press.filter(|_| mods.command).and_then(|p| self.nearest_seed(p));
        }
        if resp.dragged() {
            if let (Some(i), Some(p)) = (self.dragged_seed, press) {
                let (x, y) = self.view.to_world(p);
                let clamp = |v: i64| v.clamp(0, DOMAIN as i64 - 1) as u64;
                self.edit_seeds()[i] = (clamp(x), clamp(y));
            } else {
                let d = resp.drag_delta() * self.view.ppp;
                self.view.pan_pixels(d.x as f64, d.y as f64);
            }
        }
        if resp.drag_stopped() {
            self.dragged_seed = None;
        }
        if resp.clicked()
            && mods.shift
            && let Some(p) = press
        {
            let (x, y) = self.view.to_world(p);
            if (0..DOMAIN as i64).contains(&x) && (0..DOMAIN as i64).contains(&y) {
                self.edit_seeds().push((x as u64, y as u64));
            }
        }
        if resp.secondary_clicked()
            && self.params.seeds.len() > 1
            && let Some(i) = press.and_then(|p| self.nearest_seed(p))
        {
            self.edit_seeds().remove(i);
        }

        if !ctx.egui_wants_keyboard_input() {
            use egui::Key::*;
            let key = |k| ctx.input(|i| i.key_down(k)) as i32 as f64;
            let (dx, dy, dz) = (key(ArrowLeft) - key(ArrowRight), key(ArrowUp) - key(ArrowDown), key(Plus).max(key(Equals)) - key(Minus));
            let dt = ctx.input(|i| i.stable_dt.min(0.05)) as f64 * 6.0;
            if dx != 0.0 || dy != 0.0 {
                let step = 0.1 * (rect.size().min_elem() * self.view.ppp) as f64 * dt;
                self.view.pan_pixels(dx * step, dy * step);
                ctx.request_repaint();
            }
            if dz != 0.0 {
                self.view.zoom_at(rect.center(), (1.5 * dt * dz).exp());
                ctx.request_repaint();
            }
            if ctx.input(|i| i.key_pressed(R)) {
                self.view.reset();
            }
        }

        self.update_tiles(&ctx, self.view.pixel_offset(pointer));
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, BACKGROUND);
        self.paint_tiles(&painter);

        if self.show_seeds && self.params.seeds.len() <= 2000 {
            for (&(x, y), &color) in self.params.seeds.iter().zip(&self.job.palette) {
                let p = self.view.to_screen(x as i64, y as i64);
                if rect.expand(8.0).contains(p) {
                    painter.circle(p, 4.5, color, Stroke::new(1.5, Color32::BLACK));
                    painter.circle_stroke(p, 6.0, Stroke::new(1.0, Color32::WHITE));
                }
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("controls").default_size(270.0).resizable(false).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
        });
        egui::CentralPanel::no_frame().show(ui, |ui| self.canvas(ui));
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 860.0]).with_title("Aldous coloring"),
        ..Default::default()
    };
    eframe::run_native("aldous_coloring", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
