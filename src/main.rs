mod process;

use eframe::egui;
use egui::{Color32, ColorImage, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions, Vec2};
use process::{DOMAIN, MAX_LEVEL, Stats, Tiles, View, World, level_for, mix, unit};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

const BACKGROUND: [u8; 3] = [18, 18, 22];
const PICK_RADIUS: f32 = 14.0;

struct Job {
    id: u64,
    world: Arc<World>,
    view: View,
    density: f64,
    aa: u32,
    palette: Arc<Vec<[u8; 3]>>,
}

#[derive(Clone, Copy)]
struct Frame {
    id: u64,
    view: View,
    stats: Stats,
    ms: f64,
    aa: u32,
}

#[derive(Default)]
struct Shared {
    job: Mutex<Option<Job>>,
    wake: Condvar,
    latest: AtomicU64,
    done: Mutex<Option<(Frame, ColorImage)>>,
}

fn worker(sh: Arc<Shared>, ctx: egui::Context) {
    let mut last_world: Option<Arc<World>> = None;
    loop {
        let job = {
            let mut g = sh.job.lock().unwrap();
            while g.is_none() {
                g = sh.wake.wait(g).unwrap();
            }
            g.take().unwrap()
        };
        let latest = &sh.latest;
        let stale = || latest.load(Relaxed) != job.id;
        let never = || false;
        let draw = |level: u32, aa: u32, tiles: Tiles, cancel: &(dyn Fn() -> bool + Sync)| {
            let t0 = Instant::now();
            let out = process::render(&job.world, &job.view, level, aa, &job.palette, BACKGROUND, tiles, cancel);
            let Some((rgba, stats)) = out else { return false };
            // Same layout as Color32, so this collect reuses the allocation.
            let pixels = rgba.into_iter().map(|[r, g, b, a]| Color32::from_rgba_premultiplied(r, g, b, a)).collect();
            let image = ColorImage::new([job.view.w, job.view.h], pixels);
            let frame = Frame { id: job.id, view: job.view, stats, ms: t0.elapsed().as_secs_f64() * 1e3, aa };
            *sh.done.lock().unwrap() = Some((frame, image));
            ctx.request_repaint();
            true
        };

        let level = level_for(job.view.scale, job.density);
        // A new world (seeds edited) has an empty cache: finish its first picture even if
        // newer jobs arrive, so that dragging a seed keeps updating the screen.
        let fresh = !last_world.as_ref().is_some_and(|w| Arc::ptr_eq(w, &job.world));
        last_world = Some(job.world.clone());
        if !draw(level, 1, Tiles::CachedOnly, &never) {
            // Instant preview from a neighboring cached level while the exact one is computed.
            let near = [level + 1, level.saturating_sub(1), level.saturating_sub(2)];
            for l in near.into_iter().filter(|&l| l != level && l <= MAX_LEVEL) {
                if draw(l, 1, Tiles::CachedOnly, &never) {
                    break;
                }
            }
            if !draw(level, 1, Tiles::Compute, if fresh { &never } else { &stale }) {
                continue;
            }
        }
        if job.aa > 1 && !draw(level, job.aa, Tiles::CachedOnly, &stale) {
            continue;
        }
        if level < MAX_LEVEL {
            job.world.prefetch(&job.view, level + 1, &stale);
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Layout {
    Random,
    Circle,
    Grid,
    Manual,
}

impl Layout {
    const ALL: [Layout; 4] = [Layout::Random, Layout::Circle, Layout::Grid, Layout::Manual];

    fn label(self) -> &'static str {
        match self {
            Layout::Random => "random",
            Layout::Circle => "circle",
            Layout::Grid => "grid",
            Layout::Manual => "manual",
        }
    }
}

fn make_palette(k: usize, salt: u64) -> Vec<[u8; 3]> {
    let offset = unit(mix(salt));
    (0..k)
        .map(|i| {
            let i = i as f64;
            let h = offset + i * 0.618_033_988_75;
            let s = 0.50 + 0.35 * (i * 0.381_966).fract();
            let v = 0.70 + 0.28 * (i * 0.754_877_666).fract();
            egui::ecolor::rgb_from_hsv((h.fract() as f32, s as f32, v as f32)).map(|t| (t * 255.0).round() as u8)
        })
        .collect()
}

fn unit_to_fixed(t: f64) -> u64 {
    ((t.clamp(0.0, 1.0) * DOMAIN as f64) as u64).min(DOMAIN - 1)
}

fn clamp_to_domain(v: i64) -> u64 {
    v.clamp(0, DOMAIN as i64 - 1) as u64
}

fn random_seed(salt: u64, i: usize) -> (u64, u64) {
    (mix(salt ^ mix(2 * i as u64)) >> 2, mix(salt ^ mix(2 * i as u64 + 1)) >> 2)
}

struct App {
    shared: Arc<Shared>,
    next_id: u64,
    dirty: bool,

    view: View,

    layout: Layout,
    seed_salt: u64,
    seeds: Vec<(u64, u64)>,
    rain: u64,
    world: Arc<World>,
    palette_salt: u64,
    palette: Arc<Vec<[u8; 3]>>,

    density: f64,
    aa: u32,
    show_seeds: bool,

    tex: Option<TextureHandle>,
    shown: Option<Frame>,
    dragged_seed: Option<usize>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let shared = Arc::new(Shared::default());
        let (sh, ctx) = (shared.clone(), cc.egui_ctx.clone());
        std::thread::spawn(move || worker(sh, ctx));
        let seed_salt = 1;
        let seeds: Vec<_> = (0..6).map(|i| random_seed(seed_salt, i)).collect();
        let rain = 2017;
        App {
            shared,
            next_id: 0,
            dirty: true,
            view: View { cx: (DOMAIN / 2) as i64, cy: (DOMAIN / 2) as i64, scale: 1.0, w: 0, h: 0 },
            layout: Layout::Random,
            seed_salt,
            world: Arc::new(World::new(rain, seeds.clone())),
            palette: Arc::new(make_palette(seeds.len(), 0)),
            seeds,
            rain,
            palette_salt: 0,
            density: 1.0,
            aa: 2,
            show_seeds: true,
            tex: None,
            shown: None,
            dragged_seed: None,
        }
    }

    fn fit_scale(&self) -> f64 {
        DOMAIN as f64 / (self.view.w.min(self.view.h).max(1) as f64 * 0.94)
    }

    fn reset_view(&mut self) {
        self.view.cx = (DOMAIN / 2) as i64;
        self.view.cy = (DOMAIN / 2) as i64;
        self.view.scale = self.fit_scale();
        self.dirty = true;
    }

    fn regenerate_seeds(&mut self, k: usize) {
        self.seeds = match self.layout {
            Layout::Random => (0..k).map(|i| random_seed(self.seed_salt, i)).collect(),
            Layout::Circle => {
                let phase = unit(mix(self.seed_salt));
                (0..k)
                    .map(|i| {
                        let a = std::f64::consts::TAU * (i as f64 / k as f64 + phase);
                        (unit_to_fixed(0.5 + 0.3 * a.cos()), unit_to_fixed(0.5 + 0.3 * a.sin()))
                    })
                    .collect()
            }
            Layout::Grid => {
                let m = (k as f64).sqrt().ceil() as usize;
                let rows = k.div_ceil(m);
                (0..k)
                    .map(|i| {
                        (
                            unit_to_fixed(((i % m) as f64 + 0.5) / m as f64),
                            unit_to_fixed(((i / m) as f64 + 0.5) / rows as f64),
                        )
                    })
                    .collect()
            }
            Layout::Manual => {
                let mut s = std::mem::take(&mut self.seeds);
                s.truncate(k);
                let n = s.len();
                s.extend((n..k).map(|i| random_seed(self.seed_salt, i)));
                s
            }
        };
        self.rebuild_world();
    }

    /// Applies a hand edit to the seeds.
    fn edit_seeds(&mut self, edit: impl FnOnce(&mut Vec<(u64, u64)>)) {
        edit(&mut self.seeds);
        self.layout = Layout::Manual;
        self.rebuild_world();
    }

    fn rebuild_world(&mut self) {
        self.world = Arc::new(World::new(self.rain, self.seeds.clone()));
        self.rebuild_palette();
    }

    fn rebuild_palette(&mut self) {
        self.palette = Arc::new(make_palette(self.seeds.len(), self.palette_salt));
        self.dirty = true;
    }

    fn submit(&mut self) {
        self.next_id += 1;
        let job = Job {
            id: self.next_id,
            world: self.world.clone(),
            view: self.view,
            density: self.density,
            aa: self.aa,
            palette: self.palette.clone(),
        };
        self.shared.latest.store(self.next_id, Relaxed);
        *self.shared.job.lock().unwrap() = Some(job);
        self.shared.wake.notify_one();
        self.dirty = false;
    }

    /// Offset of screen point `p` from the view center, in physical pixels.
    fn pixel_offset(&self, rect: Rect, ppp: f32, p: Pos2) -> (f64, f64) {
        (
            ((p.x - rect.min.x) * ppp) as f64 - self.view.w as f64 * 0.5,
            ((p.y - rect.min.y) * ppp) as f64 - self.view.h as f64 * 0.5,
        )
    }

    fn to_world(&self, rect: Rect, ppp: f32, p: Pos2) -> (i64, i64) {
        let (ux, uy) = self.pixel_offset(rect, ppp, p);
        (
            self.view.cx.saturating_add((ux * self.view.scale).round() as i64),
            self.view.cy.saturating_add((uy * self.view.scale).round() as i64),
        )
    }

    fn to_screen(&self, rect: Rect, ppp: f32, x: i64, y: i64) -> Pos2 {
        let ux = x.saturating_sub(self.view.cx) as f64 / self.view.scale + self.view.w as f64 * 0.5;
        let uy = y.saturating_sub(self.view.cy) as f64 / self.view.scale + self.view.h as f64 * 0.5;
        rect.min + Vec2::new(ux as f32, uy as f32) / ppp
    }

    fn nearest_seed(&self, rect: Rect, ppp: f32, p: Pos2) -> Option<usize> {
        self.seeds
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| (i, self.to_screen(rect, ppp, x as i64, y as i64).distance(p)))
            .filter(|&(_, d)| d < PICK_RADIUS)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    fn zoom_at(&mut self, rect: Rect, ppp: f32, p: Pos2, factor: f64) {
        let (wx, wy) = self.to_world(rect, ppp, p);
        let min_scale = (DOMAIN >> (MAX_LEVEL + 1)) as f64;
        let max_scale = self.fit_scale() * 4.0;
        self.view.scale = (self.view.scale / factor).clamp(min_scale, max_scale);
        let (ux, uy) = self.pixel_offset(rect, ppp, p);
        self.view.cx = wx - (ux * self.view.scale).round() as i64;
        self.view.cy = wy - (uy * self.view.scale).round() as i64;
        self.dirty = true;
    }

    fn pan_pixels(&mut self, dx: f64, dy: f64) {
        let lim = DOMAIN as i64;
        self.view.cx = (self.view.cx - (dx * self.view.scale).round() as i64).clamp(-lim / 2, lim + lim / 2);
        self.view.cy = (self.view.cy - (dy * self.view.scale).round() as i64).clamp(-lim / 2, lim + lim / 2);
        self.dirty = true;
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Aldous coloring");
        ui.label("Poisson rain, n = ∞ at pixel precision");
        ui.separator();

        ui.strong("Seeds");
        let old_layout = self.layout;
        egui::ComboBox::from_label("layout").selected_text(self.layout.label()).show_ui(ui, |ui| {
            for layout in Layout::ALL {
                ui.selectable_value(&mut self.layout, layout, layout.label());
            }
        });
        let mut k = self.seeds.len();
        let k_changed = ui.add(egui::Slider::new(&mut k, 1..=5000).logarithmic(true).text("count")).changed();
        let mut reshuffle = false;
        ui.horizontal(|ui| {
            if ui.button("New positions").clicked() {
                self.seed_salt = mix(self.seed_salt);
                reshuffle = true;
            }
            ui.checkbox(&mut self.show_seeds, "show");
        });
        if reshuffle && self.layout == Layout::Manual {
            self.layout = Layout::Random;
        }
        if reshuffle || k_changed || self.layout != old_layout {
            self.regenerate_seeds(k);
        }
        if ui.button("New colors").clicked() {
            self.palette_salt = mix(self.palette_salt ^ 0x5bd1);
            self.rebuild_palette();
        }

        ui.separator();
        ui.strong("Rain");
        ui.horizontal(|ui| {
            ui.label("realization");
            if ui.add(egui::DragValue::new(&mut self.rain)).changed() {
                self.rebuild_world();
            }
            if ui.button("New").clicked() {
                self.rain = mix(self.rain) % 1_000_000;
                self.rebuild_world();
            }
        });

        ui.separator();
        ui.strong("Quality");
        let density = egui::Slider::new(&mut self.density, 0.25..=4.0).logarithmic(true).text("particles / pixel (min)");
        if ui.add(density).changed() {
            self.dirty = true;
        }
        if ui.add(egui::Slider::new(&mut self.aa, 1..=4).text("supersampling")).changed() {
            self.dirty = true;
        }

        ui.separator();
        ui.strong("View");
        let zoom = self.fit_scale() / self.view.scale;
        ui.label(format!("zoom ×{zoom:.3e}"));
        if let Some(f) = self.shown {
            ui.label(format!("n ≈ {:.3e} rain points", f.stats.s_t));
            ui.label(format!("level {} · {} tiles", f.stats.level, f.stats.tiles));
            ui.label(format!("{:.0} ms · {}×{} samples/px", f.ms, f.aa, f.aa));
            ui.label(format!(
                "cache: {:.1} M particles · {} memoized ancestors",
                f.stats.cached_particles as f64 / 1e6,
                f.stats.memo
            ));
        }
        if ui.button("Reset view (R)").clicked() {
            self.reset_view();
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
        let ppp = ctx.pixels_per_point();
        let (w, h) = (((rect.width() * ppp).round() as usize).max(1), ((rect.height() * ppp).round() as usize).max(1));
        if (w, h) != (self.view.w, self.view.h) {
            let first = self.view.w == 0;
            self.view.w = w;
            self.view.h = h;
            if first {
                self.reset_view();
            }
            self.dirty = true;
        }

        let (scroll, pinch, mods) = ctx.input(|i| (i.smooth_scroll_delta().y, i.zoom_delta(), i.modifiers));
        let pointer = resp.hover_pos().unwrap_or(rect.center());
        if resp.hovered() && (scroll != 0.0 || pinch != 1.0) {
            let factor = pinch as f64 * (scroll as f64 * 0.004).exp();
            self.zoom_at(rect, ppp, pointer, factor);
        }

        if resp.drag_started() && mods.command {
            self.dragged_seed = resp.interact_pointer_pos().and_then(|p| self.nearest_seed(rect, ppp, p));
        }
        if resp.dragged() {
            if let (Some(i), Some(p)) = (self.dragged_seed, resp.interact_pointer_pos()) {
                let (x, y) = self.to_world(rect, ppp, p);
                self.edit_seeds(|s| s[i] = (clamp_to_domain(x), clamp_to_domain(y)));
            } else {
                let d = resp.drag_delta() * ppp;
                self.pan_pixels(d.x as f64, d.y as f64);
            }
        }
        if resp.drag_stopped() {
            self.dragged_seed = None;
        }
        if resp.clicked() && mods.shift
            && let Some(p) = resp.interact_pointer_pos()
        {
            let (x, y) = self.to_world(rect, ppp, p);
            if (0..DOMAIN as i64).contains(&x) && (0..DOMAIN as i64).contains(&y) {
                self.edit_seeds(|s| s.push((x as u64, y as u64)));
            }
        }
        if resp.secondary_clicked() && self.seeds.len() > 1
            && let Some(i) = resp.interact_pointer_pos().and_then(|p| self.nearest_seed(rect, ppp, p))
        {
            self.edit_seeds(|s| {
                s.remove(i);
            });
        }

        if !ctx.egui_wants_keyboard_input() {
            let step = 0.1 * w.min(h) as f64;
            let keys = ctx.input(|i| {
                [
                    i.key_down(egui::Key::ArrowLeft),
                    i.key_down(egui::Key::ArrowRight),
                    i.key_down(egui::Key::ArrowUp),
                    i.key_down(egui::Key::ArrowDown),
                    i.key_down(egui::Key::Plus) || i.key_down(egui::Key::Equals),
                    i.key_down(egui::Key::Minus),
                    i.key_pressed(egui::Key::R),
                ]
            });
            let dt = ctx.input(|i| i.stable_dt).min(0.05) as f64 * 6.0;
            if keys[0] || keys[1] || keys[2] || keys[3] {
                let dx = (keys[0] as i32 - keys[1] as i32) as f64 * step * dt;
                let dy = (keys[2] as i32 - keys[3] as i32) as f64 * step * dt;
                self.pan_pixels(dx, dy);
                ctx.request_repaint();
            }
            if keys[4] || keys[5] {
                let f = if keys[4] { (1.5 * dt).exp() } else { (-1.5 * dt).exp() };
                self.zoom_at(rect, ppp, rect.center(), f);
                ctx.request_repaint();
            }
            if keys[6] {
                self.reset_view();
            }
        }

        if self.dirty {
            self.submit();
        }

        if let Some((frame, image)) = self.shared.done.lock().unwrap().take()
            && self.shown.is_none_or(|s| frame.id >= s.id)
        {
            match &mut self.tex {
                Some(t) => t.set(image, TextureOptions::LINEAR),
                None => self.tex = Some(ctx.load_texture("coloring", image, TextureOptions::LINEAR)),
            }
            self.shown = Some(frame);
        }

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_rgb(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2]));
        if let (Some(tex), Some(f)) = (&self.tex, self.shown) {
            // Re-project the last finished frame onto the current view while the next one renders.
            let half_w = (f.view.w as f64 * 0.5 * f.view.scale) as i64;
            let half_h = (f.view.h as f64 * 0.5 * f.view.scale) as i64;
            let img_rect = Rect::from_min_max(
                self.to_screen(rect, ppp, f.view.cx - half_w, f.view.cy - half_h),
                self.to_screen(rect, ppp, f.view.cx + half_w, f.view.cy + half_h),
            );
            painter.image(tex.id(), img_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }

        if self.show_seeds && self.seeds.len() <= 2000 {
            for (&(x, y), c) in self.seeds.iter().zip(self.palette.iter()) {
                let p = self.to_screen(rect, ppp, x as i64, y as i64);
                if rect.expand(8.0).contains(p) {
                    painter.circle(p, 4.5, Color32::from_rgb(c[0], c[1], c[2]), Stroke::new(1.5, Color32::BLACK));
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
