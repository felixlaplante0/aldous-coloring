//! Aldous' Poissonian coloring on the unit square, generated lazily at every scale.
//!
//! Rain particles form a Poisson process on `[0,1]^2 x [0, inf)` with intensity
//! `ds dz`; ordered by arrival time `s` they are i.i.d. uniform points, so this is
//! exactly the "elementary" model of the paper with `n = inf`. Level `l` holds the
//! arrivals with `s` in `[4^l, 4^(l+1))` (level 0: `[0, 4)`), on a `2^l x 2^l` grid;
//! each cell is generated on demand from a hash of `(rain_seed, l, cx, cy)`.

use eframe::egui::{Color32, ColorImage, Rgba, emath::OrderedFloat};
use quick_cache::{Weighter, sync::Cache};
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson};
use rand_xoshiro::SplitMix64;
use rayon::prelude::*;
use std::convert::Infallible;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, LazyLock};

/// The unit square is `[0, 2^BITS)^2` in fixed point.
const BITS: u32 = 62;
pub const DOMAIN: u64 = 1 << BITS;
/// Deepest level: cells of side `2^-48` still keep 14 random bits per coordinate.
pub const MAX_LEVEL: u32 = 48;
/// Texels per side of a raster tile, without its 1-texel gutter on each edge.
pub const RASTER: usize = 256;
/// Cells per side of a rain tile.
const TILE: i64 = 64;
/// Neighbor cells loaded around a rain tile so that lines of descent can be followed locally.
const PAD: i64 = 6;
/// Search radius, in cells, trusted before falling back to the global search.
const REACH: i64 = 3;
/// A level-`l` picture shows the arrivals before `RHO * 4^l`, i.e. `RHO` particles per cell.
const RHO: f64 = 2.0;
/// Upper bound on cached particles.
const CACHE_BUDGET: u64 = 40_000_000;

const UNKNOWN: u32 = u32::MAX;
const EMPTY: u32 = u32::MAX - 1;
const MIXED: u32 = u32::MAX - 2;

static POISSON: LazyLock<[Poisson<f64>; 2]> = LazyLock::new(|| [4.0, 3.0].map(|mean| Poisson::new(mean).unwrap()));

#[derive(Clone, Copy)]
struct Particle {
    x: u64,
    y: u64,
    /// Arrival time; seeds use `s = -(index + 1)`.
    s: f64,
}

impl Particle {
    fn dist2(&self, x: u64, y: u64) -> f64 {
        let (dx, dy) = (self.x.abs_diff(x) as f64, self.y.abs_diff(y) as f64);
        dx * dx + dy * dy
    }
}

fn pow4(l: u32) -> f64 {
    4f64.powi(l as i32)
}

fn level_start(l: u32) -> f64 {
    if l == 0 { 0.0 } else { pow4(l) }
}

/// Number of rain particles shown by a picture at `level`.
pub fn s_top(level: u32) -> f64 {
    RHO * pow4(level)
}

fn cell_particles(rain: u64, l: u32, cx: u64, cy: u64) -> impl Iterator<Item = Particle> {
    let hash = [l as u64, cx, cy].into_iter().fold(rain, |h, v| SplitMix64::seed_from_u64(h ^ v).next_u64());
    let mut rng = SplitMix64::seed_from_u64(hash);
    let n = POISSON[(l != 0) as usize].sample(&mut rng) as usize;
    let b = BITS - l;
    let times = level_start(l)..pow4(l + 1);
    (0..n).map(move |_| Particle {
        x: (cx << b) | (rng.next_u64() >> (64 - b)),
        y: (cy << b) | (rng.next_u64() >> (64 - b)),
        s: rng.random_range(times.clone()),
    })
}

/// Colored particles of a padded block around `TILE x TILE` cells of one level: all arrivals
/// before `s_top(level)`, bucketed by cell and sorted by arrival time within each cell.
struct Tile {
    shift: u32,
    x0: i64,
    y0: i64,
    w: i64,
    h: i64,
    start: Vec<u32>,
    parts: Vec<Particle>,
    /// `UNKNOWN` beyond `REACH` cells around the core.
    colors: Vec<u32>,
    /// Color of the core cells whose samples can only take that color, else `MIXED`.
    solid: Vec<u32>,
}

impl Tile {
    fn cell(&self, cx: i64, cy: i64) -> Option<usize> {
        let (gx, gy) = (cx - self.x0, cy - self.y0);
        ((0..self.w).contains(&gx) && (0..self.h).contains(&gy)).then(|| (gy * self.w + gx) as usize)
    }

    fn cell_of(&self, x: u64, y: u64) -> Option<usize> {
        self.cell((x >> self.shift) as i64, (y >> self.shift) as i64)
    }

    fn bucket(&self, c: usize) -> Range<usize> {
        self.start[c] as usize..self.start[c + 1] as usize
    }

    /// Index of the nearest particle with `s < s_lim`, or `None` if it may lie outside the
    /// block or beyond `REACH` rings of cells.
    fn nearest(&self, x: u64, y: u64, s_lim: f64) -> Option<usize> {
        let side = 1u64 << self.shift;
        let n = 1i64 << (BITS - self.shift);
        let (bx, by) = ((x >> self.shift) as i64, (y >> self.shift) as i64);
        let (fx, fy) = (x & (side - 1), y & (side - 1));
        let inner = fx.min(side - 1 - fx).min(fy).min(side - 1 - fy) as f64;
        let (mut best, mut found) = (f64::INFINITY, None);
        for r in 0..=REACH {
            for cy in (by - r).max(0)..=(by + r).min(n - 1) {
                let step = if (cy - by).abs() == r { 1 } else { 2 * r };
                for cx in (bx - r..=bx + r).step_by(step as usize).filter(|cx| (0..n).contains(cx)) {
                    let c = self.cell(cx, cy)?;
                    for i in self.bucket(c) {
                        let p = &self.parts[i];
                        if p.s >= s_lim {
                            break;
                        }
                        let d = p.dist2(x, y);
                        if d < best {
                            (best, found) = (d, Some(i));
                        }
                    }
                }
            }
            let reach = r as f64 * side as f64 + inner;
            if best <= reach * reach {
                return found;
            }
        }
        None
    }

    fn color_near(&self, x: u64, y: u64) -> Option<u32> {
        let c = self.cell_of(x, y)?;
        match self.solid[c] {
            MIXED => self.nearest(x, y, f64::INFINITY).map(|i| self.colors[i]),
            color => Some(color),
        }
    }

    fn color_of(&self, p: &Particle) -> u32 {
        let mut bucket = self.bucket(self.cell_of(p.x, p.y).unwrap());
        self.colors[bucket.find(|&i| self.parts[i].x == p.x && self.parts[i].y == p.y).unwrap()]
    }

    /// Particles in the 3x3 neighborhood put the nearest one within REACH cells, so a single
    /// color among those REACH cells decides every sample of the cell.
    fn fill_solid(&mut self, core_x: Range<i64>, core_y: Range<i64>) {
        let summary: Vec<u32> = (0..self.solid.len())
            .map(|c| match self.colors[self.bucket(c)].split_first() {
                None => EMPTY,
                Some((first, rest)) if rest.iter().all(|o| o == first) => *first,
                _ => MIXED,
            })
            .collect();
        let merge = |a: u32, b: u32| match (a, b) {
            (EMPTY, c) | (c, EMPTY) => c,
            _ if a == b => a,
            _ => MIXED,
        };
        let (x1, y1) = (self.x0 + self.w, self.y0 + self.h);
        let mut solid = vec![MIXED; summary.len()];
        for cy in core_y {
            // Merging columns first takes about 16 reads per cell instead of 49.
            let rows = (cy - REACH).max(self.y0)..(cy + REACH + 1).min(y1);
            let merged_column = |gx| rows.clone().map(|gy| summary[self.cell(gx, gy).unwrap()]).fold(EMPTY, merge);
            let columns: Vec<u32> = (self.x0..x1).map(merged_column).collect();
            for cx in core_x.clone() {
                let window = (cx - REACH).max(self.x0)..(cx + REACH + 1).min(x1);
                let color = window.map(|gx| columns[(gx - self.x0) as usize]).fold(EMPTY, merge);
                let occupied = |gx, gy| self.cell(gx, gy).is_some_and(|c| summary[c] != EMPTY);
                if (cy - 1..=cy + 1).any(|gy| (cx - 1..=cx + 1).any(|gx| occupied(gx, gy))) {
                    solid[self.cell(cx, cy).unwrap()] = color;
                }
            }
        }
        self.solid = solid;
    }

    /// Colors particle `i` and the line of descent leading to it, as far as it is unknown.
    fn resolve(&mut self, i: usize, world: &World, coarse: &TileSet, s_old: f64, path: &mut Vec<usize>) {
        let mut j = i;
        let color = loop {
            if self.colors[j] != UNKNOWN {
                break self.colors[j];
            }
            path.push(j);
            let p = self.parts[j];
            if p.s < 0.0 {
                break seed_index(p);
            }
            if p.s < s_old {
                break coarse.at((p.x >> (self.shift + 1)) as i64, (p.y >> (self.shift + 1)) as i64).color_of(&p);
            }
            match self.nearest(p.x, p.y, p.s) {
                Some(k) => j = k,
                None => break world.color_of(p),
            }
        };
        for k in path.drain(..) {
            self.colors[k] = color;
        }
    }

    fn compute(world: &World, (level, tx, ty): TileKey, coarse: &TileSet) -> Tile {
        let n = 1i64 << level;
        let shift = BITS - level;
        let (px, py) = (padded(tx, tx, n), padded(ty, ty, n));
        let (x0, y0, w, h) = (px.start, py.start, px.end - px.start, py.end - py.start);
        let mut tile = Tile { shift, x0, y0, w, h, start: vec![], parts: vec![], colors: vec![], solid: vec![MIXED; (w * h) as usize] };

        let s_lim = s_top(level);
        let mut parts: Vec<Particle> = world.seed_particles().filter(|p| tile.cell_of(p.x, p.y).is_some()).collect();
        for l in 0..=level {
            let d = level - l;
            for gy in y0 >> d..=(py.end - 1) >> d {
                for gx in x0 >> d..=(px.end - 1) >> d {
                    let rain = cell_particles(world.rain, l, gx as u64, gy as u64);
                    parts.extend(rain.filter(|p| p.s < s_lim && tile.cell_of(p.x, p.y).is_some()));
                }
            }
        }
        parts.shrink_to_fit();
        let cell = |p: &Particle| tile.cell_of(p.x, p.y).unwrap();
        parts.sort_unstable_by_key(|p| (cell(p), OrderedFloat(p.s)));
        tile.start = (0..=(w * h) as usize).map(|c| parts.partition_point(|p| cell(p) < c) as u32).collect();
        tile.colors = vec![UNKNOWN; parts.len()];
        tile.parts = parts;

        let s_old = if level == 0 { f64::NEG_INFINITY } else { s_top(level - 1) };
        let core = |t: i64| t * TILE..((t + 1) * TILE).min(n);
        let (core_x, core_y) = (core(tx), core(ty));
        let near = |v: u64, core: &Range<i64>| (core.start - REACH..core.end + REACH).contains(&((v >> shift) as i64));
        let mut path = Vec::new();
        for i in 0..tile.parts.len() {
            let p = tile.parts[i];
            if near(p.x, &core_x) && near(p.y, &core_y) {
                tile.resolve(i, world, coarse, s_old, &mut path);
            }
        }
        tile.fill_solid(core_x, core_y);
        tile
    }
}

/// Cells of tiles `t0..=t1` along one axis, with the margin loaded around them.
fn padded(t0: i64, t1: i64, n: i64) -> Range<i64> {
    (t0 * TILE - PAD).max(0)..((t1 + 1) * TILE + PAD).min(n)
}

fn seed_index(p: Particle) -> u32 {
    (-p.s) as u32 - 1
}

/// A rectangle of rain tiles of one level, kept alive whatever the cache evicts.
#[derive(Default)]
struct TileSet {
    tx0: i64,
    ty0: i64,
    ntx: i64,
    tiles: Vec<Arc<Tile>>,
}

impl TileSet {
    fn at(&self, cx: i64, cy: i64) -> &Tile {
        &self.tiles[((cy / TILE - self.ty0) * self.ntx + cx / TILE - self.tx0) as usize]
    }
}

type TileKey = (u32, i64, i64);
/// `(z, x, y)`: raster tile `(x, y)` of the `2^z x 2^z` grid over the domain.
pub type TileId = (u32, i64, i64);

#[derive(Clone)]
struct ByParticles;

impl Weighter<TileKey, Arc<Tile>> for ByParticles {
    fn weight(&self, _: &TileKey, tile: &Arc<Tile>) -> u64 {
        tile.parts.len() as u64 + 1
    }
}

/// One realization: rain and seeds, with a cache of rain tiles.
pub struct World {
    rain: u64,
    seeds: Vec<(u64, u64)>,
    tiles: Cache<TileKey, Arc<Tile>, ByParticles>,
}

impl World {
    pub fn new(rain: u64, seeds: Vec<(u64, u64)>) -> Self {
        World { rain, seeds, tiles: Cache::with_weighter(4096, CACHE_BUDGET, ByParticles) }
    }

    pub fn cached_particles(&self) -> u64 {
        self.tiles.weight()
    }

    fn seed_particles(&self) -> impl Iterator<Item = Particle> {
        self.seeds.iter().enumerate().map(|(i, &(x, y))| Particle { x, y, s: -(i as f64) - 1.0 })
    }

    /// The tiles covering inclusive ranges of level-`level` cells, computed on demand along
    /// with the coarser tiles holding the colors of their older particles. Finished tiles
    /// stay cached if `cancel` interrupts the call.
    fn tile_set(&self, level: u32, xs: (i64, i64), ys: (i64, i64), cancel: &AtomicBool) -> Option<TileSet> {
        if cancel.load(Relaxed) {
            return None;
        }
        let (tx0, tx1, ty0, ty1) = (xs.0 / TILE, xs.1 / TILE, ys.0 / TILE, ys.1 / TILE);
        let keys: Vec<TileKey> = (ty0..=ty1).flat_map(|ty| (tx0..=tx1).map(move |tx| (level, tx, ty))).collect();
        let tiles = match keys.iter().map(|k| self.tiles.get(k)).collect::<Option<Vec<_>>>() {
            Some(tiles) => tiles,
            None => {
                let coarse = if level == 0 {
                    TileSet::default()
                } else {
                    let halve = |t0, t1| {
                        let cells = padded(t0, t1, 1i64 << level);
                        (cells.start >> 1, (cells.end - 1) >> 1)
                    };
                    self.tile_set(level - 1, halve(tx0, tx1), halve(ty0, ty1), cancel)?
                };
                // Tile::compute stays serial: a rayon steal while this key is locked could deadlock.
                keys.par_iter()
                    .map(|&k| {
                        let compute = || Ok::<_, Infallible>(Arc::new(Tile::compute(self, k, &coarse)));
                        self.tiles.get_or_insert_with(&k, compute).unwrap()
                    })
                    .collect()
            }
        };
        Some(TileSet { tx0, ty0, ntx: tx1 - tx0 + 1, tiles })
    }

    /// Nearest seed or rain particle with arrival time `< s_lim`.
    fn nearest_before(&self, x: u64, y: u64, s_lim: f64) -> Particle {
        let mut hit = self.seed_particles().min_by_key(|p| OrderedFloat(p.dist2(x, y))).expect("at least one seed");
        let mut best = hit.dist2(x, y);
        for l in (0..=MAX_LEVEL).take_while(|&l| level_start(l) < s_lim) {
            let b = BITS - l;
            let side = 1u64 << b;
            let gap = |v: u64, lo: u64| v.abs_diff(v.clamp(lo, lo + side - 1)) as f64;
            let r = best.sqrt() * (1.0 + 1e-9) + 4096.0;
            let cells = |v: u64| ((v as f64 - r).max(0.0) as u64 >> b)..=((v as f64 + r).min((DOMAIN - 1) as f64) as u64 >> b);
            for cy in cells(y) {
                let gy = gap(y, cy << b);
                for cx in cells(x) {
                    let gx = gap(x, cx << b);
                    if gx * gx + gy * gy >= best {
                        continue;
                    }
                    for p in cell_particles(self.rain, l, cx, cy).filter(|p| p.s < s_lim) {
                        let d = p.dist2(x, y);
                        if d < best {
                            (best, hit) = (d, p);
                        }
                    }
                }
            }
        }
        hit
    }

    /// Color (seed index) of a particle, following its line of descent.
    fn color_of(&self, mut p: Particle) -> u32 {
        while p.s >= 0.0 {
            p = self.nearest_before(p.x, p.y, p.s);
        }
        seed_index(p)
    }
}

fn texel_size(z: u32) -> f64 {
    (DOMAIN >> z) as f64 / RASTER as f64
}

/// Coarsest level giving at least `density` particles per texel at raster zoom `z`.
pub fn level_for(z: u32, density: f64) -> u32 {
    let a = texel_size(z) / DOMAIN as f64;
    (0..MAX_LEVEL).find(|&l| s_top(l) * a * a >= density).unwrap_or(MAX_LEVEL)
}

/// Domain coordinates of the `aa` samples of each texel of tile index `t`, gutter included.
/// Gutter samples past the domain edge are clamped to it, so textures never hold background.
fn sample_coords(t: i64, z: u32, aa: u32) -> Vec<u64> {
    let (origin, texel) = (t * (DOMAIN >> z) as i64, texel_size(z));
    (0..RASTER + 2)
        .flat_map(|i| (0..aa).map(move |k| (i, k)))
        .map(|(i, k)| {
            let offset = (i as f64 - 1.0 + (k as f64 + 0.5) / aa as f64) * texel;
            (origin + offset.round() as i64).clamp(0, DOMAIN as i64 - 1) as u64
        })
        .collect()
}

/// Renders raster tile `(z, tx, ty)` from the rain of `level`, with `aa x aa` samples per texel,
/// or returns `None` if `cancel` fired while computing the rain.
pub fn render_tile(world: &World, (z, tx, ty): TileId, level: u32, aa: u32, palette: &[Color32], cancel: &AtomicBool) -> Option<ColorImage> {
    let shift = BITS - level;
    let (xs, ys) = (sample_coords(tx, z, aa), sample_coords(ty, z, aa));
    let cells = |v: &[u64]| ((v[0] >> shift) as i64, (v[v.len() - 1] >> shift) as i64);
    let rain = world.tile_set(level, cells(&xs), cells(&ys), cancel)?;
    let color = &|x: u64, y: u64| {
        let tile = rain.at((x >> shift) as i64, (y >> shift) as i64);
        tile.color_near(x, y).unwrap_or_else(|| world.color_of(Particle { x, y, s: s_top(level) }))
    };
    let pixels = ys
        .par_chunks(aa as usize)
        .flat_map_iter(|py| {
            xs.chunks(aa as usize).map(|px| {
                let samples = || py.iter().flat_map(|&y| px.iter().map(move |&x| color(x, y)));
                let mut rest = samples();
                let first = rest.next().unwrap();
                if rest.all(|c| c == first) {
                    return palette[first as usize];
                }
                let sum = samples().fold(Rgba::TRANSPARENT, |sum, c| sum + Rgba::from(palette[c as usize]));
                Color32::from(sum * (1.0 / (aa * aa) as f32))
            })
        })
        .collect();
    Some(ColorImage::new([RASTER + 2, RASTER + 2], pixels))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tile-based colors must agree with the direct search along lines of descent.
    #[test]
    fn tiles_match_direct_search() {
        let seeds: Vec<(u64, u64)> = (1..6u64).map(|i| (i * (DOMAIN / 7), (i * i % 6 + 1) * (DOMAIN / 8))).collect();
        let palette: Vec<Color32> = (0..5).map(|i| Color32::from_rgb(i as u8 * 40 + 1, 7, 9)).collect();
        let world = World::new(7, seeds.clone());
        let fresh = World::new(7, seeds);
        for (z, density) in [(0, 1.0), (5, 1.0), (14, 2.0), (30, 1.0)] {
            let id = (z, ((DOMAIN / 3) >> (BITS - z)) as i64, ((DOMAIN / 2) >> (BITS - z)) as i64);
            let level = level_for(z, density);
            let image = render_tile(&world, id, level, 1, &palette, &AtomicBool::new(false)).unwrap();
            let (xs, ys) = (sample_coords(id.1, z, 1), sample_coords(id.2, z, 1));
            for j in (0..xs.len()).step_by(7) {
                for i in (0..xs.len()).step_by(7) {
                    let direct = fresh.color_of(Particle { x: xs[i], y: ys[j], s: s_top(level) });
                    assert_eq!(image.pixels[j * xs.len() + i], palette[direct as usize], "z {z}, texel ({i}, {j})");
                }
            }
        }
    }
}
