//! Aldous' Poissonian coloring on the unit square, generated lazily at every scale.
//!
//! Rain particles form a Poisson process on `[0,1]^2 x [0, inf)` with intensity
//! `ds dz`; ordered by arrival time `s` they are i.i.d. uniform points, so this is
//! exactly the "elementary" model of the paper with `n = inf`. Level `l` holds the
//! arrivals with `s` in `[4^l, 4^(l+1))` (level 0: `[0, 4)`), on a `2^l x 2^l` grid;
//! each cell is generated on demand from a hash of `(rain_seed, l, cx, cy)`.

use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

/// The unit square is `[0, 2^BITS)^2` in fixed point.
const BITS: u32 = 62;
pub const DOMAIN: u64 = 1 << BITS;
/// Deepest level: cells of side `2^-48` still keep 14 random bits per coordinate.
pub const MAX_LEVEL: u32 = 48;

#[derive(Clone, Copy, Debug)]
struct Particle {
    x: u64,
    y: u64,
    /// Arrival time; seeds use `s = -(index + 1)`.
    s: f64,
}

impl Particle {
    #[inline]
    fn seed_index(&self) -> Option<u32> {
        (self.s < 0.0).then(|| (-self.s) as u32 - 1)
    }

    #[inline]
    fn key(&self) -> u128 {
        ((self.x as u128) << 64 | self.y as u128) ^ (self.s.to_bits() as u128) << 23
    }
}

#[inline]
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Maps random bits to a uniform float in `[0, 1)`.
#[inline]
pub fn unit(z: u64) -> f64 {
    (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// SplitMix64.
struct Rng(u64);

impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        let z = self.0;
        self.0 = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(z)
    }

    #[inline]
    fn unif(&mut self) -> f64 {
        unit(self.next())
    }
}

#[inline]
fn pow4(l: u32) -> f64 {
    f64::from_bits((1023 + 2 * l as u64) << 52)
}

#[inline]
fn level_start(l: u32) -> f64 {
    if l == 0 { 0.0 } else { pow4(l) }
}

#[inline]
fn level_end(l: u32) -> f64 {
    pow4(l + 1)
}

/// Poisson CDFs for means 4 (level 0) and 3 (other levels).
static POISSON_CDF: std::sync::LazyLock<[[f64; 24]; 2]> = std::sync::LazyLock::new(|| {
    let mut t = [[1.0; 24]; 2];
    for (row, mean) in t.iter_mut().zip([4.0f64, 3.0]) {
        let (mut p, mut cdf) = ((-mean).exp(), 0.0);
        for (n, v) in row.iter_mut().enumerate().take(23) {
            if n > 0 {
                p *= mean / n as f64;
            }
            cdf += p;
            *v = cdf;
        }
    }
    t
});

/// Calls `f` on every rain particle of cell `(cx, cy)` at level `l`.
#[inline]
fn cell_particles(rain: u64, l: u32, cx: u64, cy: u64, mut f: impl FnMut(Particle)) {
    let mut rng = Rng(mix(mix(mix(mix(rain) ^ l as u64) ^ cx) ^ cy));
    let cdf = &POISSON_CDF[(l != 0) as usize];
    let u = rng.unif();
    let mut n = 0;
    while u > cdf[n] {
        n += 1;
    }
    let b = BITS - l;
    let (s0, s1) = (level_start(l), level_end(l));
    for _ in 0..n {
        let x = (cx << b) | (rng.next() >> (64 - b));
        let y = (cy << b) | (rng.next() >> (64 - b));
        let s = s0 + (s1 - s0) * rng.unif();
        f(Particle { x, y, s });
    }
}

#[inline]
fn dist2(ax: u64, ay: u64, bx: u64, by: u64) -> f64 {
    let dx = (ax as i64 - bx as i64) as f64;
    let dy = (ay as i64 - by as i64) as f64;
    dx * dx + dy * dy
}

/// Distance from `v` to the interval `[lo, lo + side)`.
#[inline]
fn gap(v: u64, lo: u64, side: u64) -> f64 {
    if v < lo {
        (lo - v) as f64
    } else if v >= lo + side {
        (v - (lo + side - 1)) as f64
    } else {
        0.0
    }
}

/// Visits the cells at Chebyshev distance exactly `r` from `(bx, by)` inside `[0, w) x [0, h)`.
#[inline]
fn ring(bx: i64, by: i64, r: i64, w: i64, h: i64, mut f: impl FnMut(i64, i64)) {
    if r == 0 {
        if bx >= 0 && bx < w && by >= 0 && by < h {
            f(bx, by);
        }
        return;
    }
    for cx in (bx - r).max(0)..=(bx + r).min(w - 1) {
        if by - r >= 0 {
            f(cx, by - r);
        }
        if by + r < h {
            f(cx, by + r);
        }
    }
    for cy in (by - r + 1).max(0)..=(by + r - 1).min(h - 1) {
        if bx - r >= 0 {
            f(bx - r, cy);
        }
        if bx + r < w {
            f(bx + r, cy);
        }
    }
}

/// Search radius, in cells, trusted by the tile renderer; views load this margin around them.
const REACH: i64 = 3;

/// Scans the cells of side `2^shift` ring by ring around `(x, y)` until the nearest point
/// is certain, and returns its squared distance. `visit` scans one cell, lowering `best`,
/// and returns false for a cell that is not loaded; the search then gives up with `None`.
#[inline]
fn ring_search(x: u64, y: u64, shift: u32, max_r: i64, mut visit: impl FnMut(i64, i64, &mut f64) -> bool) -> Option<f64> {
    let n = 1i64 << (BITS - shift);
    let side = 1u64 << shift;
    let (bx, by) = ((x >> shift) as i64, (y >> shift) as i64);
    let (fx, fy) = (x & (side - 1), y & (side - 1));
    let inner = fx.min(side - 1 - fx).min(fy).min(side - 1 - fy) as f64;
    let mut best = f64::INFINITY;
    for r in 0..=max_r {
        let mut loaded = true;
        ring(bx, by, r, n, n, |cx, cy| loaded &= visit(cx, cy, &mut best));
        if !loaded {
            return None;
        }
        let reach = r as f64 * side as f64 + inner;
        if best <= reach * reach {
            return Some(best);
        }
    }
    None
}

struct Seeds {
    pts: Vec<(u64, u64)>,
    level: u32,
    cells: Vec<Vec<u32>>,
}

impl Seeds {
    fn new(pts: Vec<(u64, u64)>) -> Self {
        assert!(!pts.is_empty());
        let mut level = 0;
        while level < 10 && 4usize.pow(level + 1) <= pts.len() {
            level += 1;
        }
        let n = 1usize << level;
        let mut cells = vec![Vec::new(); n * n];
        for (i, &(x, y)) in pts.iter().enumerate() {
            let b = BITS - level;
            cells[(y >> b) as usize * n + (x >> b) as usize].push(i as u32);
        }
        Seeds { pts, level, cells }
    }

    /// Index and squared distance of the nearest seed.
    fn nearest(&self, x: u64, y: u64) -> (u32, f64) {
        let n = 1i64 << self.level;
        let mut bi = 0;
        let best = ring_search(x, y, BITS - self.level, n, |cx, cy, best| {
            for &i in &self.cells[(cy * n + cx) as usize] {
                let (sx, sy) = self.pts[i as usize];
                let d = dist2(x, y, sx, sy);
                if d < *best {
                    *best = d;
                    bi = i;
                }
            }
            true
        });
        (bi, best.expect("the whole seed grid is loaded"))
    }
}

#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("memo keys are u128");
    }
    fn write_u128(&mut self, k: u128) {
        self.0 = mix((k as u64) ^ ((k >> 64) as u64).rotate_left(29));
    }
}

type Memo = DashMap<u128, u32, BuildHasherDefault<KeyHasher>>;

type TileKey = (u32, i64, i64);

/// One realization: rain + seeds, with cached tiles and a memo of particle colors.
pub struct World {
    rain: u64,
    seeds: Seeds,
    memo: Memo,
    tiles: Mutex<HashMap<TileKey, Arc<Tile>>>,
    clock: AtomicU64,
}

enum Hit {
    Seed(u32),
    Rain(Particle),
}

impl World {
    pub fn new(rain: u64, seeds: Vec<(u64, u64)>) -> Self {
        World {
            rain,
            seeds: Seeds::new(seeds),
            memo: Memo::default(),
            tiles: Mutex::new(HashMap::new()),
            clock: AtomicU64::new(0),
        }
    }

    /// Nearest seed or rain particle with arrival time `< s_lim`.
    fn nearest_before(&self, x: u64, y: u64, s_lim: f64) -> Hit {
        let (si, mut best) = self.seeds.nearest(x, y);
        let mut hit = Hit::Seed(si);
        for l in 0..=MAX_LEVEL {
            if level_start(l) >= s_lim {
                break;
            }
            let b = BITS - l;
            let side = 1u64 << b;
            let r = best.sqrt() * (1.0 + 1e-9) + 4096.0;
            let lo = |v: u64| ((v as f64 - r).max(0.0) as u64) >> b;
            let hi = |v: u64| (((v as f64 + r).min((DOMAIN - 1) as f64)) as u64) >> b;
            for cy in lo(y)..=hi(y) {
                let gy = gap(y, cy << b, side);
                if gy * gy >= best {
                    continue;
                }
                for cx in lo(x)..=hi(x) {
                    let gx = gap(x, cx << b, side);
                    if gx * gx + gy * gy >= best {
                        continue;
                    }
                    cell_particles(self.rain, l, cx, cy, |p| {
                        if p.s < s_lim {
                            let d = dist2(x, y, p.x, p.y);
                            if d < best {
                                best = d;
                                hit = Hit::Rain(p);
                            }
                        }
                    });
                }
            }
        }
        hit
    }

    /// Color (seed index) of any particle, following its line of descent.
    fn color_of(&self, p: Particle) -> u32 {
        let mut path: Vec<u128> = Vec::new();
        let mut cur = p;
        let c = loop {
            if let Some(i) = cur.seed_index() {
                break i;
            }
            let k = cur.key();
            if let Some(c) = self.memo.get(&k) {
                break *c;
            }
            path.push(k);
            match self.nearest_before(cur.x, cur.y, cur.s) {
                Hit::Seed(i) => break i,
                Hit::Rain(q) => cur = q,
            }
        };
        for k in path {
            self.memo.insert(k, c);
        }
        c
    }

    fn color_of_point(&self, x: u64, y: u64, s_lim: f64) -> u32 {
        match self.nearest_before(x, y, s_lim) {
            Hit::Seed(i) => i,
            Hit::Rain(q) => self.color_of(q),
        }
    }
}

/// Cells per tile side.
const TILE: i64 = 64;
const _: () = assert!(TILE.count_ones() == 1);
/// Neighbor cells loaded around a tile so that lines of descent can be followed locally.
const PAD: i64 = 8;
/// A level-`l` picture shows the arrivals before `RHO * 4^l`, i.e. `RHO` particles per cell.
const RHO: f64 = 2.0;
/// Upper bound on cached particles before least-recently-used tiles are dropped.
const CACHE_BUDGET: usize = 20_000_000;
/// Upper bound on memoized colors of particles reached by the global search.
const MEMO_BUDGET: usize = 40_000_000;

const RESOLVED: u32 = 1 << 31;
const UNKNOWN: u32 = RESOLVED - 1;
const EMPTY: u32 = u32::MAX;
const MIXED: u32 = u32::MAX - 1;

#[inline]
fn s_top(level: u32) -> f64 {
    RHO * pow4(level)
}

/// Coarsest level giving at least `density` particles per sample of side `sample` units.
pub fn level_for(sample: f64, density: f64) -> u32 {
    let a = sample / DOMAIN as f64;
    let mut l = 0;
    while l < MAX_LEVEL && s_top(l) * a * a < density {
        l += 1;
    }
    l
}

#[inline]
fn tile_of(level: u32, c: i64) -> i64 {
    if (1i64 << level) <= TILE { 0 } else { c.div_euclid(TILE) }
}

/// First cell and width of tile `t` along one axis.
#[inline]
fn tile_extent(level: u32, t: i64) -> (i64, i64) {
    let n = 1i64 << level;
    if n <= TILE { (0, n) } else { (t * TILE, TILE) }
}

/// Cells `[a, b]` along one axis, with the padding used to compute a tile.
#[inline]
fn padded(level: u32, t: i64) -> (i64, i64) {
    let n = 1i64 << level;
    let (c0, w) = tile_extent(level, t);
    ((c0 - PAD).max(0), (c0 + w - 1 + PAD).min(n - 1))
}

/// Level-`l - 1` cells holding the older particles that tile `(tx, ty)` of level `l` reads.
fn coarse_cells(l: u32, tx: i64, ty: i64) -> ((i64, i64), (i64, i64)) {
    let ((xa, xb), (ya, yb)) = (padded(l, tx), padded(l, ty));
    ((xa >> 1, xb >> 1), (ya >> 1, yb >> 1))
}

/// Concatenates rows of `(items, end offset of each cell)` into one cell-indexed layout.
fn concat_rows<T: Copy>(rows: Vec<(Vec<T>, Vec<u32>)>) -> (Vec<u32>, Vec<T>) {
    let mut start = vec![0];
    let mut items = Vec::with_capacity(rows.iter().map(|r| r.0.len()).sum());
    for (row, ends) in rows {
        let base = items.len() as u32;
        start.extend(ends.iter().map(|e| e + base));
        items.extend_from_slice(&row);
    }
    (start, items)
}

/// Particles with `s < s_lim` in a block of level-`level` cells, bucketed by cell and
/// sorted by arrival time within each cell.
struct Grid {
    shift: u32,
    bx0: i64,
    by0: i64,
    wb: i64,
    hb: i64,
    start: Vec<u32>,
    parts: Vec<Particle>,
}

impl Grid {
    fn build(world: &World, level: u32, s_lim: f64, (bx0, bx1): (i64, i64), (by0, by1): (i64, i64)) -> Grid {
        let shift = BITS - level;
        let (wb, hb) = (bx1 - bx0 + 1, by1 - by0 + 1);

        let mut seed_rows: Vec<Vec<Particle>> = vec![Vec::new(); hb as usize];
        for (i, &(x, y)) in world.seeds.pts.iter().enumerate() {
            let (gx, gy) = ((x >> shift) as i64, (y >> shift) as i64);
            if (bx0..=bx1).contains(&gx) && (by0..=by1).contains(&gy) {
                seed_rows[(gy - by0) as usize].push(Particle { x, y, s: -(i as f64) - 1.0 });
            }
        }

        let rows: Vec<(Vec<Particle>, Vec<u32>)> = seed_rows
            .into_par_iter()
            .enumerate()
            .map(|(r, mut row)| {
                let gy = by0 + r as i64;
                for l in 0..=level {
                    if level_start(l) >= s_lim {
                        break;
                    }
                    let d = level - l;
                    for cx in (bx0 >> d)..=(bx1 >> d) {
                        cell_particles(world.rain, l, cx as u64, (gy >> d) as u64, |p| {
                            let gx = (p.x >> shift) as i64;
                            if p.s < s_lim && (p.y >> shift) as i64 == gy && (bx0..=bx1).contains(&gx) {
                                row.push(p);
                            }
                        });
                    }
                }
                row.sort_unstable_by(|a, b| (a.x >> shift).cmp(&(b.x >> shift)).then(a.s.total_cmp(&b.s)));
                let mut k = 0;
                let ends = (bx0..=bx1)
                    .map(|cx| {
                        while k < row.len() && (row[k].x >> shift) as i64 == cx {
                            k += 1;
                        }
                        k as u32
                    })
                    .collect();
                (row, ends)
            })
            .collect();
        let (start, parts) = concat_rows(rows);
        Grid { shift, bx0, by0, wb, hb, start, parts }
    }

    /// Index of the nearest particle to `(x, y)` with `s < s_lim`, or `None` if it may lie
    /// outside the grid or beyond `max_r` rings.
    #[inline]
    fn nearest(&self, x: u64, y: u64, s_lim: f64, max_r: i64) -> Option<u32> {
        let mut found = 0;
        ring_search(x, y, self.shift, max_r, |cx, cy, best| {
            let (gx, gy) = (cx - self.bx0, cy - self.by0);
            if gx < 0 || gy < 0 || gx >= self.wb || gy >= self.hb {
                return false;
            }
            let c = (gy * self.wb + gx) as usize;
            for i in self.start[c]..self.start[c + 1] {
                let p = &self.parts[i as usize];
                if p.s >= s_lim {
                    break;
                }
                let d = dist2(x, y, p.x, p.y);
                if d < *best {
                    *best = d;
                    found = i;
                }
            }
            true
        })
        .map(|_| found)
    }
}

#[derive(Clone, Copy)]
struct Colored {
    x: u64,
    y: u64,
    color: u32,
}

/// Colored particles of one tile at one level: all arrivals before `s_top(level)`.
struct Tile {
    level: u32,
    cx0: i64,
    cy0: i64,
    w: i64,
    start: Vec<u32>,
    points: Vec<Colored>,
    /// Per cell: `EMPTY`, `MIXED` or the common color of its particles.
    cell_color: Vec<u32>,
    last_used: AtomicU64,
}

impl Tile {
    #[inline]
    fn cell_points(&self, cx: i64, cy: i64) -> &[Colored] {
        let c = ((cy - self.cy0) * self.w + (cx - self.cx0)) as usize;
        &self.points[self.start[c] as usize..self.start[c + 1] as usize]
    }

    #[inline]
    fn find(&self, p: &Particle) -> Option<u32> {
        let shift = BITS - self.level;
        let cell = self.cell_points((p.x >> shift) as i64, (p.y >> shift) as i64);
        cell.iter().find(|q| q.x == p.x && q.y == p.y).map(|q| q.color)
    }

    fn compute(world: &World, level: u32, tx: i64, ty: i64, coarse: Option<&TileSet>) -> Tile {
        let n = 1i64 << level;
        let (cx0, w) = tile_extent(level, tx);
        let (cy0, _) = tile_extent(level, ty);
        let s_old = if level == 0 { f64::NEG_INFINITY } else { s_top(level - 1) };
        let grid = Grid::build(world, level, s_top(level), padded(level, tx), padded(level, ty));
        let max_r = if n <= TILE { n } else { REACH };

        let state: Vec<AtomicU32> = (0..grid.parts.len()).into_par_iter().map(|_| AtomicU32::new(UNKNOWN)).collect();
        let link = |i: usize| -> u32 {
            let p = grid.parts[i];
            if let Some(c) = p.seed_index() {
                return RESOLVED | c;
            }
            if p.s < s_old {
                // Older particles were already colored one level up.
                let c = coarse.and_then(|t| t.find(&p)).unwrap_or_else(|| world.color_of(p));
                return RESOLVED | c;
            }
            grid.nearest(p.x, p.y, p.s, max_r).unwrap_or_else(|| RESOLVED | world.color_of(p))
        };
        let resolve = |i: usize, path: &mut Vec<usize>| -> u32 {
            path.clear();
            let mut j = i;
            let c = loop {
                let mut v = state[j].load(Relaxed);
                if v == UNKNOWN {
                    v = link(j);
                    state[j].store(v, Relaxed);
                }
                if v & RESOLVED != 0 {
                    break v & !RESOLVED;
                }
                path.push(j);
                j = v as usize;
            };
            for &k in path.iter() {
                state[k].store(RESOLVED | c, Relaxed);
            }
            c
        };

        let rows: Vec<(Vec<Colored>, Vec<u32>)> = (0..w)
            .into_par_iter()
            .map_init(Vec::new, |path, r| {
                let gy = cy0 + r - grid.by0;
                let mut points = Vec::new();
                let mut ends = Vec::with_capacity(w as usize);
                for cx in cx0..cx0 + w {
                    let c = (gy * grid.wb + cx - grid.bx0) as usize;
                    for i in grid.start[c] as usize..grid.start[c + 1] as usize {
                        let p = grid.parts[i];
                        points.push(Colored { x: p.x, y: p.y, color: resolve(i, path) });
                    }
                    ends.push(points.len() as u32);
                }
                (points, ends)
            })
            .collect();
        let (start, points) = concat_rows(rows);
        let cell_color = start
            .windows(2)
            .map(|ab| match points[ab[0] as usize..ab[1] as usize].split_first() {
                None => EMPTY,
                Some((q, rest)) if rest.iter().all(|r| r.color == q.color) => q.color,
                _ => MIXED,
            })
            .collect();
        Tile { level, cx0, cy0, w, start, points, cell_color, last_used: AtomicU64::new(0) }
    }
}

/// A rectangle of tiles of one level, all present.
struct TileSet {
    level: u32,
    tx0: i64,
    ty0: i64,
    ntx: i64,
    nty: i64,
    tiles: Vec<Arc<Tile>>,
}

impl TileSet {
    #[inline]
    fn tile(&self, cx: i64, cy: i64) -> Option<&Tile> {
        let n = 1i64 << self.level;
        if cx < 0 || cy < 0 || cx >= n || cy >= n {
            return None;
        }
        let (tx, ty) = (tile_of(self.level, cx) - self.tx0, tile_of(self.level, cy) - self.ty0);
        if tx < 0 || ty < 0 || tx >= self.ntx || ty >= self.nty {
            return None;
        }
        Some(&self.tiles[(ty * self.ntx + tx) as usize])
    }

    #[inline]
    fn find(&self, p: &Particle) -> Option<u32> {
        let shift = BITS - self.level;
        self.tile((p.x >> shift) as i64, (p.y >> shift) as i64)?.find(p)
    }

    /// Color of cell `(cx, cy)`; `EMPTY` outside the domain, `MIXED` outside the set.
    #[inline]
    fn cell_color(&self, cx: i64, cy: i64) -> u32 {
        let n = 1i64 << self.level;
        if cx < 0 || cy < 0 || cx >= n || cy >= n {
            return EMPTY;
        }
        match self.tile(cx, cy) {
            Some(t) => t.cell_color[((cy - t.cy0) * t.w + (cx - t.cx0)) as usize],
            None => MIXED,
        }
    }

    /// Color of the nearest particle to `(x, y)`, or `None` if it may lie outside the set.
    #[inline]
    fn nearest_color(&self, x: u64, y: u64) -> Option<u32> {
        let mut col = EMPTY;
        ring_search(x, y, BITS - self.level, REACH, |cx, cy, best| {
            let Some(t) = self.tile(cx, cy) else { return false };
            for q in t.cell_points(cx, cy) {
                let d = dist2(x, y, q.x, q.y);
                if d < *best {
                    *best = d;
                    col = q.color;
                }
            }
            true
        })
        .map(|_| col)
    }
}

impl World {
    fn tiles_covering(level: u32, (xa, xb): (i64, i64), (ya, yb): (i64, i64)) -> (i64, i64, i64, i64) {
        let (tx0, tx1) = (tile_of(level, xa), tile_of(level, xb));
        let (ty0, ty1) = (tile_of(level, ya), tile_of(level, yb));
        (tx0, ty0, tx1 - tx0 + 1, ty1 - ty0 + 1)
    }

    fn tile_keys(level: u32, xr: (i64, i64), yr: (i64, i64)) -> impl Iterator<Item = (i64, i64)> {
        let (tx0, ty0, ntx, nty) = Self::tiles_covering(level, xr, yr);
        (ty0..ty0 + nty).flat_map(move |ty| (tx0..tx0 + ntx).map(move |tx| (tx, ty)))
    }

    /// The cached tiles covering the cell ranges, if all are present.
    fn cached(&self, level: u32, xr: (i64, i64), yr: (i64, i64), stamp: u64) -> Option<TileSet> {
        let (tx0, ty0, ntx, nty) = Self::tiles_covering(level, xr, yr);
        let cache = self.tiles.lock().unwrap();
        let tiles = Self::tile_keys(level, xr, yr)
            .map(|(tx, ty)| {
                let t = cache.get(&(level, tx, ty))?;
                t.last_used.store(stamp, Relaxed);
                Some(t.clone())
            })
            .collect::<Option<Vec<_>>>()?;
        Some(TileSet { level, tx0, ty0, ntx, nty, tiles })
    }

    /// Computes the missing tiles covering the cell ranges (and the coarser tiles they
    /// depend on). Finished tiles stay cached even if `cancel()` interrupts the call.
    fn ensure(&self, level: u32, xr: (i64, i64), yr: (i64, i64), cancel: &(dyn Fn() -> bool + Sync)) -> Option<TileSet> {
        let stamp = self.clock.fetch_add(1, Relaxed) + 1;
        if let Some(ts) = self.cached(level, xr, yr, stamp) {
            return Some(ts);
        }
        let mut want: Vec<(i64, i64)> = Self::tile_keys(level, xr, yr).collect();
        let mut missing: Vec<Vec<(i64, i64)>> = vec![Vec::new(); level as usize + 1];
        for l in (0..=level).rev() {
            let miss: Vec<(i64, i64)> = {
                let cache = self.tiles.lock().unwrap();
                want.into_iter()
                    .filter(|&(tx, ty)| match cache.get(&(l, tx, ty)) {
                        Some(t) => {
                            t.last_used.store(stamp, Relaxed);
                            false
                        }
                        None => true,
                    })
                    .collect()
            };
            want = if l == 0 {
                Vec::new()
            } else {
                let mut next: Vec<_> = miss
                    .iter()
                    .flat_map(|&(tx, ty)| {
                        let (cxr, cyr) = coarse_cells(l, tx, ty);
                        Self::tile_keys(l - 1, cxr, cyr)
                    })
                    .collect();
                next.sort_unstable();
                next.dedup();
                next
            };
            missing[l as usize] = miss;
            if want.is_empty() {
                break;
            }
        }

        for (l, miss) in missing.iter().enumerate().filter(|(_, m)| !m.is_empty()) {
            let l = l as u32;
            if cancel() {
                return None;
            }
            let done: Vec<Option<(TileKey, Arc<Tile>)>> = miss
                .par_iter()
                .map(|&(tx, ty)| {
                    if cancel() {
                        return None;
                    }
                    let coarse = (l > 0).then(|| {
                        let (cxr, cyr) = coarse_cells(l, tx, ty);
                        self.cached(l - 1, cxr, cyr, stamp)
                    });
                    let t = Tile::compute(self, l, tx, ty, coarse.flatten().as_ref());
                    t.last_used.store(stamp, Relaxed);
                    Some(((l, tx, ty), Arc::new(t)))
                })
                .collect();
            let complete = done.iter().all(Option::is_some);
            self.tiles.lock().unwrap().extend(done.into_iter().flatten());
            if !complete {
                return None;
            }
        }
        let ts = self.cached(level, xr, yr, stamp);
        self.evict(stamp);
        ts
    }

    /// Drops least-recently-used tiles (never those stamped `keep`) and, past its budget, the memo.
    fn evict(&self, keep: u64) {
        if self.memo.len() > MEMO_BUDGET {
            self.memo.clear();
        }
        let mut cache = self.tiles.lock().unwrap();
        let mut total: usize = cache.values().map(|t| t.points.len()).sum();
        if total <= CACHE_BUDGET {
            return;
        }
        let mut entries: Vec<(u64, TileKey, usize)> =
            cache.iter().map(|(k, t)| (t.last_used.load(Relaxed), *k, t.points.len())).collect();
        entries.sort_unstable();
        for (used, k, len) in entries {
            if total <= CACHE_BUDGET || used >= keep {
                break;
            }
            cache.remove(&k);
            total -= len;
        }
    }

    /// Computes ahead of time the tiles that the central half of `view` needs at `level`.
    pub fn prefetch(&self, view: &View, level: u32, cancel: &(dyn Fn() -> bool + Sync)) {
        let half = View { w: view.w / 2, h: view.h / 2, ..*view };
        if let Some((xr, yr)) = half.cells(level) {
            self.ensure(level, xr, yr, cancel);
        }
    }
}

/// What to render: `w x h` pixels, pixel `(i, j)` centered at
/// `center + ((i + 0.5 - w/2) * scale, (j + 0.5 - h/2) * scale)` in fixed-point units.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct View {
    pub cx: i64,
    pub cy: i64,
    pub scale: f64,
    pub w: usize,
    pub h: usize,
}

impl View {
    #[inline]
    fn sample_offset(&self, i: usize, sub: u32, aa: u32, n: usize) -> f64 {
        (i as f64 + (sub as f64 + 0.5) / aa as f64 - n as f64 * 0.5) * self.scale
    }

    /// Level-`level` cells under the view plus a margin of `REACH`, or `None` if the view misses the domain.
    fn cells(&self, level: u32) -> Option<((i64, i64), (i64, i64))> {
        let n = 1i64 << level;
        let shift = BITS - level;
        let axis = |c: i64, len: usize| {
            let half = len as f64 * 0.5 * self.scale;
            let (lo, hi) = (c as f64 - half, c as f64 + half);
            if hi < 0.0 || lo >= DOMAIN as f64 {
                return None;
            }
            let cell = |v: f64| (v.clamp(0.0, DOMAIN as f64 - 1.0) as u64 >> shift) as i64;
            Some(((cell(lo) - REACH).max(0), (cell(hi) + REACH).min(n - 1)))
        };
        Some((axis(self.cx, self.w)?, axis(self.cy, self.h)?))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub level: u32,
    pub s_t: f64,
    pub tiles: usize,
    pub cached_particles: usize,
    pub memo: usize,
}

pub enum Tiles {
    /// Use only tiles already in the cache.
    CachedOnly,
    /// Compute missing tiles (interruptible).
    Compute,
}

/// Renders `view` as RGBA pixels with `aa x aa` samples per pixel from the level-`level`
/// tiles; `palette` must hold one color per seed. Returns `None` if tiles are missing
/// (with `Tiles::CachedOnly`) or `cancel()` fired.
#[allow(clippy::too_many_arguments)]
pub fn render(
    world: &World,
    view: &View,
    level: u32,
    aa: u32,
    palette: &[[u8; 3]],
    background: [u8; 3],
    tiles: Tiles,
    cancel: &(dyn Fn() -> bool + Sync),
) -> Option<(Vec<[u8; 4]>, Stats)> {
    let aa = aa.max(1);
    let s_t = s_top(level);
    let opaque = |c: [u8; 3]| [c[0], c[1], c[2], 255];
    let ts = match view.cells(level) {
        None => None,
        Some((xr, yr)) => Some(match tiles {
            Tiles::CachedOnly => world.cached(level, xr, yr, world.clock.load(Relaxed))?,
            Tiles::Compute => world.ensure(level, xr, yr, cancel)?,
        }),
    };
    let stats = Stats {
        level,
        s_t,
        tiles: ts.as_ref().map_or(0, |ts| ts.tiles.len()),
        cached_particles: world.tiles.lock().unwrap().values().map(|t| t.points.len()).sum(),
        memo: world.memo.len(),
    };
    let Some(ts) = ts else {
        return Some((vec![opaque(background); view.w * view.h], stats));
    };
    let mut img = vec![[0u8; 4]; view.w * view.h];
    let n = 1i64 << level;
    let shift = BITS - level;

    let to_linear = |c: [u8; 3]| c.map(|v| (v as f32 / 255.0).powf(2.2));
    let lin: Vec<[f32; 3]> = palette.iter().map(|&c| to_linear(c)).collect();
    let bg_lin = to_linear(background);
    let encode: Vec<u8> = (0..=4096).map(|i| ((i as f32 / 4096.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8).collect();

    let pixel = |i: usize, j: usize| -> [u8; 3] {
        let mut acc = [0f32; 3];
        let mut first = EMPTY;
        let mut same = true;
        for sy in 0..aa {
            let oy = view.sample_offset(j, sy, aa, view.h);
            for sx in 0..aa {
                let ox = view.sample_offset(i, sx, aa, view.w);
                let (xf, yf) = (view.cx as f64 + ox, view.cy as f64 + oy);
                let (c, col) = if xf < 0.0 || yf < 0.0 || xf >= DOMAIN as f64 || yf >= DOMAIN as f64 {
                    (EMPTY, bg_lin)
                } else {
                    let x = (view.cx + ox.round() as i64).clamp(0, DOMAIN as i64 - 1) as u64;
                    let y = (view.cy + oy.round() as i64).clamp(0, DOMAIN as i64 - 1) as u64;
                    let c = ts.nearest_color(x, y).unwrap_or_else(|| world.color_of_point(x, y, s_t));
                    (c, lin[c as usize])
                };
                if sx == 0 && sy == 0 {
                    first = c;
                } else if c != first {
                    same = false;
                }
                for k in 0..3 {
                    acc[k] += col[k];
                }
            }
        }
        if same {
            return if first == EMPTY { background } else { palette[first as usize] };
        }
        let inv = 1.0 / (aa * aa) as f32;
        acc.map(|v| encode[((v * inv).clamp(0.0, 1.0) * 4096.0) as usize])
    };

    // A block whose samples all have a particle within REACH cells, and whose REACH-cell
    // neighborhood holds a single color, takes that color without per-sample searches.
    const BLOCK: usize = 16;
    let uniform = |i0: usize, i1: usize, j0: usize, j1: usize| -> Option<u32> {
        let (ox0, ox1) = (view.sample_offset(i0, 0, aa, view.w), view.sample_offset(i1 - 1, aa - 1, aa, view.w));
        let (oy0, oy1) = (view.sample_offset(j0, 0, aa, view.h), view.sample_offset(j1 - 1, aa - 1, aa, view.h));
        let inside = |c: i64, o: f64| (0.0..DOMAIN as f64).contains(&(c as f64 + o));
        if !(inside(view.cx, ox0) && inside(view.cx, ox1) && inside(view.cy, oy0) && inside(view.cy, oy1)) {
            return None;
        }
        let cell = |c: i64, o: f64| (c + o.round() as i64).clamp(0, DOMAIN as i64 - 1) >> shift;
        let (xa, xb) = (cell(view.cx, ox0), cell(view.cx, ox1));
        let (ya, yb) = (cell(view.cy, oy0), cell(view.cy, oy1));
        let mut col = EMPTY;
        for cy in (ya - REACH).max(0)..=(yb + REACH).min(n - 1) {
            for cx in (xa - REACH).max(0)..=(xb + REACH).min(n - 1) {
                match ts.cell_color(cx, cy) {
                    EMPTY => {}
                    MIXED => return None,
                    v if col == EMPTY => col = v,
                    v if v != col => return None,
                    _ => {}
                }
            }
        }
        for cy in ya..=yb {
            for cx in xa..=xb {
                if ts.cell_color(cx, cy) == EMPTY {
                    let found = ((cy - 1).max(0)..=(cy + 1).min(n - 1))
                        .any(|yy| ((cx - 1).max(0)..=(cx + 1).min(n - 1)).any(|xx| ts.cell_color(xx, yy) != EMPTY));
                    if !found {
                        return None;
                    }
                }
            }
        }
        (col != EMPTY).then_some(col)
    };

    img.par_chunks_mut(view.w * BLOCK).enumerate().try_for_each(|(bj, rows)| {
        if cancel() {
            return None;
        }
        let j0 = bj * BLOCK;
        let j1 = (j0 + BLOCK).min(view.h);
        for i0 in (0..view.w).step_by(BLOCK) {
            let i1 = (i0 + BLOCK).min(view.w);
            let fill = uniform(i0, i1, j0, j1).map(|c| palette[c as usize]);
            for j in j0..j1 {
                for i in i0..i1 {
                    rows[(j - j0) * view.w + i] = opaque(fill.unwrap_or_else(|| pixel(i, j)));
                }
            }
        }
        Some(())
    })?;
    Some((img, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(zoom: f64, density: f64) {
        let seeds: Vec<(u64, u64)> = (0..5).map(|i| (mix(i) >> 2, mix(i + 100) >> 2)).collect();
        let world = World::new(7, seeds.clone());
        let fit = DOMAIN as f64 / 1000.0;
        let view = View { cx: (DOMAIN / 3) as i64, cy: (DOMAIN / 2) as i64, scale: fit / zoom, w: 96, h: 64 };
        let level = level_for(view.scale, density);
        let palette: Vec<[u8; 3]> = (0..5).map(|i| [i as u8 * 40 + 1, 7, 9]).collect();
        let (img, _) = render(&world, &view, level, 1, &palette, [0; 3], Tiles::Compute, &|| false).unwrap();
        let fresh = World::new(7, seeds);
        let mut bad = 0;
        for j in 0..view.h {
            for i in 0..view.w {
                let x = (view.cx + view.sample_offset(i, 0, 1, view.w).round() as i64) as u64;
                let y = (view.cy + view.sample_offset(j, 0, 1, view.h).round() as i64) as u64;
                let c = fresh.color_of_point(x, y, s_top(level));
                bad += (img[j * view.w + i][0] != palette[c as usize][0]) as usize;
            }
        }
        assert_eq!(bad, 0, "zoom {zoom}, level {level}");
    }

    /// Tile-based colors must agree with the direct search along lines of descent.
    #[test]
    fn tiles_match_direct_search() {
        check(1.0, 1.0);
        check(40.0, 1.0);
        check(3e4, 2.0);
        check(1e9, 1.0);
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn timings() {
        let seeds: Vec<(u64, u64)> = (0..8).map(|i| (mix(i) >> 2, mix(i + 50) >> 2)).collect();
        let world = World::new(2017, seeds);
        let palette: Vec<[u8; 3]> = (0..8).map(|i| [i as u8 * 30, 0, 0]).collect();
        let fit = DOMAIN as f64 / 1000.0;
        for zoom in [1.0, 1e3, 1e6, 1e9, 1e11] {
            let mut view = View { cx: (DOMAIN / 3) as i64, cy: (DOMAIN / 2) as i64, scale: fit / zoom, w: 1920, h: 1080 };
            let level = level_for(view.scale, 1.0);
            for (name, aa, shift_px) in [("cold", 1, 0.0), ("warm", 1, 0.0), ("aa2", 2, 0.0), ("pan 40px", 1, 40.0)] {
                view.cx += (shift_px * view.scale) as i64;
                let t = Instant::now();
                let (_, st) =
                    render(&world, &view, level, aa, &palette, [0; 3], Tiles::Compute, &|| false).unwrap();
                println!(
                    "zoom {zoom:.0e} {name:9}: {:7.1} ms  level {} tiles {} cached {}",
                    t.elapsed().as_secs_f64() * 1e3,
                    st.level,
                    st.tiles,
                    st.cached_particles
                );
            }
        }
    }
}

#[cfg(test)]
mod snapshots {
    use super::*;

    /// Writes PPM renders of successive zooms onto a boundary into `$SNAP_DIR`.
    #[test]
    #[ignore]
    fn zoom_sequence() {
        let dir = std::env::var("SNAP_DIR").unwrap();
        let seeds: Vec<(u64, u64)> = (0..6).map(|i| (mix(i + 1) >> 2, mix(i + 77) >> 2)).collect();
        let world = World::new(2017, seeds);
        let palette = [[230, 90, 80], [70, 140, 220], [240, 200, 70], [90, 190, 120], [170, 110, 210], [240, 150, 60]];
        let size = 640;
        let mut view = View { cx: (DOMAIN / 2) as i64, cy: (DOMAIN / 2) as i64, scale: DOMAIN as f64 / size as f64, w: size, h: size };
        let mut zoom = 1.0f64;
        for k in 0..=3 {
            for _ in 0..if k == 0 { 1 } else { 4 } {
                zoom *= if k == 0 { 1.0 } else { 20.0f64.min(1e11 / zoom).max(1.0) };
                view.scale = DOMAIN as f64 / size as f64 / zoom;
                let level = level_for(view.scale, 1.0);
                let (img, _) = render(&world, &view, level, 2, &palette, [18, 18, 22], Tiles::Compute, &|| false).unwrap();
                // Recenter on the boundary pixel closest to the center.
                let at = |i: usize, j: usize| img[j * size + i];
                let mut best = None;
                for j in 1..size - 1 {
                    for i in 1..size - 1 {
                        if at(i, j) != at(i + 1, j) {
                            let d = (i as i64 - size as i64 / 2).pow(2) + (j as i64 - size as i64 / 2).pow(2);
                            if best.is_none_or(|(bd, _, _)| d < bd) {
                                best = Some((d, i, j));
                            }
                        }
                    }
                }
                if let Some((_, i, j)) = best {
                    view.cx += view.sample_offset(i, 0, 1, size).round() as i64;
                    view.cy += view.sample_offset(j, 0, 1, size).round() as i64;
                }
            }
            let level = level_for(view.scale, 1.0);
            let (img, _) = render(&world, &view, level, 2, &palette, [18, 18, 22], Tiles::Compute, &|| false).unwrap();
            let mut ppm = format!("P6 {size} {size} 255\n").into_bytes();
            for px in &img {
                ppm.extend_from_slice(&px[..3]);
            }
            std::fs::write(format!("{dir}/zoom{k}.ppm"), ppm).unwrap();
            println!("zoom{k}: x{zoom:.1e} level {level}");
        }
    }
}
