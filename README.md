# Aldous coloring

Interactive explorer for Aldous' Poissonian coloring at n = ∞, with zoom down to about 1e11.

Requires Rust 1.95 or newer.

## Build

```sh
cargo build --release
```

## Run

```sh
cargo run --release
```

## Test

```sh
cargo test --release
```

## Controls

| Input | Action |
| --- | --- |
| Drag, arrows | Pan |
| Wheel, pinch, `+` / `-` | Zoom |
| `R` | Reset view |
| Shift+click | Add a seed |
| Right click | Remove a seed |
| Ctrl+drag | Move a seed |
