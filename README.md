# Aldous coloring

Interactive explorer for Aldous' Poissonian coloring at n = ∞, with zoom down to about 1e11.

Requires Rust 1.95 or newer.

## Video

[Watch the explanatory video in French](aldous_explained.mp4) (9 min 36, 1080p, with embedded subtitles and no voice-over).

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

## Usage

### Moving around

| Input | Action |
| --- | --- |
| Drag, arrow keys | Pan |
| Wheel, pinch | Zoom around the pointer |
| `+` / `-` | Zoom around the center of the view |
| `R`, or **Reset view** | Fit the whole square in the window |

A blurry area sharpens as its tiles finish rendering. The **View** section of the panel shows the zoom, the number of rain points and how many tiles are still missing.

### Editing seeds

| Input | Action |
| --- | --- |
| Shift+click | Add a seed at the pointer, inside the square |
| Ctrl+drag (Cmd on macOS) | Move the seed under the pointer |
| Right click | Remove the seed under the pointer, keeping at least one |

Start a drag or a right click within about 14 points of a seed. Editing a seed by hand switches the layout to **Manual**.

### Panel

| Setting | Effect |
| --- | --- |
| **layout** | Place the seeds at random, on a circle, on a grid, or keep them as edited (**Manual**) |
| **count** | Number of seeds, from 1 to 5000 |
| **New positions** | Draw new seed positions (leaves **Manual** for **Random**) |
| **show** | Show the seed markers, which are always hidden above 2000 seeds |
| **New colors** | Draw a new palette |
| **realization**, **New** | Choose or draw the rain |
| **particles / texel** | Detail of the picture: higher is finer but slower |
| **supersampling** | Samples per texel along each axis: higher gives smoother edges but is slower |
