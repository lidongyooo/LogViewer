# AK Log Viewer

High-performance multi-file log viewer/search prototype in C + SDL3 + Metal.

## Build

macOS + Metal is required.

```sh
brew install sdl3
cd tools/log_viewer
make
```

Outputs:

- `build/aklv` - GUI viewer.
- `build/aklv_selftest` - core indexing/search self-test.

Run:

```sh
./build/aklv
```

The app starts without opening any file. Open files with the `Open` button,
`Ctrl+O`, or drag/drop them into the window. Use `Close` to close the active
file; each tab also has an `x`.

## UI

The layout and palette are intentionally close to a compact 010 Editor style:

- top file navigation bar: open, switch, close files
- central virtual text view
- bottom search panel with draggable splitter
- search input in bottom panel
- clickable search results; clicking a result jumps the text view to that line
- double-click text in the central view to highlight matching visible text
- switching files preserves each file's text viewport and selected line
- the text view has a fixed gap below the navigation bar to avoid overlap
- clicked search results keep a selected-row highlight

Useful keys:

- `Ctrl+O`: open files
- `Ctrl+F` / `Cmd+F`: focus search box
- `Ctrl/Cmd+C`, `Ctrl/Cmd+X`, `Ctrl/Cmd+V`, `Ctrl/Cmd+A`: copy/cut/paste/select-all in the search box
- `Ctrl/Cmd+C` in the text view: copy highlighted text, or the selected line if no highlight is active
- `Enter`: submit search; editing the search box does not search live
- arrows/page/home/end: move selected text line
- mouse wheel: vertical scroll
- shift + wheel: horizontal scroll

## Performance design

### File access

- Files are opened read-only with `mmap`.
- The UI never copies whole files.
- Huge line payloads are sliced directly from mapped memory.

### Line index

The index is optimized for exact search over very large trace/log files:

- block-based line-offset index, 65536 lines per block
- ASCII-derived 3-byte fragment inverted index backed by lightweight
  Roaring-style bitmaps of line numbers
- finished indexes are cached under `indexs/` as `<md5>_N.index` shards, where
  the MD5 input is the resolved full path plus basename; shards are capped at
  1 GiB serialized size
- reopening an unchanged file restores line offsets from cache and loads only
  one index shard per file at a time while searching
- index construction is split into a sequential line-offset pass and a
  parallel per-line-range gram build/merge pass
- search-effective line content skips the prefix from line start through `!` when the line starts with `[`
- CRLF is normalized at line-view boundaries

The Roaring inverted index is built from every 3-byte window inside ASCII
(`0x00..0x7f`) runs in the effective line region; non-ASCII bytes split
indexable runs. 1-byte and 2-byte fragments are intentionally not indexed,
because the backend assumes submitted searches are at least 3 characters.
During indexing, duplicate fragments inside a single line are deduplicated
with a reusable per-worker dense bitset before posting updates, avoiding
per-line allocation/sort churn. Posting lists use adaptive Roaring-style
containers: small sorted arrays for sparse ranges, full bitmaps for
high-cardinality ranges, run containers for contiguous ranges, and full
containers for saturated ranges. The index is a
candidate generator only; every candidate line is still validated against the
mmap-backed original line.

### Search

- Main thread only handles SDL events and rendering.
- File opening and index building are done by persistent loader workers.
- Search is handled by a persistent coordinator plus worker thread pool.
- New searches cancel stale generations without blocking the UI.
- Search work is split into line-range tasks so multiple worker threads can
  iterate Roaring candidates in parallel and append result batches as tasks
  complete.
- Search extracts 3-byte ASCII fragments from the case-folded query, chooses the
  lowest-cardinality posting as the driver after sorting query postings by
  cardinality, intersects remaining postings at Roaring-container granularity
  with bitmap/full-container AND fast paths, then performs mmap-backed SIMD/BMH
  exact validation on candidate lines.
- Queries with no 3-byte ASCII fragment produce no Roaring candidates and
  therefore return no matches; the log-viewer backend does not run whole-file
  searches without inverted-index candidates.
- Priority results are merged, sorted by file/line, then rotated so the first result is the nearest next match after the currently selected visible line.
- Result snippets are capped at 300 characters and marked with `...` when truncated.

### Rendering

- SDL3 is used for window/input.
- Metal is used directly through `SDL_Metal_CreateView`.
- Text rendering is virtual: each frame only generates glyphs for visible rows.
- The renderer batches rectangles, scrollbars, tabs, lines, and glyph quads into one vertex stream and submits one Metal draw call per frame.
- Glyph atlas is generated once at startup from Menlo/Monaco via CoreText.

## Source layout

- `src/main.c` - UI state, events, layout, virtual rendering orchestration.
- `src/aklv_metal.m` - Metal batch renderer and font atlas.
- `src/aklv_index.c` - mmap, line index, Roaring-style inverted index, line slicing.
- `src/aklv_search.c` - persistent multi-threaded search service.
- `src/aklv_loader.c` - persistent background file/index loader.
- `src/aklv_platform.m` - macOS open-file dialog.
- `src/selftest.c` - core non-UI self-tests.

## Current constraints

- ASCII-oriented fast case folding and ASCII-fragment indexing; non-ASCII text still renders as `?` in the current atlas path.
- Search is substring search, case-insensitive for ASCII.
- Initial index build does a newline/offset pass plus a parallel gram pass, so
  first open of a tens-of-GB file is bounded by storage bandwidth, line count,
  and distinct ASCII fragment volume.
- macOS-only renderer path.
