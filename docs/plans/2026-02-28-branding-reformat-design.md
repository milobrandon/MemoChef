# Branding Reformat Design — Subtext Brand Theme

## Overview

Add a post-update pipeline step that reformats the entire output memo to align
with the Subtext Brand Theme (.thmx). This runs after all metric/schedule
updates are applied and covers every slide in the deck.

## Subtext Brand Theme Spec

**Colors (from theme1.xml):**
| Role     | Hex       | Description        |
|----------|-----------|--------------------|
| dk1      | `#2B2825` | Near-black brown   |
| lt1      | `#FFFFFF` | White              |
| dk2      | `#16352E` | Deep forest green  |
| lt2      | `#F7F1E3` | Warm cream         |
| accent1  | `#C1D100` | Lime/chartreuse    |
| accent2  | `#16352E` | Forest green       |
| accent3  | `#A95818` | Burnt orange/rust  |
| accent4  | `#512213` | Dark mahogany      |
| accent5  | `#2B2825` | Near-black brown   |
| accent6  | `#F7F1E3` | Warm cream         |
| hlink    | `#C1D100` | Lime green         |

**Fonts:**
- Major (headings): Pragmatica Bold
- Minor (body): Pragmatica Book

## Scope

Applied to EVERY slide in the output memo, both original and newly updated content.

### 1. Theme XML Replacement

Replace the PPTX's embedded theme XML (`ppt/theme/theme1.xml`) with the Subtext
Brand Theme. This ensures all scheme-referenced colors (`schemeClr` elements)
and theme fonts resolve to Subtext values automatically.

### 2. Font Reformat

Walk every text run in every shape (text boxes, placeholders, table cells) and
set the font family:

- **Pragmatica Bold**: Title placeholders, text with font size >= 18pt
- **Pragmatica Book**: All other body text

This replaces whatever font was previously set (Calibri, Arial, etc.).

### 3. Color Remapping

Map existing hard-coded RGB text/fill colors to the nearest Subtext brand color.
Uses perceptual color distance (Euclidean in RGB space) with a threshold to
avoid mangling colors that aren't close to any brand color (e.g., chart-specific
colors, images).

**Threshold**: If the nearest brand color is more than ~80 RGB units away
(Euclidean distance), leave the color unchanged.

**Elements to remap:**
- Text run font colors (`run.font.color.rgb`)
- Cell fill colors (solid fills)
- Shape fill colors (solid fills)

**Elements to skip:**
- Images/pictures
- Chart elements
- Gradient fills (too complex to reliably remap)

## Pipeline Position

```
apply_updates() -> apply_branding() -> write_change_log()
```

Always-on, no UI toggle needed.

## Implementation Components

### `apply_branding(memo_path, theme_path, cfg)` in `memo_automator.py`

1. Open the PPTX as a zip, replace `ppt/theme/theme1.xml` with theme from .thmx
2. Re-open with python-pptx, iterate all slides/shapes/runs
3. Apply font rules (heading vs body detection)
4. Apply color remapping with distance threshold
5. Save

### Config

```yaml
branding:
  theme_path: "path/to/Subtext Brand Theme.thmx"
  heading_size_threshold: 18  # pt, >= this uses heading font
  color_distance_threshold: 80  # RGB Euclidean, skip if further
```

### `app.py` Changes

- Add `apply_branding()` call after `apply_updates()`
- Progress bar step: "Applying Subtext branding..."
- Theme file path from config or bundled with app

## Risks and Mitigations

- **Missing Pragmatica fonts**: If not installed on the machine opening the
  memo, PowerPoint will substitute. This is standard behavior.
- **Color remapping artifacts**: The distance threshold prevents aggressive
  remapping. Conservative default of 80 units.
- **Table header detection**: Some table header rows use fill colors rather than
  font size. May need to treat row 0 of tables as heading.

## Date

2026-02-28
