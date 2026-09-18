# SOLWEIG Studio — light-theme visual design plan

Status: design plan (tokens + layout + component spec). This document owns the **visual
system**. Interaction flows (edit submission, collaboration protocol UX, dialogs
sequencing) are owned by the companion interaction-flows plan in this directory.

Sources read: `examples/incremental_design_tool/index.html`, `styles.css` (dark token
system being replaced), `renderer.mjs` (`HEAT_STOPS`), `assets/site_base.webp` (canvas
base), `README.md` (offline/connected modes, capability-driven UI),
`docs/incremental_design_tool/realtime_collaboration/service_level_contract.md`
(result classes, degradation ladder),
`docs/incremental_design_tool/realtime_collaboration/collaborative_state.md`
(operation/actor vocabulary).

Non-negotiables from the product owner: **light theme**, **modern**, **pro-grade
feature density** ("must feel like a power tool, not a toy"), **not fixated on
minimalism**, **the center canvas stays realistic** (real orthophoto + UTCI thermal
overlay), **undergrad-comprehensible**.

---

## 1. Design thesis

SOLWEIG Studio is chrome around one loud, beautiful thing: a real campus orthophoto
wearing a live UTCI field. The visual system is therefore an **instrument on a
drafting table**. The center is a photographic *plate* laid on faint survey graph
paper; everything around it is a quiet, dense, high-legibility **meteorological
instrument panel** — cool paper surfaces, hairline rules, condensed small-caps station
labels, monospaced telemetry readouts, and a single *ultramarine survey-ink* accent
used only where the UI speaks (focus, selection, links, active tool). No chrome color
ever appears in the UTCI ramp or the orthophoto, so the eye always knows what is
*world* (canvas) and what is *instrument* (UI). The identity is borrowed from the
subject's own artifacts: synoptic station plots, solar-path diagrams, pyranometer
readouts, engineering pads, and the UTCI assessment scale — not from dashboards.

**Deliberate aesthetic risk:** a permanent, full-width 32 px **revision transport
rail** ("the Metronome", §5) pinned to the bottom of the app, spending prime chrome
on the workspace/fast/exact revision triple and an epoch heartbeat. Most products
would hide that in a tooltip; here the approximate-now / verified-later heartbeat
*is* the product's core science, so it gets an instrument of its own. Secondary risk
carried by the same bet: condensed-caps instrument microtype becomes the dominant
label voice of the whole UI.

## 2. Color tokens

### 2.1 Named palette (8)

| # | Name | Hex | Role |
|---|------|-----|------|
| 1 | **Instrument Paper** | `#F3F5F3` | Page ground. Cool green-gray, zero warmth — explicitly *not* cream. |
| 2 | **Plate White** | `#FFFFFF` | Raised panels, cards, floating chips. The orthophoto plate sits *on* paper, cards sit *on* panels. |
| 3 | **Survey Ink** | `#1C2622` | Primary text, primary buttons (ink fill). Near-black with a cool green cast that rhymes with the domain. 14.1:1 on paper. |
| 4 | **Ultramarine** | `#2B49CC` | The one accent: focus rings, selection, links, active tool, slider fills, epoch ticks. A pigmented drafting ultramarine, not a framework blue. 7.2:1 on white. |
| 5 | **Viridian** | `#0E6E4E` | `fast_exact` / verified. Deep, blue-leaning green — chromatically far from both the ramp's grass/lime greens and the orthophoto vegetation. 6.3:1. |
| 6 | **Ochre** | `#96590A` | `fast_qualified` / provisional / warning. Deep earth amber, far darker and duller than the ramp's bright yellows. 5.6:1. |
| 7 | **Brick** | `#C03526` | Danger / failure. Deeper and duller than the ramp's terminal red `#DD473E`. 5.6:1. |
| 8 | **Slate** | `#5F6A66` | `visual_pending` / stale / disabled-strong. Neutral instrument gray. 5.6:1. |

Supporting neutrals (not "named" heroes): `--line: #D9DED9` (hairline),
`--line-strong: #C3CAC4` (control borders), `--panel-sunken: #EAEEEB` (input wells,
tab strips), `--workspace-ground: #E7EBE8` (drafting-table field behind the plate).

### 2.2 Canvas-adjacency rules (why this cannot fight the canvas)

The canvas owns these hues; chrome never competes with them:

- **Ramp territory** (teal `#299992` → grass → yellow → orange → red `#DD473E`):
  chrome greens/ambers/reds are all **deep, low-lightness, desaturated** versions
  (Viridian/Ochre/Brick), used only in small chips and thin rules — never as fills
  larger than a badge.
- **Orthophoto territory** (asphalt grays `#5A6268–#6B7175`, vegetation
  `#4CA64C–#66B34E`, olive fields `#7A9A5E`, roofs `#C9CCC7–#DDDDD8`, tan dirt,
  teal pond `#3E7E8C`): chrome surfaces are near-neutral (≤ 6% chroma) so the
  photographic plane is the only "material" in the room.
- **Ultramarine is the one hue absent from both** the ramp and the base — which is
  exactly why interaction ink is blue: any blue in the UI is unambiguously *the
  instrument talking*, never a data value.
- Chrome quietness budget: panel surface saturation stays ≤ 6%; the accent appears
  at most once per view region (one focus, one selection, one active tool).

### 2.3 The result-class trio (colorblind-safe by construction)

State color is never the only signal. Every result-class surface carries **color +
glyph + border style + fill texture**. Under deuteranopia/protanopia Viridian and
Ochre converge toward brown — they remain separated by lightness, glyph, border, and
fill; under tritanopia all three hold. Display words are plain English; the service
term rides in a tooltip for the undergrad audience.

| Class (data-state) | Display word | Color | Glyph (inline SVG) | Border | Fill |
|---|---|---|---|---|---|
| `fast_exact` / `exact_reconciled` | **Fast exact** ● / **Exact** ■ | Viridian | circle-check (solid stroke) | 1px solid | `rgba(14,110,78,.10)` tint |
| `fast_qualified` | **Qualified estimate** ◐~ | Ochre | tilde-wave | 1px **dashed** | `rgba(150,89,10,.10)` tint |
| `visual_pending` (incl. reserved/unknown classes displayed conservatively, per `effectiveClass`) | **Preview (owed)** ◌ | Slate | hollow ring with a gap | 1px solid `--line-strong` | no tint; 45° **hatch** `repeating-linear-gradient(135deg, rgba(95,106,102,.14) 0 1px, transparent 1px 5px)` |

The display WORD SET shown above is canonical per `interaction_simulations.md`
§5 D2 (the definition of record for badge copy): `fast_exact` → **Fast exact**
●, `fast_qualified` → **Qualified estimate** ◐~, `visual_pending` → **Preview
(owed)** ◌, `exact_reconciled` → **Exact** ■; a superseded result appends
**(stale)**. This section's earlier word set — Verified / Estimate ≈ / Awaiting
exact — is retired. Connected-but-unresolved states reuse the trio:
`catching_up`/`reconnecting` → slate + spinner glyph + "Replaying…", `failed` →
Brick + cross glyph, `stale` → slate hollow ring + "(stale)" suffix.

## 3. Typography

No build step, no webfont fetch (the studio must run fully offline per README) —
system stacks only, with one characterful system-available choice.

| Token | Stack | Role |
|---|---|---|
| `--font-display` | `"Avenir Next Condensed", "Arial Narrow", "Roboto Condensed", "Liberation Sans Narrow", var(--font-ui)` | The instrument voice: brand wordmark, eyebrows, section headings, badges/chips (caps), SOLAR TIME readout, hero metrics. Condensed grotesque in tracked caps is the synoptic-station-plots / flight-instrument label tradition; all four fallbacks ship on macOS/Windows/Linux respectively. |
| `--font-ui` | `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif` | Body, controls, prose, section copy. Native and crisp; Inter is dropped so the UI doesn't default to the app-template look where installed. |
| `--font-data` | `ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "DejaVu Sans Mono", monospace` | Every number that changes: revisions, UTCI values, latencies, cell counts, ids, payloads, table `dd`s, time ticks. Instruments speak in mono. |

Type scale (7 steps; the current 7–9 px floor is retired — pro density comes from
*structure*, not from sub-10 px type):

| Token | Size/Line | Tracking | Usage |
|---|---|---|---|
| `--text-xs` | 10 / 1.4 | +0.12em, caps (display) | Eyebrows, badges, chips, rail labels, ticks |
| `--text-sm` | 11 / 1.5 | normal | Captions, meta, secondary copy, `small` |
| `--text-base` | 12.5 / 1.55 | normal | Body, labels, buttons, inputs |
| `--text-md` | 14 / 1.4 | −0.005em | Row titles, compact section headings, component names |
| `--text-lg` | 16 / 1.35 | −0.01em | Panel headings ("Thermal impact") |
| `--text-xl` | 20 / 1.25 | −0.01em | Dialog/mismatch titles |
| `--text-hero` | 30 / 1.05 | +0.01em (display) | SOLAR TIME readout, primary metric |

Numeric rules: all live numerics get `font-variant-numeric: tabular-nums`; anything
that ticks (rail counters, metric values, outputs) is `--font-data`. The 30 px time
readout may use the condensed display face with `tabular-nums` where supported —
it changes discretely per slider commit, so width jitter is acceptable.

## 4. Layout concept

Keep the 3-pane shell — it is right for this app — and evolve it into a four-band
**console**: topbar / left tools / canvas / right analysis, plus a new full-width
bottom transport rail (§5). The shell becomes an instrument chassis: panes are flat
paper separated by hairlines; only the scene plate and true overlays cast shadows.

```
grid-template-rows: 56px minmax(0, 1fr) 32px;
grid-template-columns: var(--pane-left) minmax(560px, 1fr) var(--pane-right);
--pane-left: 280px;  --pane-right: 336px;   (≤1280 px: 256 / 312)
```

- **Topbar 56 px** (was 68): brand + scenario switcher left; connection/worker pills
  and actions right. Revision/class telemetry **moves out** into the rail.
- **Left pane 280 px**: 32 px inner padding budget fits the component-card grid
  (52 px thumb + copy + add) and slider editors at min touch width; 248 px usable is
  the floor for two-line family summaries with mono meta.
- **Right pane 336 px**: metric 2-up cells get ~150 px each — enough for
  `35,480 / 250,000 cells` at `--text-sm` mono on one line, which 320 px could not
  guarantee.
- **Canvas band**: 14 px padding; scene plate (radius 14, hairline, shadow-2) fills
  the remaining height above the 72 px timeline strip.
- **Workspace backdrop** = `--workspace-ground` with the **survey grid texture**
  (the drafting-table conceit): 1 px lines at 24 px pitch, `rgba(28,38,34,.045)`,
  plus a 120 px major line at `.06`. Decorative only; never inside panels; honored
  under forced-colors by dropping to flat.

**What collapses, in order** (user-triggered via pane-edge chevrons, `[` and `]`
keys, remembered per session): right analysis → left library → timeline strip
(48 px collapsed to readout-only). Both panes animate width to 0 in 160 ms
(instant under reduced motion).

**Responsive floor stays 1180 px.** This is a desktop instrument — no phone layout
is attempted (a floor, not a failure). At 1180 the grid resolves
256 + 560 + 312 = 1128 ✓ with room for the chassis padding; the current hidden
overflow quirk (grid min 1246 under an 1180 floor) is fixed by the minmax. Below
1180 the viewport scrolls horizontally, as today, with a fixed topbar+rail.

### Main screen

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ ◠ SOLWEIG STUDIO        SCENARIO ⌄ Campus shade study    ●server ●worker  ↺ ⬇ │ 56
├──────────┬──────────────────────────────────────────────────────┬──────────────┤
│DESIGN|LAY│ ↖ ✥ │▁▄ compare baseline                             │ LIVE ANALYSIS │
│          │                                                      │ Thermal impact│
│COMPONENTS│              ┌────────────────────────────┐           │  v214  ✓fast  │
│ ▣ Shade  │              │                            │           │ ┌───────────┐ │
│ ▣ Broad  │              │   ORTHOPHOTO + UTCI FIELD  │           │ │ −0.8 °C   │ │
│ ▣ Evergr.│              │     (realistic canvas)     │           │ └───────────┘ │
│          │              │                            │           │ ┌────┐┌─────┐│
│EDITORS   │              │                            │           │ │−3.4││8,940││
│ ▾ canopy │              │             N ▲  ▬▬ 100 m  │           │ └────┘└─────┘│
│ ▾ met    │              └────────────────────────────┘           │ SELECTION     │
│          │  ◐ Analysis synced                       0.84 s      │ height ──●─── │
│LAYERS    │ ┌──────────────────────────────────────────────────┐ │ JOB ①②③④     │
│ ☑ UTCI   │ │ SOLAR TIME 12:00 ╞════╪══════════╗ 30° ▓▓▓ 44°C  │ │ DEPENDENCY   │
│ ☑ shadow │ └──────────────────────────────────────────────────┘ │ MODEL SCOPE ✓ │
├──────────┴──────────────────────────────────────────────────────┴──────────────┤
│ ✓ VERIFIED   W 0214 · F 0213 · E 0213   ⌇   ◉AS ◉JR   RTT 84 ms   CONNECTED   │ 32
└────────────────────────────────────────────────────────────────────────────────┘
     280 px                ≥ 560 px (canvas)                             336 px
```

### Degraded / collaboration state (visual_pending, reconnect, held edits)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ ◠ SOLWEIG STUDIO        SCENARIO ⌄ Campus shade study    ●server ◌worker  ↺ ⬇ │
├──────────┬──────────────────────────────────────────────────────┬──────────────┤
│DESIGN|LAY│ ↖ ✥ │▁▄ compare                                     │ LIVE ANALYSIS │
│          │ ┌──────────────────────────────────────────────┐    │ v217 ◔await   │
│ ▣ Broad ◂│ │ ▶ REPLAYING — 3 held edits, backoff 2 s…     │    │ ┌───────────┐ │
│          │ │                                              │    │ │ −0.8 °C ▒ │ │← value
│EDITORS   │ │  ORTHOPHOTO + LAST EXACT UTCI                 │    │ └───────────┘ │  goes
│ ▾ met  ◂ │ │  (stale ROI cased ┄┄ awaiting exact ┄┄)       │    │ SELECTION     │ slate+
│          │ │                        N ▲  ▬▬ 100 m         │    │ …             │ shimmer
│          │ └──────────────────────────────────────────────┘    │               │
│          │  ◔ Exact owed +2                             0.3 s  │               │
├──────────┴──────────────────────────────────────────────────────┴──────────────┤
│ ◔ AWAITING EXACT  W 0217 · F 0217 · E 0215 +2 owed  ⌇⌇  ◉AS ◉JR ◉MW  RTT —  ↻ │
└────────────────────────────────────────────────────────────────────────────────┘
```

`◂` markers: pending/held edits mark their source rows (component card, family
editor) with a 3 px Slate left bar + hollow-ring glyph until reconciled — the same
trio language everywhere. Banners and metric values carry the class; nothing else
changes hue.

## 5. Signature element — the Metronome (revision transport rail)

A full-width 32 px instrument strip pinned under all three panes. It is the topbar's
`#realtimePill` grown up and given a home: the id and its `data-state` vocabulary
(`awaiting`, `visual_pending`, `fast_exact`, `exact_reconciled`, `stale`,
`catching_up`, `reconnecting`, `failed`) are preserved — the element relocates.

**Anatomy** — `display: grid; grid-template-columns: auto auto 1fr auto; gap: 14px;
padding: 0 12px; align-items: center;` background `--panel-sunken` tinted
(`#EEF1EF`), 1 px `--line` top border, `--font-data` 11 px, `--text-xs` caps labels.

1. **Epoch LED column** (far left): a 3 × 16 px bar, Ultramarine at 90% opacity,
   that blips to full for 160 ms on each received `canonical_revision` frame,
   throttled to one visible blip per ≥ 240 ms (a 100 ms epoch must not strobe).
   Between blips it rests at 25% opacity. States: *receiving* (blipping),
   *idle/subscribed-quiet* (static 25%), *offline* (Slate, static).
2. **Class lamp**: the result-class chip (§2.3) — glyph + word, e.g.
   `[✓ VERIFIED]`. This is the relocated `#realtimePill` body.
3. **Revision triple**: `W 0214 · F 0213 · E 0213`. Letters `--text-xs` caps
   Slate; numbers mono 11 px tabular Survey Ink; separators `·` faint. Each letter
   carries a plain-language tooltip: *W workspace — everyone's edits, merged*,
   *F fast — the ~1 s analysis*, *E exact — the full SOLWEIG run*. When
   `E < F`, E renders in Ochre with a trailing `+2 owed` (mono 10 px). The triple
   can never show `E > F` (contract invariant) — the rail treats a violation as a
   `failed`-class fault display.
4. **Pendulum** (right of a 1fr spacer gap, left of actors): a 16 px SVG metronome
   whose arm sweeps ±24° at 0.8 Hz **only while frames are arriving**; it freezes
   when the stream is quiet and hides offline (replaced by the word `OFFLINE`).
5. **Session cluster**: actor chips — 18 px circles, Plate White, hairline
   `--line-strong`, mono 9 px initials, −4 px overlap, max three then `+2`;
   hover → names. `RTT 84 ms` mono; mode word caps display (`CONNECTED` /
   `LOCAL` / `OFFLINE`). Offline mode the rail reads
   `OFFLINE PREVIEW · local kernel — no collaboration` end-to-end, keeping the
   instrument identity consistent when the server is absent.

**Ladder readout**: hovering the class lamp reveals the degradation-ladder position
(service contract §"Degradation ladder") in the tooltip, e.g. *lane 2/6 — bounded
specialized kernel*. Power users get the ladder; undergrads get the word.

Reduced motion: LED and pendulum become static lamps (state shown by opacity step
only); the reconciliation underline below becomes a persistent 2 px Viridian
underline while `E == F`.

## 6. Component spec

### 6.0 Elevation and borders (light surfaces — no dark-mode shadows)

| Level | Where | Surface | Border | Shadow |
|---|---|---|---|---|
| L0 | Page ground | Instrument Paper + grid texture (workspace only) | — | — |
| L1 | Structural: topbar, rail, side panes, timeline | Paper / sunken strip | 1 px `--line` between panes | **none** — hairlines do the work |
| L2 | Cards on panes: component cards, family cards, metric cards, view cards, well groups | Plate White | 1 px `--line` | `--shadow-1: 0 1px 2px rgba(23,32,28,.05), 0 1px 3px rgba(23,32,28,.04)` |
| L3 | Over-canvas floats: scene-context, sync chip, banners, toolbar, popovers, tooltips | `rgba(255,255,255,.94)` + `backdrop-filter: blur(10px)` | 1 px `--line` | `--shadow-2: 0 2px 6px rgba(23,32,28,.08), 0 8px 24px rgba(23,32,28,.10)` |
| L4 | Modal: site-mismatch, future dialogs | Plate White, scrim `rgba(28,38,34,.32)` | 1 px `--line-strong` | `--shadow-2` |

Focus, everywhere: `:focus-visible { outline: 2px solid var(--ink-accent);
outline-offset: 1px; }` — on ink-filled surfaces the outline is white with a 1 px
ink casing. Radii: `--r-xs 4px` (chips, inputs) · `--r-sm 6px` (buttons) ·
`--r-md 10px` (cards) · `--r-lg 14px` (plate, floats, modals).

### 6.1 Shared control state matrix

Referenced by every control family below instead of repeating.

| State | Surface | Border | Text | Notes |
|---|---|---|---|---|
| Default | Plate White | 1 px `--line-strong` | Ink | |
| Hover | `--panel-sunken` tint | `--line-strong` | Ink | 120 ms |
| Active/pressed | `--panel-sunken` | 1 px Ultramarine | Ink | |
| Selected | `rgba(43,73,204,.08)` | 1 px Ultramarine | Ultramarine (or Ink + 2 px Ultramarine left/under bar) | |
| Focus-visible | — | 2 px Ultramarine ring, offset 1 | — | Never removed |
| Disabled | Paper, 55% opacity | `--line` | `--ink-faint` (≥ 3:1 kept) | `cursor: not-allowed`, no shadow |
| Loading | inert (not hidden), 14 px ring spinner Ink/Ultramarine | — | label + verb | Controls stay visible |
| Degraded | per result-class trio (§2.3) | per trio | per trio | Tooltip carries ladder reason |

### 6.2 Topbar & brand

Flat L1, hairline bottom. **Brand mark** (31 px): Plate White rounded square,
hairline border; inside, a 2 px **Ultramarine solar-arc** (a shallow arc rising
left-to-right, SOLWEIG = "sun path") over a 1 px Survey Ink horizon line with three
tiny tick marks — the sun-path diagram miniaturized; replaces the three-bar glyph.
Wordmark `SOLWEIG STUDIO` display caps 13 px; sub-label "Thermal design workspace"
`--text-sm` Slate. Scenario switcher: eyebrow + button per matrix (no border in
default; hover well). Actions right: pills (below), Reset ghost, **Export snapshot
primary (ink fill, white text)**.

**Live pills** (connection `#connectionPill`, worker `#workerPill`): 10 px caps
display chips, L1-inset wells, dot lamps — connected Viridian solid, running/local
Ochre pulsing, failed Brick, idle Slate. `#realtimePill` migrates to the rail (§5);
its topbar slot is removed, not reused.

### 6.3 Left pane — library & editors

- **Tabs** (Design/Layers): text tabs with a 2 px Ultramarine underline on active
  (replaces the pill-well); inactive Slate → Ink on hover.
- **Section headings**: eyebrow (display caps `--text-xs` Slate) + `--text-md`
  heading; right-aligned head actions (icon-button per matrix).
- **Component cards** (L2): 64 px min-height grid `52px 1fr auto` + drag dots.
  Tree thumbnails re-based for light: ground = `--workspace-ground` + micro grid
  texture; trunks `#7A5A3C`; canopies desaturated one step to sit in paper chrome.
  Drag: card lifts (shadow-2, 1 px translate) and cursor-grab; the in-flight ghost
  is the card at 90% with a dashed Ultramarine outline. Pending-add (held edit) →
  Slate left bar per §4.
- **Family editors** (capability-driven): `<details>` cards, L2; summary = name
  (`--text-md`) + mono meta line (`--text-xs` Slate, e.g. `tree-edits-v1 · 3 ops`);
  chevron rotates 90° open (no animation under reduced motion). **Property rows**:
  label `--text-sm` + **mono output value** right-aligned (11 px tabular, live);
  sliders = 4 px `--panel-sunken` track, Ultramarine fill to thumb, 14 px Plate
  White thumb with `--line-strong` border and Ultramarine ring on focus; schema
  min/max tick marks 1 px `--line-strong`. Numeric met inputs carry unit suffixes
  (`°C`, `m/s`) mono inside the right of the well. Invalid (`422` field): 2 px Brick
  outline + server message inline `--text-sm` Brick. **Rejected properties**: Ochre
  dashed-border card carrying the schema's rejection reason verbatim — never
  paraphrased. **Class chips**: pill chips per matrix; pressed = selected state;
  fenced classes hidden (valid − fenced, as today). **No-transport families**:
  normal card + status-pill `BLOCKED` (Ochre) + the payload disclosure in a mono
  pre (L2, `--text-xs`, wrap) — typed refusal stays visible, as required.
- **Layer rows**: 14 px swatches — heat = mini ramp; shadow = Survey Ink 80%;
  ROI = Slate dashed on Ultramarine-tint. Checkbox `accent-color` Ultramarine.
- **Footnote**: Paper inset well, `i` lamp Slate, `--text-sm` Slate.
- **View panel** (Layers tab): field labels `--text-xs` caps Slate over select/input
  per matrix; view cards L2 with mono `dt/dd` rows; time chips mono per matrix;
  zero-pill uses the trio (free/verified when server flags say so, neutral otherwise).

### 6.4 Workspace — canvas chrome

- **Toolbar** (L3 float, top-left): tool buttons 30 px square per matrix; active
  tool = selected state; divider 1 px `--line`; Compare toggle is a labeled button
  whose pressed state shows a half-ink/half-Ultramarine split glyph.
- **Scene plate**: radius 14, 1 px `--line-strong`, shadow-2. Canvases unchanged
  (the realism is the product). The realism **must not be dimmed** — no scrims over
  the plate except explicit states (drop hint, banners).
- **Scene-context chip** (L3, top-right): eyebrow LIVE MODEL + time `--text-md` +
  met line with mono values (`31.6 °C · RH 47% · wind 2.4 m/s`).
- **Sync chip** (`#analysisToast`, L3, bottom-left): spinner→lamp states as today;
  title `--text-base`, detail `--text-sm` Slate, duration mono 11 px; during
  visual_pending the whole chip adopts the Slate trio treatment (hollow-ring lamp,
  hatch-free — hatch is reserved for chips/badges).
- **North arrow & scale bar**: pure cartographic treatment — **Survey Ink glyphs
  with a 1.5 px white halo** (`paint-order`-style casing), no chip background;
  legible on any orthophoto without adding surface over the imagery.
- **Drop hint**: 2 px dashed Ultramarine inset border + `rgba(255,255,255,.55)`
  scrim; center disc Ultramarine, white plus; enters per motion §7.
- **Failure banner** (L3, top-center): white .97, 3 px Brick left edge, cross-glyph
  lamp, message `--text-base`, Retry text-button Brick. Appears/fades per §7.
- **Site-mismatch banner** (L4 modal over scrim): title `--text-xl` display,
  Brick lamp; identity diff as a cased table (§6.7); primary action ink-filled.

### 6.5 Timeline (time scrubber)

L1 strip, 72 px, grid `110px 1fr 220px`. Readout: eyebrow SOLAR TIME + 30 px
display `12:00` (tabular). Slider: 8–18 h, custom track — 4 px `--panel-sunken`,
Ultramarine fill left of a 16 px white thumb; 1 h ticks 1 px `--line-strong`;
keyboard step 1 h. While the exact result for the chosen hour is owed (time rides a
no-op edit), the readout carries a small Ochre `≈` until reconciliation. **Legend**:
the ramp itself is `HEAT_STOPS`, unchanged — 8 px bar, endpoint ticks mono with
white halos (`30 °C` … `44 °C UTCI`), labels Comfortable/Extreme heat `--text-xs`
caps Slate. The legend is canvas furniture, not chrome: it may sit on the plate's
sunken strip but its colors are data colors and are never themed.

### 6.6 Right pane — analysis

- **Header**: eyebrow LIVE ANALYSIS, `--text-lg` title, subtitle `--text-sm` Slate,
  version chip mono (`v214`, L2 well).
- **Metric grid** (2-up, L2): label `--text-xs` caps Slate; value **mono 18 px**
  (primary 22 px); caption `--text-sm` Slate. **Metrics carry the result class**:
  verified → Survey Ink values; estimate → Ochre values + `≈`; awaiting → Slate
  values + shimmer (§7). Primary card gets a 2 px class-color top border instead of
  a tint — the class is legible even for red-green users because the trio's
  glyph/border rules ride along in the sync chip and rail.
- **Selection**: heading + danger icon-button (Brick text, matrix states);
  property rows as §6.3.
- **Job steps**: 18 px circular **stage lamps**, mono numerals — future: hairline
  circle Slate; running: Ochre ring pulse; complete: Viridian tint fill + check.
  Detail line `--text-sm` Slate with mono values.
- **Dependency impact**: rows eyebrow + chips (mono `--text-xs`): `changed` = Ochre
  solid border, `recomputed` = Viridian tint, `reused` = plain hairline Slate —
  same semantic ladder as the trio. Scope line in a Paper well, `--text-sm`, mono
  emphasis values.
- **Model scope / `#exactBadge`**: card L2; badge states — `exact` = Viridian trio
  chip, `preview` = Ochre dashed chip, `connecting` = Slate hollow chip (nothing
  verified yet). `dl` rows: dt `--text-sm` Slate, dd mono right. Scope list
  `--text-sm`; footnote italic Slate.

### 6.7 Tables & met editor (data readouts)

One pattern for `scope-meta`, identity-diff, and view-card rows: rows on Plate White
(L2 container) separated by 1 px `--line`; keys `--text-sm` Slate left; values mono
11 px right, `word-break` for ids. Identity-diff changed rows: Brick-tinted
background + 2 px Brick left bar (color + bar, never color alone); pinned values
Ochre, current values Viridian — each with its glyph. The **met editor** is the
meteorological-forcing family editor (§6.3) with unit-suffixed mono inputs and a
note that time is set via the timeline, not the editor.

### 6.8 Toasts, dialogs, tooltips

Toasts (L3, bottom-center above rail): white .94, blur, shadow-2, auto-dismiss 4 s
+ hover-pause; icon lamp carries the semantic color; verb-noun copy ("Snapshot
exported"). Dialogs (L4): radius 14, scrim, `--text-xl` display title, actions row
right (ink primary, ghost secondary, text cancel); Escape and scrim-click close when
safe. Tooltips (L3): `--text-sm` on white .97, hairline, 4 px radius, 8 px padding,
150 ms delay — the plain-language glossary layer for service vocabulary (rail
letters, class lamps, ladder lane).

## 7. Motion — five moments, all `prefers-reduced-motion`-safe

| # | Moment | Spec | Duration | Reduced motion |
|---|---|---|---|---|
| 1 | **Epoch metronome** | Rail LED blip + pendulum sweep (§5), throttled ≥ 240 ms | 160 ms blip / 0.8 Hz sweep | Static lamps; opacity step only |
| 2 | **Fast-lane flash** | On a new fast result, the dirty-ROI casing flashes once in the current class color (white→class 80%→white) | 200 ms, one-shot | No flash; casing steps to class color |
| 3 | **Pending shimmer** | `visual_pending` metric values: a 2.5 s luminance sweep (Slate 100→85→100%); trio chips' hatch is static | 2.5 s loop | Static Slate value |
| 4 | **Reconciliation snap** | E counter rolls to F/W; 2 px Viridian underline flashes under the triple and fades | 180 ms | Underline appears statically, persists 2 s |
| 5 | **Overlay entrances** | Toasts, banners, drop hint: rise 4 px + fade in; exits reverse | 160 ms | Opacity step only |

Global tokens: `--dur-fast 120ms; --dur-base 180ms; --dur-slow 320ms;
--ease: cubic-bezier(.2,0,0,1)`. Everything else in the chrome is static. One
`@media (prefers-reduced-motion: reduce)` block sets all animations/transitions to
near-zero and swaps the four looping behaviors for their static fallbacks.

## 8. Anti-generic check

The three AI-default looks, and how each is avoided:

1. **Warm cream + serif display + terracotta** — absent. The ground is *cool*
   Instrument Paper `#F3F5F3` (green-gray, zero warmth); there is no serif anywhere;
   the accent is ultramarine, not terracotta, and terracotta-adjacent Ochre exists
   only as a small semantic state color.
2. **Near-black + acid green** — this is a light theme; the only near-black is text
   and the ink primary button. Green appears only as deep Viridian in badges/lamps,
   never as a neon accent.
3. **Broadsheet hairlines + zero radius + newspaper columns** — the closest
   temptation, since this UI is dense and hairline-led. Deliberate differentiators:
   a real radius system (4/6/10/14), console-pane structure (chassis bands + plate)
   rather than columns; condensed instrument caps and monospace telemetry rather
   than editorial serifs; the survey-grid workspace texture; and color used strictly
   as semantics (trio + ultramarine), never as decoration.

The **one bold move** is the Metronome rail (§5) — permanent full-width
instrument chrome for the revision triple — with the drafting-table plate-and-grid
workspace as its supporting bet. Everything else stays disciplined so those two can
be loud.

## Appendix A — token block (copy-pasteable)

```css
:root {
  color-scheme: light;

  /* — Surfaces — */
  --paper: #F3F5F3;                 /* page ground (instrument paper) */
  --panel: #FFFFFF;                 /* raised panels & cards (plate white) */
  --panel-sunken: #EAEEEB;          /* input wells, tab strips, rail ground */
  --workspace-ground: #E7EBE8;      /* drafting-table field behind the plate */
  --float-surface: rgba(255, 255, 255, 0.94); /* L3 over-canvas floats */
  --scrim: rgba(28, 38, 34, 0.32);

  /* — Ink hierarchy — */
  --ink: #1C2622;                   /* primary text, ink buttons */
  --ink-muted: #55605B;             /* secondary text (5.9:1 on paper) */
  --ink-faint: #7C8580;             /* decorative / disabled labels only */
  --ink-accent: #2B49CC;            /* ultramarine survey ink */

  /* — Lines — */
  --line: #D9DED9;                  /* hairline */
  --line-strong: #C3CAC4;           /* control borders */

  /* — Semantic states — */
  --accent-soft: rgba(43, 73, 204, 0.08);
  --ok: #0E6E4E;                    /* viridian — verified */
  --ok-soft: rgba(14, 110, 78, 0.10);
  --warn: #96590A;                  /* ochre — qualified / provisional */
  --warn-soft: rgba(150, 89, 10, 0.10);
  --danger: #C03526;                /* brick — failure */
  --danger-soft: rgba(192, 53, 38, 0.08);
  --pending: #5F6A66;               /* slate — visual_pending / stale */
  --pending-hatch: repeating-linear-gradient(
    135deg, rgba(95, 106, 102, 0.14) 0 1px, transparent 1px 5px);

  /* Result-class aliases (use these, not the hues, in components) */
  --class-fast-exact: var(--ok);
  --class-fast-qualified: var(--warn);
  --class-visual-pending: var(--pending);

  /* — Elevation (light only; no dark-mode shadows) — */
  --shadow-1: 0 1px 2px rgba(23, 32, 28, 0.05), 0 1px 3px rgba(23, 32, 28, 0.04);
  --shadow-2: 0 2px 6px rgba(23, 32, 28, 0.08), 0 8px 24px rgba(23, 32, 28, 0.10);

  /* — Radii — */
  --r-xs: 4px;  --r-sm: 6px;  --r-md: 10px;  --r-lg: 14px;

  /* — Type — */
  --font-display: "Avenir Next Condensed", "Arial Narrow", "Roboto Condensed",
    "Liberation Sans Narrow", -apple-system, "Segoe UI", sans-serif;
  --font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
  --font-data: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas,
    "DejaVu Sans Mono", monospace;

  --text-xs: 10px;    --lh-xs: 1.4;   /* caps display: eyebrows, badges, ticks */
  --text-sm: 11px;    --lh-sm: 1.5;
  --text-base: 12.5px; --lh-base: 1.55;
  --text-md: 14px;    --lh-md: 1.4;
  --text-lg: 16px;    --lh-lg: 1.35;
  --text-xl: 20px;    --lh-xl: 1.25;
  --text-hero: 30px;  --lh-hero: 1.05; /* display face */
  --tracking-caps: 0.12em;

  /* — Motion — */
  --dur-fast: 120ms;  --dur-base: 180ms;  --dur-slow: 320ms;
  --ease: cubic-bezier(0.2, 0, 0, 1);

  /* — Shell — */
  --pane-left: 280px;   /* ≤1280px viewport: 256px */
  --pane-right: 336px;  /* ≤1280px viewport: 312px */
  --topbar-h: 56px;
  --rail-h: 32px;
  --timeline-h: 72px;
}
```

## Appendix B — migration notes for `styles.css`

**Reuse as-is (structural):** the `.app-shell` grid concept, pane overflow/scroll
rules, canvas stacking (`#glCanvas` z1 / `#overlayCanvas` z2, crosshair/copy
cursors), absolute-overlay positions, `.scene-card.is-dragging` behavior, the
`.live-pill[hidden]` UA-rule fix, `.visually-hidden`, the compact-height media
query, and every id/data-state hook (`#realtimePill` relocates into the rail; ids
otherwise unchanged so `app.mjs` wiring survives).

**Replace:** every color/shadow/radius custom property (Appendix A), the font
stacks, the entire type scale (7–9 px sizes are retired; `--text-xs` 10 px is the
floor), translucent-white-on-dark panel recipes (light surfaces use solid fills +
hairlines), the dark radial body gradient (now the survey-grid texture on
`--workspace-ground`, workspace band only), and the pill-tab control (underline
tabs). The UTCI legend gradient and layer swatch ramp stay — they are data, not
chrome.
