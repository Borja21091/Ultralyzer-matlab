(keyboard-shortcuts)=
# Appendix

This appendix keeps reference material separate from the main workflow pages.

## Keyboard shortcuts

The list below reflects the current shortcut registrations in the GUI.

### View and overlay controls

| Action | Shortcut |
| :--- | :--- |
| Show artery overlay | `1` |
| Show optic-disc overlay | `2` |
| Show vein overlay | `3` |
| Show all vessels | `4` |
| Show all masks | `5` |
| Hide mask overlay | `6` |
| Change mask opacity | `Ctrl + Scroll` |
| Fit the current view | Double-click in the canvas |

### Editing tools

| Action | Shortcut |
| :--- | :--- |
| Brush | `B` |
| Smart paint | `Shift + B` |
| Eraser | `E` |
| Swap artery/vein classification | `C` |
| Fovea location tool | `F` |
| Increase brush size | `=` |
| Decrease brush size | `-` |
| Change brush size | `Shift + Scroll` |
| Save edits | `Ctrl + S` |
| Undo | `Ctrl + Z` |
| Redo | `Ctrl + Shift + Z` |
| Temporary pan while editing | Hold `Space` |

### Navigation

| Action | Shortcut |
| :--- | :--- |
| Previous image | `Left Arrow` |
| Next image | `Right Arrow` |

(metric-definitions)=
## Metric definitions

The tables below describe the current metric families documented by the active calculators. Metric availability depends on the presence of the required landmarks, ROI selection, and geometry support.

:::{note}
Geometry-dependent metrics such as `disc_area_mm2`, `*_area_mm2`, `*_density_area`, and geodesic distance conversions may be skipped when the geometry backend is unavailable or incomplete.
:::

### Landmark metrics

These are image-level metrics that do not depend on the selected ROI combination.

| Metric key | Description |
| :--- | :--- |
| `laterality` | Eye laterality inferred from the disc-fovea relationship. |
| `disc_center_x`, `disc_center_y` | Optic-disc center coordinates in image space. |
| `disc_area_px` | Optic-disc area in pixels. |
| `disc_area_mm2` | Optic-disc area in square millimetres when geometry area support is available. |
| `disc_diameter_px`, `disc_diameter_um` | Equivalent optic-disc diameter in pixels and micrometres. |
| `disc_major_axis_px`, `disc_major_axis_um` | Major axis of the fitted optic-disc ellipse in pixels and micrometres. |
| `disc_minor_axis_px`, `disc_minor_axis_um` | Minor axis of the fitted optic-disc ellipse in pixels and micrometres. |
| `disc_orientation_deg` | Orientation of the fitted optic-disc ellipse in degrees. |
| `disc_circularity` | Circularity estimate for the optic-disc region. |
| `disc_eccentricity` | Eccentricity of the fitted optic-disc ellipse. |
| `fovea_center_x`, `fovea_center_y` | Fovea coordinates saved for the current image. |
| `disc_fovea_distance_px`, `disc_fovea_distance_um` | Disc-to-fovea distance in pixels and micrometres. |
| `disc_fovea_angle_deg` | Angle between the disc and fovea reference line. |
| `crae_**` | Central retinal artery equivalent (CRAE) estimate using the Knudtson formula in pixels and micrometres. |
| `crve_**` | Central retinal vein equivalent (CRVE) estimate using the Knudtson formula in pixels and micrometres. |
| `avr_**` | Artery-to-vein ratio (AVR) estimate using the Knudtson formula in pixels and micrometres. |

### ROI definitions

ROI-dependent metrics are currently organized around these definitions:

| ROI code | Display name | Purpose |
| :--- | :--- | :--- |
| `full` | Full image | Whole-image vessel summary. |
| `central` | Central retina | Central retinal measurements around the optic-disc landmark. |
| `mid_periphery` | Mid-periphery | Template-based mid-peripheral measurements. |

### ROI metric key patterns

For ROI-dependent metrics, the calculators use three prefixes:

- `vessel_` for all vessels combined
- `a_` for arteries only
- `v_` for veins only

The current key families are:

| Metric pattern | Description |
| :--- | :--- |
| `roi_area_**` | Area of the selected ROI in pixels and square millimitres. |
| `*_density` | Vessel-pixel fraction inside the active ROI. |
| `*_fractal_dimension_sandbox` | Fractal dimension estimated with the sandbox method. |
| `*_fractal_dimension_boxcount` | Fractal dimension estimated with box counting. |
| `*_area_mm2` | Vessel area in square millimetres when geometry area support is available. |
| `*_density_area` | Vessel area divided by ROI area when both geometry areas are available. |
| `*_tortuosity_density` | Mean density-style tortuosity measure across extracted vessel paths. |
| `*_tortuosity_distance` | Mean chord-versus-curve tortuosity measure across extracted vessel paths. |
| `*_width_px` | Mean vessel width in pixels. |
| `*_width_um` | Mean vessel width in micrometres. |
| `*_width_gradient_px` | Linear width gradient in image space. |
| `*_width_gradient_um` | Linear width gradient in real-space micrometres. |
| `*_width_intercept_px` | Linear width-fit intercept in pixels. |
| `*_width_intercept_um` | Linear width-fit intercept in micrometres. |

### Relationship metrics

| Metric key | Description |
| :--- | :--- |
| `av_ratio` | Ratio of artery density to vein density when both are available. |
| `av_crossings` | Number of artery-vein crossings estimated from overlapping skeleton pixels. |
| `av_arcade_concavity` | Central-ROI arcade concavity estimate when the necessary landmarks and vessel data are available. |

### Notes on exported workbooks

- Landmark metrics are saved once per image.
- ROI metrics are saved per ROI selection.
- Some database columns may remain empty when the corresponding landmarks or geometry operations are unavailable for a given image.
