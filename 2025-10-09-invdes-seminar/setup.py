"""Utilities for constructing and analyzing dual-layer grating coupler simulations.

This script acts as a centralized configuration file, defining the physical
constants, geometric constraints, and core simulation-building functions used
throughout the seminar notebooks.

Notes
-----
* All lengths are specified in micrometers to match the fabrication-scale
  discussion in the accompanying tutorial notebooks.
* ``autograd.numpy`` is used instead of standard NumPy so that the same
  functions seamlessly support gradient-based optimization workflows.
* ``getval`` extracts plain floats from Autograd tracers whenever we pass
  values into Tidy3D constructors that expect concrete numbers.
"""

from __future__ import annotations

from typing import Sequence

import autograd.numpy as np
import tidy3d as td
from autograd.tracer import getval

inf = 1000
buffer_left = 3.0
buffer_right = 3.0
buffer_bot = 2.0
buffer_top = 0.5

substrate_index = 3.47
box_index = 1.44
si_index = 3.47
sin_index = 2.0

substrate = td.Medium(permittivity=substrate_index**2)
box = td.Medium(permittivity=box_index**2)
si = td.Medium(permittivity=si_index**2)
sin = td.Medium(permittivity=sin_index**2)

# Number of grating elements balances efficiency gains with simulation cost and
# a manageable optimization search space for a standard C-band coupler.
num_elements = 15

# Representative design-rule constraints to mirror ones typically found in silicon photonics processes.
# Maximums are not strictly necessary but are included to keep things within reasonable bounds.
min_width_si = 0.1
min_gap_si = 0.2
min_width_sin = 0.2
min_gap_sin = 0.3
max_width_si = 1.0
max_gap_si = 1.0
max_width_sin = 1.0
max_gap_sin = 1.0

first_gap_si = -0.7  # First gap in silicon is effectively the layer offset.
first_gap_sin = 1.5 * min_gap_sin
default_spacer_thickness = 0.3  # Vertical separation between the two functional layers.


def _make_teeth_structure(
    centers: np.ndarray,
    widths: np.ndarray,
    *,
    center_z: float,
    thickness: float,
    medium: td.Medium,
    name: str,
) -> tuple[td.Structure, np.ndarray]:
    """Construct a ``td.Structure`` representing repeating grating teeth."""
    teeth = [
        td.Box(center=(center, 0, center_z), size=(width, inf, thickness))
        for center, width in zip(centers, widths)
    ]
    structure = td.Structure(
        geometry=td.GeometryGroup(geometries=teeth),
        medium=medium,
        name=name,
    )
    extent = centers + widths / 2
    return structure, extent


center_wavelength = 1.55
min_steps_per_wvl = 20
run_time = 1e-12


def widths_gaps_to_centers(
    widths: Sequence[float],
    gaps: Sequence[float],
    *,
    first_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert widths and gaps into center locations for rectangular grating teeth.

    Parameters
    ----------
    widths : Sequence[float]
        Ordered list of grating tooth widths in micrometers.
    gaps : Sequence[float]
        Ordered list of gaps between adjacent teeth in micrometers. The
        sequence can have the same length as ``widths``; any extra entry is
        automatically ignored.
    first_gap : float
        Offset between the start of the simulation coordinate system and the
        leading edge of the first tooth, in micrometers.

    Returns
    -------
    centers : np.ndarray
        Array of tooth center positions in micrometers, aligned with the
        simulation ``x`` axis.
    widths : np.ndarray
        Copy of the widths as a NumPy array (matching the Autograd backend).
    """
    widths = np.array(widths)
    gaps = np.array(gaps)
    n = int(widths.size)

    gaps_interior = gaps[: n - 1] if n > 1 else gaps[:0]
    if n == 0:
        return widths[:0], widths

    combined = widths[:-1] + gaps_interior
    cumulative = np.cumsum(combined)
    prefix_offset = np.zeros(1, dtype=widths.dtype)
    prefix = np.concatenate((prefix_offset, cumulative)) if cumulative.size else prefix_offset
    centers = first_gap + prefix + widths / 2
    return centers, widths


def make_grating_structures(
    widths_si: Sequence[float],
    gaps_si: Sequence[float],
    widths_sin: Sequence[float],
    gaps_sin: Sequence[float],
    *,
    first_gap_si: float,
    first_gap_sin: float,
    box_thickness: float,
    si_thickness: float,
    spacer_thickness: float,
    sin_thickness: float,
) -> tuple[list[td.Structure], dict[str, float]]:
    """Return tidy3d structures for the dual-layer grating.

    Parameters
    ----------
    widths_si : Sequence[float]
        Silicon tooth widths in micrometers.
    gaps_si : Sequence[float]
        Silicon gaps in micrometers measured along ``x``.
    widths_sin : Sequence[float]
        Silicon nitride tooth widths in micrometers.
    gaps_sin : Sequence[float]
        Silicon nitride gaps in micrometers.
    first_gap_si : float
        Offset from the waveguide start to the first silicon tooth, in
        micrometers.
    first_gap_sin : float
        Offset from the silicon nitride waveguide to its first tooth, in
        micrometers.
    box_thickness : float
        Buried oxide thickness in micrometers.
    si_thickness : float
        Silicon device layer thickness in micrometers.
    spacer_thickness : float
        Spacer between silicon and silicon nitride layers in micrometers.
    sin_thickness : float
        Silicon nitride layer thickness in micrometers.

    Returns
    -------
    structures : list[td.Structure]
        Collection of Tidy3D structures representing the full coupler stack.
    geometry_info : dict[str, float]
        Dictionary with geometry references used to size the simulation domain
        and place sources/monitors. The keys are ``"c_sin"`` (center of the
        silicon nitride waveguide) and ``"x_gc"`` (end of the patterned region).
    """
    c_si, w_si = widths_gaps_to_centers(widths_si, gaps_si, first_gap=first_gap_si)
    c_sin, w_sin = widths_gaps_to_centers(widths_sin, gaps_sin, first_gap=first_gap_sin)

    structures: list[td.Structure] = []
    substrate_geom = td.Box.from_bounds((-inf, -inf, -inf), (inf, inf, 0))
    structures.append(
        td.Structure(
            geometry=substrate_geom,
            medium=substrate,
            name="substrate",
        )
    )

    si_center_z = substrate_geom.bounds[1][2] + box_thickness + si_thickness / 2
    si_teeth, si_extents = _make_teeth_structure(
        c_si,
        w_si,
        center_z=si_center_z,
        thickness=si_thickness,
        medium=si,
        name="si_teeth",
    )
    structures.append(si_teeth)

    sin_waveguide_geom = td.Box.from_bounds(
        (-inf, -inf, si_center_z + si_thickness / 2 + spacer_thickness),
        (0, inf, si_center_z + si_thickness / 2 + spacer_thickness + sin_thickness),
    )
    structures.append(
        td.Structure(
            geometry=sin_waveguide_geom,
            medium=sin,
            name="sin_waveguide",
        )
    )

    sin_teeth, sin_extents = _make_teeth_structure(
        c_sin,
        w_sin,
        center_z=sin_waveguide_geom.center[2],
        thickness=sin_thickness,
        medium=sin,
        name="sin_teeth",
    )
    structures.append(sin_teeth)

    return structures, {
        "c_sin": sin_waveguide_geom.center,
        "x_gc": np.maximum(sin_extents.max(), si_extents.max()),
    }


def make_simulation(
    widths_si: Sequence[float],
    gaps_si: Sequence[float],
    widths_sin: Sequence[float],
    gaps_sin: Sequence[float],
    *,
    first_gap_si: float = first_gap_si,
    first_gap_sin: float = first_gap_sin,
    box_thickness: float = 2.0,
    si_thickness: float = 0.09,
    spacer_thickness: float = default_spacer_thickness,
    sin_thickness: float = 0.4,
    center_wavelength: float = center_wavelength,
    bandwidth: float = 0.1,
    freq_points: int = 101,
    beam_offset_x: float = 5.0,
    beam_height: float = 2.0,
    beam_mfd: float = 9.2,
    beam_angle_deg: float = 10,
    include_field_monitor: bool = False,
) -> td.Simulation:
    """Assemble a tidy3d simulation for the dual-layer grating coupler.

    Parameters
    ----------
    widths_si, gaps_si, widths_sin, gaps_sin : Sequence[float]
        Geometry parameters in micrometers defining the silicon and silicon
        nitride tooth widths and gaps.
    first_gap_si : float, optional
        Offset between the waveguide start and first silicon tooth in
        micrometers.
    first_gap_sin : float, optional
        Offset between the silicon nitride guide and its first tooth.
    box_thickness : float, optional
        Buried oxide thickness in micrometers.
    si_thickness : float, optional
        Silicon layer thickness in micrometers.
    spacer_thickness : float, optional
        Vertical spacing between silicon and silicon nitride layers in
        micrometers.
    sin_thickness : float, optional
        Silicon nitride layer thickness in micrometers.
    center_wavelength : float, optional
        Central wavelength in micrometers.
    bandwidth : float, optional
        Spectral span around the central wavelength in micrometers.
    freq_points : int, optional
        Number of frequency samples across the bandwidth.
    beam_offset_x : float, optional
        Horizontal displacement of the incident Gaussian beam center, in
        micrometers.
    beam_height : float, optional
        Distance between the silicon nitride surface and the beam waist center,
        in micrometers.
    beam_mfd : float, optional
        Mode field diameter of the Gaussian beam in micrometers.
    beam_angle_deg : float, optional
        Incident angle of the Gaussian beam in degrees.
    include_field_monitor : bool, optional
        If ``True``, include a 2D field monitor slicing through ``x``-``z``.

    Returns
    -------
    td.Simulation
        Fully defined Tidy3D simulation ready to run.
    """
    structures, geometry_info = make_grating_structures(
        widths_si,
        gaps_si,
        widths_sin,
        gaps_sin,
        first_gap_si=first_gap_si,
        first_gap_sin=first_gap_sin,
        box_thickness=box_thickness,
        si_thickness=si_thickness,
        spacer_thickness=spacer_thickness,
        sin_thickness=sin_thickness,
    )

    freq0 = td.C_0 / center_wavelength
    freqs = td.C_0 / np.linspace(
        center_wavelength - bandwidth / 2,
        center_wavelength + bandwidth / 2,
        freq_points,
    )

    source_z = geometry_info["c_sin"][2] + sin_thickness / 2 + beam_height
    source = td.GaussianBeam(
        center=(beam_offset_x, 0, source_z),
        size=(inf, inf, 0),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
        pol_angle=np.pi / 2,
        angle_theta=np.deg2rad(beam_angle_deg),
        direction="-",
        waist_radius=beam_mfd / 2,
        name="input_beam",
    )

    monitors = [
        td.ModeMonitor(
            center=(-buffer_left + 0.5, 0, getval(geometry_info["c_sin"][2])),
            size=(0, inf, 3),
            freqs=freqs,
            mode_spec=td.ModeSpec(num_modes=1),
            name="mode_monitor",
        )
    ]

    if include_field_monitor:
        monitors.append(
            td.FieldMonitor(
                center=(0, 0, 0),
                size=(inf, 0, inf),
                freqs=freq0,
                fields=("Ey",),
                name="field_monitor",
            )
        )

    x_min = getval(-buffer_left)
    x_max = getval(geometry_info["x_gc"] + buffer_right)
    z_min = getval(-buffer_bot)
    z_max = getval(source_z + buffer_top)

    simulation = td.Simulation(
        center=((x_min + x_max) / 2, 0, (z_min + z_max) / 2),
        size=(x_max - x_min, 0, z_max - z_min),
        structures=structures,
        sources=(source,),
        monitors=monitors,
        medium=box,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.periodic(),
            z=td.Boundary.pml(),
        ),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl),
        run_time=run_time,
    )
    return simulation


def get_mode_monitor_power(
    sim_data: td.SimulationData,
    *,
    mode_index: int = 0,
    direction: str = "-",
    monitor_name: str = "mode_monitor",
    power_floor: float = 1e-12,
) -> td.DataArray:
    """Return a clipped power spectrum from a mode monitor.

    Parameters
    ----------
    sim_data : td.SimulationData
        Simulation result returned by ``td.Simulation.run()``.
    mode_index : int, optional
        Index of the guided mode to extract.
    direction : str, optional
        Propagation direction label (``"+"`` or ``"-"``) used by Tidy3D.
    monitor_name : str, optional
        Name of the mode monitor inside the simulation.
    power_floor : float, optional
        Minimum power value (in Watts) used to avoid taking the logarithm of
        zero downstream. Set to ``None`` to disable clipping.

    Returns
    -------
    td.DataArray
        Absolute squared amplitudes corresponding to the requested mode.
    """
    monitor = sim_data[monitor_name]
    amps = monitor.amps.sel(mode_index=mode_index, direction=direction)
    power = np.abs(amps) ** 2
    if power_floor is not None:
        power = power.clip(min=power_floor)
    return power
