"""Compare boundary reflections for a slanted waveguide crossing one boundary.

This is an exploratory companion to ``DivergedFDTDSimulation.ipynb``.  It builds
a small straight waveguide that exits the simulation through one boundary at an
angle, then compares the default extruded PML treatment with an Absorber on that
boundary.

By default this script only constructs and validates the simulations.  Pass
``--run`` to submit the jobs to the web API, or ``--load`` to summarize existing
``.hdf5`` results.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
import tidy3d as td

WAVELENGTH_UM = 1.55
FREQ0 = td.C_0 / WAVELENGTH_UM
FREQS = np.linspace(FREQ0 * 0.96, FREQ0 * 1.04, 21)
FWIDTH = FREQ0 / 20

WG_WIDTH = 0.45
WG_HEIGHT = 0.22
WG_PERMITTIVITY = 3.47**2
CLAD_PERMITTIVITY = 1.44**2

SIM_SIZE = (7.0, 7.0, 2.4)
SIM_CENTER_X_PLUS = (0.0, 0.0, 0.0)
SIM_CENTER_Y_PLUS = (0.0, -1.0, 0.0)

SOURCE_X = -1.8
REFLECTION_X = -2.4
THROUGH_X = 1.2

ABSORBER_LAYERS = 60
PML_LAYERS = 12
PERMITTIVITY_MONITOR_NAME = "eps_boundary"

ExitBoundary = Literal["x+", "y+"]


@dataclass(frozen=True)
class Case:
    """One boundary-condition case in the comparison."""

    name: str
    test_boundary: Union[td.PML, td.Absorber]
    description: str


def centerline_y(x: float, theta_rad: float) -> float:
    """Return the y coordinate of the slanted guide centerline."""

    return x * math.tan(theta_rad)


def simulation_center(exit_boundary: ExitBoundary) -> tuple[float, float, float]:
    """Return a simulation center chosen so the guide exits only the test boundary."""

    if exit_boundary == "x+":
        return SIM_CENTER_X_PLUS
    return SIM_CENTER_Y_PLUS


def waveguide_vertices(theta_rad: float) -> list[tuple[float, float]]:
    """Return a long strip polygon following the angled waveguide centerline."""

    x0 = -20.0
    x1 = 20.0
    direction = np.array([math.cos(theta_rad), math.sin(theta_rad)])
    normal = np.array([-direction[1], direction[0]])
    p0 = np.array([x0, centerline_y(x0, theta_rad)])
    p1 = np.array([x1, centerline_y(x1, theta_rad)])
    half_width = 0.5 * WG_WIDTH
    vertices = [
        p0 - half_width * normal,
        p1 - half_width * normal,
        p1 + half_width * normal,
        p0 + half_width * normal,
    ]
    return [(float(x), float(y)) for x, y in vertices]


def make_mode_spec(theta_rad: float) -> td.ModeSpec:
    """Mode spec aligned to the slanted guide on an x-normal source plane."""

    return td.ModeSpec(num_modes=1, angle_theta=theta_rad, angle_phi=0.0)


def make_waveguide(theta_rad: float) -> td.Structure:
    """Create the single slanted dielectric waveguide."""

    return td.Structure(
        geometry=td.PolySlab(
            vertices=waveguide_vertices(theta_rad),
            slab_bounds=(-WG_HEIGHT / 2, WG_HEIGHT / 2),
            axis=2,
        ),
        medium=td.Medium(permittivity=WG_PERMITTIVITY),
        name="slanted_waveguide",
    )


def boundary_cases(include_no_extrusion: bool) -> list[Case]:
    """Boundary cases to compare on the selected test boundary."""

    cases = [
        Case(
            name="pml_extruded",
            test_boundary=td.PML(num_layers=PML_LAYERS, extrude_structures=True),
            description="default PML with structure extrusion enabled",
        ),
        Case(
            name="absorber",
            test_boundary=td.Absorber(num_layers=ABSORBER_LAYERS),
            description="geometry-preserving absorber",
        ),
    ]
    if include_no_extrusion:
        cases.append(
            Case(
                name="pml_no_extrusion",
                test_boundary=td.PML(num_layers=PML_LAYERS, extrude_structures=False),
                description="PML with structure extrusion disabled",
            )
        )
    return cases


def make_boundary_spec(
    test_boundary: Union[td.PML, td.Absorber], exit_boundary: ExitBoundary
) -> td.BoundarySpec:
    """Use absorbers everywhere except the selected positive test boundary."""

    absorber = td.Absorber(num_layers=ABSORBER_LAYERS)
    if exit_boundary == "x+":
        x_boundary = td.Boundary(plus=test_boundary, minus=absorber)
        y_boundary = td.Boundary(plus=absorber, minus=absorber)
    else:
        x_boundary = td.Boundary(plus=absorber, minus=absorber)
        y_boundary = td.Boundary(plus=test_boundary, minus=absorber)

    return td.BoundarySpec(
        x=x_boundary,
        y=y_boundary,
        z=td.Boundary.pml(),
    )


def permittivity_monitor_geometry(
    theta_deg: float, exit_boundary: ExitBoundary, sim_center: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a zoomed monitor centered near the selected test boundary."""

    theta_rad = math.radians(theta_deg)
    if exit_boundary == "x+":
        x_plus = sim_center[0] + SIM_SIZE[0] / 2
        y_at_boundary = centerline_y(x_plus, theta_rad)
        return (x_plus - 0.65, y_at_boundary, 0.0), (2.4, 2.4, 0.0)

    x_plus = sim_center[0] + SIM_SIZE[0] / 2
    y_plus = sim_center[1] + SIM_SIZE[1] / 2
    x_center = min(boundary_crossing(theta_deg, exit_boundary), x_plus - 0.8)
    return (x_center, y_plus - 0.5, 0.0), (1.9, 1.4, 0.0)


def make_simulation(
    theta_deg: float,
    case: Case,
    *,
    exit_boundary: ExitBoundary,
    min_steps_per_wvl: float,
    run_time: float,
    with_field_monitor: bool,
    with_permittivity_monitor: bool,
) -> td.Simulation:
    """Build one slanted-guide simulation for a boundary case."""

    theta_rad = math.radians(theta_deg)
    mode_spec = make_mode_spec(theta_rad)
    sim_center = simulation_center(exit_boundary)
    source_center = (SOURCE_X, centerline_y(SOURCE_X, theta_rad), 0.0)
    reflection_center = (REFLECTION_X, centerline_y(REFLECTION_X, theta_rad), 0.0)
    through_center = (THROUGH_X, centerline_y(THROUGH_X, theta_rad), 0.0)

    monitors: list[Union[td.ModeMonitor, td.FieldMonitor]] = [
        td.ModeMonitor(
            center=reflection_center,
            size=(0.0, 3.0, 1.6),
            freqs=FREQS,
            mode_spec=mode_spec,
            name="reflection",
        ),
        td.ModeMonitor(
            center=through_center,
            size=(0.0, 3.0, 1.6),
            freqs=FREQS,
            mode_spec=mode_spec,
            name="through",
        ),
    ]
    if with_field_monitor:
        monitors.append(
            td.FieldMonitor(
                center=(0.0, sim_center[1], 0.0),
                size=(SIM_SIZE[0], SIM_SIZE[1], 0.0),
                freqs=[FREQ0],
                name="field_z0",
            )
        )
    if with_permittivity_monitor:
        monitor_center, monitor_size = permittivity_monitor_geometry(
            theta_deg, exit_boundary, sim_center
        )
        monitors.append(
            td.PermittivityMonitor(
                center=monitor_center,
                size=monitor_size,
                freqs=[FREQ0],
                name=PERMITTIVITY_MONITOR_NAME,
            )
        )

    return td.Simulation(
        center=sim_center,
        size=SIM_SIZE,
        medium=td.Medium(permittivity=CLAD_PERMITTIVITY),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl),
        structures=[make_waveguide(theta_rad)],
        sources=[
            td.ModeSource(
                center=source_center,
                size=(0.0, 3.0, 1.6),
                source_time=td.GaussianPulse(freq0=FREQ0, fwidth=FWIDTH),
                direction="+",
                mode_spec=mode_spec,
                mode_index=0,
                name="source",
            )
        ],
        monitors=monitors,
        run_time=run_time,
        boundary_spec=make_boundary_spec(case.test_boundary, exit_boundary),
        shutoff=1e-6,
        symmetry=(0, 0, 0),
    )


def safe_boundary_name(exit_boundary: ExitBoundary) -> str:
    """Stable boundary name for filenames and task names."""

    return exit_boundary.replace("+", "plus")


def task_name(theta_deg: float, case_name: str, exit_boundary: ExitBoundary) -> str:
    """Stable web task name for one run."""

    return f"slanted_wg_{theta_deg:g}deg_{safe_boundary_name(exit_boundary)}_{case_name}"


def result_path(
    path_dir: Path, theta_deg: float, case_name: str, exit_boundary: ExitBoundary
) -> Path:
    """Local result path for one run."""

    safe_angle = f"{theta_deg:g}".replace(".", "p")
    boundary = safe_boundary_name(exit_boundary)
    return path_dir / f"slanted_wg_{safe_angle}deg_{boundary}_{case_name}.hdf5"


def sim_size_summary(sim: td.Simulation) -> str:
    """Short summary of grid and run size."""

    cells = sim.num_cells
    return (
        f"size={tuple(float(value) for value in sim.size)}, "
        f"num_cells={cells}, total_cells={int(np.prod(cells)):,}, "
        f"run_time={sim.run_time:.3e} s"
    )


def mode_power(sim_data: td.SimulationData, monitor_name: str, direction: str) -> np.ndarray:
    """Return TE0 mode power for one mode monitor and direction."""

    amps = sim_data[monitor_name].amps.sel(mode_index=0, direction=direction)
    return np.abs(amps.values.squeeze()) ** 2


def monitor_freqs(sim_data: td.SimulationData, monitor_name: str) -> np.ndarray:
    """Return frequency coordinates for a mode monitor."""

    return np.asarray(sim_data[monitor_name].amps.coords["f"].values)


def wavelength_um(freqs: np.ndarray) -> np.ndarray:
    """Convert frequency coordinates to wavelength in microns."""

    return td.C_0 / freqs


def permittivity_slice(sim_data: td.SimulationData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, eps_zz from the 2D permittivity monitor."""

    eps_data = sim_data[PERMITTIVITY_MONITOR_NAME].eps_zz
    if "f" in eps_data.dims:
        eps_data = eps_data.sel(f=FREQ0, method="nearest")
    if "z" in eps_data.dims:
        eps_data = eps_data.sel(z=0.0, method="nearest")
    eps_data = eps_data.squeeze(drop=True).transpose("x", "y")
    return eps_data.x.values, eps_data.y.values, np.real(eps_data.values)


def summarize_log(sim_data: td.SimulationData) -> list[str]:
    """Extract boundary/divergence lines from a solver log."""

    log = getattr(sim_data, "log", "") or ""
    needles = (
        "extrud",
        "translational",
        "pml",
        "absorber",
        "diverg",
    )
    return [line for line in log.splitlines() if any(needle in line.lower() for needle in needles)]


def summarize_result(path: Path) -> dict[str, Union[float, str, list[str]]]:
    """Load one result and compute reflection metrics."""

    sim_data = td.SimulationData.from_file(str(path))
    reflection = mode_power(sim_data, "reflection", "-")
    through = mode_power(sim_data, "through", "+")
    max_reflection = float(np.max(reflection))
    max_reflection_db = float(10 * np.log10(max(max_reflection, 1e-300)))
    return {
        "path": str(path),
        "max_reflection": max_reflection,
        "max_reflection_db": max_reflection_db,
        "mean_reflection": float(np.mean(reflection)),
        "min_through": float(np.min(through)),
        "mean_through": float(np.mean(through)),
        "log_lines": summarize_log(sim_data),
    }


def print_comparison(
    results: dict[tuple[float, str], dict[str, Union[float, str, list[str]]]],
) -> None:
    """Print a compact comparison table and log hints."""

    print("\nReflection summary")
    print("angle_deg  case              max_R        max_R_dB   mean_R       min_T")
    for (theta_deg, case_name), metrics in sorted(results.items()):
        print(
            f"{theta_deg:9g}  {case_name:16s}  "
            f"{metrics['max_reflection']:.3e}  "
            f"{metrics['max_reflection_db']:8.2f}  "
            f"{metrics['mean_reflection']:.3e}  "
            f"{metrics['min_through']:.3e}"
        )

    print("\nPML extrusion excess over absorber")
    for theta_deg in sorted({theta for theta, _ in results}):
        pml = results.get((theta_deg, "pml_extruded"))
        absorber = results.get((theta_deg, "absorber"))
        if pml is None or absorber is None:
            continue
        pml_reflection = float(pml["max_reflection"])
        absorber_reflection = float(absorber["max_reflection"])
        ratio_db = 10 * np.log10(max(pml_reflection, 1e-300) / max(absorber_reflection, 1e-300))
        excess = pml_reflection - absorber_reflection
        print(f"{theta_deg:g} deg: excess={excess:.3e}, ratio={ratio_db:.2f} dB")

    log_lines = [
        (theta_deg, case_name, line)
        for (theta_deg, case_name), metrics in sorted(results.items())
        for line in metrics["log_lines"]
    ]
    if log_lines:
        print("\nRelevant solver log lines")
        for theta_deg, case_name, line in log_lines:
            print(f"[{theta_deg:g} deg, {case_name}] {line}")


def plot_result_figures(
    *,
    theta_deg: float,
    exit_boundary: ExitBoundary,
    path_dir: Path,
    plot_dir: Path,
) -> None:
    """Create notebook-ready geometry and reflection comparison figures."""

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plot_dir.mkdir(parents=True, exist_ok=True)
    case_names = ("pml_extruded", "absorber")
    sim_data = {
        case_name: td.SimulationData.from_file(
            str(result_path(path_dir, theta_deg, case_name, exit_boundary))
        )
        for case_name in case_names
    }
    sim_center = simulation_center(exit_boundary)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True, sharey=True)
    for ax, case_name, title in zip(axes, case_names, ("PML, extrusion enabled", "Absorber")):
        x, y, eps = permittivity_slice(sim_data[case_name])
        image = ax.pcolormesh(x, y, eps.T, shading="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_aspect("equal")
        if exit_boundary == "x+":
            ax.axvline(sim_center[0] + SIM_SIZE[0] / 2, color="white", lw=1.0, ls="--")
        else:
            ax.axhline(sim_center[1] + SIM_SIZE[1] / 2, color="white", lw=1.0, ls="--")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(image, cax=cax, label=r"$\epsilon_{zz}$")
    axes[0].set_ylabel("y (um)")
    geometry_path = plot_dir / f"diverged-fdtd-slanted-wg-{theta_deg:g}deg-eps.png"
    fig.savefig(geometry_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    for case_name, label in (("pml_extruded", "PML, extrusion enabled"), ("absorber", "Absorber")):
        freqs = monitor_freqs(sim_data[case_name], "reflection")
        reflection = mode_power(sim_data[case_name], "reflection", "-")
        ax.plot(wavelength_um(freqs), 10 * np.log10(np.maximum(reflection, 1e-300)), label=label)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Backward mode power (dB)")
    ax.set_title(f"Reflection from a {theta_deg:g} deg waveguide at {exit_boundary}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    reflection_path = plot_dir / f"diverged-fdtd-slanted-wg-{theta_deg:g}deg-reflection.png"
    fig.savefig(reflection_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {geometry_path}")
    print(f"Wrote {reflection_path}")


def build_cases(
    angles_deg: Iterable[float],
    *,
    exit_boundary: ExitBoundary,
    include_no_extrusion: bool,
    min_steps_per_wvl: float,
    run_time: float,
    with_field_monitor: bool,
    with_permittivity_monitor: bool,
) -> dict[tuple[float, str], td.Simulation]:
    """Build all requested simulations."""

    sims = {}
    for theta_deg in angles_deg:
        for case in boundary_cases(include_no_extrusion):
            sims[(theta_deg, case.name)] = make_simulation(
                theta_deg,
                case,
                exit_boundary=exit_boundary,
                min_steps_per_wvl=min_steps_per_wvl,
                run_time=run_time,
                with_field_monitor=with_field_monitor,
                with_permittivity_monitor=with_permittivity_monitor,
            )
    return sims


def run_web_jobs(
    sims: dict[tuple[float, str], td.Simulation],
    *,
    exit_boundary: ExitBoundary,
    path_dir: Path,
    overwrite: bool,
) -> None:
    """Submit requested web jobs sequentially."""

    import tidy3d.web as web

    path_dir.mkdir(parents=True, exist_ok=True)
    for (theta_deg, case_name), sim in sorted(sims.items()):
        path = result_path(path_dir, theta_deg, case_name, exit_boundary)
        if path.exists() and not overwrite:
            print(f"Skipping existing result: {path}")
            continue

        name = task_name(theta_deg, case_name, exit_boundary)
        print(f"Running {name} -> {path}")
        try:
            web.run(simulation=sim, task_name=name, path=str(path), verbose=True)
        except Exception as exc:
            print(f"{name} failed: {exc}")


def load_available_results(
    sims: dict[tuple[float, str], td.Simulation],
    *,
    exit_boundary: ExitBoundary,
    path_dir: Path,
) -> dict[tuple[float, str], dict[str, Union[float, str, list[str]]]]:
    """Load all available result files for requested simulations."""

    results = {}
    for theta_deg, case_name in sorted(sims):
        path = result_path(path_dir, theta_deg, case_name, exit_boundary)
        if not path.exists():
            print(f"Missing result: {path}")
            continue
        results[(theta_deg, case_name)] = summarize_result(path)
    return results


def boundary_crossing(theta_deg: float, exit_boundary: ExitBoundary) -> float:
    """Return the centerline coordinate where the guide reaches the test boundary."""

    theta_rad = math.radians(theta_deg)
    sim_center = simulation_center(exit_boundary)
    if exit_boundary == "x+":
        x_plus = sim_center[0] + SIM_SIZE[0] / 2
        return centerline_y(x_plus, theta_rad)

    y_plus = sim_center[1] + SIM_SIZE[1] / 2
    return y_plus / math.tan(theta_rad)


def boundary_crossing_summary(theta_deg: float, exit_boundary: ExitBoundary) -> str:
    """Short description of where the guide exits the selected boundary."""

    crossing = boundary_crossing(theta_deg, exit_boundary)
    if exit_boundary == "x+":
        return f"x+ crossing y={crossing:.3f}"
    return f"y+ crossing x={crossing:.3f}"


def print_dry_run(
    sims: dict[tuple[float, str], td.Simulation],
    *,
    exit_boundary: ExitBoundary,
    path_dir: Path,
) -> None:
    """Report what would be run."""

    print("Built slanted-waveguide boundary comparison simulations.")
    for theta_deg, case in sorted(sims):
        sim = sims[(theta_deg, case)]
        print(
            f"{task_name(theta_deg, case, exit_boundary)}: {sim_size_summary(sim)}, "
            f"{boundary_crossing_summary(theta_deg, exit_boundary)}, "
            f"result={result_path(path_dir, theta_deg, case, exit_boundary)}"
        )
    print("\nPass --run to submit these jobs, or --load after results exist.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angles-deg",
        nargs="+",
        type=float,
        default=[45.0],
        help="Slant angles to test, measured from +x toward +y.",
    )
    parser.add_argument(
        "--path-dir",
        type=Path,
        default=Path("data/slanted_waveguide_pml_reflection"),
        help="Directory for downloaded .hdf5 results.",
    )
    parser.add_argument(
        "--exit-boundary",
        choices=("x+", "y+"),
        default="x+",
        help="Positive boundary crossed by the slanted guide.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Submit simulations through tidy3d.web.run.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load existing result files and print reflection metrics.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run jobs even if the local result file exists.",
    )
    parser.add_argument(
        "--include-no-extrusion",
        action="store_true",
        help="Also run the PML(extrude_structures=False) divergence control.",
    )
    parser.add_argument(
        "--with-field-monitor",
        action="store_true",
        help="Add a z=0 field monitor for visual inspection.",
    )
    parser.add_argument(
        "--with-permittivity-monitor",
        action="store_true",
        help="Add a z=0 permittivity monitor near the selected test boundary.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate geometry and reflection figures from downloaded results.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("img"),
        help="Directory for generated figures when --plot is passed.",
    )
    parser.add_argument(
        "--min-steps-per-wvl",
        type=float,
        default=12.0,
        help="Grid resolution passed to GridSpec.auto.",
    )
    parser.add_argument(
        "--run-time",
        type=float,
        default=1.5e-12,
        help="FDTD run time in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for CLI use."""

    args = parse_args()
    sims = build_cases(
        args.angles_deg,
        exit_boundary=args.exit_boundary,
        include_no_extrusion=args.include_no_extrusion,
        min_steps_per_wvl=args.min_steps_per_wvl,
        run_time=args.run_time,
        with_field_monitor=args.with_field_monitor,
        with_permittivity_monitor=args.with_permittivity_monitor,
    )

    if args.run:
        run_web_jobs(
            sims,
            exit_boundary=args.exit_boundary,
            path_dir=args.path_dir,
            overwrite=args.overwrite,
        )

    if args.load:
        results = load_available_results(
            sims,
            exit_boundary=args.exit_boundary,
            path_dir=args.path_dir,
        )
        if results:
            print_comparison(results)
        if args.plot:
            for theta_deg in args.angles_deg:
                plot_result_figures(
                    theta_deg=theta_deg,
                    exit_boundary=args.exit_boundary,
                    path_dir=args.path_dir,
                    plot_dir=args.plot_dir,
                )
        return

    if not args.run:
        print_dry_run(sims, exit_boundary=args.exit_boundary, path_dir=args.path_dir)


if __name__ == "__main__":
    main()
