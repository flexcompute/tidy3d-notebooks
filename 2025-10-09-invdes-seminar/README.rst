Inverse Design Seminar Demos
============================

These notebooks track the inverse-designed dual-layer grating coupler workflow presented during the October 9, 2025 seminar. Start with the simulation setup, follow the optimization and robustness studies, and finish with a calibration example that ties measurements back into the digital twin.

Seminar recording: `YouTube link <https://www.youtube.com/watch?v=OpVBJmomzoo>`_

Repository Layout
-----------------
- ``00_setup_guide.ipynb`` - builds the baseline Tidy3D simulation for a dual-layer grating coupler and visualizes the initial, uniform geometry.
- ``01_bayes.ipynb`` - performs a five-parameter Bayesian optimization to locate a high-performing uniform grating without gradient information.
- ``02_adjoint.ipynb`` - expands to per-tooth parameters and applies adjoint gradients with Adam to apodize the grating and boost peak efficiency.
- ``03_sensitivity.ipynb`` - quantifies fabrication variability through plus or minus 20 nm bias sweeps, Monte Carlo sampling, and adjoint-based sensitivity analysis.
- ``04_adjoint_robust.ipynb`` - optimizes the adjoint design against nominal, over, and under etch corners by penalizing performance variance.
- ``05_robust_comparison.ipynb`` - reruns the Monte Carlo experiment with the robust and nominal designs side by side to measure yield improvements.
- ``06_measurement_calibration.ipynb`` - demonstrates how adjoint gradients can back-fit SiN widths so simulated spectra line up with measured (synthetic) data.

Supporting assets
-----------------
- ``setup.py`` - shared simulation utilities, geometry constraints, and helper routines used across the series.
- ``optim.py`` - lightweight, autograd-friendly Adam implementation plus parameter clipping helpers.
- ``results/`` - JSON snapshots of intermediate designs (Bayesian best guess, adjoint refinements, robust solution) consumed by later notebooks.

Getting Started
---------------
#. Install dependencies (Python 3.10 or newer recommended):

   .. code-block:: bash

      pip install tidy3d bayes_opt autograd pandas matplotlib scipy

   You also need an active Tidy3D account and API access since every notebook submits jobs with ``tidy3d.web.run``.

#. Launch Jupyter and open the notebooks in numerical order; each one assumes the prior results exist in ``results/``.

Suggested Workflow
------------------
- Use ``00_setup_guide.ipynb`` to verify your environment and understand the baseline geometry.
- Iterate through optimization (``01`` to ``04``) to see how global and local methods complement each other.
- Leverage the sensitivity and comparison notebooks (``03`` and ``05``) when you need wafer-level statistics.
- Apply ``06_measurement_calibration.ipynb`` after you gather measured spectra to keep your model synced with hardware.

Enjoy the seminar content, and reach out if you adapt these workflows to your own devices.
