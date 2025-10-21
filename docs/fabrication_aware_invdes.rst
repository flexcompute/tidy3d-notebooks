**********************************
Fabrication-aware inverse design
**********************************

The October 9, 2025 seminar walks through a complete dual-layer grating coupler workflow: start from a uniform baseline, pull a strong seed design with Bayesian optimization, switch to adjoint gradients for per-tooth control, study fabrication sensitivities, and close the loop with measurement-driven calibration. Everything runs inside Tidy3D, so you can rerun the exact same jobs or adapt the utilities to your own device stack.

Seminar recording: `YouTube link <https://www.youtube.com/watch?v=OpVBJmomzoo>`_

Notebook lineup
================
* ``00_setup_guide.ipynb`` - builds the nominal SiN stack, launches the reference simulation, and visualizes the initial geometry so the later notebooks can reuse the cached job ID.
* ``01_bayes.ipynb`` - uses a five-parameter Bayesian search to quickly find a good uniform grating. This provides a practical baseline before investing in gradients.
* ``02_adjoint.ipynb`` - expands to per-tooth parameters and applies Adam with adjoint sensitivities to apodize the grating and boost efficiency.
* ``03_sensitivity.ipynb`` - sweeps :math:`\pm 20` nm etch bias, runs Monte Carlo samples, and logs adjoint-derived sensitivity units (:math:`\Delta` objective / :math:`\Delta` parameter) so readers understand what the gradients mean physically.
* ``04_adjoint_robust.ipynb`` - penalizes variance across nominal/over/under corners, illustrating a fabrication-aware adjoint loop that matches what we demoed live.
* ``05_robust_comparison.ipynb`` - reruns the Monte Carlo campaign for both nominal and robust devices to quantify yield improvements.
* ``06_measurement_calibration.ipynb`` - demonstrates gradient-based calibration of tooth widths against (synthetic) spectra, showing that we can also optimize over fabrication corners using the same adjoint machinery.

How to run the series
=====================
1. Install ``tidy3d`` and configure your API key; every notebook submits jobs with ``tidy3d.web.run``.
2. Execute the notebooks in order; each step writes results into ``results/`` and later notebooks assume those JSON files exist.

Supporting assets
=================
* ``setup.py`` - shared simulation builders, fabrication constraints, and helper functions.
* ``optim.py`` - a lightweight, autograd-friendly Adam implementation with parameter clipping.
* ``results/`` - JSON checkpoints (Bayes best point, adjoint refinements, robust design) consumed by subsequent notebooks.

Walkthrough notebooks
=====================

.. toctree::
   :maxdepth: 1

   notebooks/2025-10-09-invdes-seminar/00_setup_guide
   notebooks/2025-10-09-invdes-seminar/01_bayes
   notebooks/2025-10-09-invdes-seminar/02_adjoint
   notebooks/2025-10-09-invdes-seminar/03_sensitivity
   notebooks/2025-10-09-invdes-seminar/04_adjoint_robust
   notebooks/2025-10-09-invdes-seminar/05_robust_comparison
   notebooks/2025-10-09-invdes-seminar/06_measurement_calibration
