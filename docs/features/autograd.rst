Inverse Design
~~~~~~~~~~~~~~

The adjoint method is an extremely powerful tool for photonics optimization, allowing gradient computation of a cost function in just two simulation steps, regardless of the number of free parameters. While powerful, its implementation can be quite complex. Tidy3D leverages the Autograd package to automatically differentiate a Tidy3D simulation using the adjoint method, making it straightforward to implement adjoint optimization techniques. This section introduces the adjoint method and the inverse design plugin and provides a comprehensive list of application examples.

.. toctree::
    :class: example-notebook-toc
    :maxdepth: 1

    Inverse design plugin <https://www.flexcompute.com/tidy3d/examples/notebooks/InverseDesign>
    Inverse design quickstart <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd0Quickstart>
    Autograd, automatic differentiation, and adjoint optimization: basics <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd1Intro>
    Adjoint analysis of a multi-layer slab <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd2GradientChecking>
    Inverse design optimization of a mode converter <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd3InverseDesign>
    Multi-objective adjoint optimization <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd4MultiObjective>
    Inverse design optimization of a waveguide taper <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd5BoundaryGradients>
    Inverse design optimization of a compact grating coupler <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd6GratingCoupler>
    Inverse design optimization of a metalens <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd7Metalens>
    Adjoint-based shape optimization of a waveguide bend <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd8WaveguideBend>
    Adjoint optimization of a wavelength division multiplexer <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd9WDM>
    Parameterized level set optimization of a y-branch <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd10YBranchLevelSet>
    Diffractive metasurface inverse design with topology optimization <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd13Metasurface>
    Adjoint inverse design of a quantum emitter light extractor <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd12LightExtractor>
    Inverse design optimization of a plasmonic nanoantenna metasurface <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd15Antenna>
    Inverse design optimization of a bilayer grating coupler <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd16BilayerCoupler>
    Adjoint optimization of an integrated bandpass filter <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd17BandPassFilter>
    Topology optimization of a waveguide bend <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd18TopologyBend>
    Inverse design of an apodized grating coupler through shape optimization <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd19ApodizedCoupler>
    Design and shape optimization of a metalens-assisted waveguide taper <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd20MetalensWaveguideTaper>
    Inverse design of a GaP photon extractor for nitrogen-vacancy centers in diamond  <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd21GaPLightExtractor>
    Adjoint optimization of a photonic crystal <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd22PhotonicCrystal>
    Fabrication-aware inverse design (FAID) of a wavelength division multiplexer <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd23FabricationAwareInvdes>
