Inverse Design
~~~~~~~~~~~~~~

The adjoint method is an extremely powerful tool for photonics optimization, allowing gradient computation of a cost function in just two simulation steps, regardless of the number of free parameters. While powerful, its implementation can be quite complex. Tidy3D leverages the Autograd package to automatically differentiate a Tidy3D simulation using the adjoint method, making it straightforward to implement adjoint optimization techniques. This section introduces the adjoint method and the inverse design plugin and provides a comprehensive list of application examples.

.. toctree::
    :class: example-notebook-toc
    :maxdepth: 1

    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/InverseDesign
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd0Quickstart
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd1Intro
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd2GradientChecking
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd3InverseDesign
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd4MultiObjective
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd5BoundaryGradients
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd6GratingCoupler
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd7Metalens
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd8WaveguideBend
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd9WDM
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd10YBranchLevelSet
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd13Metasurface
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd12LightExtractor
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd15Antenna
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd16BilayerCoupler
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd17BandPassFilter
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd18TopologyBend
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd19ApodizedCoupler
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd20MetalensWaveguideTaper
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd21GaPLightExtractor
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd22PhotonicCrystal
    Test <https://www.flexcompute.com/tidy3d/examples/notebooks/Autograd23FabricationAwareInvdes
