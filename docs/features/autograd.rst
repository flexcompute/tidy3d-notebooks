Inverse Design
~~~~~~~~~~~~~~

The adjoint method is an extremely powerful tool for photonics optimization, allowing gradient computation of a cost function in just two simulation steps, regardless of the number of free parameters. While powerful, its implementation can be quite complex. Tidy3D leverages the Autograd package to automatically differentiate a Tidy3D simulation using the adjoint method, making it straightforward to implement adjoint optimization techniques. This section introduces the adjoint method and the inverse design plugin and provides a comprehensive list of application examples.

.. toctree::
    :class: example-notebook-toc
    :maxdepth: 1

    ../../InverseDesign
    ../../Autograd0Quickstart
    ../../Autograd1Intro
    ../../Autograd2GradientChecking
    ../../Autograd3InverseDesign
    ../../Autograd4MultiObjective
    ../../Autograd5BoundaryGradients
    ../../Autograd6GratingCoupler
    ../../Autograd7Metalens
    ../../Autograd8WaveguideBend
    ../../Autograd9WDM
    ../../Autograd13Metasurface
    ../../Autograd15Antenna
    ../../Autograd16BilayerCoupler
    ../../Autograd17BandPassFilter
    ../../Autograd18TopologyBend
    ../../Autograd19ApodizedCoupler
    ../../Autograd20MetalensWaveguideTaper
