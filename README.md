# pyoculus

![version](https://img.shields.io/badge/version-1.0.0-blue)

The eye into chaos: a comprehensive diagnostic package for non-integrable, toroidal magnetic fields (and more general 1 1/2-D or 2D Hamiltonian system), analytic and more general maps that rely on integration. Started as a python version of the original package [Oculus](https://github.com/SRHudson/Oculus/). Oculus is the Latin word for *eye*.

## Package Installation

> [!TIP]
> We recommend the use of a virtual environment.

You can obtain the package from PYPI by:

```bash
pip3 install pyoculus
```

or for a specific user:

```bash
pip3 install --user pyoculus
```

Alternatively, you can clone this repository. By default the installation is minimal and one might need to get some additional dependencies. They may be installed directly using .[JAX] or .[SIMSOPT] or .[JAX,SIMSOPT].

### SPEC

In this case, additional steps may be needed if you want to use [SPEC] to compile the FORTRAN interfaces for SPEC magnetic field and PJH.
Some additional steps are needed to run pyoculus on outputs generated by the Stepped Pressure Equilibrium Code or [SPEC](https://princetonuniversity.github.io/SPEC/).

### SIMSOPT

On Windows, you may need to do the following changes to the package before being able to install it.

## Documentation

The documentation of pyoculus is managed by [Sphinx](https://www.sphinx-doc.org/) and can be found on [Github-Pages](https://zhisong.github.io/pyoculus/). This documentation will be updated regularly, however for latest features, you may need to refer to the API Reference. You may also generate the documentation from source by following Sphinx's instructions.
