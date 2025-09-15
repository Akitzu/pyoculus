# pyoculus
A Python version of Oculus - The eye into the chaos: a comprehensive magnetic field diagnostic package for non-integrable, toroidal magnetic fields (and more general 1 1/2-D or 2D Hamiltonian system). Oculus is the Latin word for 'eye'.

## Installation

You can obtain the package from PYPI by

```
python -m pip install pyoculus
```

or you can clone the repo and install with: 

```
python -m pip install .
```

If you want an editable installation, you will need the following command: 

```
python -m pip install -v --no-build-isolation --editable .
```
and you will need all build dependencies to also installed in your python environment. This is likely to be: 
```
pip install meson-python ninja
```


## Usage

To use the package, simply import it in Python:
```python
import pyoculus
```

Examples can be found in the `examples` subfolder.

## Documentation

The documentation of pyoculus is managed by [Doxygen](https://www.doxygen.nl/index.html).

You can find the documentation on Github Page: https://zhisong.github.io/pyoculus/

This documentation will be updated regularly but may not be most up-to-date. To generate the documentation from source, please run

```
make doxygen
```

The documentation will appear in the subfolder `doc/html`. Please open `doc/html/index.html` in your browser.


## Developers
 - Zhisong Qu 
 - Christopher Berg Smiet (chris.smiet@gmail.com)

## Link to the original Oculus package:

Github: https://github.com/SRHudson/Oculus

Documentation: https://w3.pppl.gov/~shudson/Oculus/oculus.pdf

