from .cylindrical_bfield import CylindricalBfield
import matplotlib.pyplot as plt
from functools import partial
from jax import config

config.update("jax_enable_x64", True)

from jax import jit, jacfwd
import jax.numpy as jnp
import numpy as np

# ## Decorator for jacfwd in cylindrical coordinates
# ## Not needed as pyoculus takes the dBdRphiZ as an input

# def cyljacfwd(f):
#     """Decorator to use the jax.jacfwd differentiation in cylindrical coordinates."""

#     @wraps(f)
#     def dfun(rr, *args, **kwargs):
#         B = f(rr, *args, **kwargs)
#         jac = jnp.array(jacfwd(f)(rr, *args, **kwargs))
#         jac = jac.at[:, 1].multiply(1 / rr[0]**2)
#         return jac
#         # Correction with the Christoffel symbols
#         # correction = jnp.array([[0, -B[1], 0],[B[1], B[0]/rr[0], 0],[0, 0, 0]])/rr[0]
#         # return jac + correction
#         # correction = jnp.array([-B[1], B[0], 0]) / rr[0]
#         # return jac.at[:, 1].add(correction)

#     return dfun


## Equilibrium fields


def equ_squared(rr, R, Z, sf, shear):
    """
    Returns the B field derived from the Psi and F flux functions derived with the fluxes:
    $$
        \psi = (z-Z)^{2} + \left(R - r\right)^{2}
        F = \left(2 sf + 2 shear \left(z^{2} + \left(R - r\right)^{2}\right)\right) \sqrt{R^{2} - z^{2} - \left(R - r\right)^{2}}
    $$
    using the relation B = grad x A, with A_\phi = \psi / r and B_\phi = F / r for an axisymmetric field in cylindrical coordinates.
    """
    temp = jnp.maximum(R**2 - (R - rr[0]) ** 2 - (Z - rr[2]) ** 2, 0.0)
    sgn = (1 + jnp.sign(R**2 - (R - rr[0]) ** 2 - (Z - rr[2]) ** 2)) / 2
    return sgn * jnp.array(
        [
            -(-2 * Z + 2 * rr[2]) / rr[0],
            (2 * sf + 2 * shear * ((R - rr[0]) ** 2 + (Z - rr[2]) ** 2))
            * jnp.sqrt(temp)
            / rr[0] ** 2,
            (-2 * R + 2 * rr[0]) / rr[0],
        ]
    )


def equ_squared_ellipse(rr, R, Z, sf, shear, A, B):
    """
    Returns the B field derived from the Psi and F flux functions derived with the fluxes:
    $$
        \psi = (z-Z)^{2}/B^2 + \left(R - r\right)^{2}/A^2
        F = \left(2 sf + 2 shear \left(\frac{\left(Z - z\right)^{2}}{B^{2}} + \frac{\left(R - r\right)^{2}}{A^{2}}\right)\right) \sqrt{R^{2} - \frac{\left(Z - z\right)^{2}}{B^{2}} - \frac{\left(R - r\right)^{2}}{A^{2}}}
    $$
    using the relation B = grad x A, with A_\phi = \psi / r and B_\phi = F / r for an axisymmetric field in cylindrical coordinates.
    """
    temp = jnp.maximum(
        R**2 - (Z - rr[2]) ** 2 / B**2 - (R - rr[0]) ** 2 / A**2, 0.0
    )
    sgn = (
        1 + jnp.sign(R**2 - (Z - rr[2]) ** 2 / B**2 - (R - rr[0]) ** 2 / A**2)
    ) / 2

    return sgn * jnp.array(
        [
            -(-2 * Z + 2 * rr[2]) / (B**2 * rr[0]),
            (
                2 * sf
                + 2 * shear * ((Z - rr[2]) ** 2 / B**2 + (R - rr[0]) ** 2 / A**2)
            )
            * jnp.sqrt(temp)
            / rr[0] ** 2,
            (-2 * R + 2 * rr[0]) / (A**2 * rr[0]),
        ]
    )


## Perturbation fields


def pert_maxwellboltzmann(rr, R, Z, d, m, n):
    """Maxwell-Boltzmann distributed perturbation field.

    Args:
        rr (array): Position vector in cylindrical coordinates
        R (float): R coordinate of the center of the Maxwell-Boltzmann distribution
        Z (float): Z coordinate of the center of the Maxwell-Boltzmann distribution
        d (float): Standard deviation of the Maxwell-Boltzmann distribution
        m (int): Poloidal mode number
        n (int): Toroidal mode number
    """
    return jnp.array(
        [
            jnp.sqrt(2)
            * (
                d**2
                * (
                    m
                    * ((R - rr[0]) ** 2 + (Z - rr[2]) ** 2)
                    * jnp.imag(
                        (-R + rr[0] - 1j * (Z - rr[2])) ** (m - 1)
                        * jnp.exp(1j * n * rr[1])
                    )
                    + 2
                    * (Z - rr[2])
                    * jnp.real(
                        (-R + rr[0] - 1j * (Z - rr[2])) ** m * jnp.exp(1j * n * rr[1])
                    )
                )
                - (Z - rr[2])
                * ((R - rr[0]) ** 2 + (Z - rr[2]) ** 2)
                * jnp.real(
                    (-R + rr[0] - 1j * (Z - rr[2])) ** m * jnp.exp(1j * n * rr[1])
                )
            )
            * jnp.exp(
                (
                    -(R**2)
                    + 2 * R * rr[0]
                    - Z**2
                    + 2 * Z * rr[2]
                    - rr[0] ** 2
                    - rr[2] ** 2
                )
                / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**5 * rr[0]),
            0,
            jnp.sqrt(2)
            * (
                d**2
                * (
                    m
                    * ((R - rr[0]) ** 2 + (Z - rr[2]) ** 2)
                    * jnp.real(
                        (-R + rr[0] - 1j * (Z - rr[2])) ** (m - 1)
                        * jnp.exp(1j * n * rr[1])
                    )
                    - 2
                    * (R - rr[0])
                    * jnp.real(
                        (-R + rr[0] - 1j * (Z - rr[2])) ** m * jnp.exp(1j * n * rr[1])
                    )
                )
                + (R - rr[0])
                * ((R - rr[0]) ** 2 + (Z - rr[2]) ** 2)
                * jnp.real(
                    (-R + rr[0] - 1j * (Z - rr[2])) ** m * jnp.exp(1j * n * rr[1])
                )
            )
            * jnp.exp(
                (
                    -(R**2)
                    + 2 * R * rr[0]
                    - Z**2
                    + 2 * Z * rr[2]
                    - rr[0] ** 2
                    - rr[2] ** 2
                )
                / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**5 * rr[0]),
        ]
    )


def pert_gaussian(rr, R, Z, d, m, n):
    """Gaussian distributed perturbation field.

    Args:
        rr (array): Position vector in cylindrical coordinates
        R (float): R coordinate of the center of the Gaussian distribution
        Z (float): Z coordinate of the center of the Gaussian distribution
        d (float): Standard deviation of the Gaussian distribution
        m (int): Poloidal mode number
        n (int): Toroidal mode number
    """
    return jnp.array(
        [
            jnp.sqrt(2)
            * (
                d**2
                * m
                * jnp.imag(
                    (-R + rr[0] - 1j * (Z - rr[2])) ** (m - 1) * jnp.exp(1j * n * rr[1])
                )
                - (Z - rr[2])
                * jnp.real(
                    (-R + rr[0] - 1j * (Z - rr[2])) ** m * jnp.exp(1j * n * rr[1])
                )
            )
            * jnp.exp(
                (
                    -(R**2)
                    + 2 * R * rr[0]
                    - Z**2
                    + 2 * Z * rr[2]
                    - rr[0] ** 2
                    - rr[2] ** 2
                )
                / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**3 * rr[0]),
            0,
            jnp.sqrt(2)
            * (
                d**2
                * m
                * jnp.real(
                    (-R + rr[0] - 1j * (Z - rr[2])) ** (m - 1) * jnp.exp(1j * n * rr[1])
                )
                + (R - rr[0])
                * jnp.real(
                    (-R + rr[0] - 1j * (Z - rr[2])) ** m * jnp.exp(1j * n * rr[1])
                )
            )
            * jnp.exp(
                (
                    -(R**2)
                    + 2 * R * rr[0]
                    - Z**2
                    + 2 * Z * rr[2]
                    - rr[0] ** 2
                    - rr[2] ** 2
                )
                / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**3 * rr[0]),
        ]
    )


def ellpe(m):
    """Complete elliptic integral of the second kind"""
    P_coeffs = jnp.array(
        [
            1.53552577301013293365e-4,
            2.50888492163602060990e-3,
            8.68786816565889628429e-3,
            1.07350949056076193403e-2,
            7.77395492516787092951e-3,
            7.58395289413514708519e-3,
            1.15688436810574127319e-2,
            2.18317996015557253103e-2,
            5.68051945617860553470e-2,
            4.43147180560990850618e-1,
            1.00000000000000000299e0,
        ]
    )

    Q_coeffs = jnp.array(
        [
            3.27954898576485872656e-5,
            1.00962792679356715133e-3,
            6.50609489976927491433e-3,
            1.68862163993311317300e-2,
            2.61769742454493659583e-2,
            3.34833904888224918614e-2,
            4.27180926518931511717e-2,
            5.85936634471101055642e-2,
            9.37499997197644278445e-2,
            2.49999999999888314361e-1,
        ]
    )
    x = 1 - m
    # if x <= 0.0 or x > 1.0:
    #     if x == 0.0:
    #         return 1.0
    #     else:
    #         raise ValueError("ellpe: input out of domain")
    return jnp.polyval(P_coeffs, x) - jnp.log(x) * (x * jnp.polyval(Q_coeffs, x))


def ellpk(m):
    """Complete elliptic integral of the first kind"""
    P_coeffs = jnp.array(
        [
            1.37982864606273237150e-4,
            2.28025724005875567385e-3,
            7.97404013220415179367e-3,
            9.85821379021226008714e-3,
            6.87489687449949877925e-3,
            6.18901033637687613229e-3,
            8.79078273952743772254e-3,
            1.49380448916805252718e-2,
            3.08851465246711995998e-2,
            9.65735902811690126535e-2,
            1.38629436111989062502e0,
        ]
    )

    Q_coeffs = jnp.array(
        [
            2.94078955048598507511e-5,
            9.14184723865917226571e-4,
            5.94058303753167793257e-3,
            1.54850516649762399335e-2,
            2.39089602715924892727e-2,
            3.01204715227604046988e-2,
            3.73774314173823228969e-2,
            4.88280347570998239232e-2,
            7.03124996963957469739e-2,
            1.24999999999870820058e-1,
            4.99999999999999999821e-1,
        ]
    )

    x = 1 - m
    return jnp.polyval(P_coeffs, x) - jnp.log(x) * jnp.polyval(Q_coeffs, x)


def field_circularcurrentloop(rr, R, Z):
    """Field generated at rr = rphiz by a circular current loop located at radius R and height Z."""
    # alpha2 = R**2 + rr[0] ** 2 + (rr[2] - Z) ** 2 - 2 * R * rr[0]
    alpha2 = (R - rr[0]) ** 2 + (rr[2] - Z) ** 2
    beta2 = alpha2 + 4 * R * rr[0]
    m = 1 - alpha2 / beta2
    E = ellpe(m)
    K = ellpk(m)

    # singularity = jnp.sqrt((rr[0] - R)**2 + (rr[2] - Z)**2) < tolerance

    return jnp.array(
        [
            (rr[2] - Z)
            / (2 * jnp.pi * rr[0] * alpha2 * jnp.sqrt(beta2))
            * ((R**2 + rr[0] ** 2 + (rr[2] - Z) ** 2) * E - alpha2 * K),
            0,
            1
            / (2 * jnp.pi * alpha2 * jnp.sqrt(beta2))
            * ((R**2 - rr[0] ** 2 - (rr[2] - Z) ** 2) * E + alpha2 * K),
        ]
    )


# class definition


class AnalyticCylindricalBfield(CylindricalBfield):
    """Analytical Bfield problem class that allows adding analytical perturbations to an analytical equilibrium field. The equilibrium field is
    defined by the function `equ_squared(R, sf, shear)` or its ellipsoid equivalent and the perturbations can be choosen from the type dictionary. The possible types are:
        - "maxwell-boltzmann": Maxwell-Boltzmann distributed perturbation
        - "gaussian": Normally distributed perturbation
        - "circular-current-loop": Field generated by a constant current toroidal loop
        - "squared-circle": Field generated by a squared circle (in fact from equ_squared)
        - "squared-ellipse": Field generated by a squared ellipse (in fact from equ_squared_ellipse)

    Attributes:
        sf (float): Safety factor on the magnetic axis
        shear (float): Shear factor
        perturbations_args (list): List of dictionaries with the arguments of each perturbation
        amplitude (list): List of amplitudes of the perturbations. One can set the amplitude of all perturbations
            at once by setting this attribute:
            $ myBfield.amplitudes = [1, 2, 3]
            $ myBfield.amplitudes
            >> [1, 2, 3]
        perturbations (list): List of perturbations functions. To call a certain (for instance the first) perturbation one can do:
            $ myBfield.perturbations[0](rphiz)
            >> value

    Methods:
        set_amplitude(index, value): Set the amplitude of the perturbation at index to value
        set_perturbation(index, perturbation_args): Set the perturbation at index to be defined by perturbation_args
        add_perturbation(perturbation_args): Add a new perturbation defined by perturbation_args
        B_equilibrium(rphiz): Equilibrium field function
        dBdX_equilibrium(rphiz): Gradient of the equilibrium field function
        B_perturbation(rphiz): Perturbation field function
        dBdX_perturbation(rphiz): Gradient of the perturbation field function
    """

    _pert_types_dict = {
        "maxwell-boltzmann": pert_maxwellboltzmann,
        "gaussian": pert_gaussian,
        "squared-circle": equ_squared,
        "squared-ellipse": equ_squared_ellipse,
        "circular-current-loop": field_circularcurrentloop,
    }

    def __init__(self, R, Z, sf, shear, perturbations_args=list(), A=None, B=None):
        """
        Args:
            R (float): Major radius of the magnetic axis of the equilibrium field
            Z (float): Z coordinate of the magnetic axis of the equilibrium field
            sf (float): Safety factor on the magnetic axis
            shear (float): Shear factor
            perturbations_args (list): List of dictionaries with the arguments of each perturbation

            Optional:
            If A and B are provided, the equilibrium field will be defined as an ellipse with major radius A and minor radius B.
            A (float): Major radius of the ellipse in the squared-ellipse equilibrium field
            B (float): Minor radius of the ellipse in the squared-ellipse equilibrium field

        Example:
            $ pert1_dict = {m:2, n:-1, d:1, type: "maxwell-boltzmann", amplitude: 1e-2}
            $ pert2_dict = {m:1, n:0, d:1, type: "gaussian", amplitude: -1e-2}
            $ myBfield = AnalyticCylindricalBfield(R= 3, sf = 1.1, shear=3 pert=[pert1_dict, pert2_dict])
        """

        self.sf = sf
        self.shear = shear

        # Define the equilibrium field and its gradient
        if A is not None and B is not None:
            self.B_equilibrium = partial(
                equ_squared_ellipse, R=R, Z=Z, sf=sf, shear=shear, A=A, B=B
            )
        else:
            self.B_equilibrium = partial(equ_squared, R=R, Z=Z, sf=sf, shear=shear)

        self.dBdX_equilibrium = lambda rr: jnp.array(jacfwd(self.B_equilibrium)(rr))

        # Define the perturbations and the gradient of the resulting field sum
        self._perturbations = [None] * len(perturbations_args)
        for pertdic in perturbations_args:
            if "R" not in pertdic.keys():
                pertdic.update({"R": R})
            if "Z" not in pertdic.keys():
                pertdic.update({"Z": Z})

        self.perturbations_args = perturbations_args
        self._initialize_perturbations(find_axis=False)

        # Call the CylindricalBfield constructor with (R,Z) of the axis
        super().__init__(R, Z, Nfp=1)

    @classmethod
    def without_axis(
        cls,
        R,
        Z,
        sf,
        shear,
        perturbations_args=list(),
        A=None,
        B=None,
        guess=None,
        **kwargs
    ):
        """Create an instance of the class without knowing the magnetic axis. The axis is found by creating a temporary instance and calling the CylindricalBfield.find_axis method.
        The arguments are the same as for the constructor with the addition of a guess position (default : [R,Z]) and the kwargs for the find_axis method.
        """
        if guess is None:
            guess = [R, Z]

        instance = cls(R, Z, sf, shear, perturbations_args, A, B)
        instance.find_axis(guess, **kwargs)
        return instance

    def find_axis(self, guess=None, **kwargs):
        """Tries to re-find the axis. This is useful when the perturbations have been changed.
        If the axis is not found, the previous axis is kept."""
        if guess is None:
            guess = [self._R0, self._Z0]

        defaults = {"Rbegin": guess[0] - 2, "Rend": guess[0] + 2}
        defaults.update(kwargs)

        R0, Z0 = super(AnalyticCylindricalBfield, self).find_axis(
            guess, Nfp=1, **defaults
        )
        self._R0 = R0
        self._Z0 = Z0

    @property
    def amplitudes(self):
        """List of amplitudes of the perturbations."""
        return [pert["amplitude"] for pert in self.perturbations_args]

    @amplitudes.setter
    def amplitudes(self, value):
        """Set the amplitude of all perturbations at once."""
        for i, pertdic in enumerate(self.perturbations_args):
            pertdic["amplitude"] = value[i]
        self._initialize_perturbations()

    def set_amplitude(self, index, value, find_axis=True):
        """Set the amplitude of the perturbation at index to value"""

        self.perturbations_args[index]["amplitude"] = value
        self._initialize_perturbations(index, find_axis=find_axis)

    def set_perturbation(self, index, perturbation_args, find_axis=True):
        """Set the perturbation at index to be defined by perturbation_args"""

        self.perturbations_args[index] = perturbation_args
        self.perturbations_args[index].update({"R": self._R0, "Z": self._Z0})
        self._initialize_perturbations(index, find_axis=find_axis)

    def add_perturbation(self, perturbation_args, find_axis=True):
        """Add a new perturbation defined by perturbation_args"""

        self.perturbations_args.append(perturbation_args)
        self._perturbations.append(None)
        self.perturbations_args[-1].update({"R": self._R0, "Z": self._Z0})
        self._initialize_perturbations(
            len(self.perturbations_args) - 1, find_axis=find_axis
        )

    def remove_perturbation(self, index=None, find_axis=True):
        """Remove the perturbation at index or the last one if no index is given."""
        if index is None:
            index = len(self.perturbations_args) - 1

        self.perturbations_args.pop(index)
        self._perturbations.pop(index)
        self._initialize_perturbations(find_axis=find_axis)

    def _initialize_perturbations(self, index=None, find_axis=True):
        """Initialize the perturbations functions and the gradient. Also updates the total field and its gradient."""

        if index is not None:
            indices = [index]
        else:
            indices = range(len(self.perturbations_args))

        for i in indices:
            tmp_args = self.perturbations_args[i].copy()
            tmp_args.pop("amplitude")
            tmp_args.pop("type")

            self._perturbations[i] = partial(
                self._pert_types_dict[self.perturbations_args[i]["type"]], **tmp_args
            )

        if len(self.perturbations_args) > 0:
            self.B_perturbation = lambda rr: jnp.sum(
                jnp.array(
                    [
                        pertdic["amplitude"] * self._perturbations[i](rr)
                        for i, pertdic in enumerate(self.perturbations_args)
                    ]
                ),
                axis=0,
            )
        else:
            self.B_perturbation = lambda rr: jnp.array([0, 0, 0])

        # gradient of the resulting perturbation
        self.dBdX_perturbation = lambda rr: jnp.array(jacfwd(self.B_perturbation)(rr))

        # Define the total field and its gradient
        self._B = jit(lambda rr: self.B_equilibrium(rr) + self.B_perturbation(rr))
        self._dBdX = jit(
            lambda rr: self.dBdX_equilibrium(rr) + self.dBdX_perturbation(rr)
        )

        # Refind the axis
        if find_axis:
            self.find_axis()

    @property
    def perturbations(self):
        """List of the perturbations functions (jited using jax). To call a certain (for instance the first) perturbation one can do:
        $ myBfield.perturbations[0](rphiz)
        >> value
        """
        if hasattr(self, "_jited_perturbations") and len(
            self._jited_perturbations
        ) == len(self.perturbations_args):
            return self._jited_perturbations
        else:
            self._jited_perturbations = [
                jit(lambda rr: pertdic["amplitude"] * self._perturbations[i](rr))
                for i, pertdic in enumerate(self.perturbations_args)
            ]
            return self._jited_perturbations

    # BfieldProblem methods implementation

    def B(self, rr):
        """Total field function at the point rr. Where B = B_equilibrium + B_perturbation."""
        return np.array(self._B(rr))

    def dBdX(self, rr):
        """Gradient of the total field function at the point rr. Where (dBdX)^i_j = dB^i/dX^j with i the row index and j the column index of the matrix."""
        rr = np.array(rr)
        return [self.B(rr)], np.array(self._dBdX(rr))

    def B_many(self, r, phi, z, input1D=True):
        return np.array([self._B([r[i], phi[i], z[i]]) for i in range(len(r))])

    def dBdX_many(self, r, phi, z, input1D=True):
        return self.B_many(r, phi, z).flatten(), np.array(
            [self._dBdX([r[i], phi[i], z[i]]) for i in range(len(r))]
        )

    def divB(self, rr):
        """Divergence of the total field function at the point rr."""
        b, dbdx = self.dBdX(rr)
        return dbdx[0, 0] + b[0][0] / rr[0] + dbdx[1, 1] / rr[0] ** 2 + dbdx[2, 2]


## Additional plotting functions


# Flux (Psi) functions for the perturbations

def psi_gaussian(rr, R, Z, d, m, n):
    return (
        np.sqrt(2)
        * (-R + rr[0] + 1j * (rr[2] - Z)) ** m
        * np.exp(-((Z - rr[2]) ** 2 + (R - rr[0]) ** 2) / (2 * d**2))
        * np.exp(1j * n * rr[1])
        / (2 * np.sqrt(np.pi) * d)
    )


def psi_maxwellboltzmann(rr, R, Z, d, m, n):
    return (
        np.sqrt(2)
        * ((Z - rr[2]) ** 2 + (R - rr[0]) ** 2)
        * (-R + rr[0] + 1j * (rr[2] - Z)) ** m
        * np.exp(-((Z - rr[2]) ** 2 + (R - rr[0]) ** 2) / (2 * d**2))
        * np.exp(1j * n * rr[1])
        / (np.sqrt(np.pi) * d**3)
    )


def plot_intensities(pyoproblem, ax = None, rw=[2, 5], zw=[-2, 2], nl=[100, 100], RZ_manifold = None, N_levels=50, alpha = 0.5):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
    else:
        fig = ax.get_figure()

    r = np.linspace(rw[0], rw[1], nl[0])
    z = np.linspace(zw[0], zw[1], nl[1])

    R, Z = np.meshgrid(r, z)
    psi = 0
    for pertdic in pyoproblem.perturbations_args:
        tmp_dict = pertdic.copy()
        tmp_dict.pop("amplitude")
        tmp_dict.pop("type")
        if pertdic["type"] == "maxwell-boltzmann":
            tmp_psi = np.array(
                [
                    psi_maxwellboltzmann([r, 0.0, z], **tmp_dict)/r
                    for r, z in zip(R.flatten(), Z.flatten())
                ]
            ).reshape(R.shape)
        elif pertdic["type"] == "gaussian":
            tmp_psi = np.array(
                [
                    psi_gaussian([r, 0.0, z], **tmp_dict)/r
                    for r, z in zip(R.flatten(), Z.flatten())
                ]
            ).reshape(R.shape)
        else:
            tmp_psi = np.zeros(R.shape)

        psi += pertdic["amplitude"] * np.real(tmp_psi)

    if len(pyoproblem.perturbations_args) == 0:
        psi = np.zeros(R.shape)
    mappable = ax.contourf(R, Z, psi, levels=N_levels, alpha=alpha)
    fig.colorbar(mappable)

    if RZ_manifold is not None:
        Bs = np.array([pyoproblem.B_perturbation([R, 0.0, Z]) for R, Z in RZ_manifold])
        # max_norm = np.max(np.linalg.norm(Bs, axis=1))
        # Bs = Bs / max_norm
        
        ax.quiver(RZ_manifold[:,0], RZ_manifold[:,1], Bs[:,0] / np.linalg.norm(Bs, axis=1), Bs[:,2] / np.linalg.norm(Bs, axis=1), alpha=alpha, color='black', linewidth=0.5)

    return fig, ax
