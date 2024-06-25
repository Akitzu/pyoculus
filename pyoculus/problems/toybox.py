from .cylindrical_bfield import CylindricalBfield
import matplotlib.pyplot as plt
from functools import partial, wraps
from jax import config

config.update("jax_enable_x64", True)

from jax import jit, jacfwd
from jax.lax import cond
import jax.numpy as jnp
import numpy as np

## Decorators


def psitob(f):
    """Decorator to calculate the contribution of a Psi function to a B field
    using the relation B = grad x A, with A_\phi e_\phi = \psi / r e_\phi, where
    g(e_\phi, e_\phi) = 1
    """

    @wraps(f)
    def dfun(rr, *args, **kwargs):
        deriv = jacfwd(f)(rr, *args, **kwargs)
        return jnp.array([-1 * deriv[2], 0.0, deriv[0]]) / rr[0]

    return dfun


def rot(f, from_holonomous=True):
    """Decorator that calculate the curl in cylindrical coordinates, useful to get
    the B field from a vector potential A using the relation B = Nabla x A. It takes the
    holonomous component (or non-holonomous, orthonormal basis)
    and returns the holonomous components (cylindrical metric).
    """

    @wraps(f)
    def dfun(rr, *args, **kwargs):
        if not from_holonomous:
            exp = 1
        else:
            exp = 2

        a = lambda rr, *args, **kwargs: jnp.multiply(
            jnp.array([1, rr[0] ** exp, 1]), jnp.array(f(rr, *args, **kwargs))
        )
        deriv = jacfwd(a)(rr, *args, **kwargs)

        return (
            jnp.array(
                [
                    deriv[2][1] - deriv[1][2],
                    deriv[0][2] - deriv[2][0],
                    deriv[1][0] - deriv[0][1],
                ]
            )
            / rr[0]
        )

    return dfun


## Equilibrium

# Equilibrium with q-profile = sf + shear * rho^2 with rho^2 = (rr[0]-R)^2 + (rr[2]-R)^2


def psi_squared(rr: jnp.array, R: float, Z: float) -> jnp.float64:
    """Psi flux function for the squared circle equilibrium field."""
    return (Z - rr[2]) ** 2 + (R - rr[0]) ** 2


def A_r_squared(
    rr: jnp.array, R: float, Z: float, sf: float, shear: float
) -> jnp.float64:
    """A_r vector potential (giving the poloidal flux F) for the squared circle equilibrium field."""

    def a(rr):
        return jnp.real(
            (1 / (4 * rr[0]))
            * (
                (
                    4 * sf
                    + shear
                    * (
                        5 * rr[0] ** 2
                        - 10 * rr[0] * R
                        + 4 * R**2
                        + 2 * (rr[2] - Z) ** 2
                    )
                )
                * jnp.sqrt(-rr[0] ** 2 + 2 * rr[0] * R - (rr[2] - Z) ** 2)
                * (rr[2] - Z)
                - 1j
                * rr[0]
                * (rr[0] - 2 * R)
                * (4 * sf + (3 * rr[0] ** 2 - 6 * rr[0] * R + 4 * R**2) * shear)
                * jnp.log(
                    -1j * rr[2]
                    + jnp.sqrt(-rr[0] * (rr[0] - 2 * R) - (rr[2] - Z) ** 2)
                    + 1j * Z
                )
            )
        )

    return cond(
        R**2 - (R - rr[0]) ** 2 - (Z - rr[2]) ** 2 > 0.0,
        lambda rr: a(rr),
        lambda rr: 0.0,
        jnp.array(rr),
    )


def A_squared(
    rr: jnp.array, R: float, Z: float, sf: float, shear: float
) -> jnp.ndarray:
    """Holonomous component of the vector potential for the squared circle equilibrium field."""
    return jnp.array([0.0, psi_squared(rr, R, Z) / rr[0] ** 2, 0.0]) + jnp.array(
        [A_r_squared(rr, R, Z, sf, shear), 0.0, 0.0]
    )


# # Equilibrium with q-profile = sf + shear * b^2 with b^2 = (rr[0]-R)^2/A^2 + (rr[2]-R)^2/B^2
#
# def psi_ellipse(rr: jnp.array, R: float, Z: float, A: float, B: float) -> jnp.float64:
#     """Psi flux function for the squared ellipse equilibrium field."""
#     return (Z - rr[2]) ** 2 / B**2 + (R - rr[0]) ** 2 / A**2
#
# def A_z_ellipse(rr, R, Z, sf, shear, A, B):
#     return -0.25 * (
#         (
#             (
#                 -5 * B**2 * (rr[0] - R) ** 2 * shear
#                 + A**2
#                 * (B**2 * (-4 * sf + R**2 * shear) - 2 * shear * (rr[2] - Z) ** 2)
#             )
#             * jnp.sqrt(
#                 ((rr[0] + (-1 + A) * R) * (-rr[0] + R + A * R)) / A**2 - (rr[2] - Z) ** 2 / B**2
#             )
#             * (rr[2] - Z)
#         )
#         / (A**2 * B**2)
#         + (
#             B
#             * (rr[0] + (-1 + A) * R)
#             * (rr[0] - (1 + A) * R)
#             * (3 * (rr[0] - R) ** 2 * shear + A**2 * (4 * sf + R**2 * shear))
#             * (jnp.pi/2 - jnp.arctan(
#                 (
#                     B
#                     * jnp.sqrt(
#                         ((rr[0] + (-1 + A) * R) * (-rr[0] + R + A * R)) / A**2
#                         - (rr[2] - Z) ** 2 / B**2
#                     )
#                 )
#                 / (rr[2] - Z)
#             ))
#         )
#         / A**4
#     ) / rr[0]
#
# def F_ellipse(rr, R, Z, sf, shear, A, B):
#     """F flux function for the squared ellipse equilibrium field."""
#     temp = jnp.maximum(
#         R**2 - (Z - rr[2]) ** 2 / B**2 - (R - rr[0]) ** 2 / A**2, 0.0
#     )
#     return (
#         2 * sf + 2 * shear * ((Z - rr[2]) ** 2 / B**2 + (R - rr[0]) ** 2 / A**2)
#     ) * jnp.sqrt(temp)

## Perturbations

# Maxwell-Boltzmann distributed perturbation


def psi_maxwellboltzmann(
    rr: jnp.array,
    R: float,
    Z: float,
    d: float,
    m: int,
    n: int,
    phase_poloidal: float = 0.0,
    phase_toroidal: float = 0.0,
    A: float = 1.0,
    B: float = 1.0,
) -> jnp.float64:
    """Maxwell-Boltzmann distributed Psi flux function.

    Args:
        rr (array): Position vector in cylindrical coordinates
        R (float): R coordinate of the center of the Maxwell-Boltzmann distribution
        Z (float): Z coordinate of the center of the Maxwell-Boltzmann distribution
        d (float): Standard deviation of the Maxwell-Boltzmann distribution
        m (int): Poloidal mode number
        n (int): Toroidal mode number
        phase_poloidal (float): Poloidal phase of the perturbation
        phase_toroidal (float): Toroidal phase of the perturbation
        A (float): Scaling factor for the R coordinate
        B (float): Scaling factor for the Z coordinate
    """

    rho2 = (R - rr[0]) ** 2 / A**2 + (Z - rr[2]) ** 2 / B**2

    def psi_mb(rr):
        return (
            jnp.sqrt(2)
            / (jnp.sqrt(jnp.pi) * d**3)
            * rho2
            * jnp.exp(-rho2 / (2 * d**2))
            * jnp.cos(
                jnp.arctan2((rr[2] - Z) / B, (rr[0] - R) / A) * m + phase_poloidal
            )
            * jnp.cos(rr[1] * n + phase_toroidal)
        )

    return cond(
        rho2**2 > jnp.finfo(jnp.float32).tiny,
        lambda rr: psi_mb(rr),
        lambda rr: 0.0,
        jnp.array(rr),
    )


def A_maxwellboltzmann(
    rr: jnp.array,
    R: float,
    Z: float,
    d: float,
    m: int,
    n: int,
    phase_poloidal: float = 0.0,
    phase_toroidal: float = 0.0,
    A: float = 1.0,
    B: float = 1.0,
) -> jnp.ndarray:
    """Holonomous component of the vector potential for the Maxwell-Boltzmann distributed perturbation."""
    return jnp.array(
        [
            0.0,
            psi_maxwellboltzmann(
                rr, R, Z, d, m, n, phase_poloidal, phase_toroidal, A, B
            )
            / rr[0] ** 2,
            0.0,
        ]
    )


# Gaussian distributed perturbation


def psi_gaussian(
    rr: jnp.array,
    R: float,
    Z: float,
    mu: float,
    sigma: float,
    m: int,
    n: int,
    phase_poloidal: float = 0.0,
    phase_toroidal: float = 0.0,
    A: float = 1.0,
    B: float = 1.0,
) -> jnp.float64:
    """Gaussian distributed Psi flux function.

    Args:
        rr (array): Position vector in cylindrical coordinates
        R (float): R coordinate of the center of the Gaussian distribution
        Z (float): Z coordinate of the center of the Gaussian distribution
        mu (float): Mean of the Gaussian distribution
        sigma (float): Standard deviation of the Gaussian distribution
        m (int): Poloidal mode number
        n (int): Toroidal mode number
        phase_poloidal (float): Poloidal phase of the perturbation
        phase_toroidal (float): Toroidal phase of the perturbation
        A (float): Scaling factor for the R coordinate
        B (float): Scaling factor for the Z coordinate
    """
    rho2 = (R - rr[0]) ** 2 / A**2 + (Z - rr[2]) ** 2 / B**2

    def psi_g(rr):
        return (
            jnp.sqrt(2)
            / (2 * jnp.sqrt(np.pi) * sigma)
            * jnp.exp(-((jnp.sqrt(rho2) - mu) ** 2) / (2 * sigma**2))
            * jnp.cos(
                jnp.arctan2((rr[2] - Z) / B, (rr[0] - R) / A) * m + phase_poloidal
            )
            * jnp.cos(rr[1] * n + phase_toroidal)
        )

    return cond(
        rho2**2 > jnp.finfo(jnp.float32).tiny,
        lambda rr: psi_g(rr),
        lambda rr: 0.0,
        jnp.array(rr),
    )


def A_gaussian(
    rr: jnp.array,
    R: float,
    Z: float,
    mu: float,
    sigma: float,
    m: int,
    n: int,
    phase_poloidal: float = 0.0,
    phase_toroidal: float = 0.0,
    A: float = 1.0,
    B: float = 1.0,
) -> jnp.ndarray:
    """Holonomous component of the vector potential for the Gaussian distributed perturbation."""
    return jnp.array(
        [
            0.0,
            psi_gaussian(
                rr, R, Z, mu, sigma, m, n, phase_poloidal, phase_toroidal, A, B
            )
            / rr[0] ** 2,
            0.0,
        ]
    )


## Circular current loop perturbation


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


def psi_circularcurrentloop(rr: jnp.array, R: float, Z: float) -> jnp.float64:
    """Vector potential (\phi coordinates) generated at rr = rphiz by a circular current loop located at radius R and height Z."""
    alpha2 = (R - rr[0]) ** 2 + (rr[2] - Z) ** 2
    beta2 = alpha2 + 4 * R * rr[0]
    k2 = 1 - alpha2 / beta2
    E = ellpe(k2)
    K = ellpk(k2)

    # Note psi = R*A^\phi
    return rr[0] * R * ((2 - k2) * K - 2 * E) / (jnp.sqrt(beta2) * k2 * jnp.pi)


def A_circularcurrentloop(rr: jnp.array, R: float, Z: float) -> jnp.ndarray:
    """Holonomous component of the vector potential for the circular current loop perturbation."""
    return jnp.array([0.0, psi_circularcurrentloop(rr, R, Z) / rr[0] ** 2, 0.0])


# class definition


class AnalyticCylindricalBfield(CylindricalBfield):
    """Analytical Bfield problem class that allows adding analytical perturbations to an analytical equilibrium field. The equilibrium field is
    derived from the vector potential `A_squared(R, Z, sf, shear)` (circular quadratic q-profile) and the perturbations can be choosen from the
    type dictionary. The possible types are:
        - "maxwell-boltzmann": Maxwell-Boltzmann distributed perturbation
        - "gaussian": Normally distributed perturbation
        - "circular-current-loop": Field generated by a constant current toroidal loop
        - "squared-circle": Field generated by a squared circle (in fact from equ_squared)

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
        remove_perturbation(index): Remove the perturbation at index, (default: -1, the last one)
        B_equilibrium(rphiz): Equilibrium field function
        dBdX_equilibrium(rphiz): Gradient of the equilibrium field function
        B_perturbation(rphiz): Perturbation field function
        dBdX_perturbation(rphiz): Gradient of the perturbation field function
    """

    _field_types_dict = {
        "squared-circle": rot(A_squared),
        "maxwell-boltzmann": psitob(psi_maxwellboltzmann),
        "gaussian": psitob(psi_gaussian),
        "circular-current-loop": psitob(psi_circularcurrentloop),
    }

    _pot_types_dict = {
        "squared-circle": A_squared,
        "maxwell-boltzmann": A_maxwellboltzmann,
        "gaussian": A_gaussian,
        "circular-current-loop": A_circularcurrentloop,
    }

    def __init__(self, R, Z, sf, shear, perturbations_args=list(), **kwargs):
        """
        Args:
            R (float): Major radius of the magnetic axis of the equilibrium field
            Z (float): Z coordinate of the magnetic axis of the equilibrium field
            sf (float): Safety factor on the magnetic axis
            shear (float): Shear factor
            perturbations_args (list): List of dictionaries with the arguments of each perturbation

        Example:
            $ pert1_dict = {m:2, n:-1, d:1, type: "maxwell-boltzmann", amplitude: 1e-2}
            $ pert2_dict = {m:1, n:0, mu:0, sigma:1, type: "gaussian", amplitude: -1e-2}
            $ myBfield = AnalyticCylindricalBfield(R= 3, sf = 1.1, shear=3 pert=[pert1_dict, pert2_dict])
        """

        self.sf = sf
        self.shear = shear

        # Define the equilibrium vector potential, the field and its gradient
        self.B_equilibrium = partial(
            self._field_types_dict["squared-circle"], R=R, Z=Z, sf=sf, shear=shear
        )
        self.A_equilibrium = partial(
            self._pot_types_dict["squared-circle"], R=R, Z=Z, sf=sf, shear=shear
        )
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
        super().__init__(phi0=0., R0=R, Z0=Z, Nfp=1, **kwargs)

    @classmethod
    def without_axis(
        cls, R, Z, sf, shear, perturbations_args=list(), guess=None, finderargs=dict(), **kwargs
    ):
        """Create an instance of the class without knowing the magnetic axis. The axis is found by creating a temporary instance and calling the CylindricalBfield.find_axis method.
        The arguments are the same as for the constructor with the addition of a guess position (default : [R,Z]) and the kwargs for the find_axis method.
        """
        if guess is None:
            guess = [R, Z]

        instance = cls(R, Z, sf, shear, perturbations_args, **kwargs)
        instance.find_axis(guess, **finderargs)
        return instance

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
        self.perturbations_args[index].update({"R": self.R0, "Z": self.Z0})
        self._initialize_perturbations(index, find_axis=find_axis)

    def add_perturbation(self, perturbation_args, find_axis=True):
        """Add a new perturbation defined by perturbation_args"""

        self.perturbations_args.append(perturbation_args)
        self._perturbations.append(None)
        self.perturbations_args[-1].update({"R": self.R0, "Z": self.Z0})
        self._initialize_perturbations(
            len(self.perturbations_args) - 1, find_axis=find_axis
        )

    def remove_perturbation(self, index=-1, find_axis=True):
        """Remove the perturbation at index or the last one if no index is given."""

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
                self._field_types_dict[self.perturbations_args[i]["type"]], **tmp_args
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

            self.A_pertrubation = lambda rr: jnp.sum(
                jnp.array(
                    [
                        pertdic["amplitude"]
                        * self._pot_types_dict[self.perturbations_args[i]["type"]](
                            rr,
                            **{
                                k: v
                                for k, v in pertdic.items()
                                if k not in ["type", "amplitude"]
                            }
                        )
                        for i, pertdic in enumerate(self.perturbations_args)
                    ]
                ),
                axis=0,
            )
        else:
            self.B_perturbation = lambda rr: jnp.array([0, 0, 0])
            self.A_pertrubation = lambda rr: jnp.array([0, 0, 0])

        # gradient of the resulting perturbation
        self.dBdX_perturbation = lambda rr: jnp.array(jacfwd(self.B_perturbation)(rr))

        # Define the total field and its gradient
        self._A = jit(lambda rr: self.A_equilibrium(rr) + self.A_pertrubation(rr))
        self._B = jit(lambda rr: self.B_equilibrium(rr) + self.B_perturbation(rr))
        self._dBdX = jit(
            lambda rr: self.dBdX_equilibrium(rr) + self.dBdX_perturbation(rr)
        )

        # Find the axis
        if find_axis:
            self.find_axis(guess=[self.R0, self.Z0])

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

    def B(self, coords, *args):
        """Total field function at the point rr. Where B = B_equilibrium + B_perturbation."""
        return np.array(self._B(coords))

    def dBdX(self, coords, *args):
        """Gradient of the total field function at the point coords. Where (dBdX)^i_j = dB^i/dX^j with i the row index and j the column index of the matrix."""
        rr = np.array(coords)
        return [self.B(rr)], np.array(self._dBdX(coords))

    def A(self, coords, *args):
        """Total vector potential function at the point rr."""
        return np.array(self._A(coords))

    # def B_many(self, r, phi, z, input1D=True):
    #     return np.array([self._B([r[i], phi[i], z[i]]) for i in range(len(r))])
        
    # def dBdX_many(self, r, phi, z, input1D=True):
    #     return self.B_many(r, phi, z).flatten(), np.array(
    #         [self._dBdX([r[i], phi[i], z[i]]) for i in range(len(r))]
    #     )

    def divB(self, rr):
        """Divergence of the total field function at the point rr."""
        b, dbdx = self.dBdX(rr)
        return dbdx[0, 0] + b[0][0] / rr[0] + dbdx[1, 1] / rr[0] ** 2 + dbdx[2, 2]

    ## Additional plotting functions

    def plot_intensities(
        self,
        rw,
        zw,
        nl=[100, 100],
        RZ_manifold=None,
        N_levels=50,
        alpha=0.5,
        ax=None,
    ):
        """Plot the perturbation psi flux function and the perturbation B field in the RZ plane at the provided points.
        The perturbation psi is the sum of the perturbations defined in the perturbations_args attribute.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 5))
        else:
            fig = ax.get_figure()

        r = np.linspace(rw[0], rw[1], nl[0])
        z = np.linspace(zw[0], zw[1], nl[1])

        psi_mb = jit(psi_maxwellboltzmann)
        psi_g = jit(psi_gaussian)

        R, Z = np.meshgrid(r, z)
        psi = 0
        for pertdic in self.perturbations_args:
            tmp_dict = pertdic.copy()
            tmp_dict.pop("amplitude")
            tmp_dict.pop("type")
            if pertdic["type"] == "maxwell-boltzmann":
                tmp_psi = np.array(
                    [
                        psi_mb([r, 0.0, z], **tmp_dict) / r
                        for r, z in zip(R.flatten(), Z.flatten())
                    ]
                ).reshape(R.shape)
            elif pertdic["type"] == "gaussian":
                tmp_psi = np.array(
                    [
                        psi_g([r, 0.0, z], **tmp_dict) / r
                        for r, z in zip(R.flatten(), Z.flatten())
                    ]
                ).reshape(R.shape)
            else:
                tmp_psi = np.zeros(R.shape)

            psi += pertdic["amplitude"] * np.real(tmp_psi)

        if len(self.perturbations_args) == 0:
            psi = np.zeros(R.shape)
        mappable = ax.contourf(R, Z, psi, levels=N_levels, alpha=alpha)
        fig.colorbar(mappable)

        if RZ_manifold is not None:
            bfuncts = self.perturbations
            Bs = np.zeros(shape=(RZ_manifold.shape[0], 3))
            for i, pertdic in enumerate(self.perturbations_args):
                if pertdic["type"] != "circular-current-loop":
                    Bs += np.array([bfuncts[i]([R, 0.0, Z]) for R, Z in RZ_manifold])

            norms = np.linalg.norm(Bs, axis=1)

            ax.quiver(
                RZ_manifold[:, 0],
                RZ_manifold[:, 1],
                Bs[:, 0] / np.linalg.norm(Bs, axis=1),
                Bs[:, 2] / np.linalg.norm(Bs, axis=1),
                norms,
                alpha=alpha,
                linewidth=0.5,
            )

        return fig, ax
