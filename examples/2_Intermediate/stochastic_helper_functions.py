from dataclasses import dataclass

import numpy as np
from sympy import Symbol, lambdify, exp

from simsopt._core.json import GSONable
from simsopt._core.util import RealArray

import simsoptpp as sopp
from simsopt.geo.curve import Curve
from simsopt.geo import (create_equally_spaced_curves)
from simsopt.field.coil import Current, CurrentBase


__all__ = ['GaussianSampler', 'PerturbationSample', 'CurvePerturbed_jsonfix', 'curve_fourier_fit',
           'CurrentPerturbed', 'hessian']


@dataclass
class GaussianSampler(GSONable):
    r"""
    Generate a periodic gaussian process on the interval [0, 1] on a given list of quadrature points.
    The process has standard deviation ``sigma`` a correlation length scale ``length_scale``.
    Large values of ``length_scale`` correspond to smooth processes, small values result in highly oscillatory
    functions.
    Also has the ability to sample the derivatives of the function.

    We consider the kernel

    .. math::

        \kappa(d) = \sigma^2 \exp(-d^2/l^2)

    and then consider a Gaussian process with covariance

    .. math::

        Cov(X(s), X(t)) = \sum_{i=-\infty}^\infty \sigma^2 \exp(-(s-t+i)^2/l^2)

    the sum is used to make the kernel periodic and in practice the infinite sum is truncated.

    Args:
        points: the quadrature points along which the perturbation should be computed.
        sigma: standard deviation of the underlying gaussian process
               (measure for the magnitude of the perturbation).
        length_scale: length scale of the underlying gaussian process
                      (measure for the smoothness of the perturbation).
        n_derivs: number of derivatives of the gaussian process to sample.
    """

    points: RealArray
    sigma: float
    length_scale: float
    n_derivs: int = 1

    def __post_init__(self):
        xs = self.points
        n = len(xs)
        cov_mat = np.zeros((n*(self.n_derivs+1), n*(self.n_derivs+1)))

        def kernel(x, y):
            return sum((self.sigma**2)*exp(-(x-y+i)**2/(self.length_scale**2)) for i in range(-5, 6))

        XX, YY = np.meshgrid(xs, xs, indexing='ij')
        x = Symbol("x")
        y = Symbol("y")
        f = kernel(x, y)
        for ii in range(self.n_derivs+1):
            for jj in range(self.n_derivs+1):
                if ii + jj == 0:
                    lam = lambdify((x, y), f, "numpy")
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX, YY)

        # we need to compute the sqrt of the covariance matrix. we used to do this using scipy.linalg.sqrtm,
        # but it seems sometime between scipy 1.11.1 and 1.11.2 that function broke/changed behaviour.
        # So we use a LDLT decomposition instead. See als https://github.com/hiddenSymmetries/simsopt/issues/349
        # from scipy.linalg import sqrtm, ldl
        # self.L = np.real(sqrtm(cov_mat))
        from scipy.linalg import ldl
        lu, d, _ = ldl(cov_mat)
        self.L = lu @ np.sqrt(np.maximum(d, 0))

    def draw_sample(self, randomgen=None):
        """
        Returns a list of ``n_derivs+1`` arrays of size ``(len(points), 3)``, containing the
        perturbation and the derivatives.
        """
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random.Generator(np.random.PCG64DXSM())
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]


class PerturbationSample(GSONable):
    """
    This class represents a single sample of a perturbation.  The point of
    having a dedicated class for this is so that we can apply the same
    perturbation to multipe curves (e.g. in the case of multifilament
    approximations to finite build coils).
    The main way to interact with this class is via the overloaded ``__getitem__``
    (i.e. ``[ ]`` indexing).
    For example::

        sample = PerturbationSample(...)
        g = sample[0] # get the values of the perturbation
        gd = sample[1] # get the first derivative of the perturbation
    """

    def __init__(self, sampler, randomgen=None, sample=None):
        self.sampler = sampler
        self.randomgen = randomgen   # If not None, most likely fail with serialization
        # Store generator state for serialization
        if randomgen is not None:
            self._generator_state = randomgen.bit_generator.state
            self._generator_type = type(randomgen.bit_generator).__name__
        else:
            self._generator_state = None
            self._generator_type = None
        if sample:
            self._sample = sample
        else:
            self.resample()

    def resample(self):
        # Reconstruct generator if needed for resampling
        if self.randomgen is None and self._generator_state is not None:
            if self._generator_type == 'PCG64DXSM':
                from numpy.random import PCG64DXSM, Generator
                bit_gen = PCG64DXSM()
                bit_gen.state = self._generator_state
                self.randomgen = Generator(bit_gen)
            # Add other generator types as needed
        self._sample = self.sampler.draw_sample(self.randomgen)

    def __getitem__(self, deriv):
        """
        Get the perturbation (if ``deriv=0``) or its ``deriv``-th derivative.
        """
        assert isinstance(deriv, int)
        if deriv >= len(self._sample):
            raise ValueError(f"""
The sample only has {len(self._sample)-1} derivatives.
Adjust the `n_derivs` parameter of the sampler to access higher derivatives.
""")
        return self._sample[deriv]

    def as_dict(self, serial_objs_dict):
        """Custom serialization to handle random generator state."""
        d = super().as_dict(serial_objs_dict)
        # Store generator state instead of generator object
        if hasattr(self, '_generator_state'):
            d['_generator_state'] = self._generator_state
            d['_generator_type'] = self._generator_type
        # Don't serialize the randomgen object itself
        if 'randomgen' in d:
            del d['randomgen']
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        """Custom deserialization to reconstruct random generator state."""
        # Extract generator state info before calling parent constructor
        generator_state = d.get('_generator_state')
        generator_type = d.get('_generator_type')
        
        # Create a clean dict without generator attributes for parent constructor
        clean_d = {k: v for k, v in d.items() if not k.startswith('_generator_')}
        
        obj = super().from_dict(clean_d, serial_objs_dict, recon_objs)
        
        # Set generator state attributes after construction
        if generator_state is not None:
            obj._generator_state = generator_state
            obj._generator_type = generator_type
            # Reconstruct generator if state was stored
            if generator_type == 'PCG64DXSM':
                from numpy.random import PCG64DXSM, Generator
                bit_gen = PCG64DXSM()
                bit_gen.state = generator_state
                obj.randomgen = Generator(bit_gen)
        
        return obj


class CurvePerturbed_jsonfix(sopp.Curve, Curve):

    """A perturbed curve."""

    def __init__(self, curve, sample):
        r"""
        Perturb a underlying :mod:`simsopt.geo.curve.Curve` object by drawing a perturbation from a
        :obj:`GaussianSampler`.

        Comment:
        Doing anything involving randomness in a reproducible way requires care.
        Even more so, when doing things in parallel.
        Let's say we have a list of :mod:`simsopt.geo.curve.Curve` objects ``curves`` that represent a stellarator,
        and now we want to consider ``N`` perturbed stellarators. Let's also say we have multiple MPI ranks.
        To avoid the same thing happening on the different MPI ranks, we could pick a different seed on each rank.
        However, then we get different results depending on the number of MPI ranks that we run on. Not ideal.
        Instead, we should pick a new seed for each :math:`1\le i\le N`. e.g.

        .. code-block:: python

            from np.random import SeedSequence, PCG64DXSM, Generator
            import numpy as np
            curves = ...
            sigma = 0.01
            length_scale = 0.2
            sampler = GaussianSampler(curves[0].quadpoints, sigma, length_scale, n_derivs=1)
            globalseed = 1
            N = 10 # number of perturbed stellarators
            seeds = SeedSequence(globalseed).spawn(N)
            idx_start, idx_end = split_range_between_mpi_rank(N) # e.g. [0, 5) on rank 0, [5, 10) on rank 1
            perturbed_curves = [] # this will be a List[List[Curve]], with perturbed_curves[i] containing the perturbed curves for the i-th stellarator
            for i in range(idx_start, idx_end):
                rg = Generator(PCG64DXSM(seeds_sys[j]))
                stell = []
                for c in curves:
                    pert = PerturbationSample(sampler_systematic, randomgen=rg)
                    stell.append(CurvePerturbed(c, pert))
                perturbed_curves.append(stell)
        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self.sample = sample

    def resample(self):
        self.sample.resample()
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.curve.gamma() + self.sample[0]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)
    

def curve_fourier_fit(base_curves_pert,s,order):
 
    ncoils = len(base_curves_pert)
    base_curves_fit = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=0.5, R1=1.0, order=order)

    theta = np.array(base_curves_fit[0].quadpoints)  # This gives you 0 to 1 (not 0 to 2π)
    for c in range(ncoils):
        #for each coordinate, find coefficients
        coeffs_for_all_coords = []
        for coordinate in range(3):
            x = base_curves_pert[c].gamma()[:,coordinate]
            # x = np.append(x,x[0]) #enforce periodicity for lstsq solver
            basis = []
            for m in range(order+1):
                if m == 0:
                    basis.append(np.ones_like(theta))  # Constant term (m=0)
                else:
                    basis.append(np.sin(2*np.pi*m*theta))  # sin(2π*m*phi) 
                    basis.append(np.cos(2*np.pi*m*theta))  # cos(2π*m*phi)
            A = np.column_stack(basis)
            coeffs, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
            coeffs_for_all_coords = np.append(coeffs_for_all_coords,coeffs)
        base_curves_fit[c].x = coeffs_for_all_coords

    # Plot base curves obtained from fitted coefficients
    # curves_to_vtk(base_curves_fit, OUT_DIR / f"base_curves_init_fit")
    
    #print rms error
    # More detailed error analysis
    fit_error = []
    for c in range(ncoils):
        original_points = base_curves_pert[c].gamma()
        fitted_points = base_curves_fit[c].gamma()
        
        # Interpolate fitted curve to match original points for fair comparison
        from scipy.interpolate import interp1d
        
        # Create parameterization for fitted curve
        theta_fitted = np.linspace(0, 2*np.pi, fitted_points.shape[0], endpoint=False)
        
        # Interpolate each coordinate
        fitted_interp = []
        for coord in range(3):
            f_interp = interp1d(theta_fitted, fitted_points[:, coord], kind='cubic', assume_sorted=True)
            theta_original = np.linspace(0, 2*np.pi, original_points.shape[0], endpoint=False)
            fitted_interp.append(f_interp(theta_original))
        
        fitted_interp = np.column_stack(fitted_interp)
        
        # Now calculate error with same number of points
        point_errors = np.sqrt(np.sum((original_points - fitted_interp)**2, axis=1))
        rms_error = np.sqrt(np.mean(point_errors**2))
        
        # Relative error
        curve_size = np.max(np.linalg.norm(original_points, axis=1)) - np.min(np.linalg.norm(original_points, axis=1))
        relative_error = rms_error / curve_size
        
        fit_error.append(rms_error)
        
        print(f"Coil {c}: RMS error: {rms_error:.6f}, Relative: {relative_error:.6f}")
        print(f"Max point error: {np.max(point_errors):.6f}, Min point error: {np.min(point_errors):.6f}")

    print(f"Overall fit errors: {fit_error}")
    print(f"Mean fit error: {np.mean(fit_error):.6f}")

    # return fitted curves and the mean fit error
    return base_curves_fit, np.mean(fit_error)

class CurrentPerturbed(sopp.CurrentBase, CurrentBase):
    """A perturbed current with fixed perturbation value."""
    
    def __init__(self, current, sample):
        self.current = current  # Reference to underlying current
        self.sample = sample  # Fixed perturbation for this sample
        sopp.CurrentBase.__init__(self)
        CurrentBase.__init__(self, depends_on=[current])
        
    def get_value(self):
        """Return current value + fixed perturbation"""
        return self.current.get_value() + self.sample
    
    def vjp(self, v_current):
        """Pass through VJP to underlying current"""
        return self.current.vjp(v_current)
    
    
def hessian(fun, dofs, eps=1e-6):
    x = np.asarray(dofs, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    for j in range(n):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[j] += eps
        x_bwd[j] -= eps
        _, g_fwd = fun(x_fwd)
        _, g_bwd = fun(x_bwd)
        H[:,j] = (g_fwd - g_bwd)/(2*eps)   
    return 0.5*(H + H.T)
    
    