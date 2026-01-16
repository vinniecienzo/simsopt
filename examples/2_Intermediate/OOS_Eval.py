#### Work in Progress ###

'''
 
This is the out of sample evaluation, here we use simsopt CurvePerturbed function
to take base_curves (either before or after optimization) to create random continous changes in our coils
which then enables to look at there performance for any plasma surface, s. These systematic and statistical
perturbations mathematically model the errors either manufacturing or assembly (determining this), thus 
investigating the performance with random errors gives an idea to the robustness of the coil solution.
Commonly used in stochastic code comparing robustness of deterministic and stochastic coil solutions.

The return values are is an array of Squared flux (which can be used to show the kernel, b dot n vs. number of solutions)
and a random selection of perturbed coil samples.

'''

def OOS_eval(base_curves,
            s,
            N_OOS=1000, 
            SIGMA_OOS=5e-3, 
            L_OOS=0.5,
            curves,
            seed,
             ):

    from numpy.random import PCG64DXSM, Generator
    from simsopt.geo import CurvePerturbed_jsonfix, GaussianSampler
    from simsopt.field import Coil, BiotSavart

    seed = seed
    squared_flux_data = []
    curves_pert_oos = []
    rg = Generator(PCG64DXSM(seed))
    sampler = GaussianSampler(curves[0].quadpoints, SIGMA_OOS, L_OOS, n_derivs=1)

    for i in range(N_OOS):
        # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
        base_curves_perturbed = [CurvePerturbed_jsonfix(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
        coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
        # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
        coils_pert = [Coil(CurvePerturbed_jsonfix(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
        # Squared Flux calculation
        bs_pert = BiotSavart(coils_pert)
        bs_pert.set_points(s.gamma().reshape((-1, 3)))
        squared_flux_data.append(SquaredFlux(s, bs_pert).J())
        #only save first 15 samples, for first initial guess
        if slurm_array_int==0 and i<10: 
            curves_pert_oos.append([c.curve for c in coils_pert])

return squared_flux_data, curves_pert_oos
