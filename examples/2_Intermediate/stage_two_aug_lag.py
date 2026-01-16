###Edited from base code due to the different base on Pedros Github

"""
auglag_alan.py
===============

This script performs coil optimization for stellarator devices using the Augmented Lagrangian Method (ALM). The optimization aims to design coil shapes that generate a target magnetic surface, subject to engineering and physics constraints. The script leverages the Simsopt library for geometry, field, and optimization routines.

Main Features:
--------------
- Reads a VMEC equilibrium file to define the target magnetic surface.
- Initializes a set of non-planar coils with configurable symmetry and Fourier order.
- Defines an objective function based on the squared normal magnetic field (squared flux) on the target surface.
- Adds constraints and penalties for engineering requirements such as coil length, coil-to-coil distance, coil-to-surface distance, and curvature.
- Implements the Augmented Lagrangian optimization loop, updating Lagrange multipliers and penalty parameters.
- Outputs VTK files for visualization of the surface and coil shapes at various stages.

Usage:
------
- Configure the optimization parameters and constraints in the script.
- Run the script directly to perform optimization using the Augmented Lagrangian or traditional method.
- Output files are saved in the './output/' directory for post-processing and visualization.

Dependencies:
-------------
- simsopt
- numpy
- scipy
- matplotlib

"""

import numpy as np
import os
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo import LinkingNumber, ArclengthVariation
from simsopt.geo import CurveLength, CurveCurveDistance, \
    LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature, GaussianSampler
from augmented_lagrangian import *
from simsopt.field import BiotSavart
from simsopt.field.force import LpCurveForce
from simsopt.field import Current, coils_via_symmetries, Coil
# from simsopt.util import calculate_modB_on_major_radius
from pathlib import Path
from numpy.random import PCG64DXSM, Generator
from simsopt.util import in_github_actions,proc0_print, comm_world 
from stochastic_helper_functions import *
import time

# Define the output directory   
OUT_DIR = "./auglag/"
os.makedirs(OUT_DIR, exist_ok=True)

# Define the test directory
TEST_DIR = '/scratch/projects/kaptanoglulab/VG/simsopt/examples/2_Intermediate/inputs'

# Define the filename
filename = TEST_DIR + '/input.20210406-01-002-nfp4_QH_000_000240'

order = 16
ncoils = 5

SIGMA_OOS, L_OOS = 5e-3, 0.5
N_OOS = 1000

# Define the number of phi and theta points
nphi = 32
ntheta = 32

# define the iterations of optimizations and the aug lag iterations
MAXITER =  1500 # 1500 for high-resolution
MAXITER_lag = 30  # 30 for high-resolution

input_str = f"Coils: Order: {order}, no. {ncoils}. Iterations: Optimization Iters: {MAXITER}, Aug Lag Iters: {MAXITER_lag}"

# Define the surface
s = SurfaceRZFourier.from_vmec_input(
    filename,
    range="full torus",
    nphi=nphi,
    ntheta=ntheta)

qphi = 4 * nphi
qtheta = 4 * ntheta
quadpoints_phi = np.linspace(0, 1, qphi)
quadpoints_theta = np.linspace(0, 1, qtheta)
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    range="full torus",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta)

# Define the number of coils, rotation order, and non-planar base curves
R0 = 1
R1 = 0.5 
curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)
base_currents = [Current(3e5/ncoils*1e-5)*1e5 for i in range(ncoils)]
base_currents[0].fix_all()
base_curves = curves[:ncoils]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
currents = [c.current for c in coils]
print("Number of coils:", len(coils))

# Define the upper and lower bounds for the constraints
LENGTH_THRESHOLD = 20 # comically large length upper bound
FLUX_THRESHOLD = 1e-15
CC_THRESHOLD = 0.1
CS_THRESHOLD = 0.1
CURVATURE_THRESHOLD = 16
MSC_THRESHOLD = 16
# FORCE_THRESHOLD = 0.02  # units of MN/


input_str += f"Length Thresh:{LENGTH_THRESHOLD}, CS/CC Thresh:{CC_THRESHOLD},{CS_THRESHOLD}, Curvature/MSC Thresh:{CURVATURE_THRESHOLD}, {MSC_THRESHOLD}.\n"

print(input_str)

# Save the biot-savart field data
bs = BiotSavart(coils)
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
bs.set_points(s_plot.gamma().reshape((-1, 3))) 
pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1))}
#             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
#modB = calculate_modB_on_major_radius(bs, s)
#print(modB)

# Define the individual terms objective function:
bs.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs, definition = 'normalized' ,threshold=FLUX_THRESHOLD)
Jls = [CurveLength(c) for c in base_curves]
#Jl = sum(QuadraticPenalty(jj, LENGTH_THRESHOLD, "identity") for jj in Jls)
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jlink = LinkingNumber(curves)
Jals = [ArclengthVariation(c) for c in base_curves]

# Jforce = LpCurveForce(base_coils, coils, p=2.0, threshold=FORCE_THRESHOLD)

# Main optimization function
# f = Weight(0.0) * Jf

# Constraint list
c_list = [Jf,
          QuadraticPenalty(sum(Jls), LENGTH_THRESHOLD, "max"),
          Jccdist, 
          Jcsdist,  
          sum(Jcs), 
          sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs),
          # sum(Jals)
          # Jlink
          # Jforce
]

start_time = time.time()
x, fnc, lag_mul = augmented_lagrangian_method(
    equality_constraints=c_list,
    MAXITER=MAXITER,
    MAXITER_lag=MAXITER_lag
)

end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")
print('Final CS-Sep constraint:', Jcsdist.J())
print('Final CS-sep minimum distance:', Jcsdist.shortest_distance())
print('Final CC-Sep constraint:', Jccdist.J())
print('Final CC-sep minimum distance:', Jccdist.shortest_distance())
print('Final Len constraint:', [f"{J.J():.1f}" for J in Jls])
print('Final Max Curvatures:', [np.max(c.kappa()) for c in base_curves])
print('Final MSC constraint:', [f'{J.J():.2f}' for J in Jmscs])
print('Final Link constraint:', Jlink.J())
print('Final Lengths:', [CurveLength(c).J() for c in base_curves], sum(Jls).J())
# print('Final Force constraint:', Jforce.J())

curves_to_vtk(curves, OUT_DIR + "optimized_coils_auglag")
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1))}
#        "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_optimized_auglag", extra_data=pointData)
bs.set_points(s_plot.gamma().reshape((-1, 3)))
max_BdotN_overB = np.max(np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)))
avg_BdotN_over_B = BdotN / bs.AbsB().mean()

print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("FINAL LAGRANGE MULTIPLIERS:", lag_mul)
print("--------------------------------------------------------------------------------------------------------------------------------------------")
bs.set_points(s.gamma().reshape((-1, 3)))
print("Final LOW RES NORMALIZED SQUARED FLUX:", Jf.J())

seed = 0
squared_flux_data = []
curves_pert_oos = []
rg = Generator(PCG64DXSM(seed+1))
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
    #if slurm_array_int==0 and i<15: 
    #    curves_pert_oos.append([c.curve for c in coils_pert])
    #    curves_to_vtk(curves_pert_oos[i], OUT_DIR / f"curves_pert_oos_{loop_label}_sample_{i}")
    #print progress
    if (i+1) % (N_OOS/10) == 0:
        proc0_print(f"Finished {i+1}/{N_OOS} Out-of-Sample Evaluations")

input_str += f"OOS Data: Sigma: {SIGMA_OOS}, No. of OOSs: {N_OOS}, L: {L_OOS}"

np.savez(OUT_DIR + f"results_aug_lag.npz",
         saved_parameter = input_str,
         sq_flux_value = Jf.J(),
         perturbed_sq_flux_data = squared_flux_data,
         )

#store main results in string, print and save
main_results_str = f"Flux Objective for exact coils    : {Jf.J():.3e}\n"
bs.set_points(s_plot.gamma().reshape((-1, 3)))
main_results_str += f"<B> : {bs.AbsB().mean():.3e}\n"
main_results_str+= f"High Res B_N: {BdotN:.3e}\n"
main_results_str += f"Out-of-sample flux value                  : {np.mean(squared_flux_data):.3e}\n"
main_results_str += f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}, Max BdotN/|B| = {max_BdotN_overB:.2e}\n"
bs.set_points(s.gamma().reshape((-1, 3)))
main_results_str += f"Quality Number: {Jf.J()/np.mean(squared_flux_data):.3f}\n"

proc0_print(main_results_str)


