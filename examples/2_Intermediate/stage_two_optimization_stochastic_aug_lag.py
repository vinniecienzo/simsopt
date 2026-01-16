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

import os
import time
from pathlib import Path
from numpy.random import PCG64DXSM, Generator
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.geo import (CurveLength, CurveCurveDistance, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier,
                         MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, ArclengthVariation, GaussianSampler, 
                         CurvePerturbed, PerturbationSample, LinkingNumber)
from simsopt.objectives import QuadraticPenalty, MPIObjective, SquaredFlux
from simsopt.field.force import LpCurveForce
from simsopt.util import in_github_actions, proc0_print, comm_world
import json

from stochastic_helper_functions import *
from augmented_lagrangian import *

start_time = time.time()

# Define the output directory   
OUT_DIR = "./Temp_Storage"
os.makedirs(OUT_DIR, exist_ok=True)

slurm_array_int = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# Define the test directory
TEST_DIR = Path(__file__).parent / '../' / '../' / '../' / 'tests/test_files'


#######################################################
# Specify input parameters.
#######################################################

# Number of Fourier modes describing each Cartesian component of each coil:
order = 16

# Number of samples to approximate the mean
N_SAMPLES = 4 

# Standard deviation for the coil errors
# Length scale for the coil errors
SIGMA, L = 5e-3, 0.5

# Out-of-sample evaluation parameters
N_OOS = 1000
SIGMA_OOS = SIGMA
L_OOS = L

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 2000

# Pick which configuration you want
CONFIG_NAME = "QH5" 

RUN_MODE = 'normal'

#######################################################
# End of input parameters.
#######################################################

if RUN_MODE == 'pert_init':
    # Initial guess perturbation parameters
    proc0_print("Running initial guess perturbation scan")
    SIGMA_INITIAL_GUESS = SIGMA # Standard deviation for the initial guess perturbation
    L_INITIAL_GUESS = L # Length scale for the initial guess perturbation
    fourier_fit = False #use curves with perturbed fourier coefficients
    loop_label = slurm_array_int #specify what to label results for each run
    proc0_print(loop_label)
    seed_initial_guess = slurm_array_int #assign seed using slurm array number
    save_param = slurm_array_int #relevant parameters to save correspond with saved data
    
elif RUN_MODE == 'sigma_l_scan':
    #scan sigma and L values for optimization
    proc0_print("Running sigma and l scan")
    sigma_values = np.linspace(1e-3, 1e-2, 8) #sigma values to scan
    L_values = np.linspace(0.5, 0.5, 1) #L values to scan
    sigma_and_L = [(sigma, L) for sigma in sigma_values for L in L_values] #pairs of sigma and L
    SIGMA, L= sigma_and_L[slurm_array_int] #assign sigma and L using slurm array number
    loop_label = f"Sigma={SIGMA:.3f};L={L:.3f}" #specify what to label results for each run
    save_param = (SIGMA,L) #relevant parameters to save correspond with saved data
    proc0_print(loop_label)
    if slurm_array_int >= len(sigma_and_L):
        raise ValueError(f"SLURM_ARRAY_TASK_ID {slurm_array_int} out of range for {len(sigma_and_L)} orders")
    
elif RUN_MODE == 'order_scan':
    #scan order values 
    proc0_print("Running order scan")
    order_values = [int(i) for i in range(4,36,4)] #order values to scan
    order = order_values[slurm_array_int] #assign order using slurm array number
    loop_label = f"order={order}" #specify what to label results for each run
    save_param = order #relevant parameters to save correspond with saved data
    proc0_print(loop_label)
    if slurm_array_int >= len(order_values):
        raise ValueError(f"SLURM_ARRAY_TASK_ID {slurm_array_int} out of range for {len(order_values)} orders")
    
elif RUN_MODE == 'normal':
    #Run one optimization, no scanning
    proc0_print("Running normal mode")
    loop_label = ""
    save_param = 0
    
else:
    #no proper run mode defined --> dont execute code
    raise ValueError("No run mode defined")

#load configuration
with open("000.input_parameters.json") as f:
    all_configs = json.load(f)
config = all_configs[CONFIG_NAME]
globals().update(config) # Assign all keys as variables

loop_numerical_data_label = slurm_array_int

# Define the filename
# filename = TEST_DIR / 'input.LandremanPaul2021_QA_lowres' The json should replace the need for this

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surf_filename = TEST_DIR / config["surface_filename"]

# Directory for output
out_dir_path = f"output_stage_two_optimization_stochastic_curves_{CONFIG_NAME}_{N_SAMPLES}nsamp_{RUN_MODE}_"

if RUN_MODE == 'pert_init':
    if fourier_fit == True:
        out_dir_path += "_ffit"
    
if MAXITER != 2000:
    out_dir_path += f"_{MAXITER/1000}kiter"

OUT_DIR = Path(out_dir_path)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Create the subdirectory
SUB_DIR = OUT_DIR / "Non-VTK_Data"
SUB_DIR.mkdir(parents=True, exist_ok=True)


# Define the number of phi and theta points
nphi = 64
ntheta = 16

# Pick the correct constructor dynamically
surface_constructor = getattr(SurfaceRZFourier, config["surface_method"])
s = surface_constructor(
    filename=surf_filename,
    range="full torus",
    nphi=nphi,
    ntheta=ntheta
)

qphi = 2 * nphi
qtheta = 64
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
s_plot = surface_constructor(
    surf_filename,
    range="full torus",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Define the upper and lower bounds for the constraints
#LENGTH_TARGET = 100  # comically large length upper bound
FLUX_THRESHOLD = N_SAMPLES * 1e-6 # Kept from Aug Lag script
# CC_THRESHOLD = 0.1
#CS_THRESHOLD = 0.3
#CURVATURE_THRESHOLD = 5.0
#FORCE_THRESHOLD = 0.02  # units of MN/m

# Define the number of coils, rotation order, and non-planar base curves
base_curves_init = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
curves_to_vtk(base_curves_init, OUT_DIR / f"base_curves_init")

# Perturb coils for checking different x_0

if RUN_MODE == "pert_init":
    

    rg_initial_guess = Generator(PCG64DXSM(seed_initial_guess))
    sampler_initial_guess = GaussianSampler(base_curves_init[0].quadpoints, SIGMA_INITIAL_GUESS, L_INITIAL_GUESS, n_derivs=2)
    base_curves_pert = [CurvePerturbed_jsonfix(c, PerturbationSample(sampler_initial_guess, randomgen=rg_initial_guess)) for c in base_curves_init]

    # show initial base coil after perturbation
    curves_to_vtk(base_curves_pert, OUT_DIR / f"base_curves_init_perturbed_{loop_label}")

    #fit fourier
    if fourier_fit == True: 
        base_curves, error = curve_fourier_fit(base_curves_pert, s, order)
        # show initial base coil after perturbation from fourier fit
        curves_to_vtk(base_curves, OUT_DIR / f"base_curves_init_perturbed_ffit_{loop_label}")
    else:
        base_curves = base_curves_pert
        
else:
    base_curves = base_curves_init



base_currents = [Current(1e5) for i in range(ncoils)]
base_currents[0].fix_all()
# base_curves = curves[:ncoils]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym)
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
currents = [c.current for c in coils]
#print("Number of coils:", len(coils))

# Save the biot-savart field data
bs = BiotSavart(coils)
curves_to_vtk(curves, OUT_DIR / "curves_init")

bs.set_points(s_plot.gamma().reshape((-1, 3))) 
pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1))}
             # "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR / "surf_init", extra_data=pointData)
#modB = calculate_modB_on_major_radius(bs, s)
#proc0_print(modB)

# Define the individual terms objective function:
bs.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jl = sum(QuadraticPenalty(jj, LENGTH_THRESHOLD, "max") for jj in Jls)
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
Jlink = LinkingNumber(curves) # downsample=2)
# Jforce = LpCurveForce(base_coils, coils, p=2.0, threshold=FORCE_THRESHOLD)

seed = 0
rg = Generator(PCG64DXSM(seed))
# rg = np.random.Generator(PCG64(seed, inc=0))
sampler = GaussianSampler(curves[0].quadpoints, SIGMA, L, n_derivs=1)
Jfs = []
curves_pert = []
proc0_print("Starting N_SAMPLE LOOP")
for i in range(N_SAMPLES):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed_jsonfix(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed_jsonfix(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    Jfs.append(SquaredFlux(s, bs_pert, ))
    
for k in range(len(curves_pert)):
    if k < 15:
        curves_to_vtk(curves_pert[k], OUT_DIR / f"curves_pert_n_sample_{k}")

Jmpi = MPIObjective(Jfs, comm_world, needs_splitting=True)

# Main optimization function
# f = Weight(0.0) * Jf

# Constraint list
c_list = [Jmpi, 
          Jccdist, 
          Jcsdist, 
          Jl, 
          sum(Jcs), 
          sum(Jmscs),
          sum(Jals),
          Jlink
]

x, fnc, lag_mul = augmented_lagrangian_method(
    equality_constraints=c_list,
    MAXITER=200,
    MAXITER_lag=40
)

end_time = time.time()

bs.set_points(s.gamma().reshape((-1, 3)))

proc0_print(f"Aug Lag Time taken: {end_time - start_time} seconds")
proc0_print('Final flux:', Jf.J())
proc0_print('Final CS-Sep constraint:', Jcsdist.J())
proc0_print('Final CS-sep minimum distance:', Jcsdist.shortest_distance())
proc0_print('Final CC-Sep constraint:', Jccdist.J())
proc0_print('Final CC-sep minimum distance:', Jccdist.shortest_distance())
proc0_print('Final Len constraint:', Jl.J())
proc0_print('Final Curv constraint:', sum(Jcs).J())
proc0_print('Final Link constraint:', Jlink.J())
proc0_print('Final Max Curvatures:', [np.max(c.kappa()) for c in base_curves])
proc0_print('Final Lengths:', [CurveLength(c).J() for c in base_curves], sum(Jls).J())
# print('Final Force constraint:', Jforce.J())



curves_to_vtk(curves, OUT_DIR / "optimized_coils_auglag")

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1))}
        # "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR / "surf_optimized_auglag", extra_data=pointData)
bs.set_points(s_plot.gamma().reshape((-1, 3)))
max_BdotN_overB = np.max(np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
avg_BdotN_over_B = BdotN / bs.AbsB().mean()

# now draw some fresh samples to evaluate the out-of-sample error
rg = Generator(PCG64DXSM(seed+1))
sampler = GaussianSampler(curves[0].quadpoints, SIGMA_OOS, L_OOS, n_derivs=1)
b_dot_n_pert = np.zeros((qphi, qtheta)) 
squared_flux_data = []
curves_pert_oos = []
for i in range(N_OOS):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed_jsonfix(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed_jsonfix(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    squared_flux_data.append(SquaredFlux(s, bs_pert).J())
    if slurm_array_int==0 and i<15: 
        curves_pert_oos.append([c.curve for c in coils_pert])
        curves_to_vtk(curves_pert_oos[i], OUT_DIR / f"curves_pert_oos_{loop_label}_sample_{i}")
    if (i+1) % (N_OOS/10) == 0:
        proc0_print(f"Finished {i+1}/{N_OOS} Out-of-Sample Evaluations")

proc0_print("--------------------------------------------------------------------------------------------------------------------------------------------")
proc0_print(f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}, Max BdotN/|B| = {max_BdotN_overB:.2e}")
proc0_print("FINAL LAGRANGE MULTIPLIERS:", lag_mul)
proc0_print("--------------------------------------------------------------------------------------------------------------------------------------------")
proc0_print("Final NORMALIZED SQUARED FLUX:", Jf.J())
proc0_print(f"Flux Objective for exact coils     : {Jf.J():.3e}\n")
proc0_print(f"Out-of-sample flux value                  : {np.mean(squared_flux_data):.3e}\n")
proc0_print(f"Objective Gradient (||∇J||)              : {np.linalg.norm(JF.dJ()):.3e}\n")
proc0_print(f"Mean Flux Objective across perturbed coils: {Jmpi.J():.3e}\n")
proc0_print(f"Quality Number: {Jf.J()/np.mean(squared_flux_data):.3f}\n")
proc0_print('FINISHED OPTIMIZATION')

#save data as array for plotting
np.savez(OUT_DIR / f"results_{loop_numerical_data_label}.npz",
         saved_parameter = save_param,
         sq_flux_value = Jf.J(),
         perturbed_sq_flux_data = squared_flux_data,
         gradient = np.linalg.norm(JF.dJ())
         )

#Save objective function values from outstr in fun() wrapper function
with open(SUB_DIR / 'objective_func_values.txt', 'a') as f:
    f.write(f"Run {loop_label}: \n" + last_outstr + "\n")
    
# Write input parameters to file
# Just specify the variable names you want
save_vars = ['SIGMA', 'L', 'MAXITER'
             ]

# Combine both
params = {
    'script_variables': {name: eval(name) for name in save_vars if name in locals() or name in globals()},
    'json_variables': {k: v for k, v in config.items()},
}

with open(SUB_DIR / 'input_parameters_save.json', 'w') as f:
    json.dump(params, f, indent=1)
    
end_time = time.time()
time_taken = f"Took {(end - start):.2f} for run {loop_label}."

#Save run times
with open(SUB_DIR / 'run_times.txt', 'a') as f:
            f.write(time_taken + "\n")
            
proc0_print(time_taken)