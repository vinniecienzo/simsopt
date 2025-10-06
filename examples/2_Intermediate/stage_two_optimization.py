#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
import time 
from pathlib import Path
import numpy as np
import json
from numpy.random import PCG64DXSM, Generator
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, ArclengthVariation,
                         GaussianSampler, CurvePerturbed, CurrentPerturbed,
                         PerturbationSample, LinkingNumber)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt.util import in_github_actions


start = time.time()

# assign slurm array job number to variable
slurm_array_int = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# Number of Fourier modes describing each Cartesian component of each coil:
order = 24

# Number of samples for out-of-sample evaluation
N_OOS = 1000

# Standard deviation for the coil errors
# Length scale for the coil errors
SIGMA_OOS, L_OOS = 1e-2, 0.5

# Choose and load input parameters from configuration
CONFIG_NAME = "NCSX"

RUN_MODE = 'sigma_l_scan'

if RUN_MODE == 'pert_init':
    # Initial guess perturbation parameters
    print("Running initial guess perturbation scan")
    SIGMA_INITIAL_GUESS = 1e-2 # Standard deviation for the initial guess perturbation
    L_INITIAL_GUESS = 0.15 # Length scale for the initial guess perturbation
    fourier_fit = False #use curves with perturbed fourier coefficients
    loop_label = slurm_array_int #specify what to label results for each run
    print(loop_label)
    seed_initial_guess = slurm_array_int #assign seed using slurm array number
    save_param = slurm_array_int #relevant parameters to save correspond with saved data
        
elif RUN_MODE == 'sigma_l_scan':
    #scan sigma and L values for optimization
    print("Running sigma and l scan")
    sigma_values = np.linspace(1e-3, 1e-2, 8) #sigma values to scan
    L_values = np.linspace(0.5, 0.5, 1) #L values to scan
    sigma_and_L = [(sigma, L) for sigma in sigma_values for L in L_values] #pairs of sigma and L
    SIGMA_OOS, L_OOS = sigma_and_L[slurm_array_int] #assign sigma and L using slurm array number
    loop_label = slurm_array_int #specify what to label results for each run
    save_param = (SIGMA_OOS,L_OOS) #relevant parameters to save correspond with saved data
    print(loop_label)
    if slurm_array_int >= len(sigma_and_L):
        raise ValueError(f"SLURM_ARRAY_TASK_ID {slurm_array_int} out of range for {len(sigma_and_L)} orders")
    
elif RUN_MODE == 'order_scan':
    #scan order values 
    print("Running order scan")
    order_values = [int(i) for i in range(4,36,4)] #order values to scan
    order = order_values[slurm_array_int] #assign order using slurm array number
    loop_label = f"order={order}" #specify what to label results for each run
    save_param = order #relevant parameters to save to correspond with saved data
    print(loop_label)
    if slurm_array_int >= len(order_values):
        raise ValueError(f"SLURM_ARRAY_TASK_ID {slurm_array_int} out of range for {len(order_values)} orders")

elif RUN_MODE == 'normal':
    #Run one optimization, no scanning
    print("Running normal mode")
    loop_label = ""
    save_param = 0
    
else:
    #no proper run mode defined --> dont execute code
    raise ValueError("No run mode defined")


# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 1000

#######################################################
# End of input parameters.
#######################################################

#load configuration
with open("input_parameters.json") as f:
    all_configs = json.load(f)
config = all_configs[CONFIG_NAME]
globals().update(config)  # Assign all keys as variables

#label for numerical data, like arrays or floats
#unperturbed sq flux, gradient, perturbed sq flux distribution
loop_numerical_data_label = slurm_array_int

# File for the desired boundary magnetic surface:

TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / config["surface_filename"]

# Directory for output
out_dir_path = f"output_stage_two_optimization_curves_{CONFIG_NAME}_{RUN_MODE}"

if RUN_MODE == 'pert_init':
    if fourier_fit == True:
        out_dir_path += "_ffit"
    
if MAXITER != 1000:
    out_dir_path += f"_{MAXITER/1000}kiter"
    
OUT_DIR = Path(out_dir_path)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Create the subdirectory
SUB_DIR = OUT_DIR / "Non-VTK_Data"
SUB_DIR.mkdir(parents=True, exist_ok=True)



# Initialize the boundary magnetic surface:
nphi = 64
ntheta = 16

# Pick the correct constructor dynamically
surface_constructor = getattr(SurfaceRZFourier, config["surface_method"])

s = surface_constructor(
    filename=filename,
    range="full torus",
    nphi=nphi,
    ntheta=ntheta
)

qphi = 2 * nphi
qtheta = 64
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, qtheta, endpoint=True)
s_plot = surface_constructor(
    filename,
    range="full torus",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta
)

# Create the initial coils:
base_curves_init = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
curves_to_vtk(base_curves_init, OUT_DIR / f"base_curves_init")
    
# Perturb coils
if RUN_MODE == "pert_init":
    

    rg_initial_guess = Generator(PCG64DXSM(seed_initial_guess))
    sampler_initial_guess = GaussianSampler(base_curves_init[0].quadpoints, SIGMA_INITIAL_GUESS, L_INITIAL_GUESS, n_derivs=2)
    base_curves_pert = [CurvePerturbed(c, PerturbationSample(sampler_initial_guess, randomgen=rg_initial_guess)) for c in base_curves_init]

    # show initial base coil after perturbation
    curves_to_vtk(base_curves_pert, OUT_DIR / f"base_curves_init_perturbed_{loop_label}")

    #fit fourier
    if fourier_fit == True: 
        base_curves, error = curve_fourier_fit(base_curves_pert, s, order)
        
    else:
        base_curves = base_curves_pert
        
    curves_to_vtk(base_curves, OUT_DIR / f"base_curves_init_pert_{loop_label}")
    
else:
    base_curves = base_curves_init

base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

# base_currents = [Current(1e5) for i in range(ncoils-1)]
# total_current = Current(1e5*ncoils)
# total_current.fix_all()
# base_currents += [total_current - sum(base_currents)]

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR / f"curves_init_{loop_label}") 

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR / f"surf_init_{loop_label}", extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
linkNum = LinkingNumber(curves)

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:t5
#+ LENGTH_WEIGHT * sum(Jls) \
#+ LENGTH_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD, "max") for J in Jls) \
    
JF = Jf \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_THRESHOLD, "max") \
    + CC_WEIGHT * Jccdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + ARCLENGTH_WEIGHT * sum(Jals) \
    + CS_WEIGHT * Jcsdist \
    + LINK_WEIGHT * linkNum

#J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD) for i in range(len(base_curves))])
# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize

iteration_counter = 0
def fun(dofs):
    global iteration_counter, last_outstr
    iteration_counter += 1
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    currents = JF.x[:ncoils-1]
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"Iteration {iteration_counter}/{MAXITER}-----\n"
    outstr += f"currents: {currents}\n"
    outstr += f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    last_outstr = outstr
    print(outstr)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")

f = fun
dofs = JF.x

np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)
    
print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

# Reset counter before optimization starts
iteration_counter = 0

res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol = 1e-15)

curves_to_vtk(curves, OUT_DIR / f"curves_opt_{loop_label}")
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR / f"surf_opt_{loop_label}", extra_data=pointData)
bs.set_points(s.gamma().reshape((-1, 3)))

curves_to_vtk(base_curves, OUT_DIR / f"base_curves_opt_{loop_label}")
# Save the optimized coil shapes and currents so they can be loaded into other scripts for analysis:
bs.save(OUT_DIR / f"biot_savart_opt_{loop_label}.json")

#Perturb coils
seed = 0
squared_flux_data = []
curves_pert_oos = []
rg = Generator(PCG64DXSM(seed+1))
sampler = GaussianSampler(curves[0].quadpoints, SIGMA_OOS, L_OOS, n_derivs=1)
for i in range(N_OOS):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    coils = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
    # Squared Flux calculation
    bs_pert = BiotSavart(coils_pert)
    bs_pert.set_points(s.gamma().reshape((-1, 3)))
    squared_flux_data.append(SquaredFlux(s, bs_pert).J())
    #only save first 15 samples, for first initial guess
    if slurm_array_int==0 and i<15: 
        curves_pert_oos.append([c.curve for c in coils_pert])
        curves_to_vtk(curves_pert_oos[i], OUT_DIR / f"curves_pert_oos_{loop_label}_sample_{i}")
    #print progress
    if (i+1) % (N_OOS/10) == 0:
        print(f"Finished {i+1}/{N_OOS} Out-of-Sample Evaluations")
        
#store main results in string, print and save
main_results_str = f"Flux Objective for exact coils    : {Jf.J():.3e}\n"
main_results_str += f"Out-of-sample flux value                  : {np.mean(squared_flux_data):.3e}\n"
main_results_str += f"Objective Gradient (||∇J||)              : {np.linalg.norm(JF.dJ()):.3e}\n"
main_results_str += f"Quality Number: {Jf.J()/np.mean(squared_flux_data):.3f}\n"

print(main_results_str)

with open(SUB_DIR / 'main_results.txt', 'a') as f:
    f.write(f"Run {loop_label}: \n" + main_results_str)

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
    
end = time.time()
time_taken = f"Took {(end - start):.2f} for run {loop_label}."

#Save run times
with open(SUB_DIR / 'run_times.txt', 'a') as f:
            f.write(time_taken + "\n")
            
print(f"Took {end-start}s")
