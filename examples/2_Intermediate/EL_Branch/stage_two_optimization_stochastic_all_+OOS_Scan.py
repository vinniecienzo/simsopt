#!/usr/bin/env python

r"""
In this example we solve a stochastic version of the FOCUS like Stage II coil
optimisation problem: the goal is to find coils that generate a specific target
normal field on a given surface.  In this particular case we consider a vacuum
field, so the target is just zero.

The objective is given by

    J = (1/2) Mean(\int |B dot n|^2 ds)
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)
        + ARCLENGTH_WEIGHT * ArclengthVariation

where the Mean is approximated by a sample average over perturbed coils.

The target equilibrium is the QA configuration of arXiv:2108.03711.

The coil perturbations for each coil are the sum of a 'systematic error' and a
'statistical error'.  The former satisfies rotational and stellarator symmetry,
the latter is independent for each coil.

RUN_MODE = 'sigma_oos_scan' performs a DECOUPLED out-of-sample scan: the coils
are trained once at SIGMA_CURVE, and that single fixed solution is then
evaluated against each entry of SIGMA_OOS_VALUES. Training and evaluation
perturbation sizes therefore move independently, unlike 'sigma_l_scan' where
each SLURM task retrains at its own sigma and the two are coupled.

#srun -n 1 --ntasks-per-node=1 -c 1 -t1:30:00 --mem=100000 --pty /bin/bash
"""

import os
import time
from pathlib import Path
from numpy.random import PCG64DXSM, Generator
import numpy as np
import json
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.geo import (CurveLength, CurveCurveDistance, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier,
                         MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, ArclengthVariation, GaussianSampler,
                         CurvePerturbed,
                         PerturbationSample, LinkingNumber)
from simsopt.objectives import QuadraticPenalty, MPIObjective, SquaredFlux
from simsopt.util import in_github_actions, proc0_print, comm_world
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.field.selffield import regularization_circ
from stochastic_helper_functions import *




start = time.time()

# assign slurm array job number to variable
slurm_array_int = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
job_id = int(os.environ.get("SLURM_JOB_ID", 0))
proc0_print(f"SLURM job ID: {job_id}")

# True only on rank 0 (or when running without MPI). Used to guard every file
# write, since the OOS loop below runs identically on every rank.
is_proc0 = (comm_world is None) or (comm_world.rank == 0)

#######################################################
# Specify input parameters.
#######################################################

# Number of Fourier modes describing each Cartesian component of each coil:
order = 16

# Number of samples to approximate the mean
N_SAMPLES = 50

# Standard deviation for the coil errors
# Length scale for the coil errors
# Perturbations applied to coil positions and currents
SIGMA_CURVE, L_CURVE = 5e-3, 0.5
CURRENT_BASE = 1e5
SIGMA_CURRENT = 1e-1 * CURRENT_BASE
SIGMA_CENTROID = 1e-2
SIGMA_ORIENTATION = 5*np.pi/180

# Number of out-of-sample samples per evaluation point
N_OOS = 1000

# Evaluation sigmas swept by RUN_MODE = 'sigma_oos_scan'. The trained solution
# is held fixed and evaluated against each of these in turn, so the training
# and evaluation perturbation sizes are decoupled. Ignored in other run modes.
# Cost is len(SIGMA_OOS_VALUES) * N_OOS Biot-Savart evaluations, per rank.
SIGMA_OOS_VALUES = np.linspace(1e-3, 5e-3, 9)

# Parameters for the iniital guess perturbation
SIGMA_INITIAL_GUESS = 0
L_INITIAL_GUESS = 0.2
SEED_INITIAL_GUESS = 0
fourier_fit = False


PERT_CURRENT = False
PERT_CURVE = True
PERT_CENTROID = False
PERT_ORIENTATION = False

# Pick which configuration you want
CONFIG_NAME = "QH5"

RUN_MODE = 'sigma_oos_scan'

if RUN_MODE == 'pert_init':
    # Initial guess perturbation parameters
    proc0_print("Running initial guess perturbation scan")
    SIGMA_INITIAL_GUESS = 5e-3 # Standard deviation for the initial guess perturbation
    L_INITIAL_GUESS = 0.2 # Length scale for the initial guess perturbation
    fourier_fit = False #use curves with perturbed fourier coefficients
    loop_label = slurm_array_int #specify what to label results for each run
    proc0_print(loop_label)
    SEED_INITIAL_GUESS = slurm_array_int #assign seed using slurm array number
    save_param = slurm_array_int #relevant parameters to save correspond with saved data

elif RUN_MODE == 'sigma_l_scan':
    #scan sigma and L values for optimization
    proc0_print("Running sigma and l scan")
    sigma_curves_values = np.linspace(1e-3, 1e-2, 8) #sigma values to scan
    L_curves_values = np.linspace(0.5, 0.5, 1) #L values to scan
    sigma_and_L_curves = [(sigma, L) for sigma in sigma_curves_values for L in L_curves_values] #pairs of sigma and L
    if slurm_array_int >= len(sigma_and_L_curves):
        raise ValueError(f"SLURM_ARRAY_TASK_ID {slurm_array_int} out of range for {len(sigma_and_L_curves)} pairs")
    SIGMA_CURVE , L_CURVE = sigma_and_L_curves[slurm_array_int] #assign sigma and L using slurm array number
    sigma_current_values = np.linspace(1e-2, 1e-1, 8) * CURRENT_BASE
    SIGMA_CURRENT = sigma_current_values[slurm_array_int]
    sigma_centroid_values = np.linspace(1e-3, 1e-2, 8)
    SIGMA_CENTROID = sigma_centroid_values[slurm_array_int]
    if PERT_CURRENT and PERT_CURVE and PERT_CENTROID:
        loop_label = f"Sigma_curve={SIGMA_CURVE:.3f};L_curve={L_CURVE:.3f},Sigma_current={SIGMA_CURRENT:.3f},Sigma_centroid={SIGMA_CENTROID:.3f}" #specify what to label results for each run
        save_param = (SIGMA_CURVE,L_CURVE,SIGMA_CURRENT,SIGMA_CENTROID) #relevant parameters to save correspond with saved data
    elif PERT_CURRENT:
        loop_label = f"Sigma_current={SIGMA_CURRENT:.3f}" #specify what to label results for each run
        save_param = (SIGMA_CURRENT) #relevant parameters to save correspond with saved data
    elif PERT_CURVE:
        loop_label = f"Sigma_curve={SIGMA_CURVE:.3f};L_curve={L_CURVE:.3f}" #specify what to label results for each run
        save_param = (SIGMA_CURVE,L_CURVE) #relevant parameters to save correspond with saved data
    elif PERT_CENTROID:
        loop_label = f"Sigma_centroid={SIGMA_CENTROID:.3f}" #specify what to label results for each run
        save_param = (SIGMA_CENTROID) #relevant parameters to save correspond with saved data
    proc0_print(loop_label)

elif RUN_MODE == 'sigma_oos_scan':
    # Train once, then sweep the EVALUATION sigma against that fixed solution.
    # SIGMA_CURVE (training) is left at its input value on purpose.
    proc0_print("Running sigma_OOS scan (evaluation sigma decoupled from training)")
    proc0_print(f"  training sigma held at SIGMA_CURVE = {SIGMA_CURVE:.4e}")
    proc0_print(f"  {len(SIGMA_OOS_VALUES)} evaluation sigmas x N_OOS={N_OOS} "
                f"= {len(SIGMA_OOS_VALUES)*N_OOS} Biot-Savart evaluations")
    loop_label = f"sigma_oos_scan_{slurm_array_int}"
    save_param = SIGMA_OOS_VALUES
    proc0_print(loop_label)

elif RUN_MODE == 'order_scan':
    #scan order values
    proc0_print("Running order scan")
    order_values = [int(i) for i in range(4,36,4)] #order values to scan
    if slurm_array_int >= len(order_values):
        raise ValueError(f"SLURM_ARRAY_TASK_ID {slurm_array_int} out of range for {len(order_values)} orders")
    order = order_values[slurm_array_int] #assign order using slurm array number
    loop_label = f"order={order}" #specify what to label results for each run
    save_param = order #relevant parameters to save correspond with saved data
    proc0_print(loop_label)

elif RUN_MODE == 'normal':
    #Run one optimization, no scanning
    proc0_print("Running normal mode")
    loop_label = ""
    save_param = 0

else:
    #no proper run mode defined --> dont execute code
    raise ValueError("No run mode defined")

SIGMA_CURRENT = SIGMA_CURRENT if PERT_CURRENT else 0
SIGMA_CURVE = SIGMA_CURVE if PERT_CURVE else 0
SIGMA_CENTROID = SIGMA_CENTROID if PERT_CENTROID else 0
SIGMA_ORIENTATION = SIGMA_ORIENTATION if PERT_ORIENTATION else 0

# Out-of-sample evaluation parameters. By default the evaluation perturbation
# matches the training perturbation; the sigma_oos_scan mode overrides the
# curve sigma below, one value at a time.
SIGMA_CURVE_OOS = SIGMA_CURVE
L_CURVE_OOS = L_CURVE
SIGMA_CURRENT_OOS = SIGMA_CURRENT
SIGMA_CENTROID_OOS = SIGMA_CENTROID
SIGMA_ORIENTATION_OOS = SIGMA_ORIENTATION

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 2000

#######################################################
# End of input parameters.
#######################################################

#load configuration
with open("input_parameters.json") as f:
    all_configs = json.load(f)
config = all_configs[CONFIG_NAME]
globals().update(config) # Assign all keys as variables

#label for numerical data, like arrays or floats
#unperturbed sq flux, gradient, perturbed sq flux distribution
loop_numerical_data_label = slurm_array_int

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surf_filename = TEST_DIR / config["surface_filename"]

# Directory for output
out_dir_path = f"output_stage_two_optimization_stochastic_{CONFIG_NAME}_{N_SAMPLES}nsamp_{RUN_MODE}"

if PERT_CURRENT and PERT_CURVE and PERT_CENTROID and PERT_ORIENTATION:
    out_dir_path += "_all"
if PERT_CURRENT:
    out_dir_path += "_currents"
if PERT_CURVE:
    out_dir_path += "_curves"
if PERT_CENTROID:
    out_dir_path += "_centroids"
if PERT_ORIENTATION:
    out_dir_path += "_orientations"

if RUN_MODE == 'pert_init':
    if fourier_fit == True:
        out_dir_path += "_ffit"

if MAXITER != 2000:
    out_dir_path += f"_{MAXITER/1000}kiter"
proc0_print(out_dir_path)
OUT_DIR = Path(out_dir_path)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Create the subdirectory
SUB_DIR = OUT_DIR / "Non-VTK_Data"
SUB_DIR.mkdir(parents=True, exist_ok=True)

#Subdirectory for checking perturbed objects
SUB_PERT_DIR = OUT_DIR / "Pert_Data"
SUB_PERT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the boundary magnetic surface; errors break symmetries, so consider the full torus
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

# Create the initial coils:
#initial coils for comparison, left unbothered
base_curves_init = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
curves_to_vtk(base_curves_init, OUT_DIR / f"base_curves_init")

# Perturb coils
if SIGMA_INITIAL_GUESS != 0:

    rg_initial_guess = Generator(PCG64DXSM(SEED_INITIAL_GUESS))
    sampler_initial_guess = GaussianSampler(base_curves_init[0].quadpoints, SIGMA_INITIAL_GUESS, L_INITIAL_GUESS, n_derivs=2)
    base_curves_pert = [CurvePerturbed_jsonfix(c, PerturbationSample(sampler_initial_guess, randomgen=rg_initial_guess)) for c in base_curves_init]

    # show initial base coil after perturbation
    curves_to_vtk(base_curves_pert, SUB_PERT_DIR / f"base_curves_init_perturbed_{loop_label}")

    #fit fourier
    if fourier_fit == True:
        base_curves, error = curve_fourier_fit(base_curves_pert, s, order)
        curves_to_vtk(base_curves, SUB_PERT_DIR / f"base_curves_fit{loop_label}")

    else:
        base_curves = base_curves_pert

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
currents = [c.current for c in coils]
curves_to_vtk(curves, OUT_DIR / f"curves_init_{loop_label}")

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
# s_plot.to_vtk(OUT_DIR / f"surf_init_{loop_label}", extra_data=pointData)

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

seed = 0
rg = Generator(PCG64DXSM(seed))
# rg = np.random.Generator(PCG64(seed, inc=0))
# Training sampler. Kept distinct from the evaluation sampler built later, so
# training draws and out-of-sample draws never share a stream.
sampler = GaussianSampler(curves[0].quadpoints, SIGMA_CURVE, L_CURVE, n_derivs=1)
Jfs = []
curves_pert = []
currents_pert = []
proc0_print("Starting N_SAMPLE LOOP")
for i in range(N_SAMPLES):
    # first add the 'systematic' error. this error is applied to the base curves and hence the various symmetries are applied to it.
    # systematic coil position error
    base_curves_perturbed = [CurvePerturbed_jsonfix(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
    # systematic coil centroid position error
    base_curves_centroid_perturbed = [CentroidPerturbed(c,(rg.standard_normal(3), SIGMA_CENTROID*rg.standard_normal())) for c in base_curves_perturbed]
    # systematic coil orientation error
    base_curves_orientation_perturbed = [OrientationPerturbed(c, SIGMA_ORIENTATION*rg.standard_normal(3)) for c in base_curves_centroid_perturbed]
    coils_sample = coils_via_symmetries(base_curves_orientation_perturbed, base_currents, s.nfp, True)
    # now add the 'statistical' error. this error is added to each of the final coils, and independent between all of them.
    # statistical coil position and current error
    coils_pert = [Coil(CurvePerturbed_jsonfix(c.curve, PerturbationSample(sampler, randomgen=rg)), CurrentPerturbed(c.current, SIGMA_CURRENT*rg.standard_normal())) for c in coils_sample]
    # statistical centroid position error
    coils_centroid_pert = [Coil(CentroidPerturbed(c.curve, (rg.standard_normal(3), SIGMA_CENTROID*rg.standard_normal())),c.current) for c in coils_pert]
    # statistical coil orientation error
    coils_orientation_pert = [Coil(OrientationPerturbed(c.curve, SIGMA_ORIENTATION*rg.standard_normal(3)), c.current) for c in coils_centroid_pert]
    # save curves for plotting; saves coil position perturbed and coil + centroid position perturbed curves
    curves_pert.append([c.curve for c in coils_orientation_pert])
    # save current pert current array
    currents_pert.append([c.current.get_value() for c in coils_orientation_pert])
    bs_pert = BiotSavart(coils_orientation_pert)
    Jfs.append(SquaredFlux(s, bs_pert))


if is_proc0:
    if PERT_CURVE:
        for k in range(min(15, len(curves_pert))):
            curves_to_vtk(curves_pert[k], SUB_PERT_DIR / f"curves_pert_n_sample_{k}")
    if PERT_CENTROID:
        for k in range(min(15, len(curves_pert))):
            curves_to_vtk(curves_pert[k], SUB_PERT_DIR / f"curves_centroid_pert_n_sample_{k}")
    if PERT_CURRENT:
        for k in range(min(15, len(currents_pert))):
            np.save(SUB_PERT_DIR / f"currents_pert_n_sample_{k}", currents_pert[k])
    if PERT_ORIENTATION:
        for k in range(min(15, len(curves_pert))):
            curves_to_vtk(curves_pert[k], SUB_PERT_DIR / f"curves_orientation_pert_n_sample_{k}")

Jmpi = MPIObjective(Jfs, comm_world, needs_splitting=True)

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
#+ LENGTH_WEIGHT * sum(Jls) \
#+ LENGTH_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD, "max") for J in Jls) \

JF = Jmpi \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_THRESHOLD, "max") \
    + CC_WEIGHT * Jccdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + ARCLENGTH_WEIGHT * sum(Jals) \
    + CS_WEIGHT * Jcsdist \
    + LINK_WEIGHT * linkNum


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
    jf = Jmpi.J()
    currents = [c.get_value() for c in base_currents]
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"Iteration {iteration_counter}/{MAXITER}-----\n"
    outstr += f"currents: {currents}\n"
    outstr += f"J={J:.1e}, ⟨Jf⟩={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L>=[{msc_string}], C-C-Sep={Jccdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    last_outstr = outstr
    proc0_print(outstr, flush=True)

    return J, grad

proc0_print("""
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
    proc0_print("err", (J1-J2)/(2*eps) - dJh)

# start_hess = time.time()
# proc0_print("""
# ################################################################################
# ### Perform a Hessian test######################################################
# ################################################################################
# """)

# ddJ0 = hessian(f, dofs)
# print("done with mpi hessian calc")
# for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
#     J1, _ = f(dofs+eps*h)
#     err = J1 - (J0 + eps*dJh + 0.5*eps**2*h.T@ddJ0@h)
#     proc0_print("err", eps, err)

# end_hess = time.time()
# proc0_print(f"time to run hessian test {end_hess - start_hess}")


proc0_print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

# Reset counter before optimization starts
iteration_counter = 0

res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 400}, tol=1e-15)
proc0_print("--------------------------------JF.x shape after opt:", JF.x.shape)
alen_string = ", ".join([f"{np.max(c.incremental_arclength())/np.min(c.incremental_arclength())-1:.2e}" for c in base_curves])
proc0_print(f"Final arclength variation max(|ℓ|)/min(|ℓ|) - 1=[{alen_string}]")

proc0_print("""
################################################################################
### Evaluate the obtained coils ################################################
################################################################################
""")


curves_to_vtk(curves, OUT_DIR / f"curves_opt_{loop_label}")
curves_to_vtk(base_curves, OUT_DIR / f"base_curves_opt_{loop_label}")
bs.save(OUT_DIR / "biot_savart_opt.json")

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR / f"surf_opt_{loop_label}", extra_data=pointData)

bs.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
avg_BdotN_over_B = BdotN / bs.AbsB().mean()
Jf.x = res.x

# now draw some fresh samples to evaluate the out-of-sample error.
# The trained solution is fixed from here on; everything below only evaluates it.
b_dot_n_pert = np.zeros((qphi, qtheta))
curves_pert_oos = []
perturbation_number = 5 if PERT_CURVE and PERT_CURRENT and PERT_CENTROID and PERT_ORIENTATION else 1

# In scan mode sweep the evaluation sigma; otherwise a single point at the
# nominal value, which reproduces the previous behaviour exactly.
sigma_oos_list = SIGMA_OOS_VALUES if RUN_MODE == 'sigma_oos_scan' else [SIGMA_CURVE_OOS]
sigma_oos_nominal = float(sigma_oos_list[0])

# sigma_oos -> list of perturbation_number lists, each holding N_OOS flux values
squared_flux_scan = {}

for sigma_oos in sigma_oos_list:
    # Reseed at every scan point so all points draw the same underlying standard
    # normals. Differences between points are then attributable to sigma rather
    # than to sampling noise (common random numbers). seed+1 keeps the
    # evaluation stream distinct from the training stream seeded above.
    rg = Generator(PCG64DXSM(seed+1))
    squared_flux_data = [[] for _ in range(perturbation_number)]

    for j in range(perturbation_number):
        # j = 0 : all perturbation types together
        # j = 1 : current only        j = 3 : centroid only
        # j = 2 : curve only          j = 4 : orientation only
        sig_curve  = sigma_oos             if j in (0, 2) else 0.0
        sig_curr   = SIGMA_CURRENT_OOS     if j in (0, 1) else 0.0
        sig_cent   = SIGMA_CENTROID_OOS    if j in (0, 3) else 0.0
        sig_orient = SIGMA_ORIENTATION_OOS if j in (0, 4) else 0.0

        sampler_j = GaussianSampler(curves[0].quadpoints, sig_curve, L_CURVE_OOS, n_derivs=1)

        for i in range(N_OOS):
            # --- systematic error: applied to the base coils, then propagated
            #     through the symmetries, so it is correlated across coils ---
            base_curves_perturbed = [
                CurvePerturbed_jsonfix(c, PerturbationSample(sampler_j, randomgen=rg))
                for c in base_curves]
            base_curves_centroid_perturbed = [
                CentroidPerturbed(c, (rg.standard_normal(3), sig_cent*rg.standard_normal()))
                for c in base_curves_perturbed]
            base_curves_orientation_perturbed = [
                OrientationPerturbed(c, sig_orient*rg.standard_normal(3))
                for c in base_curves_centroid_perturbed]
            coils_sys = coils_via_symmetries(base_curves_orientation_perturbed,
                                             base_currents, s.nfp, True)

            # --- statistical error: applied to each final coil independently ---
            coils_pert = [
                Coil(CurvePerturbed_jsonfix(c.curve, PerturbationSample(sampler_j, randomgen=rg)),
                     CurrentPerturbed(c.current, sig_curr*rg.standard_normal()))
                for c in coils_sys]
            coils_centroid_pert = [
                Coil(CentroidPerturbed(c.curve, (rg.standard_normal(3), sig_cent*rg.standard_normal())),
                     c.current)
                for c in coils_pert]
            coils_orientation_pert = [
                Coil(OrientationPerturbed(c.curve, sig_orient*rg.standard_normal(3)), c.current)
                for c in coils_centroid_pert]

            # Squared Flux calculation
            bs_pert = BiotSavart(coils_orientation_pert)
            bs_pert.set_points(s.gamma().reshape((-1, 3)))
            squared_flux_data[j].append(SquaredFlux(s, bs_pert).J())

            # only save the first 15 samples, and only at the nominal sigma
            if j == 0 and i < 15 and float(sigma_oos) == sigma_oos_nominal:
                curves_pert_oos.append([c.curve for c in coils_orientation_pert])
                if is_proc0:
                    curves_to_vtk(curves_pert_oos[-1],
                                  SUB_PERT_DIR / f"curves_opt_pert_oos_{loop_label}_sample_{i}")

            #print progress
            if (i+1) % max(1, N_OOS // 10) == 0:
                proc0_print(f"sigma_OOS={sigma_oos:.4f}  j={j}  "
                            f"Finished {i+1}/{N_OOS} Out-of-Sample Evaluations")

    squared_flux_scan[float(sigma_oos)] = squared_flux_data
    proc0_print(f"sigma_OOS = {sigma_oos:.4f} -> mean J_OOS = {np.mean(squared_flux_data[0]):.4e}")

# Downstream reporting uses the nominal evaluation point.
squared_flux_data = squared_flux_scan[sigma_oos_nominal]

#store main results in string, print and save
main_results_str = f"Flux Objective for exact coils     : {Jf.J():.3e}\n"
main_results_str += f"Out-of-sample flux value                  : {np.mean(squared_flux_data[0]):.3e}\n"
main_results_str += f"Objective Gradient (||∇J||)              : {np.linalg.norm(JF.dJ()):.3e}\n"
main_results_str += f"Mean Flux Objective across perturbed coils: {Jmpi.J():.3e}\n"
main_results_str += f"Quality Number: {Jf.J()/np.mean(squared_flux_data[0]):.3f}\n"
main_results_str += f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}\n"

if RUN_MODE == 'sigma_oos_scan':
    main_results_str += (f"\nDecoupled sigma_OOS scan "
                         f"(training sigma fixed at {SIGMA_CURVE:.4e}):\n")
    for sig in sigma_oos_list:
        mean_j = np.mean(squared_flux_scan[float(sig)][0])
        main_results_str += (f"  sigma_OOS = {sig:.4e}  "
                             f"J_OOS = {mean_j:.4e}  Q = {Jf.J()/mean_j:.4f}\n")

proc0_print(main_results_str)

#save data as array for plotting
save_dict = dict(
    saved_parameter = save_param,
    sq_flux_value = Jf.J(),
    perturbed_sq_flux_data = np.array(squared_flux_data),
    gradient = np.linalg.norm(JF.dJ()),
    avg_BdotN_over_B = avg_BdotN_over_B,
)

if RUN_MODE == 'sigma_oos_scan':
    sigmas_sorted = sorted(squared_flux_scan.keys())
    # sigma_oos_flux row i corresponds to sigma_oos_values[i]; shape (n_sigma, N_OOS)
    save_dict['training_sigma'] = SIGMA_CURVE
    save_dict['sigma_oos_values'] = np.array(sigmas_sorted)
    save_dict['sigma_oos_flux'] = np.array([squared_flux_scan[k][0] for k in sigmas_sorted])
    save_dict['sigma_oos_mean'] = np.array([np.mean(squared_flux_scan[k][0]) for k in sigmas_sorted])
    save_dict['sigma_oos_Q'] = np.array([Jf.J()/np.mean(squared_flux_scan[k][0]) for k in sigmas_sorted])

if is_proc0:
    np.savez(OUT_DIR / f"results_{loop_numerical_data_label}.npz", **save_dict)

    with open(SUB_DIR / 'main_results.txt', 'a') as f:
        f.write(f" Run {loop_label}: \n" + main_results_str)
    #Save objective function values from outstr in fun() wrapper function
    with open(SUB_DIR / 'objective_func_values.txt', 'a') as f:
        f.write(f"\n Run {loop_label}: \n" + last_outstr)

    # Write input parameters to file
    # Just specify the variable names you want
    save_vars = ['SIGMA_CURVE', 'L_CURVE', 'SIGMA_CURRENT', 'SIGMA_CENTROID',
                 'SIGMA_ORIENTATION', 'N_SAMPLES', 'N_OOS', 'MAXITER', 'order',
                 'RUN_MODE', 'PERT_CURVE', 'PERT_CURRENT', 'PERT_CENTROID',
                 'PERT_ORIENTATION']

    # Combine both
    params = {
        'script_variables': {name: eval(name) for name in save_vars if name in locals() or name in globals()},
        'json_variables': {k: v for k, v in config.items()},
    }

    if RUN_MODE == 'sigma_oos_scan':
        params['script_variables']['SIGMA_OOS_VALUES'] = [float(x) for x in SIGMA_OOS_VALUES]

    with open(SUB_DIR / 'input_parameters_save.json', 'w') as f:
        json.dump(params, f, indent=1, default=str)

    end = time.time()
    time_taken = f"Took {(end - start):.2f} for run {loop_label}."

    #Save run times
    with open(SUB_DIR / 'run_times.txt', 'a') as f:
                f.write(time_taken + "\n")

    proc0_print(f"Total time taken: {(end - start):.2f} seconds")
