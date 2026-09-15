#!/usr/bin/env python

r"""
Stochastic Stage-II coil optimization with a 2D (sigma_train x sigma_OOS) scan.

Each SLURM array task:
  1. optimizes coils at ONE training sigma (geometry / curve-shape errors only),
  2. sweeps sigma_OOS over SIGMA_CURVE_OOS_VALUES,
  3. runs N_OOS out-of-sample evaluations at each sigma_OOS,
  4. writes ONE .npz per training sigma containing the FULL J_b sample arrays.

No plotting here. Stack the per-task .npz files afterwards to build the
(sigma_train, sigma_OOS) -> <J_b> mesh.

Objective:

    J = (1/2) Mean(\int |B dot n|^2 ds)
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)
        + ARCLENGTH_WEIGHT * ArclengthVariation

#sbatch --array=0-7 ...
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

slurm_array_int = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
job_id = int(os.environ.get("SLURM_JOB_ID", 0))
proc0_print(f"SLURM job ID: {job_id}")

#######################################################
# Specify input parameters.
#######################################################

# Number of Fourier modes describing each Cartesian component of each coil:
order = 16

# Number of samples to approximate the mean during TRAINING
N_SAMPLES = 50

CURRENT_BASE = 1e5

# --- Geometry-only perturbations for this study ------------------------------
# Manufacturing geometry (coil-shape) errors only. Orientation, centroid (COM)
# and current perturbations are all switched off.
PERT_CURVE = True
PERT_CURRENT = False
PERT_CENTROID = False
PERT_ORIENTATION = False

# Length scale of the Gaussian coil-shape errors (held fixed across the mesh)
L_CURVE = 0.5

# Parameters for the initial guess perturbation (unused in this mode)
SIGMA_INITIAL_GUESS = 0
L_INITIAL_GUESS = 0.2
SEED_INITIAL_GUESS = 0
fourier_fit = False

# Pick which configuration you want
CONFIG_NAME = "QH5"

RUN_MODE = 'sigma_oos_mesh'

# ---------------- Mesh axes --------------------------------------------------
# Axis 1 (one SLURM array task per value): training sigma, metres.
SIGMA_CURVE_TRAIN_VALUES = np.linspace(1e-3, 1e-2, 10)

# Axis 2 (swept inside every task): out-of-sample sigma, metres.
# Same range and resolution as the training axis -> square 8x8 mesh, with the
# diagonal being "evaluated at the sigma it was trained on".
SIGMA_CURVE_OOS_VALUES = np.linspace(1e-3, 1e-2, 10)

# Number of out-of-sample perturbation draws at EACH sigma_OOS
N_OOS = 1000

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 2000

# RNG seeds. The OOS generator is re-seeded identically at every sigma_OOS and
# in every array task, so all mesh points see the SAME underlying perturbation
# realizations (common random numbers) -- differences across the mesh are then
# signal, not sampling noise.
SEED_TRAIN = 0
SEED_OOS = 1

# Save VTK for a handful of perturbed OOS coil sets (0 disables)
N_VTK_SAMPLES = 5

#######################################################
# End of input parameters.
#######################################################

if RUN_MODE != 'sigma_oos_mesh':
    raise ValueError("This script only implements RUN_MODE = 'sigma_oos_mesh'")

if slurm_array_int >= len(SIGMA_CURVE_TRAIN_VALUES):
    raise ValueError(f"SLURM_ARRAY_TASK_ID {slurm_array_int} out of range for "
                     f"{len(SIGMA_CURVE_TRAIN_VALUES)} training sigmas")

SIGMA_CURVE = SIGMA_CURVE_TRAIN_VALUES[slurm_array_int]

# Everything except curve-shape error is zeroed out.
SIGMA_CURRENT = 0.0
SIGMA_CENTROID = 0.0
SIGMA_ORIENTATION = 0.0

loop_label = f"Sigma_curve={SIGMA_CURVE:.4f};L_curve={L_CURVE:.3f}"
save_param = (SIGMA_CURVE, L_CURVE)
loop_numerical_data_label = slurm_array_int

proc0_print("Running training-sigma x OOS-sigma mesh scan (geometry errors only)")
proc0_print(loop_label)
proc0_print(f"sigma_OOS sweep (m): {SIGMA_CURVE_OOS_VALUES}")
proc0_print(f"N_OOS per sigma_OOS: {N_OOS}  ->  {len(SIGMA_CURVE_OOS_VALUES)*N_OOS} total evals")

# load configuration
with open("000.input_parameters.json") as f:
    all_configs = json.load(f)
config = all_configs[CONFIG_NAME]
globals().update(config)  # Assign all keys as variables

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
surf_filename = TEST_DIR / config["surface_filename"]

# Directory for output
out_dir_path = (f"output_stage_two_optimization_stochastic_{CONFIG_NAME}_"
                f"{N_SAMPLES}nsamp_{RUN_MODE}_curves")
if MAXITER != 2000:
    out_dir_path += f"_{MAXITER/1000}kiter"
proc0_print(out_dir_path)
OUT_DIR = Path(out_dir_path)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUB_DIR = OUT_DIR / "Non-VTK_Data"
SUB_DIR.mkdir(parents=True, exist_ok=True)

SUB_PERT_DIR = OUT_DIR / "Pert_Data"
SUB_PERT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the boundary magnetic surface; errors break symmetries, so consider the full torus
nphi = 64
ntheta = 16
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
base_curves_init = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
curves_to_vtk(base_curves_init, OUT_DIR / f"base_curves_init")
base_curves = base_curves_init

base_currents = [Current(CURRENT_BASE) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

curves = [c.curve for c in coils]
currents = [c.current for c in coils]
curves_to_vtk(curves, OUT_DIR / f"curves_init_{loop_label}")

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}

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

#######################################################
# Training samples: geometry (curve-shape) errors only
#######################################################

rg = Generator(PCG64DXSM(SEED_TRAIN))
sampler = GaussianSampler(curves[0].quadpoints, SIGMA_CURVE, L_CURVE, n_derivs=1)
Jfs = []
curves_pert = []
proc0_print("Starting N_SAMPLE LOOP")
for i in range(N_SAMPLES):
    # 'systematic' error: applied to the base curves, so the symmetries carry it through
    base_curves_perturbed = [CurvePerturbed_jsonfix(c, PerturbationSample(sampler, randomgen=rg))
                             for c in base_curves]
    coils_sym = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
    # 'statistical' error: independent on each final coil
    coils_pert = [Coil(CurvePerturbed_jsonfix(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current)
                  for c in coils_sym]
    curves_pert.append([c.curve for c in coils_pert])
    bs_pert = BiotSavart(coils_pert)
    Jfs.append(SquaredFlux(s, bs_pert))

for k in range(min(len(curves_pert), 15)):
    curves_to_vtk(curves_pert[k], SUB_PERT_DIR / f"curves_pert_n_sample_{k}")

Jmpi = MPIObjective(Jfs, comm_world, needs_splitting=True)

JF = Jmpi \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), LENGTH_THRESHOLD, "max") \
    + CC_WEIGHT * Jccdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs) \
    + ARCLENGTH_WEIGHT * sum(Jals) \
    + CS_WEIGHT * Jcsdist \
    + LINK_WEIGHT * linkNum

iteration_counter = 0
last_outstr = ""


def fun(dofs):
    global iteration_counter, last_outstr
    iteration_counter += 1
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jmpi.J()
    currents_now = [c.get_value() for c in base_currents]
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"Iteration {iteration_counter}/{MAXITER}-----\n"
    outstr += f"currents: {currents_now}\n"
    outstr += f"J={J:.1e}, ⟨Jf⟩={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J_.J():.1f}" for J_ in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J_.J():.1f}" for J_ in Jmscs)
    outstr += (f", Len=sum([{cl_string}])={sum(J_.J() for J_ in Jls):.1f}, ϰ=[{kap_string}], "
               f"∫ϰ²/L>=[{msc_string}], C-C-Sep={Jccdist.shortest_distance():.2f}")
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

proc0_print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")

iteration_counter = 0

res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 400}, tol=1e-15)
proc0_print("--------------------------------JF.x shape after opt:", JF.x.shape)
alen_string = ", ".join([f"{np.max(c.incremental_arclength())/np.min(c.incremental_arclength())-1:.2e}"
                         for c in base_curves])
proc0_print(f"Final arclength variation max(|ℓ|)/min(|ℓ|) - 1=[{alen_string}]")

proc0_print("""
################################################################################
### Evaluate the obtained coils ################################################
################################################################################
""")

curves_to_vtk(curves, OUT_DIR / f"curves_opt_{loop_label}")
curves_to_vtk(base_curves, OUT_DIR / f"base_curves_opt_{loop_label}")
bs.save(OUT_DIR / f"biot_savart_opt_{loop_numerical_data_label}.json")

bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) * s_plot.unitnormal(), axis=2)[:, :, None]}
s_plot.to_vtk(OUT_DIR / f"surf_opt_{loop_label}", extra_data=pointData)

bs.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
avg_BdotN_over_B = BdotN / bs.AbsB().mean()
Jf.x = res.x
Jf_det = Jf.J()
grad_norm = np.linalg.norm(JF.dJ())
Jmpi_final = Jmpi.J()

#######################################################
# Out-of-sample sigma sweep
#######################################################

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.size

n_oos_sigma = len(SIGMA_CURVE_OOS_VALUES)
# Raw J_b samples: shape (n_oos_sigma, N_OOS). Assembled on rank 0.
squared_flux_data = np.full((n_oos_sigma, N_OOS), np.nan)

# Split the N_OOS draws across ranks. Each rank owns a strided slice so the
# per-rank workload is balanced; results are gathered and re-ordered on rank 0.
local_idx = np.arange(mpi_rank, N_OOS, mpi_size)

for m, sigma_oos in enumerate(SIGMA_CURVE_OOS_VALUES):
    proc0_print(f"--- OOS sigma {m+1}/{n_oos_sigma}: sigma_curve_OOS = {sigma_oos:.4e} m ---")

    sampler_oos = GaussianSampler(curves[0].quadpoints, sigma_oos, L_CURVE, n_derivs=1)
    local_vals = np.zeros(local_idx.size)

    for k, i in enumerate(local_idx):
        # Re-seed per draw so draw i is the SAME underlying realization at every
        # sigma_OOS and in every array task (common random numbers).
        rg_oos = Generator(PCG64DXSM([SEED_OOS, int(i)]))

        # systematic coil-shape error (carried through the symmetries)
        base_curves_perturbed = [CurvePerturbed_jsonfix(c, PerturbationSample(sampler_oos, randomgen=rg_oos))
                                 for c in base_curves]
        coils_sym = coils_via_symmetries(base_curves_perturbed, base_currents, s.nfp, True)
        # statistical coil-shape error, independent per coil
        coils_pert = [Coil(CurvePerturbed_jsonfix(c.curve, PerturbationSample(sampler_oos, randomgen=rg_oos)),
                           c.current)
                      for c in coils_sym]

        bs_pert = BiotSavart(coils_pert)
        bs_pert.set_points(s.gamma().reshape((-1, 3)))
        local_vals[k] = SquaredFlux(s, bs_pert).J()

        if N_VTK_SAMPLES and mpi_rank == 0 and i < N_VTK_SAMPLES:
            curves_to_vtk([c.curve for c in coils_pert],
                          SUB_PERT_DIR / f"curves_opt_pert_oos_sig{m}_sample_{i}")

        # free the perturbed field before the next draw
        del bs_pert, coils_pert, coils_sym, base_curves_perturbed

        if N_OOS >= 10 and (k+1) % max(1, (local_idx.size // 10)) == 0:
            proc0_print(f"  rank0 finished {k+1}/{local_idx.size} of its OOS draws")

    # gather this sigma's results
    all_idx = mpi_comm.gather(local_idx, root=0)
    all_vals = mpi_comm.gather(local_vals, root=0)
    if mpi_rank == 0:
        for idx_chunk, val_chunk in zip(all_idx, all_vals):
            squared_flux_data[m, idx_chunk] = val_chunk
        proc0_print(f"  sigma_OOS={sigma_oos:.4e}: <J_b>={np.mean(squared_flux_data[m]):.3e}, "
                    f"p95={np.percentile(squared_flux_data[m], 95):.3e}")

squared_flux_data = mpi_comm.bcast(squared_flux_data, root=0)

#######################################################
# Save
#######################################################

oos_mean = np.mean(squared_flux_data, axis=1)
oos_std = np.std(squared_flux_data, axis=1)
oos_p95 = np.percentile(squared_flux_data, 95, axis=1)

main_results_str = f"Training sigma_curve                       : {SIGMA_CURVE:.4e} m\n"
main_results_str += f"Flux Objective for exact coils            : {Jf_det:.3e}\n"
main_results_str += f"Objective Gradient (||∇J||)               : {grad_norm:.3e}\n"
main_results_str += f"Mean Flux Objective across perturbed coils: {Jmpi_final:.3e}\n"
main_results_str += f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}\n"
for m, sigma_oos in enumerate(SIGMA_CURVE_OOS_VALUES):
    main_results_str += (f"  sigma_OOS={sigma_oos:.4e} m -> <J_b>={oos_mean[m]:.3e}, "
                         f"std={oos_std[m]:.3e}, p95={oos_p95[m]:.3e}, "
                         f"Q={Jf_det/oos_mean[m]:.3f}\n")

proc0_print(main_results_str)

if mpi_rank == 0:
    np.savez(
        OUT_DIR / f"results_mesh_{loop_numerical_data_label}.npz",
        # --- mesh coordinates ---
        sigma_curve_train=SIGMA_CURVE,
        sigma_curve_oos=SIGMA_CURVE_OOS_VALUES,
        L_curve=L_CURVE,
        train_index=slurm_array_int,
        sigma_curve_train_grid=SIGMA_CURVE_TRAIN_VALUES,
        # --- raw OOS samples, shape (n_sigma_oos, N_OOS) ---
        perturbed_sq_flux_data=squared_flux_data,
        # --- convenience reductions (recomputable from the raw array) ---
        oos_mean=oos_mean,
        oos_std=oos_std,
        oos_p95=oos_p95,
        # --- deterministic / training-side quantities ---
        sq_flux_value=Jf_det,
        mean_train_flux=Jmpi_final,
        gradient=grad_norm,
        avg_BdotN_over_B=avg_BdotN_over_B,
        # --- run metadata ---
        saved_parameter=np.array(save_param),
        n_samples=N_SAMPLES,
        n_oos=N_OOS,
        maxiter=MAXITER,
        order=order,
        config_name=CONFIG_NAME,
        seed_train=SEED_TRAIN,
        seed_oos=SEED_OOS,
        pert_flags=np.array([PERT_CURVE, PERT_CURRENT, PERT_CENTROID, PERT_ORIENTATION]),
        dofs_opt=res.x,
    )

    with open(SUB_DIR / 'main_results.txt', 'a') as f:
        f.write(f" Run {loop_label}: \n" + main_results_str)
    with open(SUB_DIR / 'objective_func_values.txt', 'a') as f:
        f.write(f"\n Run {loop_label}: \n" + last_outstr)

    save_vars = ['SIGMA_CURVE', 'L_CURVE', 'N_SAMPLES', 'N_OOS', 'MAXITER', 'order',
                 'CONFIG_NAME', 'SEED_TRAIN', 'SEED_OOS']
    params = {
        'script_variables': {name: (globals()[name].tolist() if isinstance(globals()[name], np.ndarray)
                                    else globals()[name])
                             for name in save_vars if name in globals()},
        'sigma_curve_oos_values': SIGMA_CURVE_OOS_VALUES.tolist(),
        'json_variables': {k: v for k, v in config.items()},
    }
    with open(SUB_DIR / f'input_parameters_save_{loop_numerical_data_label}.json', 'w') as f:
        json.dump(params, f, indent=1)

    end = time.time()
    time_taken = f"Took {(end - start):.2f} for run {loop_label}."
    with open(SUB_DIR / 'run_times.txt', 'a') as f:
        f.write(time_taken + "\n")
    proc0_print(f"Total time taken: {(end - start):.2f} seconds")
