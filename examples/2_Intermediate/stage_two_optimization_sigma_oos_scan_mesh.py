#!/usr/bin/env python

r"""
DETERMINISTIC baseline counterpart to stage_two_stochastic_sigma_mesh.py.

Coils are optimized against the UNPERTURBED flux objective (no stochastic
training, no sample average), then evaluated out-of-sample over the SAME
sigma_OOS sweep used by the stochastic runs.

This produces the reference row of the (sigma_train, sigma_OOS) mesh, i.e. the
sigma_train -> 0 limit. The npz schema matches the stochastic script exactly so
the files stack together for plotting; sigma_curve_train = 0.0 and
train_index = -1 mark this as the deterministic row.

The OOS RNG seeding is IDENTICAL to the stochastic script (per-draw
PCG64DXSM([SEED_OOS, i])), so draw i is the same underlying coil-shape
realization here as in every stochastic task. The deterministic-vs-stochastic
comparison is therefore paired, not just distributionally matched.

Objective:

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)
        + ARCLENGTH_WEIGHT * ArclengthVariation

No SLURM array needed -- this is a single job.
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
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt.util import in_github_actions, proc0_print
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.field.selffield import regularization_circ
from stochastic_helper_functions import *

start = time.time()

job_id = int(os.environ.get("SLURM_JOB_ID", 0))
proc0_print(f"SLURM job ID: {job_id}")

#######################################################
# Specify input parameters.
#######################################################

# Number of Fourier modes describing each Cartesian component of each coil:
order = 16

CURRENT_BASE = 1e5

# --- Geometry-only perturbations, OUT OF SAMPLE ONLY -------------------------
# Training is deterministic. These flags describe the OOS evaluation only, and
# are carried into the npz so the plotting side can match runs up.
PERT_CURVE = True
PERT_CURRENT = False
PERT_CENTROID = False
PERT_ORIENTATION = False

# Length scale of the Gaussian coil-shape errors (must match stochastic runs)
L_CURVE = 0.5

# Pick which configuration you want
CONFIG_NAME = "QH5"

RUN_MODE = 'deterministic_oos_scan'

# ---------------- Mesh axes --------------------------------------------------
# This script IS the sigma_train = 0 row.
SIGMA_CURVE = 0.0

# Out-of-sample sigma sweep -- must match SIGMA_CURVE_OOS_VALUES in the
# stochastic script for the rows to stack into a mesh.
# 10 points over 1mm-1cm = exact 1mm spacing.
SIGMA_CURVE_OOS_VALUES = np.linspace(1e-3, 1e-2, 10)

# Training grid of the stochastic runs, recorded for bookkeeping only.
SIGMA_CURVE_TRAIN_VALUES = np.linspace(1e-3, 1e-2, 10)

# Number of out-of-sample perturbation draws at EACH sigma_OOS
N_OOS = 1000

# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 2000

# Must match the stochastic script for paired (common random number) OOS draws.
SEED_OOS = 1

# Save VTK for a handful of perturbed OOS coil sets (0 disables)
N_VTK_SAMPLES = 5

#######################################################
# End of input parameters.
#######################################################

loop_label = "deterministic"
save_param = (0.0, L_CURVE)
loop_numerical_data_label = "det"

proc0_print("Running DETERMINISTIC optimization + OOS sigma scan (geometry errors only)")
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
out_dir_path = f"output_stage_two_optimization_deterministic_{CONFIG_NAME}_{RUN_MODE}_curves"
if MAXITER != 2000:
    out_dir_path += f"_{MAXITER/1000}kiter"
proc0_print(out_dir_path)
OUT_DIR = Path(out_dir_path)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUB_DIR = OUT_DIR / "Non-VTK_Data"
SUB_DIR.mkdir(parents=True, exist_ok=True)

SUB_PERT_DIR = OUT_DIR / "Pert_Data"
SUB_PERT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the boundary magnetic surface. The optimization itself is
# stellarator-symmetric, but the OOS errors break symmetry, so we use the full
# torus throughout to keep the objective definition identical to the stochastic
# script -- otherwise J_b values would not be directly comparable.
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

# Deterministic objective: the flux term is the unperturbed SquaredFlux, i.e.
# exactly the N_SAMPLES -> single-unperturbed-coil-set limit of MPIObjective.
# Every regularization term and weight is unchanged from the stochastic script.
JF = Jf \
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
    jf = Jf.J()
    currents_now = [c.get_value() for c in base_currents]
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"Iteration {iteration_counter}/{MAXITER}-----\n"
    outstr += f"currents: {currents_now}\n"
    outstr += f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
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

#######################################################
# Out-of-sample sigma sweep
#######################################################

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.rank
mpi_size = mpi_comm.size

# NOTE: with no MPIObjective the optimization above runs redundantly on every
# rank -- that is intentional, it leaves every rank holding the same optimized
# coils so the OOS draws below can be split. If you want the optimization to be
# cheap, run this with fewer ranks than the stochastic script; the eval phase is
# what benefits from the rank count here.

n_oos_sigma = len(SIGMA_CURVE_OOS_VALUES)
# Raw J_b samples: shape (n_oos_sigma, N_OOS). Assembled on rank 0.
squared_flux_data = np.full((n_oos_sigma, N_OOS), np.nan)

local_idx = np.arange(mpi_rank, N_OOS, mpi_size)

for m, sigma_oos in enumerate(SIGMA_CURVE_OOS_VALUES):
    proc0_print(f"--- OOS sigma {m+1}/{n_oos_sigma}: sigma_curve_OOS = {sigma_oos:.4e} m ---")

    sampler_oos = GaussianSampler(curves[0].quadpoints, sigma_oos, L_CURVE, n_derivs=1)
    local_vals = np.zeros(local_idx.size)

    for k, i in enumerate(local_idx):
        # Same per-draw seeding as the stochastic script -> paired comparison.
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

        del bs_pert, coils_pert, coils_sym, base_curves_perturbed

        if N_OOS >= 10 and (k+1) % max(1, (local_idx.size // 10)) == 0:
            proc0_print(f"  rank0 finished {k+1}/{local_idx.size} of its OOS draws")

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

main_results_str = "Training                                  : DETERMINISTIC (sigma = 0)\n"
main_results_str += f"Flux Objective for exact coils            : {Jf_det:.3e}\n"
main_results_str += f"Objective Gradient (||∇J||)               : {grad_norm:.3e}\n"
main_results_str += f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}\n"
for m, sigma_oos in enumerate(SIGMA_CURVE_OOS_VALUES):
    main_results_str += (f"  sigma_OOS={sigma_oos:.4e} m -> <J_b>={oos_mean[m]:.3e}, "
                         f"std={oos_std[m]:.3e}, p95={oos_p95[m]:.3e}, "
                         f"Q={Jf_det/oos_mean[m]:.3f}\n")

proc0_print(main_results_str)

if mpi_rank == 0:
    np.savez(
        OUT_DIR / f"results_mesh_{loop_numerical_data_label}.npz",
        # --- mesh coordinates (sigma_train = 0 row) ---
        sigma_curve_train=0.0,
        sigma_curve_oos=SIGMA_CURVE_OOS_VALUES,
        L_curve=L_CURVE,
        train_index=-1,
        sigma_curve_train_grid=SIGMA_CURVE_TRAIN_VALUES,
        deterministic=True,
        # --- raw OOS samples, shape (n_sigma_oos, N_OOS) ---
        perturbed_sq_flux_data=squared_flux_data,
        # --- convenience reductions (recomputable from the raw array) ---
        oos_mean=oos_mean,
        oos_std=oos_std,
        oos_p95=oos_p95,
        # --- deterministic / training-side quantities ---
        sq_flux_value=Jf_det,
        mean_train_flux=Jf_det,
        gradient=grad_norm,
        avg_BdotN_over_B=avg_BdotN_over_B,
        # --- run metadata ---
        saved_parameter=np.array(save_param),
        n_samples=1,
        n_oos=N_OOS,
        maxiter=MAXITER,
        order=order,
        config_name=CONFIG_NAME,
        seed_train=-1,
        seed_oos=SEED_OOS,
        pert_flags=np.array([PERT_CURVE, PERT_CURRENT, PERT_CENTROID, PERT_ORIENTATION]),
        dofs_opt=res.x,
    )

    with open(SUB_DIR / 'main_results.txt', 'a') as f:
        f.write(f" Run {loop_label}: \n" + main_results_str)
    with open(SUB_DIR / 'objective_func_values.txt', 'a') as f:
        f.write(f"\n Run {loop_label}: \n" + last_outstr)

    save_vars = ['SIGMA_CURVE', 'L_CURVE', 'N_OOS', 'MAXITER', 'order',
                 'CONFIG_NAME', 'SEED_OOS']
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