#!/usr/bin/env python
"""
auglag_qh_reactorscale.py
===============

This script performs coil optimization for the reactor-scale Landreman-Paul 2021 precise QH configuration 
using the Augmented Lagrangian Method (ALM).

In order to reproduce the coilsets of the paper the following thresholds/parameters should be set:
1) 4 coils solution (#1)
    ncoils : 4
    LENGTH_TARGET = 4*40 
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 0.8 
    CS_THRESHOLD = 1
    CURVATURE_THRESHOLD = 1 
    MSC_THRESHOLD = 0.1
    FORCE_THRESHOLD = 10 # units of MN/m

2) 5 coils solution (#2)
    ncoils_choice : 6
    LENGTH_TARGET = 5*35.56 
    FLUX_THRESHOLD = 1e-15
    CC_THRESHOLD = 1.1 
    CS_THRESHOLD = 1.6
    CURVATURE_THRESHOLD = 0.88 
    MSC_THRESHOLD = 0.08
    FORCE_THRESHOLD = 10 # units of MN/m

"""

import numpy as np
import os
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo import LinkingNumber
from simsopt.geo import CurveLength, CurveCurveDistance, \
    LpCurveCurvature, CurveSurfaceDistance, MeanSquaredCurvature, curves_to_vtk
from augmented_lagrangian import *
from simsopt.field import BiotSavart
from simsopt.field.force import LpCurveForce, coil_force
from simsopt.field import Current, coils_via_symmetries #, regularization_circ
from pathlib import Path
from simsopt.util import in_github_actions
import time

# Define the test directory
TEST_DIR = '/scratch/projects/kaptanoglulab/VG/simsopt/examples/2_Intermediate/inputs/'

# Define the filename

filename = TEST_DIR + 'input.LandremanPaul2021_QH_reactorScale_lowres'

# Set some parameters -- warning this is super low resolution!
if in_github_actions:
    nphi = 4
    ntheta = 4
    MAXITER = 50
    MAXITER_lag = 5
else:
    # Define the number of phi and theta points
    nphi = 32
    ntheta = 32
    MAXITER =  1000 # 1500 for high-resolution
    MAXITER_lag = 15  # 30 for high-resolution

# Define the surface
s = SurfaceRZFourier.from_vmec_input(
    filename,
    range="half period",
    nphi=nphi,
    ntheta=ntheta)

# Define a high-resolution, full-torus surface for plotting
qphi = 4 * nphi
qtheta = 4 * ntheta
quadpoints_phi = np.linspace(0, 1, qphi)
quadpoints_theta = np.linspace(0, 1, qtheta)
s_plot = SurfaceRZFourier.from_vmec_input(
    filename,
    range="full torus",
    quadpoints_phi=quadpoints_phi,
    quadpoints_theta=quadpoints_theta)

# Define the upper and lower bounds for the constraints
LENGTH_TARGET = 5 * 40  # 4 coils, 40 m per coil
FLUX_THRESHOLD = 1e-15
CC_THRESHOLD = 0.8
CS_THRESHOLD = 1 
CURVATURE_THRESHOLD = 1 
MSC_THRESHOLD = 0.1 
FORCE_THRESHOLD = 10 # units of MN/m

# Define the number of coils, rotation order, and non-planar base curves
R0 = s.x[0]
R1 = 0.5 * s.x[0]

print(float(R0),float(R1))
order = 16
ncoils = 5
curves = create_equally_spaced_curves(
    ncoils, s.nfp, stellsym=s.stellsym, R0=R0, R1=R1, order=order, numquadpoints=128)
total_current = 45642162
base_currents = [Current(total_current / ncoils * 1e-7) * 1e7 for _ in range(ncoils)]
base_currents[0].fix_all

# Define the output directory   
OUT_DIR = (f"./output_paper/qh_ncoils{ncoils}_curvature{CURVATURE_THRESHOLD}_msc{MSC_THRESHOLD}_" + \
           f"force{FORCE_THRESHOLD}_flux{FLUX_THRESHOLD}_length{LENGTH_TARGET}_" + \
           f"cc{CC_THRESHOLD}_cs{CS_THRESHOLD}/")
os.makedirs(OUT_DIR, exist_ok=True)

base_curves = curves[:ncoils]
a = 0.15  # radius of the coil
#regularizations = [regularization_circ(a) for _ in range(ncoils)]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, s.stellsym) #,regularizations=regularizations)
base_coils = coils[:ncoils]
curves = [c.curve for c in coils]
currents = [c.current for c in coils]
print("Number of coils:", len(coils))

# Save the biot-savart field data
bs = BiotSavart(coils)
curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "coils_init_qh_reactorscale")
bs.set_points(s_plot.gamma().reshape((-1, 3))) 
pointData = {"B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                               s_plot.unitnormal(), axis=2)[:, :, None] / bs.AbsB().reshape((qphi, qtheta, 1)),
             "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_init_qh_reactorscale", extra_data=pointData)

# Define the individual terms objective function:
bs.set_points(s.gamma().reshape((-1, 3)))
Jf = SquaredFlux(s, bs, definition="normalized", threshold=FLUX_THRESHOLD)
Jls = [CurveLength(c) for c in base_curves]
Jl = sum(QuadraticPenalty(jj, LENGTH_TARGET, "max") for jj in Jls)
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jlink = LinkingNumber(curves, downsample=2)
#Jforce = LpCurveForce(base_coils, coils, p=2.0, threshold=FORCE_THRESHOLD)

#force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in base_coils]
#print("Forces:")
#print(",".join(f"{f:.2e}" for f in force))

# Main optimization function
f = None

# Constraint list
c_list = [ Jf,
        Jccdist, 
        Jcsdist, 
        QuadraticPenalty(sum(Jls), LENGTH_TARGET, "max"), 
        sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs),
        sum(Jcs), 
        Jlink,
        #   Jforce
]

print('Initial normalized flux:', Jf.J())
print('Initial CS-Sep constraint:', Jcsdist.J())
print('Initial CS-sep minimum distance:', Jcsdist.shortest_distance())
print('Initial CC-Sep constraint:', Jccdist.J())
print('Initial CC-sep minimum distance:', Jccdist.shortest_distance())
print('Initial Len constraint:', Jl.J())
print('Initial Curv constraint:', sum(Jcs).J())
print('Initial Link constraint:', Jlink.J())
print('Initial Max Curvatures:', [np.max(c.kappa()) for c in base_curves])
print('Initial Lengths:', [CurveLength(c).J() for c in base_curves], sum(Jls).J())

start_time = time.time()
x, fnc, lag_mul = augmented_lagrangian_method(f=f,
    equality_constraints=c_list,
    tau=10,
    MAXITER=MAXITER,
    MAXITER_lag=MAXITER_lag,
    grad_tol=1e-8,
    c_tol=1e-8,
)

bs.save(OUT_DIR + "biot_savart_optimized_auglag_qh_reactorscale.json")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print('Final normalized flux:', Jf.J())
print('Final CS-Sep constraint:', Jcsdist.J())
print('Final CS-sep minimum distance:', Jcsdist.shortest_distance())
print('Final CC-Sep constraint:', Jccdist.J())
print('Final CC-sep minimum distance:', Jccdist.shortest_distance())
print('Final Len constraint:', Jl.J())
print('Final Curv constraint:', sum(Jcs).J())
print('Final Link constraint:', Jlink.J())
print('Final Max Curvatures:', [np.max(c.kappa()) for c in base_curves])
print('Final Lengths:', [CurveLength(c).J() for c in base_curves], sum(Jls).J())
print('Final Mean Squared Curvature', [MeanSquaredCurvature(c).J() for c in base_curves])
# print('Final Force constraint:', Jforce.J())
#force = [np.max(np.linalg.norm(coil_force(c, coils), axis=1)) for c in base_coils]
#print("Forces:")
#print(",".join(f"{f:.2e}" for f in force))
curves_to_vtk(curves, OUT_DIR + "coils_optimized_auglag_qh_reactorscale")
bs.set_points(s_plot.gamma().reshape((-1, 3)))
pointData = {"B_N": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None],
        "B_N/|B|": np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)),
        "modB": bs.AbsB().reshape((qphi, qtheta, 1))}
s_plot.to_vtk(OUT_DIR + "surf_optimized_auglag_qh_reactorscale", extra_data=pointData)
max_BdotN_overB = np.max(np.sum(bs.B().reshape((qphi, qtheta, 3)) *
                        s_plot.unitnormal(), axis=2)[:, :, None] /
        bs.AbsB().reshape((qphi, qtheta, 1)))
bs.set_points(s.gamma().reshape((-1, 3)))
BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
avg_BdotN_over_B = BdotN / bs.AbsB().mean()
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print(f"<B_N>/<|B|> = {avg_BdotN_over_B:.2e}, Max BdotN/|B| = {max_BdotN_overB:.2e}")
print("FINAL LAGRANGE MULTIPLIERS:", lag_mul)
print("--------------------------------------------------------------------------------------------------------------------------------------------")
print("Final NORMALIZED SQUARED FLUX:", Jf.J())
print('FINISHED OPTIMIZATION')