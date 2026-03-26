#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri November 08 13:42:44 2024 

@author: linecolin

Dedalus script based on the example of convection in a spherical shell (https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_shell_convection.html).
It has been modified for convection in the Moon:
    - Infinite Prandtl number
    - Phase change at the interface
    - Cooling of the core
See the UserGuide for more details.

It takes a param.toml file as input, which allows the following parameters to be varied: 
    - GAMMA: aspect ratio
    - RAYLEIGH: Rayleigh number
    - PHI: Phase change number
    - FACTOR: for the curvature of the temperature profile
    - GRAVITY: linearly radius dependant gravity (true or false)
    - DT: initial and maximum time step
    - STOP: stop simulation time
    - Nr: radial resolution
    - Nphi: resolution
    - Ntheta: resolution
    - SAFETY: safety for CFL number
    - NOISE: initial noise
    - BC: thermal coundary conditions
    - TempProf: initial temperature profile
    - DIR_SAVE: Directory to save datas


To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shell_convection.py param.toml 
"""

import numpy as np
import tomli
import logging
import time
logger = logging.getLogger(__name__)
debut = time.time()
print(f"start {debut}")
import os
import shutil
import glob
from pathlib import Path
import sys

def load_latest_checkpoint(solver, path_repository, parameters):
    checkpoint_files = glob.glob(str(path_repository / 'checkpoints/checkpoints_s*.h5'))
    
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint file found in the specified directory.")

    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_s')[-1].split('.h5')[0]))
    print(f" : {latest_checkpoint}")
    
    write, initial_timestep = solver.load_state(latest_checkpoint)
    return write, initial_timestep


def run(parameters, restart):
    import dedalus.public as d3

    # Parameters:
    with Path(parameters).open("rb") as file_in:
        par = tomli.load(file_in)
    gamma = par['GAMMA']
    Ri = gamma / (1 - gamma)
    Ro = 1 + gamma / (1 - gamma)
    print(Ri, Ro)
    Rm = Ri * par['FACTOR']
    Nphi, Ntheta, Nr = par['NPhi'], par['NTheta'], par['Nr']
    Rayleigh = par['RAYLEIGH']
    PHIC = par['PHI']
    dealias = 3/2
    timestepper = d3.SBDF1
    initial_timestep = par['DT']
    if not restart:
        max_timestep = initial_timestep
    else:
        max_timestep = par['new_DT']
    stop_sim_time = par['STOP']
    dtype = np.float64
    mesh = None
    safety = par['SAFETY']
    g_uniform = par['GRAVITY']
    BC = par['BC']
    TempProf = par['TempProf']
    noise = par['NOISE']

    Ra_print = format(Rayleigh, '.1e')
    Ra_print = Ra_print.replace('.', '-')
    safety_print = format(safety, '.1f')
    safety_print = safety_print.replace('.', '-')
    dt_print = format(initial_timestep, '.0e')
    dt_print = dt_print.replace('.', '-')
    gamma_print = format(gamma, '.2f')
    gamma_print = gamma_print.replace('.', '-')

    if PHIC == False:
        phi_print = 'inf'
    else:
        phi_print = format(PHIC, '.1e')
        phi_print = phi_print.replace('.', '-')

    if g_uniform == True:
        g_print = 'cst'
    else:
        g_print = 'r'
    
    if TempProf == 'Parabolic':
        h_print = format(par['FACTOR'], '.1f')
        h_print = h_print.replace('.', '-')
        Temp_print = f'Parabolic_h_{h_print}'
    else:
        Temp_print = TempProf
    
    save_name = f"{par['DIR_SAVE']}/Rayleigh_{Ra_print}_Phi_{phi_print}_gam_{gamma_print}_T_{Temp_print}_BC_{BC}_g_{g_print}_safety_{safety_print}_dt_{dt_print}_res_{Nphi}-{Ntheta}-{Nr}"
    
    print(save_name)
    path_repository = Path(save_name)

    # Bases
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')

    dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
    shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
    sphere = shell.outer_surface

    # Fields
    p = dist.Field(name='p', bases=shell)
    b = dist.Field(name='b', bases=shell)
    u = dist.VectorField(coords, name='u', bases=shell)
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=sphere)
    tau_b2 = dist.Field(name='tau_b2', bases=sphere)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)
    t = dist.Field()

    # Substitutions

    phi, theta, r = dist.local_grids(shell)
    er = dist.VectorField(coords, bases=shell.radial_basis)
    er['g'][2] = 1
    rvec = dist.VectorField(coords, bases=shell.radial_basis)
    rvec['g'][2] = r # rvec = r*er
    lift_basis = shell.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + rvec*lift(tau_b1) # First-order reduction
    
    strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
    shear_stress_Ro = d3.angular(d3.radial(strain_rate(r=Ro), index=1))  
    shear_stress_Ri = d3.angular(d3.radial(strain_rate(r=Ri), index=1))

    #### Problem ####
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals(), time=t)
    # mass conservation
    problem.add_equation("trace(grad_u) + tau_p = 0")
    # heat conservation
    problem.add_equation("dt(b) - div(grad_b) + lift(tau_b2) = - u@grad(b)")
    if g_uniform == True:  # NS with cst gravity
        problem.add_equation("-div(grad_u) + grad(p) - Rayleigh*b*er + lift(tau_u2) = 0")
    else: #NS with gravity linearly dependent on radius
        problem.add_equation("-div(grad_u) + grad(p) - Rayleigh*b*rvec/Ro + lift(tau_u2) = 0")

    #### Thermal boundary conditions ###
    problem.add_equation("b(r=Ro) = 0") # top temperature
    if BC == 'Dirichlet': 
        print("Dirichlet BC")
        problem.add_equation("b(r=Ri) = 1")
    else:
        print("CooreCooling BC")
        problem.add_equation("radial(grad(b)(r=Ri)) - dt(b)(r=Ri)*Ri/3 = 0", condition="(nphi==0) and (ntheta==0)")  # equation for core cooling, true for mode 0 but for mode >=1 homogeneous temperature
        problem.add_equation("b(r=Ri)=0", condition="(nphi!=0) or (ntheta!=0)") # T=0 for modes greater than 0
    #### Mechanical boundary conditions ####

    if PHIC==False: # No phase change
        problem.add_equation("(u@er)(r=Ro) = 0")
        problem.add_equation("integ(p)=0") # Pressure gauge
    else: # Phase change
        problem.add_equation("d3.radial(d3.radial(strain_rate(r=Ro))) + PHIC*radial(u(r=Ro)) - p(r=Ro) = 0")
        problem.add_equation("d3.Average(p(r=Ro),coords.S2coordsys) = 0") # Pressure gauge
    problem.add_equation("shear_stress_Ro = 0")
    
    problem.add_equation("(u@er)(r=Ri) = 0")
    problem.add_equation("shear_stress_Ri = 0")

    
    

    #### Solver ####
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time
    
    #### Initial conditions ####
    if not restart:
        b.fill_random('g', seed=42, distribution='normal', scale=noise) # Random noise
        b['g'] *= (r - Ri) * (Ro - r) # Damp noise at walls
        if TempProf == "Conductive":
            b['g'] += (Ri - Ri*Ro/r) / (Ri - Ro) # steady-state solution
        elif TempProf == "RefProf":
            b['g'] += (Ro**3-r**3)/(Ro**3-Ri**3) * (Rm**3-Ri**3)/(Rm**3-r**3) # Temperature profile from thermal evolution
        file_handler_mode = 'overwrite'
        initial_timestep = max_timestep
        if not os.path.exists(path_repository):
                os.makedirs(path_repository, exist_ok=True)
    else:
        print(restart)
        write, initial_timestep = solver.load_state(path_repository/'checkpoints/checkpoints_s1.h5')
        initial_timestep = max_timestep
        file_handler_mode = 'append'


    snapshots_dir = path_repository / 'snapshots'
    scalars_dir = path_repository / 'scalars'
    analysis_dir = path_repository / 'diagnostics'
    checkpoints_dir = path_repository / 'checkpoints'

    #### Analysis ####
    flux = er @ (-d3.grad(b) + u*b)
    ur = u@er
    radius = rvec@er
    snapshots = solver.evaluator.add_file_handler(snapshots_dir, sim_dt=stop_sim_time/1000, max_writes=1000, mode=file_handler_mode)  # sim_dt=5e-3
    # snapshots.add_task(b(r=(Ri+Ro)/2), scales=dealias, name='bmid')
    # snapshots.add_task(flux(r=Ro), scales=dealias, name='flux_r_outer')
    # snapshots.add_task(flux(r=(Ri+Ro)/2), scales=dealias, name='flux_r_mid')
    # snapshots.add_task(flux(r=Ri), scales=dealias, name='flux_r_inner')
    # snapshots.add_task(flux(phi=0), scales=dealias, name='flux_phi_start')
    snapshots.add_task(b, scales=dealias, name='temperature')
    # snapshots.add_task(b, layout='c', name='temp_c') # coefficients
    snapshots.add_task(flux, scales=dealias, name='flux')
    # snapshots.add_task(flux, layout='c', name='flux_c') # coefficients
    # snapshots.add_task(flux(phi=3*np.pi/2), scales=dealias, name='flux_phi_end')
    snapshots.add_task(d3.Average(flux, coords.S2coordsys), name='average_flux')
    snapshots.add_task(d3.Average(b,coords.S2coordsys), name='average_profile_b')
    snapshots.add_task(d3.Average(u@u,coords.S2coordsys), name='average_velocity')
    #snapshots.add_task(radius, name='rad')

    scalars = solver.evaluator.add_file_handler(scalars_dir, sim_dt=stop_sim_time/1000, max_writes=1000, mode=file_handler_mode) 
    scalars.add_task(d3.Average(flux(r=Ro),coords.S2coordsys), name='meanflux_r_outer')
    scalars.add_task(d3.Average(flux(r=Ri),coords.S2coordsys), name='meanflux_r_inner')
    scalars.add_task(d3.Average(b(r=Ri),coords.S2coordsys), name='temperature_r_inner')
    scalars.add_task(d3.Average(b(r=Ro),coords.S2coordsys), name='temperature_r_outer')
    scalars.add_task(d3.grad(b), name='grad_temperature')
    # scalars.add_task(d3.grad(b(r=Ri)), name='grad_temperature_r_inner')
    scalars.add_task(d3.Average(ur(r=Ro)*ur(r=Ro),coords.S2coordsys), name='rad_sq_ur_outer')
    scalars.add_task(d3.Average(ur(r=Ri)*ur(r=Ri),coords.S2coordsys), name='rad_sq_ur_inner')
    scalars.add_task(d3.Integrate(d3.Average(b,coords.S2coordsys))/(4*np.pi*(Ro**3.-Ri**3.)/3), name='mean_temperature')
    scalars.add_task(d3.Integrate(d3.Average((u@u)**(1/2),coords.S2coordsys))/(4*np.pi*(Ro**3.-Ri**3.)/3), name='mean_rmsvelsq')


    #### Checkpoints ####
    checkpoints = solver.evaluator.add_file_handler(checkpoints_dir, sim_dt=stop_sim_time/500, max_writes=1000, mode=file_handler_mode)
    checkpoints.add_tasks(solver.state)

    #### CFL ####
    # CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=safety, threshold=0.1,
                # max_change=1.5, min_change=.5, max_dt=max_timestep)
    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=safety, threshold=0.1,
                max_change=1., min_change=.2, max_dt=max_timestep)
    CFL.add_velocity(u)

    #### Flow properties ####
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u), name='Re')


    #### Main loop ####
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                max_Re = flow.max('Re')
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
            if np.isnan(max_Re):
                print("Error diverging code")
                break
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Read CSV file with parameters")
    parser.add_argument("filename", type=str, help="Name of the CSV file")
    parser.add_argument("--restart", action="store_true", help="Restart from the last iteration")
    args = parser.parse_args()

    run(args.filename, args.restart)


if __name__ == "__main__":
    main()




def run(parameters, restart):
    import dedalus.public as d3

    # Parameters:
    with Path(parameters).open("rb") as file_in:
        par = tomli.load(file_in)
    gamma = par['GAMMA']
    Ri = gamma / (1 - gamma)
    Ro = 1 + gamma / (1 - gamma)
    print(Ri, Ro)
    Rm = Ri * par['FACTOR']
    Nphi, Ntheta, Nr = par['NPhi'], par['NTheta'], par['Nr']
    Rayleigh = par['RAYLEIGH']
    PHIC = par['PHI']
    dealias = 3/2
    timestepper = d3.SBDF1
    max_timestep = par['DT']
    stop_sim_time = par['STOP']
    dtype = np.float64
    mesh = None
    safety = par['SAFETY']
    g_uniform = par['GRAVITY']
    BC = par['BC']
    TempProf = par['TempProf']
    noise = par['NOISE']

    Ra_print = format(Rayleigh, '.1e')
    Ra_print = Ra_print.replace('.', '-')
    safety_print = format(safety, '.1f')
    safety_print = safety_print.replace('.', '-')
    dt_print = format(max_timestep, '.0e')
    dt_print = dt_print.replace('.', '-')
    gamma_print = format(gamma, '.2f')
    gamma_print = gamma_print.replace('.', '-')

    if PHIC == False:
        phi_print = 'inf'
    else:
        phi_print = format(PHIC, '.1e')
        phi_print = phi_print.replace('.', '-')

    if g_uniform == True:
        g_print = 'cst'
    else:
        g_print = 'r'
    
    if TempProf == 'Parabolic':
        h_print = format(par['FACTOR'], '.1f')
        h_print = h_print.replace('.', '-')
        Temp_print = f'Parabolic_h_{h_print}'
    else:
        Temp_print = TempProf
    
    save_name = f"{par['DIR_SAVE']}/Rayleigh_{Ra_print}_Phi_{phi_print}_gam_{gamma_print}_T_{Temp_print}_BC_{BC}_g_{g_print}_safety_{safety_print}_dt_{dt_print}_res_{Nphi}-{Ntheta}-{Nr}"
    
    print(save_name)
    path_repository = Path(save_name)

    # Bases
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')

    dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
    shell = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
    sphere = shell.outer_surface

    # Fields
    p = dist.Field(name='p', bases=shell)
    b = dist.Field(name='b', bases=shell)
    u = dist.VectorField(coords, name='u', bases=shell)
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=sphere)
    tau_b2 = dist.Field(name='tau_b2', bases=sphere)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=sphere)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=sphere)
    t = dist.Field()

    # Substitutions

    phi, theta, r = dist.local_grids(shell)
    er = dist.VectorField(coords, bases=shell.radial_basis)
    er['g'][2] = 1
    rvec = dist.VectorField(coords, bases=shell.radial_basis)
    rvec['g'][2] = r
    lift_basis = shell.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + rvec*lift(tau_b1) # First-order reduction
    
    strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
    shear_stress_Ro = d3.angular(d3.radial(strain_rate(r=Ro), index=1))  # je ne comprends pas le index
    shear_stress_Ri = d3.angular(d3.radial(strain_rate(r=Ri), index=1))

    # Problem
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals(), time=t)
    problem.add_equation("trace(grad_u) + tau_p = 0") #conservation masse
    problem.add_equation("dt(b) - div(grad_b) + lift(tau_b2) = - u@grad(b)") #conservation  chaleur
    if g_uniform == True:  # NS with cst gravity
        problem.add_equation("-div(grad_u) + grad(p) - Rayleigh*b*er + lift(tau_u2) = 0")
    else: #NS avec gravité qui dépend du rayon Rayleigh*b*er
        problem.add_equation("-div(grad_u) + grad(p) - Rayleigh*b*rvec/Ro + lift(tau_u2) = 0")

    # Boundary conditions
    problem.add_equation("b(r=Ro) = 0") # temp top
    if BC == 'Dirichlet':
        print("Dirichlet BC")
        problem.add_equation("b(r=Ri) = 1")
    else:
        print("CooreCooling BC")
        problem.add_equation("radial(grad(b)(r=Ri)) - dt(b)(r=Ri)*Ri/3 = 0", condition="(nphi==0) and (ntheta==0)")  # equation for core cooling, vrai pour le mode 0 mais pour mode >=1 temperature homogene
        problem.add_equation("b(r=Ri)=0", condition="(nphi!=0) or (ntheta!=0)") # T=0 pour les modes superieurs à 0

    if PHIC==False:
        # problem.add_equation("radial(u(r=Ro)) = 0")   #problem.add_equation("u(r=Ro)@er = 0")  marche pas ...
        problem.add_equation("(u@er)(r=Ro) = 0")
        problem.add_equation("integ(p)=0")
    else:
        problem.add_equation("d3.radial(d3.radial(strain_rate(r=Ro))) + PHIC*radial(u(r=Ro)) - p(r=Ro) = 0")# - d3.Average(p(r=Ro),coords.S2coordsys) ") # - p(r=Ro) le terme p pose souci #= 0 ")# + PHIC*radial(u(r=Ro)) = 0")  # phase change à revoir
        problem.add_equation("d3.Average(p(r=Ro),coords.S2coordsys) = 0")
    problem.add_equation("shear_stress_Ro = 0")


    # problem.add_equation("radial(u(r=Ri)) = 0")   #problem.add_equation("u(r=Ro)@er = 0")  marche pas ...
    problem.add_equation("(u@er)(r=Ri) = 0")
    problem.add_equation("shear_stress_Ri = 0")

    # Pressure gauge
    

    # Solver
    solver = problem.build_solver(timestepper)

    solver.stop_sim_time = stop_sim_time
        # Initial conditions

    if not restart:
        b.fill_random('g', seed=42, distribution='normal', scale=noise) # Random noise
        b['g'] *= (r - Ri) * (Ro - r) # Damp noise at walls
        if TempProf == "Conductive":
            b['g'] += (Ri - Ri*Ro/r) / (Ri - Ro) # steady-state solution
        elif TempProf == "RefProf":
            b['g'] += (Ro**3-r**3)/(Ro**3-Ri**3) * (Rm**3-Ri**3)/(Rm**3-r**3)
        else:
            h = par['FACTOR']
            b['g'] += f(r, h)
        file_handler_mode = 'overwrite'
        initial_timestep = max_timestep
        if not os.path.exists(path_repository):
                os.makedirs(path_repository, exist_ok=True)
    else:
        write, initial_timestep = solver.load_state(path_repository/'checkpoints/checkpoints_s2.h5')
        # initial_timestep = par['DT2']
        # max_timestep = par['DT2']
        file_handler_mode = 'append'


    snapshots_dir = path_repository / 'snapshots'
    scalars_dir = path_repository / 'scalars'
    analysis_dir = path_repository / 'diagnostics'
    checkpoints_dir = path_repository / 'checkpoints'

    # Analysis
    flux = er @ (-d3.grad(b) + u*b)
    ur = u@er
    radius = rvec@er
    snapshots = solver.evaluator.add_file_handler(snapshots_dir, sim_dt=stop_sim_time/1000, max_writes=1000, mode=file_handler_mode)  # sim_dt=5e-3
    # snapshots.add_task(b(r=(Ri+Ro)/2), scales=dealias, name='bmid')
    # snapshots.add_task(flux(r=Ro), scales=dealias, name='flux_r_outer')
    # snapshots.add_task(flux(r=(Ri+Ro)/2), scales=dealias, name='flux_r_mid')
    # snapshots.add_task(flux(r=Ri), scales=dealias, name='flux_r_inner')
    # snapshots.add_task(flux(phi=0), scales=dealias, name='flux_phi_start')
    snapshots.add_task(b, scales=dealias, name='temperature')
    # snapshots.add_task(b, layout='c', name='temp_c') # coefficients
    snapshots.add_task(flux, scales=dealias, name='flux')
    # snapshots.add_task(flux, layout='c', name='flux_c') # coefficients
    # snapshots.add_task(flux(phi=3*np.pi/2), scales=dealias, name='flux_phi_end')
    snapshots.add_task(d3.Average(flux, coords.S2coordsys), name='average_flux')
    snapshots.add_task(d3.Average(b,coords.S2coordsys), name='average_profile_b')
    snapshots.add_task(d3.Average(u@u,coords.S2coordsys), name='average_velocity')
    #snapshots.add_task(radius, name='rad')

    scalars = solver.evaluator.add_file_handler(scalars_dir, sim_dt=stop_sim_time/1000, max_writes=1000, mode=file_handler_mode)  # verifier max_writes
    scalars.add_task(d3.Average(flux(r=Ro),coords.S2coordsys), name='meanflux_r_outer')
    scalars.add_task(d3.Average(flux(r=Ri),coords.S2coordsys), name='meanflux_r_inner')
    scalars.add_task(d3.Average(b(r=Ri),coords.S2coordsys), name='temperature_r_inner')
    scalars.add_task(d3.Average(b(r=Ro),coords.S2coordsys), name='temperature_r_outer')
    scalars.add_task(d3.grad(b), name='grad_temperature')
    # scalars.add_task(d3.grad(b(r=Ri)), name='grad_temperature_r_inner')
    scalars.add_task(d3.Average(ur(r=Ro)*ur(r=Ro),coords.S2coordsys), name='rad_sq_ur_outer')
    scalars.add_task(d3.Average(ur(r=Ri)*ur(r=Ri),coords.S2coordsys), name='rad_sq_ur_inner')
    scalars.add_task(d3.Integrate(d3.Average(b,coords.S2coordsys))/(4*np.pi*(Ro**3.-Ri**3.)/3), name='mean_temperature')
    scalars.add_task(d3.Integrate(d3.Average((u@u)**(1/2),coords.S2coordsys))/(4*np.pi*(Ro**3.-Ri**3.)/3), name='mean_rmsvelsq')


    #checkpoints
    checkpoints = solver.evaluator.add_file_handler(checkpoints_dir, sim_dt=stop_sim_time/500, max_writes=1000, mode=file_handler_mode)
    checkpoints.add_tasks(solver.state)

    # CFL
    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=safety, threshold=0.1,
                max_change=1.5, min_change=.5, max_dt=max_timestep)   # baisser la cadence et la safety
                
                # safety : nb de fois le CFL.
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u), name='Re')


    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                max_Re = flow.max('Re')
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
            if max_Re == np.nan:
                print("Error diverging code")
                break
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Read CSV file with parameters")
    parser.add_argument("filename", type=str, help="Name of the CSV file")
    parser.add_argument("--restart", action="store_true", help="Restart from the last iteration")
    args = parser.parse_args()

    run(args.filename, args.restart)


if __name__ == "__main__":
    main()

