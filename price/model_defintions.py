import numpy as np
from gurobipy import Model, GRB
import h5py
def build_model_gurobi(G, ES, buses, lines, farms, wind, load, Hw, settings, case, fixed_U):
    n_buses = len(buses)
    n_lines = len(lines)
    n_generators = len(G)
    n_farms = len(farms)
    n_ESs = len(ES)

    print(f"bus: {n_buses}")
    print(f"line: {n_lines}")
    print(f"gen: {n_generators}")
    print(f"farm: {n_farms}")
    print(f"ES: {n_ESs}")

    # Get and prepare settings
    k = settings["k"]
    mup = settings["mu_p"]
    thetap = settings["theta_p"]
    muh = settings["mu_h"]
    thetah = settings["theta_h"]
    epsilon_g = settings["epsilon_g"]
    epsilon_d = settings["epsilon_d"]
    epsilon_c = settings["epsilon_c"]
    epsilon_h = settings["epsilon_h"]
    phi_g = settings["Phi_g"]
    phi_d = settings["Phi_d"]
    phi_c = settings["Phi_c"]
    phi_h = settings["Phi_h"]
    Hmin = settings["Hmin"]
    delta_f_max = settings["Delta_fmax"]
    f0 = settings["f0"]
    rocof_max = settings["RoCoFmax"]
    E0 = settings["E0"]

    # Define Hg array
    Hg = np.zeros(54)
    Hg[:10] = 3.5
    Hg[10:35] = 4
    Hg[35:54] = 5

    # List with timesteps 1...24
    t_list = list(range(0, 24))
    t_list2 = list(range(1, 24))
    bus_list = list(range(n_buses))
    line_list = list(range(n_lines))
    G_list = list(range(n_generators))
    ES_list = list(range(n_ESs))
    wind_list = list(range(n_farms))

    # System Ssys
    Ssys = sum(G[i].Pgmax for i in G_list) + sum(ES[j].Pdmax for j in ES_list) + sum(farms[k].Pwmax for k in wind_list)

    # Initialize model
    m = Model("Gurobi Optimization Model")
    m.setParam("NonConvex", 2)

    # Create u as decision variables regardless of fixed_U availability
    u = m.addVars(G_list, t_list, vtype=GRB.BINARY, name="u")

    if len(fixed_U) != 0:
        # Add constraints to enforce the values from fixed_U
        for i in G_list:
            for t in t_list:
                m.addConstr(u[i, t] == fixed_U[i, t], name=f"fixed_u_{i}_{t}")

    # Define Variables
    Pg = m.addVars(G_list, t_list, lb=0, name="Pg")
    Pd = m.addVars(ES_list, t_list, lb=0, name="Pd")
    Pc = m.addVars(ES_list, t_list, lb=0, name="Pc")
    E = m.addVars(ES_list, t_list, lb=0, name="E")
    alpha_g = m.addVars(G_list, t_list, lb=0, ub=1, name="alpha_g")
    Cost_E = m.addVars(t_list, lb=0, name="Cost_E")
    Cost_G = m.addVars(t_list, lb=0, name="Cost_G")
    theta = m.addVars(bus_list, t_list, lb=-GRB.INFINITY, name="theta")
    bus_out_power = m.addVars(bus_list, t_list,lb=-GRB.INFINITY, name="bus_out_power")
    DC_flow = m.addVars(line_list, t_list, lb=-GRB.INFINITY, name="DC_flow")
    G_bus = m.addVars(bus_list, t_list, lb=-GRB.INFINITY, name="G_bus")
    ESwind_bus = m.addVars(bus_list, t_list, lb=-GRB.INFINITY, name="ESwind_bus")

    if case == 6:
        He = m.addVars(ES_list, t_list, lb=0, name="He")
        alpha_d = m.addVars(ES_list, t_list, lb=0, ub=1, name="alpha_d")
        alpha_c = m.addVars(ES_list, t_list, lb=0, ub=1, name="alpha_c")

    # Add constraints
    # Power constraints for G
    m.addConstrs((Pg[i, t] <= u[i, t] * G[i].Pgmax - (phi_g * thetap - mup) * alpha_g[i, t] for i in G_list for t in t_list),
                 "mu_plus")
    m.addConstrs((u[i, t] * G[i].Pgmin + (phi_g * thetap - mup) * alpha_g[i, t] <= Pg[i, t] for i in G_list for t in t_list),
                 "mu_minus")
    # DC power flow
    m.addConstrs(
        (DC_flow[i, t] == (theta[lines[i].head, t] - theta[lines[i].tail, t]) * lines[i].beta for i in line_list for t
         in t_list), "DC_power_flow")

    m.addConstrs((DC_flow[i, t] <= lines[i].u for i in line_list for t in t_list), "theta_plus")
    m.addConstrs((-lines[i].u <= DC_flow[i, t] for i in line_list for t in t_list), "theta_minus")
    m.addConstrs((theta[1, t] == 0 for t in t_list), "reference_bus")

    for i in bus_list:
        if len(buses[i].inlist) == 0 and len(buses[i].outlist) == 0:
            m.addConstrs((bus_out_power[i, t] == 0 for t in t_list))
        elif len(buses[i].inlist) != 0 and len(buses[i].outlist) == 0:
            m.addConstrs((bus_out_power[i, t] == sum(-DC_flow[k, t] for k in buses[i].inlist) for t in t_list))
        elif len(buses[i].inlist) == 0 and len(buses[i].outlist) != 0:
            m.addConstrs((bus_out_power[i, t] == sum(DC_flow[k, t] for k in buses[i].outlist) for t in t_list))
        elif len(buses[i].inlist) != 0 and len(buses[i].outlist) != 0:
            m.addConstrs((bus_out_power[i, t] == sum(-DC_flow[k, t] for k in buses[i].inlist) + sum(
                DC_flow[k, t] for k in buses[i].outlist) for t in t_list))
    for i in bus_list:
        if len(buses[i].genids) == 0 and len(buses[i].farmids) == 0:
            m.addConstrs((G_bus[i, t] == 0 for t in t_list))
            m.addConstrs((ESwind_bus[i, t] == 0 for t in t_list))
        elif len(buses[i].genids) != 0 and len(buses[i].farmids) == 0:
            m.addConstrs((G_bus[i, t] == sum(Pg[k, t] for k in buses[i].genids) for t in t_list))
            m.addConstrs((ESwind_bus[i, t] == 0 for t in t_list))
        elif len(buses[i].genids) == 0 and len(buses[i].farmids) != 0:
            m.addConstrs((G_bus[i, t] == 0 for t in t_list))
            m.addConstrs(
                (ESwind_bus[i, t] == sum(Pd[k, t] - Pc[k, t] + wind[k, t] for k in buses[i].farmids) for t in t_list))
        elif len(buses[i].genids) != 0 and len(buses[i].farmids) != 0:
            m.addConstrs((G_bus[i, t] == sum(Pg[k, t] for k in buses[i].genids) for t in t_list))
            m.addConstrs(
                (ESwind_bus[i, t] == sum(Pd[k, t] - Pc[k, t] + wind[k, t] for k in buses[i].farmids) for t in t_list))

    # Energy price
    m.addConstrs((G_bus[i, t] + ESwind_bus[i, t] == load[i, t] + bus_out_power[i, t] for i in bus_list for t in t_list), "lambda")

    if case == 1:
        # power constraints for ES
        m.addConstrs((Pd[j, t] <= ES[j].Pdmax for j in ES_list for t in t_list), "xi_plus")
        m.addConstrs((Pc[j, t] <= ES[j].Pcmax for j in ES_list for t in t_list), "nu_plus")
        # energy constraints for ES
        m.addConstrs((E[j, t] <= ES[j].Emax for j in ES_list for t in t_list), "beta_plus")
        m.addConstrs((E[j, t] >= ES[j].Emin for j in ES_list for t in t_list), "beta_minus")
        m.addConstrs((E[j, 0] == E0 for j in ES_list), "initial_energy")
        m.addConstrs((E[j, t] == E[j, t - 1] + Pc[j, t] * k - Pd[j, t] / k for j in ES_list for t in t_list2), "eta")
        # participation factor limitation
        m.addConstrs((alpha_g[i, t] <= u[i, t] for i in G_list for t in t_list), "rho_g")
        # Reserve price
        m.addConstrs((sum(alpha_g[i, t] for i in G_list) == 1 for t in t_list), "gamma")
        # Inertia price
        m.addConstrs((sum(u[i, t] * Hg[i] * G[i].Pgmax for i in G_list) >= Hmin * Ssys for t in t_list), "chi")

        # cost of G and ES
        m.addConstrs((Cost_E[t] == sum(ES[j].cd * Pd[j, t] + ES[j].cc * Pc[j, t] for j in ES_list) for t in t_list),
                     "cost_E")
        m.addConstrs((Cost_G[t] == sum(
            u[i, t] * G[i].pi1 / 100 + G[i].pi2 / 100 * (Pg[i, t] + mup * alpha_g[i, t]) + G[i].pi3 / 100 * (
                        Pg[i, t] ** 2 + 2 * mup * alpha_g[i, t] * Pg[i, t] + alpha_g[i, t] ** 2 * (thetap ** 2 + mup ** 2))
            for i in G_list) for t in t_list), "cost_G")

    elif case == 6:
        # power constraints for ES
        m.addConstrs(
            (Pd[j, t] + 2 * He[j, t] * ES[j].Pdmax * rocof_max / f0 <= ES[j].Pdmax - (phi_d * thetap - mup) * alpha_d[j, t]
             for j in ES_list for t in t_list), "xi_plus_case6")
        m.addConstrs(
            (Pc[j, t] + 2 * He[j, t] * ES[j].Pdmax * rocof_max / f0 <= ES[j].Pcmax - (phi_c * thetap - mup) * alpha_c[j, t]
             for j in ES_list for t in t_list), "nu_plus_case6")
        # energy constraints for ES
        m.addConstrs(
            (E[j, t] <= ES[j].Emax - 2 * He[j, t] * delta_f_max * ES[j].Pdmax / f0 * k for j in ES_list for t in
             t_list), "beta_plus_case6")
        m.addConstrs(
            (E[j, t] >= ES[j].Emin - 2 * He[j, t] * delta_f_max * ES[j].Pdmax / f0 / k for j in ES_list for t in
             t_list), "beta_minus_case6")
        m.addConstrs((E[j, 1] == E0 for j in ES_list), "initial_energy_case6")
        m.addConstrs((E[j, t] == E[j, t - 1] + Pc[j, t] * k - Pd[j, t] / k for j in ES_list for t in t_list2),
                     "eta_case6")
        # synthetic inertia limitation
        m.addConstrs((He[j, t] <= ES[j].Hemax for j in ES_list for t in t_list), "epsilon")
        # participation factor limitation
        m.addConstrs((alpha_g[i, t] <= u[i, t] for i in G_list for t in t_list), "rho_g_case6")
        # Reserve price
        m.addConstrs(
            (sum(alpha_g[i, t] for i in G_list) + sum(alpha_d[j, t] - alpha_c[j, t] for j in ES_list) == 1 for t in
             t_list), "gamma")
        # Inertia price
        m.addConstrs((sum(u[i, t] * Hg[i] * G[i].Pgmax for i in G_list) + sum(
            He[j, t] * ES[j].Pdmax for j in ES_list) + sum(
            (Hw[k, t] - (phi_h * thetah - muh)) * farms[k].Pwmax for k in wind_list) >= Hmin * Ssys for t in t_list),
                     "chi")
        # cost of G and ES
        m.addConstrs((Cost_E[t] == sum(
            ES[j].cd * (Pd[j, t] + mup * alpha_d[j, t]) + ES[j].cc * (Pc[j, t] + mup * alpha_c[j, t]) for j in ES_list)
                      for t in t_list), "cost_E_case6")
        m.addConstrs((Cost_G[t] == sum(
            u[i, t] * G[i].pi1 / 100 + G[i].pi2 / 100 * (Pg[i, t] + mup * alpha_g[i, t]) + G[i].pi3 / 100 * (
                        Pg[i, t] ** 2 + 2 * mup * alpha_g[i, t] * Pg[i, t] + alpha_g[i, t] ** 2 * (thetap ** 2 + mup ** 2))
            for i in G_list) for t in t_list), "cost_G_case6")

    # Objective function
    m.setObjective(sum(Cost_G[t] + Cost_E[t] for t in t_list), GRB.MINIMIZE)

    return m, u
