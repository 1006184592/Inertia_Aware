import numpy as np
import pyomo.environ as pyo
import h5py

def build_model(G, ES, buses, lines, farms, wind, load, Hw, settings, case, fixed_U):
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

    # System Ssys
    G_list = list(range(n_generators))
    ES_list = list(range(n_ESs))
    wind_list = list(range(n_farms))
    Ssys = sum(G[i].Pgmax for i in G_list) + sum(ES[j].Pdmax for j in ES_list) + sum(farms[k].Pwmax for k in wind_list)

    # Initialize model
    model = pyo.ConcreteModel()

    # Define sets
    model.T = pyo.RangeSet(0, 23)
    model.T2 = pyo.RangeSet(1, 23)
    model.Buses = pyo.RangeSet(0, n_buses - 1)
    model.Lines = pyo.RangeSet(0, n_lines - 1)
    model.Generators = pyo.RangeSet(0, n_generators - 1)
    model.ESs = pyo.RangeSet(0, n_ESs - 1)
    model.WindFarms = pyo.RangeSet(0, n_farms - 1)

    # Define variables
    model.u = pyo.Var(model.Generators, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))

    # Fix u if fixed_U is provided
    if len(fixed_U) != 0:
        for i in G_list:
            for t in model.T:
                model.u[i, t].fix(fixed_U[i, t])

    model.Pg = pyo.Var(model.Generators, model.T, within=pyo.NonNegativeReals)
    model.Pd = pyo.Var(model.ESs, model.T, within=pyo.NonNegativeReals)
    model.Pc = pyo.Var(model.ESs, model.T, within=pyo.NonNegativeReals)
    model.E = pyo.Var(model.ESs, model.T, within=pyo.NonNegativeReals)
    model.alpha_g = pyo.Var(model.Generators, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))
    model.Cost_E = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.Cost_G = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.theta = pyo.Var(model.Buses, model.T)
    model.bus_out_power = pyo.Var(model.Buses, model.T)
    model.DC_flow = pyo.Var(model.Lines, model.T)
    model.G_bus = pyo.Var(model.Buses, model.T)
    model.ESwind_bus = pyo.Var(model.Buses, model.T)

    if case == 6:
        model.He = pyo.Var(model.ESs, model.T, within=pyo.NonNegativeReals)
        model.alpha_d = pyo.Var(model.ESs, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.alpha_c = pyo.Var(model.ESs, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))

    # Constraints
    # Power constraints for G
    def mu_plus_rule(model, i, t):
        return model.Pg[i, t] <= model.u[i, t] * G[i].Pgmax - (phi_g * thetap - mup) * model.alpha_g[i, t]
    model.mu_plus = pyo.Constraint(model.Generators, model.T, rule=mu_plus_rule)

    def mu_minus_rule(model, i, t):
        return model.u[i, t] * G[i].Pgmin + (phi_g * thetap - mup) * model.alpha_g[i, t] <= model.Pg[i, t]
    model.mu_minus = pyo.Constraint(model.Generators, model.T, rule=mu_minus_rule)

    # DC power flow constraints
    def DC_power_flow_rule(model, i, t):
        head = lines[i].head
        tail = lines[i].tail
        beta = lines[i].beta
        return model.DC_flow[i, t] == (model.theta[head, t] - model.theta[tail, t]) * beta
    model.DC_power_flow = pyo.Constraint(model.Lines, model.T, rule=DC_power_flow_rule)

    def theta_plus_rule(model, i, t):
        return model.DC_flow[i, t] <= lines[i].u
    model.theta_plus = pyo.Constraint(model.Lines, model.T, rule=theta_plus_rule)

    def theta_minus_rule(model, i, t):
        return -lines[i].u <= model.DC_flow[i, t]
    model.theta_minus = pyo.Constraint(model.Lines, model.T, rule=theta_minus_rule)

    def reference_bus_rule(model, t):
        return model.theta[1, t] == 0
    model.reference_bus = pyo.Constraint(model.T, rule=reference_bus_rule)

    # Bus power balance constraints
    def bus_out_power_rule(model, i, t):
        if len(buses[i].inlist) == 0 and len(buses[i].outlist) == 0:
            return model.bus_out_power[i, t] == 0
        elif len(buses[i].inlist) != 0 and len(buses[i].outlist) == 0:
            return model.bus_out_power[i, t] == sum(-model.DC_flow[k, t] for k in buses[i].inlist)
        elif len(buses[i].inlist) == 0 and len(buses[i].outlist) != 0:
            return model.bus_out_power[i, t] == sum(model.DC_flow[k, t] for k in buses[i].outlist)
        else:
            return model.bus_out_power[i, t] == sum(-model.DC_flow[k, t] for k in buses[i].inlist) + sum(
                model.DC_flow[k, t] for k in buses[i].outlist)
    model.bus_out_power_constr = pyo.Constraint(model.Buses, model.T, rule=bus_out_power_rule)

    # Generator bus constraints
    def G_bus_rule(model, i, t):
        if len(buses[i].genids) == 0:
            return model.G_bus[i, t] == 0
        else:
            return model.G_bus[i, t] == sum(model.Pg[k, t] for k in buses[i].genids)
    model.G_bus_constr = pyo.Constraint(model.Buses, model.T, rule=G_bus_rule)

    # ES and wind bus constraints
    def ESwind_bus_rule(model, i, t):
        if len(buses[i].farmids) == 0:
            return model.ESwind_bus[i, t] == 0
        else:
            return model.ESwind_bus[i, t] == sum(
                model.Pd[k, t] - model.Pc[k, t] + wind[k, t] for k in buses[i].farmids)
    model.ESwind_bus_constr = pyo.Constraint(model.Buses, model.T, rule=ESwind_bus_rule)

    # Energy balance constraints
    def lambda_rule(model, i, t):
        return model.G_bus[i, t] + model.ESwind_bus[i, t] == load[i, t] + model.bus_out_power[i, t]
    model.lambda_constr = pyo.Constraint(model.Buses, model.T, rule=lambda_rule)

    if case == 1:
        # Power constraints for ES
        def xi_plus_rule(model, j, t):
            return model.Pd[j, t] <= ES[j].Pdmax
        model.xi_plus = pyo.Constraint(model.ESs, model.T, rule=xi_plus_rule)

        def nu_plus_rule(model, j, t):
            return model.Pc[j, t] <= ES[j].Pcmax
        model.nu_plus = pyo.Constraint(model.ESs, model.T, rule=nu_plus_rule)

        # Energy constraints for ES
        def beta_plus_rule(model, j, t):
            return model.E[j, t] <= ES[j].Emax
        model.beta_plus = pyo.Constraint(model.ESs, model.T, rule=beta_plus_rule)

        def beta_minus_rule(model, j, t):
            return model.E[j, t] >= ES[j].Emin
        model.beta_minus = pyo.Constraint(model.ESs, model.T, rule=beta_minus_rule)

        def initial_energy_rule(model, j):
            return model.E[j, 0] == E0
        model.initial_energy = pyo.Constraint(model.ESs, rule=initial_energy_rule)

        def eta_rule(model, j, t):
            if t >= 1:
                return model.E[j, t] == model.E[j, t - 1] + model.Pc[j, t] * k - model.Pd[j, t] / k
            else:
                return pyo.Constraint.Skip
        model.eta = pyo.Constraint(model.ESs, model.T, rule=eta_rule)

        # Participation factor limitation
        def rho_g_rule(model, i, t):
            return model.alpha_g[i, t] <= model.u[i, t]
        model.rho_g = pyo.Constraint(model.Generators, model.T, rule=rho_g_rule)

        # Reserve price
        def gamma_rule(model, t):
            return sum(model.alpha_g[i, t] for i in model.Generators) == 1
        model.gamma = pyo.Constraint(model.T, rule=gamma_rule)

        # Inertia price
        def chi_rule(model, t):
            return sum(model.u[i, t] * Hg[i] * G[i].Pgmax for i in G_list) >= Hmin * Ssys
        model.chi = pyo.Constraint(model.T, rule=chi_rule)

        # Cost of G and ES
        def cost_E_rule(model, t):
            return model.Cost_E[t] == sum(
                ES[j].cd * model.Pd[j, t] + ES[j].cc * model.Pc[j, t] for j in model.ESs)
        model.cost_E = pyo.Constraint(model.T, rule=cost_E_rule)

        def cost_G_rule(model, t):
            return model.Cost_G[t] == sum(
                model.u[i, t] * G[i].pi1 / 100
                + G[i].pi2 / 100 * (model.Pg[i, t] + mup * model.alpha_g[i, t])
                + G[i].pi3 / 100 * (model.Pg[i, t] ** 2 + 2 * mup * model.alpha_g[i, t] * model.Pg[i, t]
                                    + model.alpha_g[i, t] ** 2 * (thetap ** 2 + mup ** 2))
                for i in model.Generators)
        model.cost_G = pyo.Constraint(model.T, rule=cost_G_rule)

    elif case == 6:
        # Additional variables and constraints for case 6
        def xi_plus_case6_rule(model, j, t):
            return (model.Pd[j, t] + 2 * model.He[j, t] * ES[j].Pdmax * rocof_max / f0
                    <= ES[j].Pdmax - (phi_d * thetap - mup) * model.alpha_d[j, t])
        model.xi_plus_case6 = pyo.Constraint(model.ESs, model.T, rule=xi_plus_case6_rule)

        def nu_plus_case6_rule(model, j, t):
            return (model.Pc[j, t] + 2 * model.He[j, t] * ES[j].Pdmax * rocof_max / f0
                    <= ES[j].Pcmax - (phi_c * thetap - mup) * model.alpha_c[j, t])
        model.nu_plus_case6 = pyo.Constraint(model.ESs, model.T, rule=nu_plus_case6_rule)

        def beta_plus_case6_rule(model, j, t):
            return (model.E[j, t]
                    <= ES[j].Emax - 2 * model.He[j, t] * delta_f_max * ES[j].Pdmax / f0 * k)
        model.beta_plus_case6 = pyo.Constraint(model.ESs, model.T, rule=beta_plus_case6_rule)

        def beta_minus_case6_rule(model, j, t):
            return (model.E[j, t]
                    >= ES[j].Emin - 2 * model.He[j, t] * delta_f_max * ES[j].Pdmax / f0 / k)
        model.beta_minus_case6 = pyo.Constraint(model.ESs, model.T, rule=beta_minus_case6_rule)

        def initial_energy_case6_rule(model, j):
            return model.E[j, 1] == E0
        model.initial_energy_case6 = pyo.Constraint(model.ESs, rule=initial_energy_case6_rule)

        def eta_case6_rule(model, j, t):
            if t >= 1:
                return model.E[j, t] == model.E[j, t - 1] + model.Pc[j, t] * k - model.Pd[j, t] / k
            else:
                return pyo.Constraint.Skip
        model.eta_case6 = pyo.Constraint(model.ESs, model.T, rule=eta_case6_rule)

        def epsilon_rule(model, j, t):
            return model.He[j, t] <= ES[j].Hemax
        model.epsilon = pyo.Constraint(model.ESs, model.T, rule=epsilon_rule)

        def rho_g_case6_rule(model, i, t):
            return model.alpha_g[i, t] <= model.u[i, t]
        model.rho_g_case6 = pyo.Constraint(model.Generators, model.T, rule=rho_g_case6_rule)

        def gamma_rule(model, t):
            return (sum(model.alpha_g[i, t] for i in model.Generators)
                    + sum(model.alpha_d[j, t] - model.alpha_c[j, t] for j in model.ESs) == 1)
        model.gamma = pyo.Constraint(model.T, rule=gamma_rule)

        def chi_rule(model, t):
            return (sum(model.u[i, t] * Hg[i] * G[i].Pgmax for i in G_list)
                    + sum(model.He[j, t] * ES[j].Pdmax for j in ES_list)
                    + sum((Hw[k, t] - (phi_h * thetah - muh)) * farms[k].Pwmax for k in wind_list)
                    >= Hmin * Ssys)
        model.chi = pyo.Constraint(model.T, rule=chi_rule)

        def cost_E_case6_rule(model, t):
            return model.Cost_E[t] == sum(
                ES[j].cd * (model.Pd[j, t] + mup * model.alpha_d[j, t])
                + ES[j].cc * (model.Pc[j, t] + mup * model.alpha_c[j, t])
                for j in model.ESs)
        model.cost_E_case6 = pyo.Constraint(model.T, rule=cost_E_case6_rule)

        def cost_G_case6_rule(model, t):
            return model.Cost_G[t] == sum(
                model.u[i, t] * G[i].pi1 / 100
                + G[i].pi2 / 100 * (model.Pg[i, t] + mup * model.alpha_g[i, t])
                + G[i].pi3 / 100 * (model.Pg[i, t] ** 2 + 2 * mup * model.alpha_g[i, t] * model.Pg[i, t]
                                    + model.alpha_g[i, t] ** 2 * (thetap ** 2 + mup ** 2))
                for i in model.Generators)
        model.cost_G_case6 = pyo.Constraint(model.T, rule=cost_G_case6_rule)

    # Objective function
    def objective_rule(model):
        return sum(model.Cost_G[t] + model.Cost_E[t] for t in model.T)
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model
