import pandas as pd
from .input import *
from .tools import run_case_study
# from tools_Ipopt import run_case_study
def price_case(wind_data,Pwmax_value,mu_p,theta_p,case):
    line_limits= [175,175,500,175,175,175,500,500,500,175,175,175,175,175,175,175,175,175,175,175,500,175,175,175,175,175,175,175,175,175,500,500,500,175,175,500,175,500,175,175,140,175,175,175,175,175,175,175,175,500,500,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,500,175,175,500,500,500,500,500,500,500,175,175,500,175,500,175,175,500,500,175,175,175,175,175,175,175,500,175,175,175,175,175,175,500,500,175,500,500,200,200,175,175,175,500,500,175,175,500,500,500,175,500,500,175,175,175,175,175,175,175,175,175,175,200,175,175,175,175,175,175,175,175,175,500,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,500,175,175,175,500,175,175,175]
    #,读取数据并创建对象
    file_path= "./price/casedata/118bus.jld"
    lines, generators, buses = load_jld_data(file_path)

    mvaBase = 100
    thermalLimitscale = 0.9
    # Adjust line limits
    for i in range(len(lines)):
        lines[i].u = 0.99 * thermalLimitscale * line_limits[i] / mvaBase

    # Create Wind Farms
    wp = 1.25
    factor_sigma = 1.25 * wp
    selected_nodes=[2,7,10,19,23,25,30,37,42,48,52]
    farms = []

    for node in selected_nodes:
        farms.append(Farm(100.0 / 100 * wp, factor_sigma * 10.0 / 100, bus=node, Pwmax=Pwmax_value))
    # Assign farms to buses
    for i, f in enumerate(farms):
        buses[f.bus].farmids.append(i)

    # Load Energy Storage and Time Series Data
    ES = load_ES("./price/casedata")
    load_data, Hw = load_timeseries("./price/casedata")
    # Set up settings
    settings = {
        "k": 0.9,  # Efficiency of the ES
        "mu_p": mu_p,  # Mean of wind power forecast error
        "theta_p": theta_p,  # Standard deviation of wind power forecast error
        "mu_h": 0.5,  # Mean of wind inertia forecast error
        "theta_h": 1,  # Standard deviation of wind inertia forecast error
        "epsilon_g": 0.05,  # Probability of generator's power limit violations
        "epsilon_d": 0.05,  # Probability of ES's discharging power limit violations
        "epsilon_c": 0.05,  # Probability of ES's charging power limit violations
        "epsilon_h": 0.05,  # Probability of inertia limit violations
        "Phi_g": 1.65,  # the (1-epsilon_g)-quantile of the standard normal distribution
        "Phi_d": 1.65,  # the (1-epsilon_d)-quantile of the standard normal distribution
        "Phi_c": 1.65,  # the (1-epsilon_c)-quantile of the standard normal distribution
        "Phi_h": 1.65,  # the (1-epsilon_h)-quantile of the standard normal distribution
        "Hmin": 3.3,  # unit: s
        "Delta_fmax": 0.55,  # unit: Hz
        "f0": 60,  # unit: Hz
        "RoCoFmax": 0.5,  # unit: Hz/s
        "E0": 0.5,  # unit: MWh
    }

    # Placeholder for running the case study (function should be defined based on model specifics)
    Energy_price, Reserve_price, Inertia_price= run_case_study(generators, ES, buses, lines, farms, wind_data, load_data, Hw, settings, case)
    return Energy_price, Reserve_price, Inertia_price
