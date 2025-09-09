import os
from gurobipy import GRB
from pyomo.environ import *
import numpy as np
def all_variables(model):
    # 使用 Gurobi 提供的 .getVars() 方法获取所有变量
    # return model.getVars()
    # 使用 component_objects(Var) 获取模型中的所有变量对象
    return model.component_objects(Var)
def value(var):
    # 检查变量是否已经被求解
    if var.X is not None:
        return var.X  # 使用 .X 获取变量值
    else:
        return None  # 如果变量尚未被求解，返回 None

def save_results(m, u, case):
    os.makedirs("results", exist_ok=True)

    # 保存所有变量
    # with open(f"results/all_vars_case{case}.txt", "w") as f:
    #     for var_obj in all_variables(m):
    #         for index in var_obj:
    #             # 获取变量名和对应值
    #             var_name = var_obj[index].name
    #             var_value = var_obj[index].value
    #             f.write(f"{var_name} = {var_value}\n")
    # all_vars = all_variables(m)
    # with open(f"results/all_vars_case{case}.txt", "w") as f:
    #     for v in all_vars:
    #         f.write(f"{v.VarName} = {value(v)}\n")

    Energy_price = []
    Reserve_price = []
    Inertia_price = []
    Congestion_plus = []
    Congestion_minus = []
    t_list = range(1, 25)
    bus_list = range(1, 119)
    G_list = range(1, 55)
    line_list = range(1, 187)

    # 获取 Reserve_price 和 Inertia_price
    if f"gamma" in m.component_map():
        for t in m.T:
            constr_reserve = m.gamma[t]
            constr_reserve = m.dual.get(constr_reserve)
            Reserve_price.append(constr_reserve)
    if f"chi" in m.component_map():
        for t in m.T:
            constr_inertia = m.chi[t]
            constr_inertia = m.dual.get(constr_inertia)
            Inertia_price.append(constr_inertia)
    # for t in t_list:
    #     constr_reserve = m.getConstrByName(f"gamma_case6[{t-1}]")
    #     constr_inertia = m.getConstrByName(f"chi_case6[{t-1}]")
    #     if constr_reserve:
    #         Reserve_price.append(constr_reserve.Pi)
    #     if constr_inertia:
    #         Inertia_price.append(constr_inertia.Pi)

    # 获取 Energy_price
    # 提取 lambda (Power Balance Constraint) 的对偶值
    if f"lambda_constr" in m.component_map():
        for t in m.T:
            for b in m.Buses:
                constr_lambda = m.lambda_constr[b, t]
                constr_lambda = m.dual.get(constr_lambda)
                Energy_price.append(constr_lambda)
    # for t in t_list:
    #     for b in bus_list:
    #         constr_lambda = m.getConstrByName(f"lambda[{b-1},{t-1}]")
    #         if constr_lambda:
    #             Energy_price.append(constr_lambda.Pi)

    # 写入 Energy_price
    with open(f"results/Energy_price_case{case}.txt", "w") as f:
        for t in t_list:
            for b in bus_list:
                index = 118 * (t - 1) + (b - 1)
                if index < len(Energy_price):
                    f.write(f"{Energy_price[index]}\n")
            f.write("\n")

    # 写入 Reserve_price
    with open(f"results/Reserve_price_case{case}.txt", "w") as f:
        for t in t_list:
            if t - 1 < len(Reserve_price):
                f.write(f"{Reserve_price[t - 1]}\n")

    # 写入 Inertia_price
    with open(f"results/Inertia_price_case{case}.txt", "w") as f:
        for t in t_list:
            if t - 1 < len(Inertia_price):
                f.write(f"{Inertia_price[t - 1]}\n")

    # 写入 U
    # with open(f"results/U_case{case}.txt", "w") as f:
    #     for t in t_list:
    #         for g in G_list:
    #             index = 54 * (t - 1) + (g - 1)
    #             if index < len(u):
    #                 f.write(f"{u[g-1,t-1]}\n")
    #         f.write("\n")
    return Energy_price, Reserve_price, Inertia_price
