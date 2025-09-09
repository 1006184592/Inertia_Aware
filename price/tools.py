import time
import numpy as np
from gurobipy import GRB
from .output import value, save_results
from .model_defintions import build_model_gurobi
from .model_Ipopt import build_model
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def run_case_study(generators, ES, buses, lines, farms, wind_data, load_data, Hw, settings, case):
    fixed_U = []
    print(">>>> Building MIQP Model")
    m, u = build_model_gurobi(generators, ES, buses, lines, farms, wind_data, load_data, Hw, settings, case, fixed_U)

    print(">>>> Running MIQP Model")
    start_time = time.time()
    optimize_with_gurobi(m)
    print("Model status:", m.Status,"****************************")
    solvetime = time.time() - start_time
    status = m.Status  # 使用 Gurobi 的状态属性
    if status == GRB.OPTIMAL:
        print(f">>>> MIQP Model finished with status OPTIMAL in {solvetime:.2f} seconds")
    elif status == GRB.INFEASIBLE:
        print(f">>>> MIQP Model finished with status INFEASIBLE in {solvetime:.2f} seconds")
    else:
        print(f">>>> MIQP Model finished with status {status} in {solvetime:.2f} seconds")

    # 确保 u 是一个包含 Gurobi 变量的字典
    fixed_U = np.zeros((54, 24))
    for i in range(54):
        for j in range(0,24):
            if u[i, j].VarName and hasattr(u[i, j], 'X'):
                fixed_U[i, j] = u[i, j].X  # 使用 Gurobi API 获取变量的值

            # else:
                # print(f"Warning: Variable u[{i}, {j}] is not defined or has no value.")

    print(">>>> Building QP Model")
    # 第二次优化，使用从 MIQP 中得到的 fixed_U 值
    model = build_model(generators, ES, buses, lines, farms, wind_data, load_data, Hw, settings, case, fixed_U)

    # 添加对偶变量后缀
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    print(">>>> Running QP Model")
    start_time = time.time()
    opt = SolverFactory('ipopt', solver_io="python")
    # 再次使用 Ipopt 进行优化
    results = opt.solve(model, tee=True)

    # 记录并显示优化状态和时间
    solvetime = time.time() - start_time
    status = results.solver.termination_condition

    if status == pyo.TerminationCondition.optimal:
        print(f">>>> QP Model finished with status OPTIMAL in {solvetime:.2f} seconds")
    elif status == pyo.TerminationCondition.infeasible:
        print(f">>>> QP Model finished with status INFEASIBLE in {solvetime:.2f} seconds")
    else:
        print(f">>>> QP Model finished with status {status} in {solvetime:.2f} seconds")

    Energy_price, Reserve_price, Inertia_price=save_results(model,fixed_U,case)

    return Energy_price, Reserve_price, Inertia_price

def optimize_with_gurobi(model):
    model.setParam('Threads', 16)  # 例如设置为 16 个线程
    try:
        # model.setParam("FeasibilityTol", 1e-3)
        # model.setParam('MIPGap', 0.0002)
        model.optimize()  # 使用 Gurobi 提供的 optimize 方法
        if model.Status == GRB.OPTIMAL:
            print("Solution is optimal.")
        elif model.Status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()  # 计算不可行子系统

            # 将 IIS 保存到文件中（可选）
            model.write("infeasibility_report.ilp")

        else:
            print(f"Solver terminated with status: {model.Status}")
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
