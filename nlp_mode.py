import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds

N = 15
M = 5

alpha2 = 0.05
w1 = 1
w2 = 1

C = np.array([600, 300, 700, 500, 500])
D = np.array([180, 220, 100, 120, 260, 190, 140, 70, 110, 200, 90, 85, 95, 150, 95])

r = np.array([
    [1.5, 1.7, 0.9, 1.4, 1.2], [2.0, 1.6, 0.7, 2.2, 1.3], [1.6, 1.4, 1.1, 1.5, 1.2],
    [1.4, 1.8, 1.0, 1.6, 1.1], [2.2, 1.9, 0.6, 2.4, 1.5], [1.3, 1.6, 0.9, 1.5, 1.1],
    [1.2, 1.1, 1.0, 1.3, 1.0], [2.6, 1.3, 1.1, 1.8, 1.4], [1.5, 1.4, 1.0, 1.3, 1.2],
    [1.9, 1.7, 0.8, 2.1, 1.4], [1.4, 1.5, 1.1, 1.2, 1.3], [1.3, 1.25,1.05,1.1, 1.0],
    [1.45,1.35,1.1, 1.25,1.05], [1.6, 1.5, 0.95,1.45,1.2], [1.8, 1.6, 0.85,1.7, 1.25]
])

server_names = [f"Server {j}" for j in range(M)]
server_names_short = [f"S{j}" for j in range(M)]
app_names = [f"Application {i}" for i in range(N)]

r_flat = r.flatten()
num_vars = N * M

def objective_func_nlp(x):
    """The Non-Linear Objective Function (Phase 2)"""
    epsilon = 1e-9
    energy_cost = w1 * alpha2 * np.sum(x**2)
    performance_benefit = w2 * np.sum(np.log(1 + r_flat * x + epsilon))
    return energy_cost - performance_benefit

def gradient_func_nlp(x):
    """The Gradient of the NLP objective (for KKT Stationarity)"""
    epsilon = 1e-9
    grad_energy = 2 * w1 * alpha2 * x
    grad_performance = w2 * r_flat / (1 + r_flat * x + epsilon)
    return grad_energy - grad_performance

def hessian_func_nlp(x):
    """The Hessian of the NLP objective (curvature)"""
    epsilon = 1e-9
    hess_energy = 2 * w1 * alpha2
    hess_performance = (w2 * (r_flat**2)) / ((1 + r_flat * x + epsilon)**2)
    return np.diag(hess_energy + hess_performance)

def hessian_vector_product(x, p):
    """Hessian-vector product for 'Newton-CG' solver"""
    return hessian_func_nlp(x).dot(p)

def setup_constraints_nlp():
    """Defines the KKT Primal Feasibility rules for the solver"""
    constraints = []
    for j in range(M):
        A_j = np.zeros(num_vars)
        A_j[j::M] = 1.0
        constraints.append(LinearConstraint(A_j, lb=-np.inf, ub=C[j]))
        
    for i in range(N):
        A_i = np.zeros(num_vars)
        A_i[i*M : (i+1)*M] = 1.0
        constraints.append(LinearConstraint(A_i, lb=D[i], ub=np.inf))
    
    bounds = Bounds(lb=0, ub=np.inf)
    
    return constraints, bounds

def check_nlp_constraints(x_flat, C, D):
    """Checks if a given NLP solution is feasible"""
    x = x_flat.reshape((N, M))
    if np.any(x_flat < -1e-3): return False, "Negative allocation found"
    for j in range(M):
        if np.sum(x[:, j]) > C[j] + 1e-3: 
            return False, f"Server {j} violated capacity"
    for i in range(N):
        if np.sum(x[i, :]) < D[i] - 1e-3:
            return False, f"User {i} failed demand"
    return True, "All constraints met"

def calculate_metrics_nonlinear(x_flat):
    """Calculates Z-score and components using the NON-LINEAR objective"""
    x = x_flat.reshape((N, M))
    epsilon = 1e-9
    energy = w1 * alpha2 * np.sum(x**2)
    performance = w2 * np.sum(np.log(1 + r * x + epsilon))
    cost = energy - performance
    return energy, performance, cost

def print_nlp_allocation_details(x_flat, method_name):
    x = x_flat.reshape(N, M)
    print(f"\n--- Allocation Details for: {method_name} ---")
    print(f"\n{'Application':<30} {'Total CPU':<12} Server Distribution")
    print("-" * 80)
    for i in range(N):
        total = np.sum(x[i, :])
        dist = " | ".join([f"{server_names_short[j]}: {x[i,j]:.1f}" for j in range(M)])
        print(f"  {app_names[i]:<28}: {total:>6.1f} cores  [{dist}]")
    
    print(f"\n{'Server Load Summary':<30}")
    print("-" * 50)
    for j in range(M):
        load = np.sum(x[:, j])
        util = (load / C[j]) * 100
        print(f"  {server_names[j]:<25}: {load:>6.1f} / {C[j]} cores ({util:>5.1f}% utilized)")

def solve_nlp_gradient_descent(x_init):
    print("\n" + "=" * 80)
    print("PHASE 2 - METHOD 1: GRADIENT DESCENT ('CG')")
    print("=" * 80)
    print("(This is an UNCONSTRAINED method and should fail to meet constraints)")
    start_time = time.time()
    res_gd = minimize(objective_func_nlp, 
                      x_init, 
                      method='CG', 
                      jac=gradient_func_nlp, 
                      options={'disp': False, 'maxiter': 10000})
    end_time = time.time()
    
    x_gd = res_gd.x
    e_gd, t_gd, z_gd = calculate_metrics_nonlinear(x_gd)
    feasible, reason = check_nlp_constraints(x_gd, C, D)
    print(f"Finished in {end_time - start_time:.3f}s. Feasible: {feasible} ({reason})")
    return {'Z': z_gd, 'Energy': e_gd, 'Perf': t_gd, 'Feasible': "Yes" if feasible else "NO", 'Reason': reason, 'Time (s)': end_time - start_time}
    
def solve_nlp_newton(x_init):
    print("\n" + "=" * 80)
    print("PHASE 2 - METHOD 2: NEWTON'S METHOD ('Newton-CG')")
    print("=" * 80)
    print("(This is an UNCONSTRAINED method and should also fail)")
    start_time = time.time()
    res_nt = minimize(objective_func_nlp, 
                      x_init, 
                      method='Newton-CG', 
                      jac=gradient_func_nlp, 
                      hessp=hessian_vector_product, 
                      options={'disp': False, 'maxiter': 1000})
    end_time = time.time()
    
    x_nt = res_nt.x
    e_nt, t_nt, z_nt = calculate_metrics_nonlinear(x_nt)
    feasible, reason = check_nlp_constraints(x_nt, C, D)
    print(f"Finished in {end_time - start_time:.3f}s. Feasible: {feasible} ({reason})")
    return {'Z': z_nt, 'Energy': e_nt, 'Perf': t_nt, 'Feasible': "Yes" if feasible else "NO", 'Reason': reason, 'Time (s)': end_time - start_time}

def solve_nlp_trust_region(x_init):
    print("\n" + "=" * 80)
    print("PHASE 2 - METHOD 3: TRUST-REGION (KKT SOLVER)")
    print("=" * 80)
    print("(This is the correct CONSTRAINED method... this may take a moment)")
    start_time = time.time()
    
    constraints_list, bounds = setup_constraints_nlp()
    
    res_tr = minimize(objective_func_nlp, 
                      x_init, 
                      method='trust-constr', 
                      jac=gradient_func_nlp, 
                      hess=hessian_func_nlp, 
                      constraints=constraints_list, 
                      bounds=bounds,
                      options={'disp': False, 'maxiter': 10000})
    end_time = time.time()
    
    x_tr = res_tr.x
    e_tr, t_tr, z_tr = calculate_metrics_nonlinear(x_tr)
    feasible, reason = check_nlp_constraints(x_tr, C, D)
    print(f"Finished in {end_time - start_time:.3f}s. Feasible: {feasible} ({reason})")
    
    if feasible:
        print_nlp_allocation_details(x_tr, "Trust-Region (NLP)")
        
    return res_tr, {'Z': z_tr, 'Energy': e_tr, 'Perf': t_tr, 'Feasible': "Yes" if feasible else "NO", 'Reason': reason, 'Time (s)': end_time - start_time}

def main_nlp():
    print("\n\n" + "=" * 80)
    print("PHASE 2: NON-LINEAR OPTIMIZATION (NLP) ANALYSIS")
    print("This compares Gradient Descent, Newton's Method, and Trust-Region.")
    print("=" * 80)
    
    x_init_nlp = np.full(num_vars, 10.0) 
    
    nlp_results = {}
    nlp_results['Gradient Descent'] = solve_nlp_gradient_descent(x_init_nlp)
    nlp_results["Newton's Method"] = solve_nlp_newton(x_init_nlp)
    res_tr, tr_metrics = solve_nlp_trust_region(x_init_nlp)
    nlp_results['Trust-Region (KKT)'] = tr_metrics

    print("\n" + "=" * 80)
    print("PHASE 2: FINAL NLP SOLVER COMPARISON")
    print("=" * 80)
    
    df_nlp = pd.DataFrame(nlp_results).T
    df_nlp = df_nlp[['Z', 'Energy', 'Perf', 'Feasible', 'Reason', 'Time (s)']]
    df_nlp_print = df_nlp.copy()
    df_nlp_print['Z'] = df_nlp_print['Z'].map('{:,.2f}'.format)
    df_nlp_print['Energy'] = df_nlp_print['Energy'].map('{:,.2f}'.format)
    df_nlp_print['Perf'] = df_nlp_print['Perf'].map('{:,.2f}'.format)
    df_nlp_print['Time (s)'] = df_nlp_print['Time (s)'].map('{:.3f}'.format)
    print(df_nlp_print.to_string())
    
    print("\n" + "=" * 80)
    print("--- ANALYSIS OF PHASE 2 RESULTS ---")
    print("=" * 80)
    print("1. Gradient Descent & Newton's Method (Unconstrained):")
    print("   These methods are 'dumb' as they do not handle the constraints (C_j and D_i).")
    print(f"   They find a low objective value (Z~{nlp_results['Gradient Descent']['Z']:.2f}), but the solution is USELESS as it is INFEASIBLE.")
    
    print("\n2. Trust-Region (Constrained):")
    print("   This is the only method that finds the true, optimal, and FEASIBLE solution.")
    print(f"   It correctly finds the lowest objective cost while respecting all server and user constraints.")
    print(f"\n   The final optimal non-linear cost is: Z = {nlp_results['Trust-Region (KKT)']['Z']:.2f}")

    print("\n" + "=" * 80)
    print("GENERATING PHASE 2 VISUALIZATIONS...")
    print("=" * 80)

    methods_nlp = list(nlp_results.keys())
    costs_nlp = [nlp_results[m]['Z'] for m in nlp_results]
    feasibility = [nlp_results[m]['Feasible'] for m in nlp_results]
    
    plt.figure(figsize=(10, 6))
    colors_nlp = ['#FFC107', '#F44336', '#4CAF50']
    bars = plt.bar(methods_nlp, costs_nlp, color=colors_nlp, alpha=0.8)
    plt.title('Phase 2: Non-Linear Model (NLP) Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Objective Cost (Z) - Lower is Better', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}\n(Feasible: {feasibility[i]})', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.show()

    x_trust = res_tr.x.reshape(N, M)
    plt.figure(figsize=(10, 8))
    sns.heatmap(x_trust, cmap="rocket_r", annot=True, fmt=".1f", xticklabels=server_names_short, yticklabels=app_names)
    plt.title("NLP Trust-Region Allocation Heatmap", fontsize=16, fontweight='bold')
    plt.xlabel("Servers", fontsize=12)
    plt.ylabel("Applications", fontsize=12)
    plt.show()

if __name__ == "__main__":
    main_nlp()