import pulp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

N = 15
M = 5

alpha = 0.32
w1 = 1
w2 = 1

C = np.array([600, 300, 700, 500, 500])

D = np.array([180, 220, 100, 120, 260, 190, 140, 70, 110, 200, 90, 85, 95, 150, 95])

d = np.array([
    [1.6, 2.4, 5.3, 3.8, 2.0], [3.9, 2.1, 6.8, 4.1, 3.2], [1.1, 1.5, 2.9, 2.0, 1.3],
    [2.2, 2.6, 4.3, 2.5, 2.1], [5.0, 3.8, 6.2, 5.4, 3.9], [3.2, 2.7, 5.5, 4.0, 2.6],
    [2.6, 2.3, 4.1, 3.4, 1.9], [1.0, 1.3, 2.0, 1.6, 0.9], [2.4, 2.1, 3.6, 2.8, 1.8],
    [3.6, 3.2, 5.9, 4.6, 2.8], [1.8, 1.9, 2.5, 1.9, 1.7], [1.7, 1.6, 2.8, 1.8, 1.4],
    [2.0, 1.9, 3.1, 2.4, 1.6], [2.9, 2.5, 4.6, 3.9, 2.0], [3.3, 2.8, 5.0, 4.2, 2.4]
])

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

P_linear = (w1 * alpha * (d ** 2)) - (w2 * r)

def check_server_capacity(x):
    for j in range(M):
        if np.sum(x[:, j]) > C[j] + 0.01: return False
    return True

def check_user_requirements(x):
    for i in range(N):
        if np.sum(x[i, :]) < D[i] - 0.01: return False
    return True

def ensure_constraints(x):
    """A simple heuristic to adjust an allocation to be feasible."""
    x_fixed = x.copy()
    for j in range(M):
        total_load = np.sum(x_fixed[:, j])
        if total_load > C[j]:
            scale_factor = C[j] / total_load
            x_fixed[:, j] *= scale_factor
    
    for i in range(N):
        current_allocation = np.sum(x_fixed[i, :])
        if current_allocation < D[i] - 1e-6:
            deficit = D[i] - current_allocation
            for j in range(M):
                server_spare = C[j] - np.sum(x_fixed[:, j])
                if server_spare > 0:
                    additional = min(deficit, server_spare)
                    x_fixed[i, j] += additional
                    deficit -= additional
                    if deficit <= 1e-6: break
    return x_fixed

def print_allocation_details(x, method_name):
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

def calculate_metrics_linear(x):
    """Calculates Z-score using the LINEAR objective"""
    energy = np.sum(w1 * alpha * (d ** 2) * x)
    throughput = np.sum(w2 * r * x)
    cost = energy - throughput
    return energy, throughput, cost

def solve_optimal_lp():
    print("\n" + "=" * 80)
    print("PHASE 1 - METHOD 1: OPTIMAL LP SOLUTION (PuLP)")
    print("=" * 80)
    
    prob = pulp.LpProblem("Server_CPU_Allocation_LP", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(N), range(M)), lowBound=0)
    
    prob += pulp.lpSum([P_linear[i, j] * x[i][j] for i in range(N) for j in range(M)]), "Total_Net_Cost"
    
    for j in range(M):
        prob += pulp.lpSum([x[i][j] for i in range(N)]) <= C[j], f"Server_Capacity_{j}"
    for i in range(N):
        prob += pulp.lpSum([x[i][j] for j in range(M)]) >= D[i], f"User_Requirement_{i}"
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    x_optimal = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            x_optimal[i, j] = x[i][j].varValue
    
    energy, throughput, cost = calculate_metrics_linear(x_optimal)
    
    print(f"\nStatus: {pulp.LpStatus[prob.status]}")
    print(f"\nResults:")
    print(f"  Total Energy Cost: {energy:.2f}")
    print(f"  Total Throughput: {throughput:.2f}")
    print(f"  Objective Cost (Z): {cost:.2f}")
    print(f"\nConstraint Verification:")
    print(f"  All users meet minimum requirement: {check_user_requirements(x_optimal)}")
    print(f"  All servers within capacity: {check_server_capacity(x_optimal)}")
    
    print_allocation_details(x_optimal, "Optimal LP")
    return x_optimal, energy, throughput, cost

def solve_round_robin():
    print("\n" + "=" * 80)
    print("PHASE 1 - METHOD 2: ROUND ROBIN BASELINE")
    print("=" * 80)
    
    x_rr = np.zeros((N, M))
    for i in range(N):
        allocation_per_server = D[i] / M
        for j in range(M):
            x_rr[i, j] = allocation_per_server
    
    x_rr = ensure_constraints(x_rr)
    energy, throughput, cost = calculate_metrics_linear(x_rr)
    
    print(f"\nResults:")
    print(f"  Total Energy Cost: {energy:.2f}")
    print(f"  Total Throughput: {throughput:.2f}")
    print(f"  Objective Cost (Z): {cost:.2f}")
    print(f"\nConstraint Verification:")
    print(f"  All users meet minimum requirement: {check_user_requirements(x_rr)}")
    print(f"  All servers within capacity: {check_server_capacity(x_rr)}")
    
    return x_rr, energy, throughput, cost

def solve_random():
    print("\n" + "=" * 80)
    print("PHASE 1 - METHOD 3: RANDOM BASELINE")
    print("=" * 80)
    
    x_rand = np.zeros((N, M))
    for i in range(N):
        weights = np.random.random(M)
        weights /= weights.sum()
        for j in range(M):
            x_rand[i, j] = D[i] * weights[j]
    
    x_rand = ensure_constraints(x_rand)
    energy, throughput, cost = calculate_metrics_linear(x_rand)
    
    print(f"\nResults:")
    print(f"  Total Energy Cost: {energy:.2f}")
    print(f"  Total Throughput: {throughput:.2f}")
    print(f"  Objective Cost (Z): {cost:.2f}")
    print(f"\nConstraint Verification:")
    print(f"  All users meet minimum requirement: {check_user_requirements(x_rand)}")
    print(f"  All servers within capacity: {check_server_capacity(x_rand)}")
    
    return x_rand, energy, throughput, cost

def solve_greedy_performance():
    print("\n" + "=" * 80)
    print("PHASE 1 - METHOD 4: GREEDY (PERFORMANCE FIRST)")
    print("=" * 80)

    all_pairs = []
    for i in range(N):
        for j in range(M):
            all_pairs.append((r[i, j], i, j))
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    
    demand_remaining = D.copy()
    capacity_remaining = C.copy()
    x_plan = np.zeros((N, M))
    
    for (r_val, i, j) in all_pairs:
        need = demand_remaining[i]
        if need <= 1e-6: continue
        available = capacity_remaining[j]
        if available <= 1e-6: continue
            
        alloc = min(need, available)
        x_plan[i, j] = alloc
        demand_remaining[i] -= alloc
        capacity_remaining[j] -= alloc
    
    x_plan = ensure_constraints(x_plan)
    energy, throughput, cost = calculate_metrics_linear(x_plan)
    
    print(f"\nResults:")
    print(f"  (This method ignores energy (d_ij) and only chases performance (r_ij))")
    print(f"  Total Energy Cost: {energy:.2f}")
    print(f"  Total Throughput: {throughput:.2f}")
    print(f"  Objective Cost (Z): {cost:.2f}")
    print(f"\nConstraint Verification:")
    print(f"  All users meet minimum requirement: {check_user_requirements(x_plan)}")
    print(f"  All servers within capacity: {check_server_capacity(x_plan)}")
    
    return x_plan, energy, throughput, cost

def main_lp():
    print("=" * 80)
    print("ENERGY-MINIMIZED SERVER CPU ALLOCATION - PHASE 1 (LP)")
    print("=" * 80)
    
    lp_results = {}
    x_opt_lp, e_opt, t_opt, c_opt = solve_optimal_lp()
    lp_results['Optimal LP'] = {'allocation': x_opt_lp, 'energy': e_opt, 'throughput': t_opt, 'cost': c_opt}
    
    x_rr, e_rr, t_rr, c_rr = solve_round_robin()
    lp_results['Round Robin'] = {'allocation': x_rr, 'energy': e_rr, 'throughput': t_rr, 'cost': c_rr}
    
    x_rand, e_rand, t_rand, c_rand = solve_random()
    lp_results['Random'] = {'allocation': x_rand, 'energy': e_rand, 'throughput': t_rand, 'cost': c_rand}

    x_greedy, e_greedy, t_greedy, c_greedy = solve_greedy_performance()
    lp_results['Greedy (Perf. First)'] = {'allocation': x_greedy, 'energy': e_greedy, 'throughput': t_greedy, 'cost': c_greedy}
    
    print("\n" + "=" * 80)
    print("PHASE 1: COMPARATIVE ANALYSIS (LINEAR MODEL)")
    print("=" * 80)
    lp_data = []
    for method, metrics in lp_results.items():
        lp_data.append({'Method': method, 'Energy Cost': metrics['energy'], 'Throughput': metrics['throughput'], 'Objective (Z)': metrics['cost']})
    df_lp = pd.DataFrame(lp_data).set_index('Method')
    df_lp_print = df_lp.copy()
    df_lp_print['Energy Cost'] = df_lp_print['Energy Cost'].map('{:,.2f}'.format)
    df_lp_print['Throughput'] = df_lp_print['Throughput'].map('{:,.2f}'.format)
    df_lp_print['Objective (Z)'] = df_lp_print['Objective (Z)'].map('{:,.2f}'.format)
    print("\n" + df_lp_print.to_string())
    
    optimal_lp_cost = lp_results['Optimal LP']['cost']
    print("\n" + "-" * 80)
    print("IMPROVEMENT ANALYSIS (vs Optimal LP)")
    print("-" * 80)
    if abs(optimal_lp_cost) > 1e-6:
        for method in ['Round Robin', 'Random', 'Greedy (Perf. First)']:
            cost = lp_results[method]['cost']
            improvement = ((cost - optimal_lp_cost) / abs(cost) * 100)
            print(f"Optimal LP is better than {method} by: {improvement:.2f}%")
    print(f"Optimal LP achieves: Lowest objective cost (Z = {optimal_lp_cost:.2f})")

    print("\n" + "=" * 80)
    print("GENERATING PHASE 1 VISUALIZATIONS...")
    print("=" * 80)
    
    methods_lp = list(lp_results.keys())
    costs_lp = [lp_results[m]['cost'] for m in methods_lp]
    
    plt.figure(figsize=(10, 6))
    colors_lp = ['#4CAF50', '#FFC107', '#F44336', '#2196F3']
    bars = plt.bar(methods_lp, costs_lp, color=colors_lp, alpha=0.8)
    plt.title('Phase 1: Linear Model (LP) Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Objective Cost (Z) - Lower is Better', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')
    plt.show()

    plt.figure(figsize=(8, 8))
    for method in lp_results:
        e = lp_results[method]['energy']
        t = lp_results[method]['throughput']
        plt.scatter(e, t, s=200, label=method, alpha=0.7)
        plt.text(e, t + 20, method, fontsize=9)
    plt.xlabel("Total Energy Cost", fontsize=12)
    plt.ylabel("Total Throughput", fontsize=12)
    plt.title("Phase 1: Energy vs. Throughput Trade-off", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(x_opt_lp, cmap="YlGnBu", annot=True, fmt=".1f", xticklabels=server_names_short, yticklabels=app_names)
    plt.title("Optimal LP Allocation Heatmap", fontsize=16, fontweight='bold')
    plt.xlabel("Servers", fontsize=12)
    plt.ylabel("Applications", fontsize=12)
    plt.show()


if __name__ == "__main__":
    main_lp()
