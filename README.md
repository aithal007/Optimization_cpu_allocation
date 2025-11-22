# Energy-Minimized Permormance-Maximized Server CPU Allocation

**Team:** Cloud Calculus (Parikshith, Munjikesh)  
**Roll Numbers:** BT2024249, BT2024247

---

## 1. Project Overview

This project designs, implements, and analyzes a mathematical optimization model to solve the server CPU allocation problem. The primary goal is to distribute CPU resources from a pool of **M servers** to a set of **N applications** (users) in a way that simultaneously:

1.  **Minimizes Total Energy Consumption**: Accounts for server efficiency and load.
2.  **Maximizes System Performance**: Measures the total useful work or throughput.
3.  **Respects All Constraints**: Ensures no server exceeds its capacity and every application receives its minimum required CPU.

This is a constrained, multi-objective optimization task critical for the efficient operation of modern data centers, cloud gaming platforms, and large-scale AI services.

---

## 2. Methodology: A Two-Phase Approach

We tackle this problem in two distinct phases, moving from a simplified abstract model to a more realistic one.

### Phase 1: Linear Programming (LP) Model (`lp_model.py`)

In this phase, we assume that both energy cost and performance benefit scale **linearly** with the amount of CPU allocated.

-   **Objective**: Minimize a linear combination of energy cost and performance benefit.
-   **Constraints**: Server capacity, user demand, and non-negativity are all linear.
-   **Solver**: We use the **PuLP** library in Python, which models the problem and uses an industrial-strength solver (like CBC) that implements the **Simplex Method**.
-   **Key Feature**: This model is guaranteed to find the single, globally optimal solution for the simplified linear world. It provides a fast and powerful baseline for comparison.

### Phase 2: Non-Linear Programming (NLP) Model (`nlp_model.py`)

To better reflect real-world physics, this phase introduces non-linearities into the objective function.

-   **Objective**:
    -   **Quadratic Energy Cost ($E \propto x^2$)**: Correctly models the fact that server energy increases sharply at high loads(here Quadratically).
    -   **Logarithmic Performance Benefit ($T \propto \log(1+x)$)**: Captures the law of diminishing returns, where each additional unit of CPU provides less benefit than the last.
-   **Solver**: This non-linear, non-convex problem cannot be solved by the Simplex method. We use **SciPy's `trust-constr` solver**, a powerful algorithm based on two key concepts:
    1.  **The KKT Conditions**: A set of mathematical rules that define the properties of the optimal solution for a constrained non-linear problem.
    2.  **The Trust-Region Method**: A stable, iterative numerical engine that safely navigates the complex, curved solution space to find the point that satisfies the KKT conditions.

---

## 3. File Structure

The project is organized into two main files for clarity and contribution tracking:

-   `lp_model.py`: Contains all the code for Phase 1, including the Optimal LP solver and baseline methods (Round Robin, Randomized, Greedy).
-   `nlp_model.py`: Contains all the code for Phase 2, including the unconstrained solvers (Gradient Descent, Newton's Method) and the correct, constrained Trust-Region solver.
-   `requirements.txt`: A list of all Python dependencies required to run the project.
-   `README.md`: This file.

---

## 4. How to Run the Project

Follow these steps to set up and run the analysis.

### Step 1: Set up a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 2: Install Dependencies

Install all the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 3: Run the Models

You can run each phase of the project independently.

**To run the Linear Programming analysis (Phase 1):**

```bash
python lp_model.py
```
This will execute the Optimal LP solver and the three baseline methods, print a comparative analysis, and generate plots for the linear models.

**To run the Non-Linear Programming analysis (Phase 2):**

```bash
python nlp_model.py
```
This will execute the unconstrained solvers and the Trust-Region solver, print the results, and generate plots comparing the NLP methods.

---

## 5. Key Findings

1.  **Optimization is Highly Effective**: The **Optimal LP** model consistently outperforms naive allocation strategies like Round Robin (53% better) and Randomized (38% better), proving the value of a mathematical approach.
2.  **Unconstrained Solvers Fail**: In the NLP phase, "dumb" unconstrained solvers like Gradient Descent and Newton's Method find solutions that are **infeasible** and therefore useless, as they violate critical user demand or server capacity constraints.
3.  **Constrained NLP is Required for Realism**: The **Trust-Region (KKT) solver** is the only method capable of finding the true, feasible, and optimal solution for the realistic non-linear model.
4.  **LP vs. NLP Allocation Strategy**:
    -   The LP model produces an **"all-or-nothing"** allocation, loading some servers to 100% because it doesn't see the penalty for high utilization.
    -   The NLP model, aware of the quadratic ($x^2$) energy cost, produces a **"load-spreading"** allocation, which is more energy-efficient and realistic.

---

## 6. Technologies Used

-   **Python 3**
-   **PuLP**: For modeling and solving the Linear Program.
-   **SciPy**: For solving the Non-Linear Program (`scipy.optimize.minimize`).
-   **NumPy**: For numerical operations and matrix manipulation.
-   **Pandas**: For data analysis and creating results tables.
-   **Matplotlib & Seaborn**: For generating all plots and visualizations.
