# -*- coding: utf-8 -*-
"""
REALISTIC QUICK TEST
Better parameters to actually see MDP vs POMDP differences
Should take 5-10 minutes
"""

import os, sys

package_path = os.path.dirname(os.path.abspath(os.getcwd()))

sys.path.insert(0, package_path + "/test_func")
from _core_test_fun import *

sys.path.insert(0, package_path + "/experiment_func")
from _DGP_TIGER import *

os.environ["OMP_NUM_THREADS"] = "1"

print("=" * 60)
print("REALISTIC QUICK TEST")
print("Better parameters - should take 5-10 minutes")
print("=" * 60)


# Function from the original script
def one_time(seed=1, J=1,
             N=100, T=20, T_def=0,
             B=100, Q=10,
             behav_def=0, obs_def="alt",
             paras=[20, 3], weighted=True, include_reward=False,
             method="QRF"):
    ### generate data
    fixed_state_comp = (obs_def == "null")
    MDPs = simu_tiger(N=N, T=T, seed=seed,
                      behav_def=behav_def, obs_def=obs_def,
                      T_def=T_def, include_reward=include_reward,
                      fixed_state_comp=fixed_state_comp)
    T += 1
    ### Preprocess
    if fixed_state_comp:
        MDPs, fixed_state_comp = MDPs
    else:
        fixed_state_comp = None
    if T_def == 1:
        MDPs = truncateMDP(MDPs, T)
    if not include_reward:
        MDPs = [a[:2] for a in MDPs]
    N = len(MDPs)
    ### Calculate
    if paras == "CV_once":
        return lam_est(data=MDPs, J=J, B=B, Q=Q, paras=paras,
                       include_reward=include_reward,
                       fixed_state_comp=fixed_state_comp, method=method)
    return test(data=MDPs, J=J, B=B, Q=Q, paras=paras,
                include_reward=include_reward,
                fixed_state_comp=fixed_state_comp, method=method)


# Test configurations
configs = [
    {"obs_def": "null", "name": "MDP", "J_values": [1, 3]},
    {"obs_def": "alt", "name": "POMDP", "J_values": [1, 3, 5]},
]

for config in configs:
    obs_def = config["obs_def"]
    name = config["name"]
    J_values = config["J_values"]

    print(f"\n{'=' * 60}")
    print(f"Testing {name} (obs_def='{obs_def}')")
    print(f"{'=' * 60}")

    for J in J_values:
        print(f"\n--- Markov Order J={J} ---")

        # Run 5 replications (compromise between speed and reliability)
        p_values = []
        for rep in range(5):
            print(f"  Replication {rep + 1}/5...", end=" ")
            p_val = one_time(
                seed=rep,
                J=J,
                N=50,  # More trajectories than ultra-minimal
                T=20,  # Full length trajectories
                T_def=0,
                B=20,  # More bootstrap samples
                Q=5,  # More CV folds
                behav_def=0,
                obs_def=obs_def,
                paras=[20, 3],  # CORRECT: [max_depth, min_samples_leaf]
                include_reward=False,
                method="QRF"
            )
            p_values.append(p_val)
            print(f"p={p_val:.3f}")


        # Calculate rejection rate
        rejection_rate = sum(1 for p in p_values if p < 0.05) / len(p_values)
        avg_p_value = sum(p_values) / len(p_values)

        print(f"\n  Results for J={J}:")
        print(f"    Average p-value: {avg_p_value:.3f}")
        print(f"    Rejection rate: {rejection_rate:.2f} ({int(rejection_rate * 5)}/5 rejected)")

        # Interpret results
        if obs_def == "null":  # MDP
            if J == 1:
                if rejection_rate <= 0.2:
                    print(f"    ✓ CORRECT: MDP passes 1st-order Markov test")
                else:
                    print(f"    ⚠ Unexpected: MDP should pass 1st-order test")
            else:
                if rejection_rate <= 0.2:
                    print(f"    ✓ CORRECT: MDP passes higher-order test too")
        else:  # POMDP
            if J == 1:
                if rejection_rate >= 0.6:
                    print(f"    ✓ CORRECT: POMDP fails 1st-order Markov test")
                else:
                    print(f"    ⚠ Weak: POMDP should usually fail 1st-order test")
            elif J == 3:
                if rejection_rate >= 0.4:
                    print(f"    → Still rejecting at J=3 (needs higher order)")
                else:
                    print(f"    → Starting to pass at J=3")
            elif J == 5:
                if rejection_rate <= 0.3:
                    print(f"    ✓ CORRECT: POMDP passes at higher order (J=5)")
                else:
                    print(f"    → May need even higher order")

print("\n" + "=" * 60)
print("REALISTIC QUICK TEST COMPLETE!")
print("=" * 60)
print("\n SUMMARY OF EXPECTED PATTERNS:")
print("-" * 60)
print("MDP (null):")
print("  • J=1: Should PASS (low rejection ~0-20%)")
print("  • Higher J: Should also PASS")
print("\nPOMDP (alt):")
print("  • J=1: Should FAIL (high rejection ~80-100%)")
print("  • J=3: May still FAIL (moderate rejection)")
print("  • J=5: Should PASS (low rejection ~0-30%)")
print("-" * 60)
print("\nThis demonstrates the key finding:")
print("POMDPs require HIGHER-ORDER Markov models!")
print("=" * 60)
