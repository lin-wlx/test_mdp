"""
Test Script for k-th Order Moving Tiger

This script uses the MovingTigerKthOrder class from _DGP_TIGER.py
to generate k-th order data and test it with the Markov property test.

Usage:
    python test_moving_tiger_kth_order.py
"""

import os
import sys
import numpy as np

# Add parent directory to path
package_path = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.insert(0, package_path + "/test_func")
sys.path.insert(0, package_path + "/experiment_func")

from _core_test_fun import *

# Import from _DGP_TIGER.py
from _DGP_TIGER import MovingTigerKthOrder, simu_moving_tiger

# ============================================================================
# Configuration
# ============================================================================

# ============================================================================
# Configuration Presets
# ============================================================================

# RECOMMENDED: Balanced configuration (should reject J=3)
CONFIG_BALANCED = {
    'k': 4,
    'n_trajectories': 100,
    'trajectory_length': 50,
    'n_bootstrap': 50,
    'n_folds': 10,
    'paras': [50, 5],
    'random_state': 42,
}

# CONSERVATIVE: Faster but less power
CONFIG_CONSERVATIVE = {
    'k': 4,
    'n_trajectories': 75,
    'trajectory_length': 40,
    'n_bootstrap': 30,
    'n_folds': 7,
    'paras': [30, 4],
    'random_state': 42,
}

# AGGRESSIVE: Slow but highest power (for publication)
CONFIG_AGGRESSIVE = {
    'k': 5,
    'n_trajectories': 150,
    'trajectory_length': 60,
    'n_bootstrap': 100,
    'n_folds': 10,
    'paras': [100, 5],
    'random_state': 42,
}

# QUICK: For debugging only
CONFIG_QUICK = {
    'k': 4,
    'n_trajectories': 30,
    'trajectory_length': 25,
    'n_bootstrap': 10,
    'n_folds': 3,
    'paras': [20, 3],
    'random_state': 42,
}

# Select which configuration to use
# Options: CONFIG_BALANCED, CONFIG_CONSERVATIVE, CONFIG_AGGRESSIVE, CONFIG_QUICK
TEST_CONFIG = CONFIG_AGGRESSIVE  # <-- CHANGE THIS TO USE DIFFERENT CONFIG


# ============================================================================
# Test Functions
# ============================================================================

def test_single_order(data, order, config):
    """
    Test a single Markov order.

    Parameters
    ----------
    data : list of tuples
        Trajectory data from MovingTigerKthOrder
    order : int
        Markov order to test
    config : dict
        Test configuration

    Returns
    -------
    p_value : float
        Test p-value
    """
    print(f"  Testing J={order}...", end=" ", flush=True)

    p_value = test(
        data=data,
        J=order,
        B=config['n_bootstrap'],
        Q=config['n_folds'],
        paras=config['paras'],
        include_reward=False,
        fixed_state_comp=None,
        method="QRF",
        print_time=False,
    )

    reject = "REJECT" if p_value < 0.05 else "PASS"
    symbol = "✗" if p_value < 0.05 else "✓"

    print(f"p-value = {p_value:.4f} ({reject}) {symbol}")

    return p_value


def run_full_test():
    """
    Run complete test of k-th order Moving Tiger.

    Tests multiple Markov orders (J=1 through J=k+1) to verify:
    - J < k: Test REJECTS (insufficient order)
    - J = k: Test PASSES (correct order)
    - J > k: Test PASSES (more than sufficient)
    """
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  K-TH ORDER MOVING TIGER TEST".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    config = TEST_CONFIG
    k = config['k']

    # Print configuration
    print("=" * 70)
    print("Test Configuration")
    print("=" * 70)
    print(f"  k (Markov order):        {k}")
    print(f"  Number of trajectories:  {config['n_trajectories']}")
    print(f"  Trajectory length:       {config['trajectory_length']}")
    print(f"  Bootstrap samples:       {config['n_bootstrap']}")
    print(f"  CV folds:                {config['n_folds']}")
    print(f"  Random state:            {config['random_state']}")
    print(f"  RF parameters:           {config['paras']}")

    # Generate data using MovingTigerKthOrder from _DGP_TIGER.py
    print("\n" + "=" * 70)
    print("Step 1: Generating k-th Order Moving Tiger Data")
    print("=" * 70)

    sim = MovingTigerKthOrder(
        k=k,
        pattern=None,  # Use default pattern
        policy='always_listen',
        random_state=config['random_state']
    )

    # Get pattern info
    info = sim.get_pattern_info()
    print(f"\n✓ Created Moving Tiger with k={info['k']}")
    print(f"  Pattern: {info['pattern_str']}")
    print(f"  Observation accuracy: {info['obs_accuracy']} (perfect!)")

    # Generate trajectories
    data = sim.generate(
        n_trajectories=config['n_trajectories'],
        trajectory_length=config['trajectory_length']
    )

    print(f"\n✓ Generated {len(data)} trajectories")
    print(f"✓ First trajectory: {data[0][0].shape} observations, {data[0][1].shape} actions")

    # Verify pattern by generating a test trajectory
    (obs, acts), states = sim.generate_trajectory(12)
    print(f"\n✓ Sample states (first 12): {states}")
    if len(states) >= 2 * k:
        cycle1 = states[0:k]
        cycle2 = states[k:2 * k]
        if cycle1 == cycle2:
            print(f"✓ Pattern repeats correctly! Cycle: {cycle1}")

    # Test multiple orders
    print("\n" + "=" * 70)
    print("Step 2: Testing Multiple Markov Orders")
    print("=" * 70)
    print(f"\nWill test orders J = 1, 2, ..., {k + 1}")
    print(f"Expected: REJECT for J < {k}, PASS for J >= {k}\n")

    orders_to_test = list(range(1, k + 2))  # 1, 2, ..., k, k+1
    results = {}

    for j in orders_to_test:
        p_value = test_single_order(data, j, config)
        results[j] = p_value

    # Analyze results
    print("\n" + "=" * 70)
    print("Step 3: Results Analysis")
    print("=" * 70)

    print("\nSummary Table:")
    print("-" * 70)
    print(f"{'Order (J)':<12} {'P-value':<12} {'Decision':<12} {'Expected':<20} {'Match?'}")
    print("-" * 70)

    all_correct = True
    for j in orders_to_test:
        p_val = results[j]
        decision = "REJECT" if p_val < 0.05 else "PASS"

        if j < k:
            expected = "REJECT"
            expected_desc = f"J<{k}: insufficient"
        else:
            expected = "PASS"
            expected_desc = f"J>={k}: sufficient"

        match = "✓" if decision == expected else "✗"
        if decision != expected:
            all_correct = False

        print(f"{j:<12} {p_val:<12.4f} {decision:<12} {expected_desc:<20} {match}")

    print("-" * 70)

    # Final verdict
    print("\n" + "=" * 70)
    print("Final Verdict")
    print("=" * 70)

    if all_correct:
        print("\n SUCCESS! All tests passed as expected!")
        print(f"\n✓ Orders J=1 to J={k - 1}: Correctly REJECTED (insufficient order)")
        print(f"✓ Orders J={k} to J={k + 1}: Correctly PASSED (sufficient order)")
        print(f"\nConclusion: The algorithm correctly detects k={k} order Markov structure!")
    else:
        print("\n️  UNEXPECTED RESULTS!")
        print("\nSome tests did not match expected outcomes.")
        print("Possible reasons:")
        print("  - Too few trajectories (try increasing n_trajectories)")
        print("  - Too few bootstrap samples (try increasing n_bootstrap)")
        print("  - Random variation (run with different random_state)")

    print("\n" + "=" * 70)

    return results, all_correct


def test_with_wrapper_function():
    """
    Test using the wrapper function simu_moving_tiger() instead of the class.

    This demonstrates an alternative way to generate data.
    """
    print("\n" + "=" * 70)
    print("Alternative Test: Using simu_moving_tiger() Wrapper")
    print("=" * 70)

    config = TEST_CONFIG
    k = config['k']

    print(f"\nGenerating data with simu_moving_tiger() wrapper function...")

    # Use wrapper function (similar interface to simu_tiger)
    data = simu_moving_tiger(
        N=config['n_trajectories'],
        T=config['trajectory_length'],
        seed=config['random_state'],
        k=k,
        pattern=None,
        behav_def=0,  # always_listen
    )

    print(f"✓ Generated {len(data)} trajectories")
    print(f"✓ First trajectory: {data[0][0].shape}, {data[0][1].shape}")

    # Test at J=k
    print(f"\nTesting at J={k}...")
    p_value = test(
        data=data,
        J=k,
        B=config['n_bootstrap'],
        Q=config['n_folds'],
        paras=config['paras'],
        include_reward=False,
        method="QRF",
    )

    result = "PASS" if p_value > 0.05 else "REJECT"
    print(f"✓ p-value = {p_value:.4f} ({result})")

    if result == "PASS":
        print(f"✓ Wrapper function works correctly!")

    print("\n" + "=" * 70)


def quick_sanity_check():
    """Quick sanity check that everything is set up correctly."""
    print("\n" + "=" * 70)
    print("Quick Sanity Check")
    print("=" * 70)

    config = TEST_CONFIG
    k = config['k']

    print("\n1. Testing MovingTigerKthOrder class import...")
    try:
        sim = MovingTigerKthOrder(k=k, random_state=42)
        print(f"   ✓ Successfully imported from _DGP_TIGER.py")
    except Exception as e:
        print(f"   ✗ Error importing: {e}")
        return False

    print("\n2. Testing data generation...")
    try:
        data = sim.generate(n_trajectories=5, trajectory_length=20)
        print(f"   ✓ Generated {len(data)} trajectories")
    except Exception as e:
        print(f"   ✗ Error generating data: {e}")
        return False

    print("\n3. Verifying pattern repetition...")
    (obs, acts), states = sim.generate_trajectory(8)
    if len(states) >= 2 * k:
        if states[:k] == states[k:2 * k]:
            print(f"   ✓ Pattern repeats correctly")
        else:
            print(f"   ⚠ Warning: Pattern does not repeat perfectly")

    print("\n4. Testing MarkovTest function...")
    try:
        p_val = test(
            data=data,
            J=1,
            B=5,
            Q=2,
            paras=[10, 5],
            include_reward=False,
            method="QRF"
        )
        print(f"   ✓ MarkovTest working (p={p_val:.4f})")
    except Exception as e:
        print(f"   ✗ Error running test: {e}")
        return False

    print("\n" + "=" * 70)
    print("✅ All sanity checks passed! Ready for full test.")
    print("=" * 70)

    return True


def test_different_k_values():
    """Test with different k values to show generalizability."""
    print("\n" + "=" * 70)
    print("Testing Different k Values")
    print("=" * 70)

    config = TEST_CONFIG.copy()

    for k in [2, 3, 4, 5]:
        print(f"\n--- Testing k={k} ---")

        sim = MovingTigerKthOrder(k=k, random_state=42)
        data = sim.generate(n_trajectories=30, trajectory_length=30)

        print(f"Generated {len(data)} trajectories")

        # Test at J=k (should pass)
        print(f"Testing at J={k}...", end=" ")
        p_val = test(
            data=data,
            J=k,
            B=10,
            Q=3,
            paras=config['paras'],
            include_reward=False,
            method="QRF",
        )

        result = "PASS" if p_val > 0.05 else "REJECT"
        print(f"p={p_val:.4f} ({result})")

        if result == "PASS":
            print(f"✓ Correctly detected k={k} order!")

    print("\n" + "=" * 70)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("=" * 70)
    print("  K-TH ORDER MOVING TIGER - MARKOV PROPERTY TEST SUITE")
    print("  Using MovingTigerKthOrder from _DGP_TIGER.py")
    print("=" * 70)
    print("=" * 70)

    # Run sanity check
    if not quick_sanity_check():
        print("\n Sanity check failed. Please fix issues before proceeding.")
        return

    # Run main test
    results, success = run_full_test()

    # Optional: Test wrapper function
    print("\n" + "=" * 70)
    response = input("Test wrapper function simu_moving_tiger()? (y/n): ")
    if response.lower() == 'y':
        test_with_wrapper_function()

    # Optional: Test different k values
    response = input("\nTest different k values (k=2,3,4,5)? (y/n): ")
    if response.lower() == 'y':
        test_different_k_values()

    # Final summary
    print("\n\n" + "=" * 70)
    print("=" * 70)
    print("  TEST SUITE COMPLETE")
    print("=" * 70)
    print("=" * 70)

    if success:
        print("\n✅ All tests passed!")
        print(f"\nThe algorithm successfully detected k={TEST_CONFIG['k']} order Markov structure.")
        print("This demonstrates that the test can identify k-th order MDPs,")
        print("not just distinguish between 1st-order MDP and POMDP.")
    else:
        print("\n⚠️  Some tests did not match expectations.")
        print("Review the detailed results above for more information.")

    print("\n" + "=" * 70)
    print("\nFiles used:")
    print("  - Data generation: _DGP_TIGER.py (MovingTigerKthOrder)")
    print("  - Testing: _core_test_fun.py (test function)")
    print("\nTo modify test parameters, edit TEST_CONFIG at top of this file.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()