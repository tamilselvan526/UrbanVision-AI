"""
Automated Test Runner — Smart Crime Detection System
Runs all unit and integration test suites.
"""

import importlib
import inspect
import sys
import time

TEST_MODULES = [
    "tests.test_bike_cargo_analyzer",
    "tests.test_liveness_detector",
    "tests.test_reid_plate",
    "tests.test_street_profiler",
    "tests.test_pipeline",
]

def main():
    print("=" * 60)
    print("SMART CITY AI — FULL SUITE VERIFICATION")
    print("=" * 60)

    passed = 0
    failed = []
    t_start = time.perf_counter()

    for mod_name in TEST_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[FAIL] Could not import {mod_name}: {e}")
            failed.append((mod_name, "import", str(e)))
            continue

        funcs = [
            (name, obj)
            for name, obj in sorted(inspect.getmembers(mod, inspect.isfunction))
            if name.startswith("test_")
        ]

        print(f"\n--> Running {mod_name} ({len(funcs)} tests)...")
        for name, fn in funcs:
            try:
                fn()
                passed += 1
                print(f"    [PASS] {name}")
            except Exception as e:
                failed.append((mod_name, name, str(e)))
                print(f"    [FAIL] {name}: {e}")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {len(failed)} failed in {elapsed:.2f}s")
    print("=" * 60)

    if failed:
        for mod_name, name, err in failed:
            print(f"  FAILED: {mod_name}.{name} -> {err}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED SUCCESSFULLY.")
        sys.exit(0)

if __name__ == "__main__":
    main()
