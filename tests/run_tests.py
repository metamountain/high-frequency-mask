"""Standalone runner -- pytest is not installed in the ComfyUI portable python.

    ..\..\..\python_embeded\python.exe tests\run_tests.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODULES = ["test_regressions", "test_radius", "test_photo"]


def main():
    import importlib
    passed = failed = xfailed = xpassed = 0
    failures = []

    for name in MODULES:
        mod = importlib.import_module(name)
        expected_fail = getattr(mod, "XFAIL", set())
        print(f"\n{name}")
        for fn_name in sorted(d for d in dir(mod) if d.startswith("test_")):
            short = fn_name[len("test_"):]
            try:
                getattr(mod, fn_name)()
            except Exception as exc:
                if short in expected_fail:
                    xfailed += 1
                    print(f"  XFAIL {short}\n         {type(exc).__name__}: "
                          f"{str(exc).splitlines()[0][:90]}")
                else:
                    failed += 1
                    failures.append((name, fn_name, traceback.format_exc()))
                    print(f"  FAIL  {short}")
            else:
                if short in expected_fail:
                    xpassed += 1
                    print(f"  XPASS {short}  <- bug is fixed, remove it from XFAIL")
                else:
                    passed += 1
                    print(f"  ok    {short}")

    for name, fn, tb in failures:
        print(f"\n{'=' * 70}\n{name}.{fn}\n{'=' * 70}\n{tb}")

    print(f"\n{passed} passed, {failed} failed, {xfailed} xfailed, {xpassed} xpassed")
    return 1 if failed or xpassed else 0


if __name__ == "__main__":
    sys.exit(main())
