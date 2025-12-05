import os
import sys
import subprocess
import time

physics = [
    ("example_DR_2d_GS.py", 11, 0),
    ("example_burgers_1d.py", 51, 0),
    ("example_advection_1d.py", 51, 0),
]

runs = 10
inner_runs = 3


def run_task(script_path, grid_size, fine_tune):
    if not os.path.exists(script_path):
        print(f"Error: File {script_path} not found!")
        return False

    successful_runs = 0

    for i in range(1, runs + 1):
        print(f"\nRun {i}/{runs}...")

        try:
            cmd = [
                sys.executable,
                script_path,
                f"--grid_size={grid_size}",
                f"--n_run={inner_runs}",
                f"--fine_tune_data={fine_tune}"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                successful_runs += 1
            else:
                if result.stderr:
                    print(result.stderr[:200])

        except Exception as e:
            print(f"Error!!! Physics {i}: {str(e)}")

        if i < runs:
            time.sleep(1)

    print(f"\n{successful_runs}/{runs} successful runs")
    return successful_runs == runs


def main():
    total_success = 0
    total_failed = 0

    for i, (script_path, grid_size, fine_tune) in enumerate(physics, 1):
        print(f"Physics №{i} ")
        success = run_task(script_path, grid_size, fine_tune)

        if success:
            total_success += 1
        else:
            total_failed += 1

    if total_failed == 0:
        print("\nDone!")
    else:
        print(f"\n{total_failed} physics with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
