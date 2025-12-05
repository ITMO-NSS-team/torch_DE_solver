#!/bin/bash

BASH_RUNS=10
PYTHON_RUNS=3

declare -A PHYSICS_CONFIG

PHYSICS_CONFIG['DR_2d_GS']="31 0"
PHYSICS_CONFIG['Burgers_1d']="51 0"
PHYSICS_CONFIG['Advection_1d']="51 0"

declare -A PHYSICS_SCRIPTS

PHYSICS_SCRIPTS['DR_2d_GS']='example_DR_2d_GS.py'
PHYSICS_SCRIPTS['Burgers_1d']='example_burgers_1d.py'
PHYSICS_SCRIPTS['Advection_1d']='example_advection_1d.py'

echo "Multiphysics data generation"

for physics_name in "${!PHYSICS_CONFIG[@]}"; do
    IFS=' ' read -r grid_size fine_tune <<< "${PHYSICS_CONFIG[$physics_name]}"
    echo "  $physics_name: grid_size=$grid_size, fine_tune_data=$fine_tune"
done

echo ""
echo "Bash runs per one physics: $BASH_RUNS"
echo "Python experiments per one run: $PYTHON_RUNS"
echo ""

run_physics() {
    local physics_name="$1"
    local script_path="$2"
    local grid_size="$3"
    local fine_tune_data="$4"

    echo "========================================"
    echo "Physics name: $physics_name"
    echo "Parameters: grid_size=$grid_size, n_run="$n_run", fine_tune_data=$fine_tune_data"
    echo "========================================"

    for ((bash_run=1; bash_run<=BASH_RUNS; bash_run++)); do
        echo "---"
        echo "Bash startup $bash_run/$BASH_RUNS"
        echo "Time: $(date '+%H:%M:%S')"

        python3 "$script_path" \
            --grid_size="$grid_size" \
            --n_run="$PYTHON_RUNS" \
            --fine_tune_data="$fine_tune_data"

        if [ $? -eq 0 ]; then
            echo "Bash startup №$bash_run completed successfully"
        else
            echo "Error: bash startup №$bash_run"
        fi

        sleep 1
    done

    echo "Physics '$physics_name' done!"
}

for physics_name in "${!PHYSICS_CONFIG[@]}"; do
    script_path="${PHYSICS_SCRIPTS[$physics_name]}"
    IFS=' ' read -r grid_size fine_tune_data <<< "${PHYSICS_CONFIG[$physics_name]}"

    run_physics "$physics_name" "$script_path" "$grid_size" "$fine_tune_data"
done

echo "All physics is done!"
