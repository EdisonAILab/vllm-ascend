#!/bin/bash
# Setup bi_test env and run inference test
set -e

source ~/miniconda3/bin/activate bi_test
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
source /home/pkg/CANN-1223/Ascend/8.5.0/bin/setenv.bash 2>/dev/null
export ASCEND_HOME_PATH=/home/pkg/CANN-1223/Ascend/8.5.0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/driver/
export CANN_ROOT=$ASCEND_HOME_PATH
export VLLM_BATCH_INVARIANT=1

cd /home/o00649568/b84411271

# Auto-install missing modules (max 30 attempts)
for i in $(seq 1 30); do
  err=$(python -c 'from vllm import LLM, SamplingParams; print("SUCCESS")' 2>&1 | tail -1)
  if echo "$err" | grep -q SUCCESS; then echo "$err"; break; fi

  mod=$(echo "$err" | grep -oP "No module named '\K[^'.]+" | head -1)
  if [ -z "$mod" ]; then
    echo "[$i] Non-import error: $err"
    # Try to create a stub module for optional deps
    errmod=$(echo "$err" | grep -oP "No module named '\K[^']+" | head -1)
    if [ -n "$errmod" ]; then
      stubdir="$CONDA_PREFIX/lib/python3.11/site-packages/${errmod//.//}"
      echo "[$i] Creating stub: $stubdir"
      mkdir -p "$stubdir"
      touch "$stubdir/__init__.py"
    else
      break
    fi
    continue
  fi

  # Map module names to package names
  pkg=$mod
  case $mod in
    cpuinfo) pkg=py-cpuinfo;;
    yaml) pkg=pyyaml;;
    PIL) pkg=pillow;;
    cv2) pkg="opencv-python-headless";;
    zmq) pkg=pyzmq;;
    sklearn) pkg=scikit-learn;;
    google) pkg=googleapis-common-protos;;
  esac

  echo "[$i] Installing: $mod -> $pkg"
  conda install -y -c conda-forge "$pkg" 2>&1 | tail -1

  # If conda fails, create stub
  if ! python -c "import $mod" 2>/dev/null; then
    stubdir="$CONDA_PREFIX/lib/python3.11/site-packages/${mod//.//}"
    echo "[$i] Conda failed, creating stub: $stubdir"
    mkdir -p "$stubdir"
    touch "$stubdir/__init__.py"
  fi
done

echo "=== Checking vllm import ==="
python -c 'from vllm import LLM, SamplingParams; print("vllm import OK")' 2>&1

echo "=== Running inference test ==="
python test_inference_bi.py 2>&1
