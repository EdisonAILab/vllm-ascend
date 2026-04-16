#!/bin/bash
# Auto-download and install missing vllm deps
# Run on LOCAL machine, it will download wheels then scp+install on server

SERVER="o00649568@10.50.90.206"
REMOTE_DIR="/home/o00649568/b84411271"
LOCAL_DIR="/tmp/vllm_deps"
export SSH_ASKPASS=/tmp/ssh_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0

download_and_install() {
  local mod=$1
  # Map module names to package names
  local pkg=$mod
  case $mod in
    cv2) pkg=opencv-python-headless;;
    yaml) pkg=pyyaml;;
    PIL) pkg=pillow;;
    zmq) pkg=pyzmq;;
    sklearn) pkg=scikit-learn;;
    cpuinfo) pkg=py-cpuinfo;;
    google.*) pkg=googleapis-common-protos;;
  esac

  echo "  Downloading $pkg..."
  local ver=$(curl -s -k "https://pypi.org/pypi/$pkg/json" 2>/dev/null | grep -oP '"version":"[^"]+' | head -1 | grep -oP '[^"]+$')
  [ -z "$ver" ] && { echo "  NO VERSION for $pkg"; return 1; }

  local url=""
  for pat in "cp311.*x86_64" "abi3.*x86_64" "none-any"; do
    url=$(curl -s -k "https://pypi.org/pypi/$pkg/$ver/json" 2>/dev/null | \
      grep -oP "\"url\":\"[^\"]*${pat}[^\"]*\.whl\"" | \
      grep -v macosx | grep -v musllinux | head -1 | grep -oP 'https://[^"]*')
    [ -n "$url" ] && break
  done
  [ -z "$url" ] && { echo "  NO WHEEL for $pkg==$ver"; return 1; }

  local fname=$(basename "$url")
  cd "$LOCAL_DIR"
  curl -s -L -k -o "$fname" "$url" || { echo "  DOWNLOAD FAIL"; return 1; }

  scp -o StrictHostKeyChecking=no "$LOCAL_DIR/$fname" "$SERVER:$REMOTE_DIR/vllm_deps/" 2>/dev/null
  ssh -o StrictHostKeyChecking=no "$SERVER" "
    source ~/miniconda3/bin/activate haojinenv
    pip install --no-deps $REMOTE_DIR/vllm_deps/$fname 2>&1 | tail -1
  " 2>/dev/null
  echo "  Installed $pkg==$ver"
}

# Main loop
for i in $(seq 1 30); do
  err=$(ssh -o StrictHostKeyChecking=no "$SERVER" "
    source ~/miniconda3/bin/activate haojinenv
    source /home/pkg/CANN-1223/Ascend/8.5.0/bin/setenv.bash 2>/dev/null
    export ASCEND_HOME_PATH=/home/pkg/CANN-1223/Ascend/8.5.0
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:\$LD_LIBRARY_PATH
    export CANN_ROOT=\$ASCEND_HOME_PATH
    python -c 'from vllm import LLM, SamplingParams; print(\"SUCCESS\")' 2>&1 | tail -1
  " 2>/dev/null)

  if echo "$err" | grep -q SUCCESS; then
    echo "[$i] $err"
    break
  fi

  mod=$(echo "$err" | grep -oP "No module named '\K[^'.]+")
  if [ -n "$mod" ]; then
    echo "[$i] Missing: $mod"
    download_and_install "$mod"
  else
    echo "[$i] Non-import error: $err"
    break
  fi
done
