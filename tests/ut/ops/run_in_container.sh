#!/bin/bash
# Run on the .32 host: finds CI container and executes test inside it
set -e

CONTAINER=$(docker ps --format '{{.Names}}' | grep triton_CI_op | head -1)
echo "Container: $CONTAINER"

WORKSPACE=$(docker exec $CONTAINER ls /data/autotest/optest_fast_pipeline/ | tail -1)
WORKSPACE=/data/autotest/optest_fast_pipeline/$WORKSPACE
echo "Workspace: $WORKSPACE"

# Copy test files into the container
docker cp /home/b84411271/test_triton_bi_kernels.py $CONTAINER:/home/b84411271/
docker cp /home/b84411271/vllm_ascend $CONTAINER:/home/b84411271/

# Write the inner script
docker exec $CONTAINER bash -c "cat > /home/b84411271/run_test.sh << 'INNEREOF'
#!/bin/bash
source /etc/profile
source /root/.bashrc

WORKSPACE=\$(ls -d /data/autotest/optest_fast_pipeline/*/ | tail -1)
export WORKSPACE=\${WORKSPACE%/}
export HARDWARE_TYPE=NPU_A5
source \${WORKSPACE}/triton_ascend/test/script/setenv.bash

ln -sf /data/bin_blue2/debug-rel2/bishengir-compile /usr/local/python3.11.13/lib/python3.11/site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile
ln -sf /data/bin_blue2/hivmc-a5 /usr/local/python3.11.13/lib/python3.11/site-packages/triton/backends/ascend/bishengir/bin/hivmc-a5

export PATH=/data/bin_blue2/debug-rel2:/data/bin_blue2:\$PATH
export BISHENG_INSTALL_PATH=/data/bin_blue2
export TRITON_ALWAYS_COMPILE=1
export TRITON_ENABLE_TASKQUEUE=0
export PYTHONPATH=/home/b84411271:\$PYTHONPATH

cd /home/b84411271
python3 test_triton_bi_kernels.py
INNEREOF
chmod +x /home/b84411271/run_test.sh"

# Run the inner script
docker exec -e ASCEND_RT_VISIBLE_DEVICES=1 $CONTAINER bash /home/b84411271/run_test.sh
