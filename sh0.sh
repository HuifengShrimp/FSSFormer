#!/bin/bash
# File name: run_until_success.sh

while true; do
    bazel build //examples/python/... -c opt
    status=$?

    if [ $status -eq 0 ]; then
        echo "Command executed successfully."
        break
    else
        echo "Command failed with status $status. Retrying..."
    fi
    sleep 1  
done

echo "Script completed."
