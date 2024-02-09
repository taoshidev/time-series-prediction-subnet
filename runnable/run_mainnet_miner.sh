#!/bin/bash

# Delay between attempts in seconds
delay=5

# Infinite loop
while true; do
    echo "Attempting to run the Python script..."
    # Run the Python script
    
    set -e
    source /root/time-series-prediction-subnet/venv/bin/activate

    # Run the Python script with arguments
    cd /root/time-series-prediction-subnet/ && /root/time-series-prediction-subnet/venv/bin/python3 /root/time-series-prediction-subnet/neurons/miner.py --netuid 8 --subtensor.network finney --wallet.name miner --wallet.hotkey default --logging.debug --axon.port 8999 
    # Check the exit status of the Python script
    if [ $? -eq 0 ]; then
        echo "Python script succeeded."
        break
    else
        echo "Python script failed. Retrying in $delay seconds..."
        sleep $delay
    fi
done

echo "Script execution finished."


