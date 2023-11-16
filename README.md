# License
Copyright © 2023 Taoshi, LLC
```text
Taoshi All rights reserved. 
Source code produced by Taoshi, LLC may not be reproduced, modified, or distributed 
without the express permission of Taoshi, LLC.
```


# License
Bittensor source code in this repository is licensed under the MIT License.
```text
The MIT License (MIT)
Copyright © 2023 Yuma Rao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

# Overview
Overview of the Time-Series Prediction Subnet
```text
This subnet is dedicated to the Time-Series Prediction Subnet (TSPS) made by Taoshi.

Initially the TSPS will be predictions on the future movement of financial markets. We will start with predictions 
on Bitcoin's price movement, and quickly move to more crypto trade pairs and eventually into other financial markets. 
Eventually, we'll move to more topics or domains beyond financial markets but this will be the initial concentration.

Over time, we expect miners to concentrate on various topics or domains, or those that are very sophisticated to 
try and take on multiple topics (sports betting, unspecified topics sent by clients, etc.).
```

# FAQ

## How does rewarding work?

```text
Miners are rewarded based on predicting future information or live data as we classify it in the subnet. In 
order to help train miners the network will also provide training data on the various datasets (trade pairs for 
financial market data) but rewarding only occurs on data that hasn't occurred yet, or what the subnet defines as 
live data. Once the future data is known, we compare against the predictions made and reward based on performance.
```

## What processing power does it require to be a validator?
```text
In order to be a validator you simply need to have a server running in the EU (recommend Ireland, UK). This can be 
through VPN or a cloud-based server. This is because not all financial data can be accessed inside the US (for crypto). 
The actual processing power is light, as validators are really only comparing results against what occurs live 
therefore a relatively small machine can be a validator (will have exact details soon).
```

## What is the expected data input and output as a miner?
```text
For financial markets, the goal will be to predict the next 8 hours of closes on a 5m interval 
(100 closes). In order to help with the prediction, the subnet will provide the 
last 25-30 days of 5m data for the trade pair. You can expect the data to be in the format
[close_timestamp (ms), close, high, low, volume] where close, high, low, and volume are all linearly 
pre-scaled for you between 0.495 to 0.505. You can convert the linear scale to whatever format you'd like, 
but you can consistently expect the data to be on this scale.

We linearly pre-scale because future data may come from clients who want the data to remain anonymous and
we've planned ahead to account for these future cases.

Input Features: [close_timestamp (milliseconds), close, high, low, volume]
Target Feature: [close]
```


## Can I be a miner with little knowledge?
```text
Predicting on markets is very hard, but we want to help those who want to contribute to the network by providing 
a base model that can be used. This base can be used to build upon, or just run yourself to try and compete. You can
participate by running a pre-built & pre-trained model provided to all miners in mining_models/base_model.h5

This model is already built into the core logic of neurons/miner.py for you to run and compete as a miner. All you
need to do is run neurons/miner.py
```

## I'm knowledgable about creating a competing prediction model on my own, can I prepare my miner to compete?
```text
Yes, you can start from the base model (neurons/miner.py) in order to prepare for release or you can start from 
scratch and test on testnet (netuid 3). 
You can choose to use the training data provided by the subnet on each trade pair or prepare separately using your own 
training data (say on BTC to start).
```

## Where can I begin testing?
```text
You can begin testing on testnet netuid 3. You can follow the 
docs/running_on_testnet.md file inside the repo to run on testnet.
```

## Building a model

How can I build a model? 

You can choose to build a model on your own or using the infrastructure inside this repository. If you choose
to use the infra provided in here, you can choose to leverage standardized financial market indicators provided
in `mining_objects/financial_market_indicators.py` as well as a standardized LSTM model 
in `mining_objects/base_mining_model.py`. You can generate a historical Bitcoin dataset using the script 
`runnable/generate_historical_data.py` and choose the timeframe you'd like to generate for using the comments provided
inside the script.

We've provided a basis for creating and testing models using `runnable/miner_testing.py` and `runnable/miner_training.py`

Please note we are constantly adjusting these scripts and will be making improvements over time.


## Testing on testnet 
```text
You can begin testing on testnet netuid 3. You can follow the docs/running_on_testnet.md file inside the repo 
to run on testnet. Some recommendations when testing on testnet to speed up testing is pass the "test_only_historical"
argument for your validator. You can do this by using running
python neurons/validator.py --netuid 3 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug --test_only_historical 1
this will have the validator run and test against historical data instead of live. If you only
want to test against live data then dont use this flag.

When youre testing we also recommend using 2 miners as a single miner won't provide enough responses to pass for
weighing. You can do this by passing a different port for the 2nd registered miner. 
You can run the second miner using the following example command:
python neurons/miner.py --netuid 3 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug --axon.port 8095
```

## Running on mainnet 
```text
You can run on mainnet by following the instructions in docs/running_on_mainnet.md

If you are running into issues please run with --logging.debug and --logging.trace set so you can better
analyze why your miner isnt running.

The current flow of information is as follows:
1. valis request predictions hourly
2. miners make predictions
3. valis store predictions
4. valis wait until data is available to compare against (roughly a day and a half for 5m data)
5. valis compare when data is ready

so dont expect emissions on predictions immediately, you'll need to wait for the predicted information to occur to be
compared against. 
```

*Running a miner*
```text
If you're running a miner you should see two types of requests, LiveForward and LiveBackward. LiveForward will be when
your miner performs predictions, LiveBackward will be receiving back the results that occurred live in case you want
to use them for any updating purposes.

Short term, it shouldn't be expected to run your miner to be trained on the network (TrainingBackward and TrainingForward
won't be consistently running). Please train ahead of time as we'll only be using BTC/USD on the 5m to begin which you can prepare for. 
```

*Running a validator*

```text
Your validator will request predictions hourly based on your randomized interval (to distribute load). Distributing rewards will still 
happen after live results come in (32 hours after predictions are made)

_Using run script_

These validators are designed to run and update themselves automatically. To run a validator, follow these steps:

1. Install this repository, you can do so by following the steps outlined in the installation section.

2. Install PM2 and the jq package on your system. 

On Linux:
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
On Mac OS:
brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update

3. Be sure to installed venv for the repo

Inside the repository directory (time-series-prediction-subnet) run:
python3 -m venv venv
pip install -r requirements.txt
source venv/bin/activate

4. Run the run.sh script which will handle running your validator and pulling the latest updates as they are issued.
pm2 start run.sh --name sn8 -- --wallet.name <wallet> --wallet.hotkey <hotkey> --netuid 8

This will run two PM2 process: one for the validator which is called sn8 by default (you can change this in run.sh), 
and one for the run.sh script (in step 4, we named it tsps). The script will check for updates every 30 minutes, 
if there is an update then it will pull it, install it, restart tsps and then restart itself.

_Not using run script_

If there are any issues with the run script or you choose to not use it, you can run your validator with the following command:
python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>

You can also choose to simply run your script in the background and logs will get stored in nohup.out, you can do that using:
nohup python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey> &
```


## Recommended Specs to Run
```text

Requires Python 3.10 or higher.

These are recommended specs, you may be able to make miner and validator work off lesser specs.

Validator
2 vCPU + 8 GB memory
100 GB balanced persistent disk

Miner
2 vCPU + 7.5 GB memory
1 NVIDIA V100
100 GB balanced persistent disk

Helpful install commands (on linux machine)
# sudo apt install git-all
# git clone https://github.com/taoshidev/time-series-prediction-subnet.git
# sudo apt install python3-pip 
# sudo apt-get install python3-venv
# cd time-series-prediction-subnet
# python3 -m venv venv
# . venv/bin/activate
# export PIP_NO_CACHE_DIR=1
# pip install -r requirements.txt
# python -m pip install -e .
```