## License
Copyright © 2023 Taoshi, LLC
```text
Taoshi All rights reserved. 
Source code produced by Taoshi, LLC may not be reproduced, modified, or distributed 
without the express permission of Taoshi, LLC.
```


## License
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

## Overview
Overview of the Time-Series Prediction Subnet
```text
This subnet is dedicated to the Time-Series Prediction Subnet (TSPS) made by Taoshi. We will have a lot of information 
available over the next 2 weeks, but we'll start with some basic information here on the subnet and also a bit of 
information on our timeline.

Initially the TSPS will be predictions on the future movement of financial markets. We will start with predictions 
on Bitcoin's price movement, and quickly move to more crypto trade pairs and eventually into other financial markets. 
Eventually, we'll move to more topics or domains beyond financial markets but this will be the initial concentration.

Over time, we expect miners to concentrate on various topics or domains, or those that are very sophisticated to 
try and take on multiple topics (sports betting, unspecified topics sent by clients, etc.).
```

## FAQ
```text
How does rewarding work?

Miners are rewarded based on predicting future information or live data as we classify it in the subnet. In 
order to help train miners the network will also provide training data on the various datasets (trade pairs for 
financial market data) but rewarding only occurs on data that hasn't occurred yet, or what the subnet defines as 
live data. Once the future data is known, we compare against the predictions made and reward based on performance.

What processing power does it require to be a validator?

In order to be a validator you simply need to have a server running in the EU (recommend Ireland, UK). This can be 
through VPN or a cloud-based server. This is because not all financial data can be accessed inside the US (for crypto). 
The actual processing power is light, as validators are really only comparing results against what occurs live 
therefore a relatively small machine can be a validator (will have exact details soon).

What is the expected data input and output as a miner?

For financial markets, the goal will be to predict the next 4-8 hours of closes on a 5m interval 
(so between ~50-100 data points). In order to help with the prediction, the subnet will provide the 
last 25-30 days of 5m data for the trade pair. You can expect the data to be in the format
[close_timestamp (ms), close, high, low, volume] where close, high, low, and volume are all linearly 
pre-scaled for you between 0.49 to 0.51. 

Can I be a miner with little knowledge?

Predicting on markets is very hard, but we want to help those who want to contribute to the network by providing 
a base model that can be used. This base can be used to build upon, or just run yourself to try and compete. The 
model and the specifications for running a miner will be released in the upcoming week as we finalize it.

I'm knowledgable about creating a competing prediction model on my own, can I prepare my miner to compete?

Yes, you can start from the base model in order to prepare for release or you can start from scratch and test on 
testnet (netuid 3). We will run on testnet for the next 2 weeks as we prepare for release on sn8. You can choose to 
use the training data provided by the subnet on each trade pair or prepare separately using your own 
training data (say on BTC to start).

Where can I begin testing?

You can begin testing on testnet netuid 3. You can follow the 
docs/running_on_testnet.md file inside the repo to run on testnet.
```

## Building a model
```text
How can I build a model? 

You can choose to build a model on your own or using the infrastructure inside this repository. If you choose
to use the infra provided in here, you can choose to leverage standardized financial market indicators provided
in mining_objects/financial_market_indicators.py as well as a standardized LSTM model 
in mining_objects/base_mining_model.py . You can generate a historical Bitcoin dataset using the script 
runnable/generate_historical_data.py and choose the timeframe you'd like to generate for using the comments provided
inside the script.

We've provided a basis for creating and testing models using runnable/miner_testing.py and runnable/miner_training.py
```

## Testing on testnet 
```text
You can begin testing on testnet netuid 3. You can follow the docs/running_on_testnet.md file inside the repo 
to run on testnet. Some recommendations when testing on testnet to speed up testing is pass the "test_only_historical"
argument for your validator. You can do this by using running
python neurons/validator.py --netuid 3 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug --test_only_historical 1
this will have the validator run every 5 seconds and test against historical data instead of live. If you only
want to test against live data then dont use this flag.

When youre testing we also recommend using 2 miners as a single miner won't provide enough responses to pass for
weighing. You can do this by passing a different port for the 2nd registered miner. 
You can run the second miner using the following example command:
python neurons/miner.py --netuid 3 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug --axon.port 8095
```


## TODO
```text
We will provide the cpu/gpu requirements in order to run a miner and a validator.
```