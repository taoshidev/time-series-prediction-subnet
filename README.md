<p align="center">
    <a href="https://taoshi.io">
      <img width="500" alt="taoshi - subnet 8 repo logo" src="https://i.imgur.com/deBxDUm.png">
    </a>
    
</p>

<p align="center">
  <a href="https://taoshi.io">Website</a>
  ·
  <a href="#installation">Installation</a>
  ·  
  <a href="https://dashboard.taoshi.io/">Dashboard</a>
  ·
  <a href="https://twitter.com/taoshiio">Twitter</a>
    ·
  <a href="https://twitter.com/taoshiio">Bittensor</a>
</p>

---

<details>
  <summary>Table of contents</summary>
  <ol>
    <li>
      <a href="#bittensor">Bittensor</a>
      <ol>
        <li>
          <a href="#subnets">Subnets</a>
        </li>
        <li>
          <a href="#miners">Miners</a>
        </li>
        <li>
          <a href="#validators">Validators</a>
        </li>
      </ol>
    </li>
    <li><a href="#prediction-subnet">Prediction Subnet</a></li>
    <li><a href="#features">Featuers</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ol>
        <li>
          <a href="#running-a-validator">Running a Validator</a>
        </li>
        <li>
          <a href="#running-a-miner">Running a Miner</a>
        </li>
      </ol>
    </li>
    <li><a href="#building-a-model">Building A Model</a></li>
    <li><a href="#testing">Testing</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

---

# Bittensor

Bittensor is a mining network, similar to Bitcoin, that includes built-in incentives designed to encourage computers to provide access to machine learning models in an efficient and censorship-resistant manner. Bittensor is comprised of Subnets, Miners, and Validators.

**Explain Like I'm Five**

Bittensor is an API that connects machine learning models and incentivizes correctness through the power of the blockchain.

### Subnets

Subnets are decentralized networks of machines that collaborate to train and serve machine learning models.

### Miners

Miners run machine learning models. They fulfill requests from the Validators.

### Validators

Validators query and prompt the Miners. Validators also validate miner requests. Validators are also storefronts for data.

# Prediction Subnet

This repository contains the code for the Time-Series Prediction Subnet (TSPS) developed by Taoshi.

Initially, the primary focus of TSPS is to forecast the future trends of financial markets, starting with predicting the price movements of Bitcoin. Eventually, TSPS will expand to include a broader range of cryptocurrency trading pairs. Subsequently, our scope will broaden to encompass additional financial markets.

As the project evolves, we anticipate that miners will diversify their focus, specializing in different subjects or domains. Some may concentrate on specific areas, while others, particularly those with advanced skills, tackle multiple topics, such as sports betting or various client-requested issues.

# Features

🛠️&nbsp;Open Source Modeling<br>
🫰&nbsp;Intraday Bitcoin Predictions<br>
📈&nbsp;Higher Payouts<br>
📉&nbsp;Lower Registration Fees<br>
💪&nbsp;Superior Cryptocurrency Infrastructure<br>
⚡&nbsp;Faster Payouts<br>

# Prerequisites

Below are the prerequisites for validators and miners, you may be able to make miner and validator work off lesser specs.

Requires **Python 3.10 or higher.**

**Validator**

- 2 vCPU + 8 GB memory
- 100 GB balanced persistent disk

**Miner**

- 2 vCPU + 8 GB memory
- Run the miner using CPU

# Installation

On Linux

```bash
# install git and subpackages
$ sudo apt install git-all

# install pip package manager for python 3
$ sudo apt install python3-pip

# install venv virtual environment package for python 3
$ sudo apt-get install python3-venv

# clone repo
$ git clone https://github.com/taoshidev/time-series-prediction-subnet.git

# change directory
$ cd time-series-prediction-subnet

# create virtual environment
$ python3 -m venv venv

# activate the virtual environment
$ . venv/bin/activate

# disable pip cache
$ export PIP_NO_CACHE_DIR=1

# install dependencies
$ pip install -r requirements.txt

# create a local and editable installation
$ python -m pip install -e .

```

# Usage

## Running a Validator

Your validator will request predictions hourly based on your randomized interval to help distribute the load. After live results come 10 hours after predictions, rewards will be distributed.

### Using Provided Scripts

These validators run and update themselves automatically.

To run a validator, follow these steps:

1. [Install Prediction Subnet.](#installation)
2. Install [PM2](https://pm2.io) and the (jq)[https://jqlang.github.io/jq/] package on your system.

On Linux:

```bash
# update lists
$ sudo apt update

# JSON-processor
$ sudo apt install jq

# install npm
$ sudo apt install npm

# install pm2 globally
$ sudo npm install pm2 -g

# update pm2 process list
$ pm2 update
```

On MacOS:

```bash
# update lists
$ brew update

# JSON-processor
$ brew install jq

# install npm
$ brew install npm

# install pm2 globally
$ sudo npm install pm2 -g

# update pm2 process list
$ pm2 update
```

3. Be sure to install venv for the repo.

```bash
# /time-series-prediction-subnet

# create virtual environment
$ python3 -m venv venv

# activate virtual environment
$ source venv/bin/activate

# install packages
$ pip install -r requirements.txt
```

4. Run the `run.sh` script, which will run your validator and pull the latest updates as they are issued.

```bash
$ pm2 start run.sh --name sn8 -- --wallet.name <wallet> --wallet.hotkey <hotkey> --netuid 8
```

This will run two PM2 process:

1. A process for the validator, called sn8 by default (you can change this in run.sh)
2. And a process for the run.sh script (in step 4, we named it tsps). The script will check for updates every 30 minutes,
   if there is an update, it will pull, install, restart tsps, and restart itself.

### Manually

If there are any issues with the run script or you choose not to use it, run a validator manually.

```bash
$ python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey>
```

You can also run your script in the background. Logs are stored in `nohup.out`.

```bash
$ nohup python neurons/validator.py --netuid 8 --wallet.name <wallet> --wallet.hotkey <hotkey> &
```

## Running a Miner

If you're running a miner, you should see two types of requests: LiveForward and LiveBackward. LiveForward will be when your miner performs predictions; LiveBackward will receive the results that occurred live if you want to use them for any updating purposes.

You'll receive rewards for your predictions ~10 hours after making them. Therefore, if you start running on the network, you should expect a lag in receiving rewards. Predictions are reviewed and rewarded every 30 minutes.

In the short term, your miner will not be trained on the network (TrainingBackward and TrainingForward will only run sometimes). Please train beforehand as we'll only be using BTC/USD on the 5m to begin, which you can prepare for.

# Running on mainnet

You can run on mainnet by following the instructions in `docs/running_on_mainnet.md`.

If you are running into issues, please run with `--logging.debug` and `--logging.trace` set so you can better
analyze why your miner isn't running.

The current flow of information is as follows:

1. Validators request predictions every 30 minutes.
2. Miners make predictions.
3. Validators store predictions.
4. Validators wait until data is available to compare against (roughly a day and a half for 5m data).
5. Validators compare when data is ready.

So don't expect emissions on predictions immediately; you will need to wait for the predicted information to occur to be compared against.

# Building a model

You can build a model on your own or use the infrastructure inside this repository.

If you choose to use the infrastructure provided in here, you can choose to leverage standardized financial market indicators provided in `mining_objects/financial_market_indicators.py` as well as a standardized LSTM model in `mining_objects/base_mining_model.py`.

You can generate a historical Bitcoin dataset using the script
`runnable/generate_historical_data.py` and choose the timeframe you'd like to generate for using the comments provided inside the script.

We've provided a basis for creating and testing models using `runnable/miner_testing.py` and `runnable/miner_training.py`

Please note that we are constantly adjusting these scripts and will make improvements over time.

# Testing

You can begin testing on testnet netuid 3. You can follow the `docs/running_on_testnet.md` file inside the repo
to run on testnet.

A recommendation when testing on testnet to speed up testing is to pass the `test_only_historical`
argument for your validator.

You can do this by using running:

```bash
$ python neurons/validator.py --netuid 3 --subtensor.network test --wallet.name validator --wallet.hotkey default --logging.debug --test_only_historical 1
```

This will have the validator run and test against historical data instead of live. Don't use this flag if you only want to test against live data.

We also recommend using two miners when testing, as a single miner won't provide enough responses to pass for weighing. You can pass a different port for the 2nd registered miner.

You can run the second miner using the following example command:

```bash
$ python neurons/miner.py --netuid 3 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug --axon.port 8095
```

# FAQ

<details>
  <summary>How does rewarding work?</summary>
  <br>
  <p>
    Miners are rewarded based on predicting future information or live data as we classify it in the subnet. The network will also provide training data on the various datasets (trade pairs for financial market data) to increase miner accuracy. However, rewarding occurs on data that has yet to happen or what the subnet defines as live data. Once the future data is known, we compare it against the predictions made and reward based on performance.
  </p>
</details>

<details>
  <summary>What processing power is required to be a validator?</summary>
  <br>
  <p>
    To be a validator, you must have a server running in the EU (recommend Ireland, UK). This server can be through a VPN or a cloud-based server. Not all financial data can be accessed inside the US (for crypto). The actual processing power is light, as validators only compare results against what occurs live; therefore, a relatively small machine can be a validator.
  </p>
</details>

<details>
  <summary>What is the expected data input and output as a miner?</summary>
  <br>
  <p>
For financial markets, the goal will be to predict the next 8 hours of closes on a 5m interval (100 closes). The subnet will provide the trade pair's last 25-30 days of 5m data to help achieve that goal. You'll likely want to incorporate additional data into your modeling techniques.
    
    Input Features: [close_timestamp (milliseconds), close, high, low, volume]
    Target Feature: [close]
  </p>
</details>

<details>
  <summary>Can I be a miner with little knowledge?</summary>
  <br>
  <p>
    Predicting on markets is very hard, but we want to help those who want to contribute to the network by providing models that can be used. These models can be used to build upon, or just run yourself to try and compete.

    You can participate by running these pre-built & pre-trained models provided to all miners [here](https://huggingface.co/Taoshi/model_v4).

    These model are already built into the core logic of `neurons/miner.py` for you to run and compete as a miner. All you need to do is run `neurons/miner.py` and specify the model you want to run as an argument through --base_model:
    --base_model model_v4_1

  </p>

</details>

<details>
  <summary>I'm knowledgeable about creating a competing prediction model on my own. Can I prepare my miner to compete?</summary>
  <br>
  <p>
You can start using the hugging face models to prepare for release or from scratch and test on testnet (netuid 3). You can use the training data provided by the subnet or prepare separately using your training data (say on BTC to start).
  </p>
</details>

<details>
  <summary>Where can I begin testing?</summary>
  <br>
  <p>
    You can begin testing on testnet netuid 3. You can follow the
    `docs/running_on_testnet.md` file inside the repo to run on testnet.
  </p>
</details>

---

# Contributing

For instructions on how to contribute to Taoshi, see CONTRIBUTING.md and Taoshi's code of conduct.

# License

Refer to the <a href='?tab=MIT-1-ov-file'>License</a> page for information about Taoshi's licensing.

Bittensor's source code in this repository is licensed under the MIT License.
