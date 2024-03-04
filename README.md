<p align="center">
    <a href="https://taoshi.io">
      <img width="500" alt="taoshi - subnet 8 repo logo" src="https://i.imgur.com/deBxDUm.png">
    </a>
    
</p>

<div align='center'>

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.com/channels/799672011265015819/1162384774170677318)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

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
    ·
  <a href='https://huggingface.co/Taoshi'>Hugging Face</a>
</p>

---

<details>
  <summary>Table of contents</summary>
  <ol>
    <li>
      <a href="#bittensor">What is Bittensor?</a>
    </li>
    <li><a href="#prediction-subnet">Prediction Subnet</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li>
      <a href="#get-started">Get Started</a>
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

<details id='bittensor'>
  <summary>What is Bittensor?</summary>

Bittensor is a mining network, similar to Bitcoin, that includes built-in incentives designed to encourage computers to provide access to machine learning models in an efficient and censorship-resistant manner. Bittensor is comprised of Subnets, Miners, and Validators.

> Explain Like I'm Five

Bittensor is an API that connects machine learning models and incentivizes correctness through the power of the blockchain.

### Subnets

Subnets are decentralized networks of machines that collaborate to train and serve machine learning models.

### Miners

Miners run machine learning models. They fulfill requests from the Validators.

### Validators

Validators query and prompt the Miners. Validators also validate miner requests. Validators are also storefronts for data.

</details>

---

# Prediction Subnet

This repository contains the code for the Time-Series Prediction Subnet (TSPS) developed by Taoshi.

Initially, the primary focus of TSPS is to forecast the future trends of financial markets, starting with predicting the price movements of Bitcoin. Eventually, TSPS will expand to include a broader range of cryptocurrency trading pairs. Subsequently, our scope will broaden to encompass additional financial markets.

As the project evolves, we anticipate that miners will diversify their focus, specializing in different subjects or domains. Some may concentrate on specific areas, while others, particularly those with advanced skills, tackle multiple topics, such as sports betting or various client-requested issues.

## Features

🛠️&nbsp;Open Source Modeling<br>
🫰&nbsp;Intraday Bitcoin Predictions<br>
📈&nbsp;Higher Payouts<br>
📉&nbsp;Lower Registration Fees<br>
💪&nbsp;Superior Cryptocurrency Infrastructure<br>
⚡&nbsp;Faster Payouts<br>

# Get Started

### Validator Installation

Please see our [Validator Installation](https://github.com/taoshidev/time-series-prediction-subnet/blob/main/docs/validator.md) guide.

### Miner Installation

Please see our [Miner Installation](https://github.com/taoshidev/time-series-prediction-subnet/blob/main/docs/miner.md) guide.

# Building a model

You can build a model on your own or use the infrastructure inside this repository.

If you choose to use the infrastructure provided in here, you can choose to leverage standardized financial market indicators provided in `mining_objects/financial_market_indicators.py` as well as a standardized LSTM model in `mining_objects/base_mining_model.py`.

You can generate a historical Bitcoin dataset using the script
`runnable/generate_historical_data.py` and choose the timeframe you'd like to generate for using the comments provided inside the script.

We've provided a basis for creating and testing models using `runnable/miner_testing.py` and `runnable/miner_training.py`

Please note that we are constantly adjusting these scripts and will make improvements over time.

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
For financial markets, the goal will be to predict the next 8 hours of closes on a 5m interval (100 closes).
At a minimum, you should be using OHLCV. You should consider using additional data sources in order to be competitive.
    
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

For instructions on how to contribute to Taoshi, see CONTRIBUTING.md.

# License

Copyright © 2023 Taoshi Inc

```text
Taoshi All rights reserved.
Source code produced by Taoshi Inc may not be reproduced, modified, or distributed
without the express permission of Taoshi Inc.
```

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
