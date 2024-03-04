# Miner

While running a miner, you should see three types of requests: LiveForwardHash, LiveForward, and LiveBackward.

LiveForwardHash is requested first, which will be a hash of your predictions. LiveForward will be requested 60 seconds later which will request the actual predictions made (non-hashed). Using the hash and the actual predictions, validators can validate the authenticity of the predictions made, ensuring no participants are copying anothers.

LiveForwardHash - will be when you provide a hash of your predictions.
LiveForward - will be when your miner provides your actual predictions.
LiveBackward - will receive the results that occurred live if you want to use them for any updating purposes.
You'll receive rewards for your predictions ~10 hours after making them. Therefore, if you start running on the network, you should expect a lag in receiving rewards. Predictions are reviewed and rewarded every 30 minutes.

In the short term, your miner will not be trained on the network (TrainingBackward and TrainingForward will only run sometimes). Please train beforehand as we'll only be using BTC/USD on the 5m to begin, which you can prepare for.

**IMPORTANT**

Before attempting to register on mainnet, we strongly recommend that you run a validator on the testnet. To do ensure you add the appropriate testnet flags.

| Environment | Netuid |
| ----------- | -----: |
| Mainnet     |      8 |
| Testnet     |      3 |

Your incentive mechanisms running on the mainnet are open to anyone. They emit real TAO. Creating these mechanisms incur a lock_cost in TAO.

**DANGER**

- Do not expose your private keys.
- Only use your testnet wallet.
- Do not reuse the password of your mainnet wallet.
- Make sure your incentive mechanism is resistant to abuse.

# System Requirements

- Requires **Python 3.10 or higher.**
- [Bittensor](https://github.com/opentensor/bittensor#install)

Below are the prerequisites for miners. You may be able to make a miner work off lesser specs but it is not recommended.

- 2 vCPU + 8 GB memory
- Run the miner using CPU

# Getting Started

## 1. Install TSPS

Clone repository

```bash
git clone https://github.com/taoshidev/time-series-prediction-subnet.git
```

Change directory

```bash
cd time-series-prediction-subnet
```

Create Virtual Environment

```bash
python3 -m venv venv
```

Activate a Virtual Environment

```bash
. venv/bin/activate
```

Disable pip cache

```bash
export PIP_NO_CACHE_DIR=1
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a local and editable installation

```bash
python3 -m pip install -e .
```

## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your miner.

The miner will be registered to the subnet specified. This ensures that the miner can run the respective miner scripts.

Create a coldkey and hotkey for your miner wallet.

```bash
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

## 2a. (Optional) Getting faucet tokens

Faucet is disabled on the testnet. Hence, if you don't have sufficient faucet tokens, ask the Bittensor Discord community for faucet tokens.

## 3. Register keys

This step registers your subnet miner keys to the subnet, giving it the first slot on the subnet.

```bash
btcli subnet register --wallet.name miner --wallet.hotkey default
```

To register your miner on the testnet add the `--subtensor.network test` flag.

Follow the below prompts:

```bash
>> Enter netuid (0): # Enter the appropriate netuid for your environment
Your balance is: # Your wallet balance will be shown
The cost to register by recycle is τ0.000000001 # Current registration costs
>> Do you want to continue? [y/n] (n): # Enter y to continue
>> Enter password to unlock key: # Enter your wallet password
>> Recycle τ0.000000001 to register on subnet:8? [y/n]: # Enter y to register
📡 Checking Balance...
Balance:
  τ5.000000000 ➡ τ4.999999999
✅ Registered
```

## 4. Check that your keys have been registered

This step returns information about your registered keys.

Check that your miner has been registered:

```bash
btcli wallet overview --wallet.name miner
```

To check your miner on the testnet add the `--subtensor.network test` flag

The above command will display the below:

```bash
Subnet: 8 # or 3 on testnet
COLDKEY  HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
miner    default  196    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000        *      134  none  5HRPpSSMD3TKkmgxfF7Bfu67sZRefUMNAcDofqRMb4zpU4S6
1        1        1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                               Wallet balance: τ4.998999856
```

## 6. Run your Miner

Run the subnet miner:

```bash
python neurons/miner.py --netuid 8  --wallet.name miner --wallet.hotkey default --logging.debug
```

To run your miner on the testnet add the `--subtensor.network test` flag and `--netuid 3` flag.

You will see the below terminal output:

```bash
>> 2023-08-08 16:58:11.223 |       INFO       | Running miner for subnet: 8 on network: ws://127.0.0.1:9946 with config: ...
```

## 7. Get emissions flowing

Register to the root network using the `btcli`:

```bash
btcli root register
```

To register your miner to the root network on testnet use the `--subtensor.network test` flag.

Then set your weights for the subnet:

```bash
btcli root weights
```

To set your weights on testnet `--subtensor.network test` flag.

## 8. Stopping your miner

To stop your miner, press CTRL + C in the terminal where the miner is running.

# Testing

You can begin testing TSPS testnet with netuid 3.

We recommend using two miners when testing, as a single miner won't provide enough responses to pass for weighing. You can pass a different port for the 2nd registered miner.

You can run the second miner using the following example command:

```bash
$ python neurons/miner.py --netuid 3 --subtensor.network test --wallet.name miner --wallet.hotkey default --logging.debug --axon.port 8095
```
