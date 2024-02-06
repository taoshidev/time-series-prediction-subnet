# Running Subtensor Locally

- [Running docker](#running-docker)
- [Compiling your own binary](#compiling-your-own-binary)

## Running Docker

### Install git

`sudo apt install git`

### Install Docker Engine

You can follow Docker's [oficial installation guides](https://docs.docker.com/engine/install/)

## Run node-subtensor container

Clone the subtensor repository:

```bash
git clone https://github.com/opentensor/subtensor.git
```

Navigate to subtensor directory:

```bash
cd subtensor
```

To run a lite node on the mainnet:

```bash
sudo ./scripts/run/subtensor.sh -e docker --network mainnet --node-type lite
# or mainnet archive node: sudo ./scripts/run/subtensor.sh -e docker --network mainnet --node-type archive
# or testnet lite node:    sudo ./scripts/run/subtensor.sh -e docker --network testnet --node-type lite
# or testnet archive node: sudo ./scripts/run/subtensor.sh -e docker --network testnet --node-type archive
```

## Compiling your own binary

### Requirements

```bash
sudo apt install build-essential git make clang libssl-dev llvm libudev-dev protobuf-compiler -y
```

### Install Rust

Download the `rustup` installation program and use it to install Rust by running the following command:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup-init.sh
```

Make `rustup-init.sh` executable:

```bash
chmod +x rustup-init.sh
```

Run script:

```bash
./rustup-init.sh # you can select default options in the prompts you will be given
```

Update your current shell to include Cargo by running the following command:

```bash
source "$HOME/.cargo/env"
```

### Rustup update

```bash
rustup default stable && \
 rustup update && \
 rustup update nightly && \
 rustup target add wasm32-unknown-unknown --toolchain nightly
```

### Compiling

Clone the subtensor repository:

```bash
git clone https://github.com/opentensor/subtensor.git

```

Navigate to directory

```bash
cd subtensor
```

Compile subtensor by running the following command:

```bash
cargo build --release --features runtime-benchmarks
```

### Running the node

#### Mainnet / Lite node

```bash
sudo ./scripts/run/subtensor.sh -e binary --network mainnet --node-type lite
```

#### Mainnet / Archive node

```bash
sudo ./scripts/run/subtensor.sh -e docker --network mainnet --node-type archive
```

#### Testnet / Lite node

```bash
sudo ./scripts/run/subtensor.sh -e docker --network testnet --node-type lite
```

#### Testnet / Archive node

```bash
sudo ./scripts/run/subtensor.sh -e docker --network testnet --node-type archive
```
