# ssEinterpolator

**Interpolate numerical Slow Slip simulations**

This repository contains the source code for `ssEinterpolator`, a Python package for interpolating numerical Slow Slip simulations.

## Installation

Follow these instructions to set up a Conda environment and install the package in editable mode.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/yohaimagen/ssEinterpolator.git
cd ssEinterpolator
```


### 2. Create the Conda Environment

This project uses Conda to manage its dependencies. The `environment.yml` file contains all the necessary packages.

Create the environment using the following command:

```bash
conda env create -f environment.yml
```

This will create a new Conda environment named ssEinterpolator-env.

### 3. Activate the Environment
Before you can use the package, you must activate the environment:

```bash
conda activate ssEinterpolator-env
```

Your shell prompt should now indicate that the `ssEinterpolator-env` environment is active.

### 4. Install the Package
Finally, install the `ssEinterpolator` package in "editable" mode. This links the package to your source code, so any changes you make in the src/ directory will be immediately available.

```bash
pip install -e .
```

You are now ready to use the package!
