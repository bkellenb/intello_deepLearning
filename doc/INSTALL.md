# Installation

The instructions below apply to a Debian-based Linux distribution (Debian, Ubuntu, etc.) and have been tested on Ubuntu 20.04 LTS (x86). They make use of the environment manager [Conda](http://conda.io/), although alternatives like [Virtualenv](https://virtualenv.pypa.io/) can also be employed with respective adaptations.

## Environment setup

1. Install dependencies
```bash
sudo apt update && sudo apt install -y build-essential gdal-bin libgdal-dev
```

2. Download and install [Conda](http://conda.io/) (Miniconda is enough) and create a new environment:

```bash
conda create -n intello python=3.8 -y
conda activate intello
```

3. Install basic requirements:
```bash
pip install -U -r requirements.txt
conda install -c conda-forge gdal -y    # GDAL needs to be installed separately due to dependency conflicts
```