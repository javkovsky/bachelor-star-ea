# bachelor-star-ea

This repository contains the codebase for a bachelor's thesis focused on event activity and shape observables in proton-proton (pp) collisions at the STAR experiment. The workflow consists of generating Monte Carlo events using PYTHIA8, simulating STAR detector effects (such as efficiency and smearing), calculating event activity (EA) observables, and performing Bayesian and deep-learning based unfolding to correct measured data back to the true particle-level distributions.

## Environment Setup
The project relies on a Conda environment named `star-bachelor-env`.
It requires the following core dependencies:
* Python 3.11
* ROOT
* PYTHIA8
* RooUnfold
* NumPy

## Project Structure

### Jupyter Notebooks (Core Workflow)
* `pythia8_generate-tree.ipynb`: Generates simulated pp collision events using PYTHIA8 and stores particle tracks in a ROOT TTree. It also applies a detector effect simulator for the Time Projection Chamber (TPC) and Time of Flight (TOF) detectors, producing a smeared dataset.
* `plot-ea.ipynb`: Processes the generated events to extract event activity observables, specifically multiplicity, transverse spherocity, and unweighted transverse spherocity. It produces 1D distributions, response matrices, and 2D kinematic spectra required for unfolding.
* `unfolding-Bayes-ea.ipynb`: Performs Bayesian unfolding using RooUnfold to correct the smeared data back to the true underlying physical distributions. It implements different unfolding methods (including global 1D unfolding and deconvoluted 1D unfolding per $p_T$ bin) and evaluates their quality using closure tests.

### C++ Macros (`cpp/`)
The repository uses ROOT's `RDataFrame` and `gInterpreter` to execute high-performance C++ macros directly from the Python notebooks.
* `generate_events.cpp`: Handles the PYTHIA8 initialization (Detroit tune, 510 GeV) and populates the initial ROOT tree with midrapidity final state charged particles.
* `TPCandTOFefficiency.cpp`: Defines acceptance and efficiency masks for the TPC and TOF detectors based on track kinematics.
* `smearing.cpp`: Applies Gaussian smearing to true $p_T$, $\eta$, and $\phi$ variables to simulate detector resolution.
* `extract_components.cpp`: Contains helper functions for extracting kinematic components ($p_T$, $\eta$, $\phi$, mass) from track vectors.
* `spherocities.cpp`: Computes the transverse spherocity ($S_0$) and unweighted transverse spherocity ($S_0^{p_T = 1}$) of the events.
* Unfolding utilities: `applyUM.cpp`, `normalizeUM.cpp`, and `deconvoluted1Dunfolding.cpp` handle the application, mathematical normalization, and bin-by-bin projection of unfolding matrices onto 2D histograms.

### Batch System Scripts
For running generation and analysis on the Sunrise cluster at FNSPE (CTU in Prague) using batch job arrays:
* `generate_pp510.sh`: Automates executing the PYTHIA8 tree generation python script (`pythia8_generate-tree.py`) on a cluster node and copies generated smeared output files back.
* `analyse_pp510.sh`: Submits batch jobs to run the `plot-ea.py` analysis script and retrieves generated event distributions and graphs for unfolding.

## Instructions
1. **Data Generation:** Run `pythia8_generate-tree.ipynb` (or execute locally/on a cluster via `generate_pp510.sh`) to generate the true collision datasets and simulate the STAR detector response.
2. **Event Activity Analysis:** Process the datasets by running `plot-ea.ipynb` (or `analyse_pp510.sh`). This constructs multiplicity and transverse spherocity observables and also 2D and 3D response matrices.
3. **Bayesian Unfolding:** Execute `unfolding-Bayes-ea.ipynb` using two different datasets (a training set and a smeared set). The notebook generates validation plots and closure tests comparing the true datasets with the RooUnfold Bayesian-corrected datasets.
4. **MultiFold:** TO DO This pipeline runs natively on CPU or GPU. If you are running on an NVIDIA GPU and encounter XLA libdevice warnings, you can optionally accelerate training by running: `conda install -c nvidia cuda-nvcc`.

## Gitignore Configuration
The `.gitignore` explicitly prevents uploading large local `.root` data files, generated `.pdf` figures, and local `.vscode` properties JSONs to version control.
