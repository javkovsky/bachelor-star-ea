#! /bin/bash

# set the seed of the datasets we want to train the neural networks on and the seed of the dataset to be unfolded
seedTrainingBiased=2
seedTrainingMB=20000
seedUnfolding=30000

# load the libraries via sourcing the CERN Virtual Machine File System (CVMFS)
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh

# set the path to the cloned GitHub repository -- CHANGE THIS according to YOUR path to the cloned repository
projectDir="$HOME/bachelor-star-ea/"

# copy the necessary files including the datasets onto the cluster node -- ensure that YOUR batch system has the $TMPDIR variable
mkdir -p "$TMPDIR/event_analysis/data/"

cp "$projectDir/multifold-ea.py" "$TMPDIR/event_analysis/multifold-ea.py"
cp "$projectDir/omnifold.py" "$TMPDIR/event_analysis/omnifold.py"

mkdir -p "$TMPDIR/event_analysis/data/$seedTrainingBiased/"
cp "$projectDir/data/$seedTrainingBiased/"observables*.root "$TMPDIR/event_analysis/data/$seedTrainingBiased/"

mkdir -p "$TMPDIR/event_analysis/data/$seedTrainingMB/"
cp "$projectDir/data/$seedTrainingMB/"observables*.root "$TMPDIR/event_analysis/data/$seedTrainingMB/"

mkdir -p "$TMPDIR/event_analysis/data/$seedUnfolding/"
cp "$projectDir/data/$seedUnfolding/"observables*.root "$TMPDIR/event_analysis/data/$seedUnfolding/"

# enter the directory on the node and launch the MultiFold script
cd "$TMPDIR/event_analysis/" || exit 1 # the script fails if it cannot enter the temporary directory
python3 multifold-ea.py --seedTrainingBiased $seedTrainingBiased --seedTrainingMB $seedTrainingMB --seedUnfolding $seedUnfolding

# copy the created graphs back into the project directory
cp -r "$TMPDIR/event_analysis/img/" "$projectDir/"  