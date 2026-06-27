#! /bin/bash

# set the seed of the dataset we want to train the neural networks on and the seed of the dataset to be unfolded
seedTraining=2
seedUnfolding=30000

# load the libraries via sourcing the CERN Virtual Machine File System (CVMFS)
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc15-opt/setup.sh

# set the path to the cloned GitHub repository -- CHANGE THIS according to YOUR path to the cloned repository
projectDir="$HOME/bachelor-star-ea/"

# copy the necessary files including the dataset onto the cluster node -- ensure that YOUR batch system has the $TMPDIR variable
mkdir -p "$TMPDIR/event_analysis/data/"
cp "$projectDir/multifold-ea.py" "$TMPDIR/event_analysis/multifold-ea.py"
cp "$projectDir/omnifold.py" "$TMPDIR/event_analysis/omnifold.py"
cp -r "$projectDir/data/$seedTraining/" "$TMPDIR/event_analysis/data/"
cp -r "$projectDir/data/$seedUnfolding/" "$TMPDIR/event_analysis/data/"

# enter the directory on the node and launch the MultiFold script
cd "$TMPDIR/event_analysis/" || exit 1 # the script fails if it cannot enter the temporary directory
python3 multifold-ea.py --seedTraining $seedTraining --seedUnfolding $seedUnfolding

# copy the created graphs back into the project directory
cp -r "$TMPDIR/event_analysis/img/" "$projectDir/"  