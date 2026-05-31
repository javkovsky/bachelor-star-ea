#! /bin/bash

# set the seed of the dataset we want to analyse
seed=22

# ROOT and PYTHIA8 setup -- CHANGE THIS to source your script which loads ROOT and PYTHIA8
source "$HOME/thisroot2026.sh"

# set the path to the cloned GitHub repository -- CHANGE THIS according to YOUR path to the cloned repository
projectDir="$HOME/bachelor-star-ea/"

# copy the necessary files including the dataset onto the cluster node -- ensure that YOUR batch system has the $TMPDIR variable
mkdir -p "$TMPDIR/event_analysis/data/"
cp "$projectDir/plot-ea.py" "$TMPDIR/event_analysis/plot-ea.py"
cp -r "$projectDir/cpp/" "$TMPDIR/event_analysis/"
cp -r "$projectDir/data/$seed/" "$TMPDIR/event_analysis/data/"

# enter the directory on the node and launch the data analysis script
cd "$TMPDIR/event_analysis/" || exit 1 # the script fails if it cannot enter the temporary directory
python3 plot-ea.py --seed $seed

# copy the created graphs and the .root file with distributions for unfolding back into the project directory
cp "$TMPDIR/event_analysis/data/$seed/events$seed"_plots.root "$projectDir/data/$seed/"
cp "$TMPDIR/event_analysis/data/$seed/observables$seed".root "$projectDir/data/$seed/"
cp -r "$TMPDIR/event_analysis/img/" "$projectDir/"  