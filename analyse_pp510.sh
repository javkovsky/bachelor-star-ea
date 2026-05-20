#! /bin/bash

# ROOT and PYTHIA8 setup -- CHANGE THIS to source your script which loads ROOT and PYTHIA8
source "$HOME/thisroot2026.sh"

# set the path to the cloned GitHub repository -- CHANGE THIS according to YOUR path to the cloned repository
projectDir="$HOME/bachelor-star-ea/"

# copy the necessary file onto the cluster node -- ensure that YOUR batch system has the $TMPDIR variable
mkdir "$TMPDIR/event_analysis/"
cp "$projectDir/plot-ea.py" "$TMPDIR/event_analysis/plot-ea.py"
cp -r "$projectDir/cpp/" "$TMPDIR/event_analysis/"

# enter the directory on the node and launch the data analysis script
cd "$TMPDIR/event_analysis/" || exit 1 # the script fails if it cannot enter the temporary directory
python3 plot-ea.py --seed 22

# copy the created graphs and the .root file with distributions for unfolding back into the project directory
cp -r "$TMPDIR/event_analysis/data/"* "$projectDir/data/"
cp -r "$TMPDIR/event_analysis/img/" "$projectDir/"  