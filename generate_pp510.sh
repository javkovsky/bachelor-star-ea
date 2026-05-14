#! /bin/bash

# ROOT and PYTHIA8 setup -- CHANGE THIS to source your script which loads ROOT and PYTHIA8
source "$HOME/thisroot2026.sh"

# set the path to the cloned GitHub repository -- CHANGE THIS according to YOUR path to the cloned repository
projectDir="$HOME/bachelor-star-ea/"

# copy the necessary file onto the cluster node -- ensure that YOUR batch system has the $TMPDIR variable
mkdir "$TMPDIR/event_simulation/"
cp "$projectDir/pythia8_generate-tree.py" "$TMPDIR/event_simulation/pythia8_generate-tree.py"
cp -r "$projectDir/cpp/" "$TMPDIR/event_simulation/"

# enter the directory on the node and launch the data generation script
cd "$TMPDIR/event_simulation/" || exit 1 # the script fails if it cannot enter the temporary directory
python3 pythia8_generate-tree.py

# copy the generated smeared events back into the project directory (the files also include true events)
cp -r "$TMPDIR"/event_simulation/data/events*_smeared.root "$projectDir/"