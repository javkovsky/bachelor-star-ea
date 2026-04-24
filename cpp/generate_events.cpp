#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"
#include <vector>
#include <string>
#include <cmath>

// define the function for event generation
void generateEvents(int nEvents, int seed, std::string outFileName) {

    // -------------
    // PYTHIA8 setup
    // -------------

    Pythia8::Pythia pythia;

    pythia.readString("Beams:idA = 2212"); // proton
    pythia.readString("Beams:idB = 2212"); // proton
    pythia.readString("Beams:eCM = 510."); // collision energy in GeV

    pythia.readString("hardQCD:all = on"); // enable hard QCD processes
    pythia.readString("PhaseSpace:pTHatMin = 20."); // minimum pT of scattered partons in GeV
    pythia.readString("PartonLevel:MPI = on"); // enable multi-parton interactions
    pythia.readString("PartonLevel:ISR = on"); // enable initial state radiation
    pythia.readString("PartonLevel:FSR = on"); // enable final state radiation
    pythia.readString("HadronLevel:Decay = on"); // enable decay of hadronization products

    pythia.readString("Tune:pp = 33"); // the Detroit tune for RHIC

    pythia.readString("Random:setSeed = on"); // enable random seed setting
    pythia.readString("Random:seed = " + std::to_string(seed)); // set random seed to 0 (random every time) or to a specific value for reproducibility

    pythia.init(); // initialize PYTHIA8

    // --------------------------
    // ROOT TFile and TTree setup
    // --------------------------

    TFile* file = new TFile(outFileName.c_str(), "RECREATE"); // create a .root file
    TTree* tree = new TTree("events", "A tree with pp events from PYTHIA8"); // create the tree

    // ALLOCATE MEMORY FOR BRANCHES
    // scalars
    unsigned long long eventID = 0;

    // vectors
    std::vector<double> pT, eta, phi, mass;

    // CREATE BRANCHES
    tree -> Branch("eventID", &eventID);
    tree -> Branch("pT", &pT);
    tree -> Branch("eta", &eta);
    tree -> Branch("phi", &phi);
    tree -> Branch("mass", &mass);

    // ----------------
    // EVENT GENERATION
    // ----------------

    for (size_t iEvent = 0; iEvent < nEvents; iEvent++) {

        // generate an event
        if (!pythia.next()) continue;

        // clear the vectors for each event
        pT.clear(); 
        eta.clear(); 
        phi.clear(); 
        mass.clear();

        // store the event ID
        eventID = iEvent;

        // PARTICLE LOOP
        for (size_t iParticle = 0; iParticle < pythia.event.size(); iParticle++) {
            
            auto particle = pythia.event[iParticle];

            // select only final state charged particles in midrapidity, add pT cut of 0.2 GeV typical for STAR
            if (particle.isFinal() && particle.isCharged() && std::abs(particle.eta()) < 1 && particle.pT() > 0.2) {

                // fill the vectors with respective data
                pT.push_back(particle.pT());
                eta.push_back(particle.eta());
                phi.push_back(particle.phi());
                mass.push_back(particle.m());
            }
        }

        // store the event in the tree
        tree -> Fill();
    }

    // --------------
    // WRITE AND SAVE
    // --------------

    file -> Write();
    file -> Close();
    delete file;
}