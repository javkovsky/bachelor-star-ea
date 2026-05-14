// import libraries
#include <random>
#include <ROOT/RVec.hxx> // ROOT's fast array of numbers
#include <Math/Vector4D.h> // ROOT's 4-vectors -- we do not use it in this script, however it is used in PyROOT in Jupyter Notebook
#include <TMath.h> // ROOT's TMath
#include <functional> // needed to hash the eventIDs

// --------------
// TPC EFFICIENCY
// --------------

// define the efficiency function of TPC
double TPCefficiency(double pT) {
    return 0.8;
}

// define a function which gives us a mask telling us which particles to discard
ROOT::RVec<int> TPCmask(size_t nParticles, const ROOT::RVec<double>& pTEvent, unsigned long long eventID) {
    
    // random number generator setup
    std::mt19937 generator(std::hash<unsigned long long>{}(eventID + 12345)); // std::mt19937 is a random number generator ("Mersenne Twister") with seed 12345+eventID 
                                                                              // we hash the eventId to ensure a unique, well-distributed seed for the Mersenne Twister as it produces correlated numbers when fed sequential seeds
    std::uniform_real_distribution<double> dist(0.0, 1.0); // tells the generator to generate numbers between 0.0 and 1.0 according to a uniform distribution

    ROOT::RVec<int> mask(nParticles); // create an empty mask
    
    // particle loop
    for (size_t i = 0; i < nParticles; i++) {
        if (dist(generator) < TPCefficiency(pTEvent[i])) { // if the random number dist(generator) is less than TPC's efficiency, we accept the particle
            mask[i] = 1;
        }
        else { // otherwise, we reject the particle
            mask[i] = 0;
        }
    }

    return mask;
}

// ----------------
// + TOF EFFICIENCY
// ----------------

// define the efficiency function of TOF
double TOFefficiency(double pT) {

    // fit parameters from TPC-TOF matching efficiency plot provided by STAR collaborator Oliver Matonoha
    double c0 = 0.675;
    double c1 = 0.3387;
    double c2 = 0.6878;

    return c0 * (1 - TMath::Exp(-TMath::Power(pT/c1, c2)));
}

// define a function which gives us a mask telling us which particles were recorded by both TPC and TOF
ROOT::RVec<int> TPCandTOFmask(size_t nParticles, 
                              const ROOT::RVec<double>& pTEvent,
                              unsigned long long eventID, 
                              const ROOT::RVec<int>& TPCaccepted) {

    // random number generator setup
    std::mt19937 generator(std::hash<unsigned long long>{}(eventID + 54321)); // std::mt19937 is a random number generator ("Mersenne Twister") with seed 54321+eventID 
                                                                              // we hash the eventId to ensure a unique, well-distributed seed for the Mersenne Twister as it produces correlated numbers when fed sequential seeds
    std::uniform_real_distribution<double> dist(0.0, 1.0); // tells the generator to generate numbers between 0.0 and 1.0 according to a uniform distribution

    ROOT::RVec<int> mask(nParticles); // create an empty mask
    
    // particle loop
    for (size_t i = 0; i < nParticles; i++) {
        if (dist(generator) < TOFefficiency(pTEvent[i]) && TPCaccepted[i] == 1) { // if the particle was accepted by TPC and also a random number dist(generator) is less than TOF's efficiency, we accept the particle
                                                                                  // dist(generator) MUST be on the left side so it evaluates first, guaranteeing the generator ticks exactly once per particle to maintain reproducibility
            mask[i] = 1;
        }
        else { // otherwise, we reject the particle
            mask[i] = 0;
        }
    }

    return mask;
}