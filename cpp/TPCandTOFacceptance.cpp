// import libraries
#include <random>
#include <ROOT/RVec.hxx> // ROOT's fast array of numbers
#include <Math/Vector4D.h> // ROOT's 4-vectors -- we do not use it in this script, however it is used in PyROOT in Jupyter Notebook
#include <functional> // needed to hash the eventIDs

// --------------
// TPC ACCEPTANCE
// --------------

// define the acceptance function of TPC
double TPCacceptance(double pT) {
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
        if (dist(generator) < TPCacceptance(pTEvent[i])) { // if the random number dist(generator) is less than TPC's acceptance, we accept the particle
            mask[i] = 1;
        }
        else { // otherwise, we reject the particle
            mask[i] = 0;
        }
    }

    return mask;
}

// ----------------
// + TOF ACCEPTANCE
// ----------------

// define the acceptance function of TOF
double TOFacceptance(double pT) {
    return 0.6;
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
        if (dist(generator) < TOFacceptance(pTEvent[i]) && TPCaccepted[i] == 1) { // if the particle was accepted by TPC and also a random number dist(generator) is less than TOF's acceptance, we accept the particle
                                                                                  // dist(generator) MUST be on the left side so it evaluates first, guaranteeing the generator ticks exactly once per particle to maintain reproducibility
            mask[i] = 1;
        }
        else { // otherwise, we reject the particle
            mask[i] = 0;
        }
    }

    return mask;
}

// asi by se jeste dalo zkombinovat do jedne funkce a jednoho for loopu, dale mi Gemini nabizi zbavit se for loopu a misto toho pouzit vektorizovane operace pomoci RVec