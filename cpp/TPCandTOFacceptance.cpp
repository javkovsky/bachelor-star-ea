// import libraries
#include <random>
#include <ROOT/RVec.hxx> // ROOT's fast array of numbers
#include <Math/Vector4D.h> // ROOT's 4-vectors -- we do not use it in this script, however it is used in PyROOT in Jupyter Notebook
#include <thread> // needed to extract CPU thread IDs
#include <functional> // needed to hash the thread IDs

// --------------
// TPC ACCEPTANCE
// --------------

// define the acceptance function of TPC
double TPCacceptance(const ROOT::RVec<double>& pTEvent) {
    return 0.8;
}

// define a function which gives us a mask telling us which particles to discard
ROOT::RVec<int> TPCmask(size_t nParticles, const ROOT::RVec<double>& pTEvent) {
    
    // random number generator setup
    static thread_local std::mt19937 generator([]{  // std::mt19937 is a random number generator ("Mersenne Twister") with seed 12345 
                                                    // static ensures that the generator is kept in RAM for the next time the function is called, otherwise the seed would reset everytime the function is called, giving us the same random numbers for all events
                                                    // thread_local makes sure that every core gets its own generator since we allow multi-threading
        size_t thread_id = std::hash<std::thread::id>{} // transforms thread ID into a size_t integer
        (std::this_thread::get_id()); // acquire thread ID (it is a special C++ class, that is why we need to hash it in order to perform math with it)
        return 12345 + thread_id; // create a unique seed for each thread to prevent having the same seed on every thread
    }()); // the () at the end tells C++ to run the temporary (lambda) function and use whatever it returns as the argument for the generator
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0); // tells the generator to generate numbers between 0.0 and 1.0 according to a uniform distribution

    ROOT::RVec<int> mask(nParticles); // create an empty mask
    
    // particle loop
    for (size_t i = 0; i < nParticles; i++) {
        if (dist(generator) < TPCacceptance(pTEvent)) { // if the random number dist(generator) is less than TPC's acceptance, we accept the particle
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
double TOFacceptance(const ROOT::RVec<double>& pTEvent) {
    return 0.6;
}

// define a function which gives us a mask telling us which particles were recorded by both TPC and TOF
ROOT::RVec<int> TPCandTOFmask(size_t nParticles, 
                              const ROOT::RVec<double>& pTEvent, 
                              const ROOT::RVec<int>& TPCaccepted) {
    static thread_local std::mt19937 generator([]{
        size_t thread_id = std::hash<std::thread::id>{}
        (std::this_thread::get_id());
        return 54321 + thread_id;
    }()); // descibed above
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0); // tells the generator to generate numbers between 0.0 and 1.0 according to a uniform distribution

    ROOT::RVec<int> mask(nParticles); // create an empty mask
    
    // particle loop
    for (size_t i = 0; i < nParticles; i++) {
        if (TPCaccepted[i] == 1 && dist(generator) < TOFacceptance(pTEvent)) { // if the particle was accepted by TPC and also a random number dist(generator) is less than TOF's acceptance, we accept the particle
            mask[i] = 1;
        }
        else { // otherwise, we reject the particle
            mask[i] = 0;
        }
    }

    return mask;
}

// asi by se jeste dalo zkombinovat do jedne funkce a jednoho for loopu, dale mi Gemini nabizi zbavit se for loopu a misto toho pouzit vektorizovane operace pomoci RVec