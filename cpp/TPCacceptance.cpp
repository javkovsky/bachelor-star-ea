// import libraries
#include <random>
#include <ROOT/RVec.hxx> // ROOT's fast array of numbers
#include <Math/Vector4D.h> // ROOT's 4-vectors

// define the acceptance function of TPC
double TPCacceptance(const ROOT::RVec<float>& pTEvent) {
    return 0.8;
}

// define a function which gives us a mask telling us which particles to discard
ROOT::RVec<int> TPCmask(size_t nParticles, const ROOT::RVec<float>& pTEvent) {
    static thread_local std::mt19937 generator(12345); // std::mt19937 is a random number generator ("Mersenne Twister") with seed 12345 
                                                       // static ensures that the generator is kept in RAM for the next time the function is called, otherwise the seed would reset everytime the function is called, giving us the same random numbers for all events
                                                       // thread_local makes sure that every core gets its own generator since we allow multi-threading
    std::uniform_real_distribution<double> dist(0.0, 1.0); // tells the generator to generate numbers between 0.0 and 1.0 according to a uniform distribution

    ROOT::RVec<int> mask(nParticles); // create an empty mask
    
    // particle loop
    for (size_t i = 0; i < nParticles; i++) {
        if (dist(generator) > TPCacceptance(pTEvent)) { // if the random number dist(generator) is greater than the TPC's acceptance, we reject the particle
            mask[i] = 0;
        }
        else { // otherwise, we accept the particle
            mask[i] = 1;
        }
    }

    return mask;
}