// import libraries
#include <cmath> // standard math functions
#include <vector> // dynamic lists
#include "TMath.h" // ROOT's TMath
#include "ROOT/RVec.hxx" // ROOT's fast array of numbers

// define the function to calculate unweighted transverse spherocity
double calc_S0pT1(const ROOT::RVec<float>& uxEvent, // const is a safety lock to ensure that the function only reads data and does not change it
                  const ROOT::RVec<float>& uyEvent, // & tells C++ to use the original data rather than making copies in RAM
                  int NchEvent) {

    double minVal = 999999.0; // set minimum value to a large number, so that it gets overwritten
    // loop over particles in an event -- loop over potential minimization axes nhat
    for (size_t i = 0; i < NchEvent; i++) {
        double sumCrossProducts = 0.0; // reset sum

        // loop over particles
        for (size_t j = 0; j < NchEvent; j++) {
            sumCrossProducts += std::abs(uxEvent[j]*uyEvent[i] - uyEvent[j]*uxEvent[i]); // calculate sums of absolute values of cross products
        }

        // overwrite minimum value if we find a lower sum
        if (sumCrossProducts < minVal) {
            minVal = sumCrossProducts;
        }
    }

    // return S0pT1
    return (TMath::Pi() * TMath::Pi() / 4.0) * std::pow(minVal / NchEvent, 2);
}