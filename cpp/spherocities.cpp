// import libraries
#include <TMath.h> // ROOT's TMath
#include <ROOT/RVec.hxx> // ROOT's fast array of numbers

// define the function to calculate unweighted transverse spherocity for an event
double calculateS0pT1(const ROOT::RVec<double>& uxEvent, // const is a safety lock to ensure that the function only reads data and does not change it
                      const ROOT::RVec<double>& uyEvent, // & tells C++ to use the original data rather than making copies in RAM
                      int NchEvent) {

    double minVal = 999999.0; // set minimum value to a large number, so that it gets overwritten
    
    // loop over particles in an event -- loop over potential minimization axes \vec{n}
    for (int i = 0; i < NchEvent; i++) {
        double sumCrossProducts = 0.0; // reset sum

        // loop over particles
        for (int j = 0; j < NchEvent; j++) {
            sumCrossProducts += TMath::Abs(uxEvent[j]*uyEvent[i] - uyEvent[j]*uxEvent[i]); // calculate sums of absolute values of cross products
        }

        // overwrite minimum value if we find a lower sum
        if (sumCrossProducts < minVal) {
            minVal = sumCrossProducts;
        }
    }

    // return S0pT1
    return (TMath::Pi() * TMath::Pi() / 4.0) * TMath::Power(minVal / NchEvent, 2);
}

// define the function to calculate transverse spherocity for an event
double calculateS0(const ROOT::RVec<double>& uxEvent, // const is a safety lock to ensure that the function only reads data and does not change it
                   const ROOT::RVec<double>& uyEvent, // & tells C++ to use the original data rather than making copies in RAM
                   const ROOT::RVec<double>& pxEvent, 
                   const ROOT::RVec<double>& pyEvent, 
                   const ROOT::RVec<double>& pTEvent,
                   int NchEvent) {

    double minVal = 999999.0; // set minimum value to a large number, so that it gets overwritten
    double pTSum = 0; // create and set the sum of absolute values of pT equal to zero
    
    // loop over particles in an event -- loop over potential minimization axes \vec{n}
    for (int i = 0; i < NchEvent; i++) {
        double sumCrossProducts = 0.0; // reset sum
        pTSum += TMath::Abs(pTEvent[i]);

        // loop over particles
        for (int j = 0; j < NchEvent; j++) {
            sumCrossProducts += TMath::Abs(pxEvent[j]*uyEvent[i] - pyEvent[j]*uxEvent[i]); // calculate sums of absolute values of cross products
        }

        // overwrite minimum value if we find a lower sum
        if (sumCrossProducts < minVal) {
            minVal = sumCrossProducts;
        }
    }

    // return S0
    return (TMath::Pi() * TMath::Pi() / 4.0) * TMath::Power(minVal / pTSum, 2);
}