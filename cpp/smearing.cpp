#include <ROOT/RVec.hxx> // ROOT's fast array of numbers
#include <Math/Vector4D.h> // ROOT's 4-vectors
#include <TMath.h> // ROOT's TMath
#include <random>
#include <functional> // needed to hash the eventIDs

// define the spread functions
double pTSpread(double pT, double mass) {

    // fit parameters from plots by STAR collaborator Subhadip Pal acquired by averaging corresponding parameters for pi- and pi+
    double c = 3793.0 * TMath::Power(10, -6);
    double d = 8726.5 * TMath::Power(10, -6);

    return TMath::Sqrt(TMath::Power(c*pT, 2) + TMath::Power(d*mass/pT, 2) + d*d);
}

double phiSpread(double phi, double mass) {
    return 0.01;
}

double etaSpread(double eta, double mass) {
    return 0.01;
}

// define the smearing function
ROOT::RVec<ROOT::Math::PtEtaPhiMVector> smearing(const ROOT::RVec<ROOT::Math::PtEtaPhiMVector>& tracks, 
                                                 unsigned long long eventID) { // by using unsigned, we promise the compiler that eventID will not be a negative number saving the sign bit, which can now be used to store a larger number
                                                                               // the long long type stands for a massive 64-bit integer to avoid overflow

    // random number generator setup
    std::mt19937 generator(std::hash<unsigned long long>{}(eventID + 98765)); // std::mt19937 is a random number generator ("Mersenne Twister") with seed 98765+eventID 
                                                                              // we hash the eventId to ensure a unique, well-distributed seed for the Mersenne Twister as it produces correlated numbers when fed sequential seeds
    std::normal_distribution<double> dist(0.0, 1.0); // tells the generator to generate numbers according to a Gaussian distribution centered around 0.0 with spread 1.0

    // create a copy of the tracks that is safe to modify
    ROOT::RVec<ROOT::Math::PtEtaPhiMVector> smeared_tracks = tracks;

    // particle loop
    for (size_t i = 0; i < tracks.size(); i++) {
        ROOT::Math::PtEtaPhiMVector& track = smeared_tracks[i];

        // extract 4vector components
        double pT = track.Pt();
        double eta = track.Eta();
        double phi = track.Phi();

        // smearing utilizing the fact that we acquire a number from Gaussian distribution centered around 1.0 with a spread of pTSpread by scaling a number from Gaussian distribution centered around 0.0 with a spread of 1.0 and sjifting it so that its centre is at 1.0
        double mass = 0.13957; // pion mass in GeV/c^2
        
        double pTSmeared = pT * (1.0 + dist(generator)*pTSpread(pT, mass)); // pT multiplicative smearing (pT has relative errors)
        double etaSmeared = eta + dist(generator)*etaSpread(eta, mass); // eta additive smearing (anbsolute errors)
        double phiSmeared = phi + dist(generator)*phiSpread(phi, mass); // phi additive smearing

        // update the tracks
        track.SetPt(pTSmeared);
        track.SetEta(etaSmeared);
        track.SetPhi(phiSmeared); // this method automatically ensures that phi stays in [-pi, pi]
    }

    return smeared_tracks;
}