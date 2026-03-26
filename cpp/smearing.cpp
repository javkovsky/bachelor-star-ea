#include <ROOT/RVec.hxx> // ROOT's fast array of numbers
#include <Math/Vector4D.h> // ROOT's 4-vectors
#include <TMath.h> // ROOT's TMath
#include <random>
#include <thread> // needed to extract CPU thread IDs
#include <functional> // needed to hash the thread IDs

// define the spread functions
double pTSpread(double pT, double mass) {

    // fit parameters from Subhadip acquired by averaging corresponding parameters for pi- and pi+
    double c = 3793.0 * TMath::Power(10, -6);
    double d = 8726.5 * TMath::Power(10, -6);

    return TMath::Sqrt(TMath::Power(c*pT, 2) + TMath::Power(d*mass/pT, 2) + d*d);
}

double phiSpread(double pT, double mass) {
    return 0.01;
}

double etaSpread(double eta, double mass) {
    return 0.01;
}

// define the smearing function
ROOT::RVec<ROOT::Math::PtEtaPhiMVector> smearing(ROOT::RVec<ROOT::Math::PtEtaPhiMVector>& tracks) {

    // random number generator setup
    static thread_local std::mt19937 generator([]{  // std::mt19937 is a random number generator ("Mersenne Twister") with seed 12345 
                                                    // static ensures that the generator is kept in RAM for the next time the function is called, otherwise the seed would reset everytime the function is called, giving us the same random numbers for all events
                                                    // thread_local makes sure that every core gets its own generator since we allow multi-threading
        size_t thread_id = std::hash<std::thread::id>{} // transforms thread ID into a size_t integer
        (std::this_thread::get_id()); // acquire thread ID (it is a special C++ class, that is why we need to hash it in order to perform math with it)
        return 98765 + thread_id; // create a unique seed for each thread to prevent having the same seed on every thread
    }()); // the () at the end tells C++ to run the temporary (lambda) function and use whatever it returns as the argument for the generator
    static thread_local std::normal_distribution<double> dist(0.0, 1.0); // tells the generator to generate numbers according to a Gaussian distribution centered around 0.0 with spread 1.0

    // particle loop
    for (size_t i = 0; i < tracks.size(); i++) {
        ROOT::Math::PtEtaPhiMVector& track = tracks[i];

        // extract 4vector components
        double pT = track.Pt();
        double eta = track.Eta();
        double phi = track.Phi();

        // smearing utilizing the fact that we acquire a number from Gaussian distribution centered around 1.0 with a spread of pTSpread by scaling a number from Gaussian distribution centered around 0.0 with a spread of 1.0 and sjifting it so that its centre is at 1.0
        double mass = 0.13957; // pion mass in GeV/c^2
        
        double pTSmeared = pT * (1.0 + dist(generator)*pTSpread(pT, mass)); // pT multiplicative smearing (pT has relative errors)
        double etaSmeared = eta + dist(generator)*etaSpread(pT, mass); // eta additive smearing (anbsolute errors -- ASI NEROZUMIM)
        double phiSmeared = phi + dist(generator)*phiSpread(pT, mass); // phi additive smearing

        // update the tracks
        track.SetPt(pTSmeared);
        track.SetEta(etaSmeared);
        track.SetPhi(phiSmeared); // this method automatically ensures that phi stays in [-pi, pi]
    }

    return tracks;
}