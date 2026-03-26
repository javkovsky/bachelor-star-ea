#include <ROOT/RVec.hxx> // ROOT's fast array of numbers
#include <Math/Vector4D.h> // ROOT's 4-vectors -- we do not use it in this script, however it is used in PyROOT in Jupyter Notebook

// pT extraction
ROOT::RVec<double> getpT  (const ROOT::RVec<ROOT::Math::PtEtaPhiMVector>& tracks) { // takes a vector of LorentzVectors
    ROOT::RVec<double> components(tracks.size()); // create an empty vector of the same size as the event where pT will be stored for each track
    
    for (size_t i = 0; i < tracks.size(); i++) { // particle loop
        components[i]=tracks[i].Pt();  // fill the pT vector
    } 
    
    return components; 
}

// eta extraction
ROOT::RVec<double> getEta  (const ROOT::RVec<ROOT::Math::PtEtaPhiMVector>& tracks) { 
    ROOT::RVec<double> components(tracks.size()); 
    
    for (size_t i = 0; i < tracks.size(); i++) {
        components[i]=tracks[i].Eta(); 
    } 
    
    return components; 
}

// phi extraction
ROOT::RVec<double> getPhi  (const ROOT::RVec<ROOT::Math::PtEtaPhiMVector>& tracks) { 
    ROOT::RVec<double> components(tracks.size()); 
    
    for (size_t i = 0; i < tracks.size(); i++) {
        components[i]=tracks[i].Phi(); 
    } 
    
    return components; 
}

// mass extraction
ROOT::RVec<double> getMass  (const ROOT::RVec<ROOT::Math::PtEtaPhiMVector>& tracks) { 
    ROOT::RVec<double> components(tracks.size()); 
    
    for (size_t i = 0; i < tracks.size(); i++) {
        components[i]=tracks[i].M(); 
    } 
    
    return components; 
}