#include <TH2D.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include <iostream>

// define the function which applies an unfolding matrix (UM) to the y-axis of a measured 2D histogram
TH2D* applyUM(TH2D* histoMeas, TMatrixD* UM) {

    // clone the original histogram to acquire its binning structure
    TH2D* histoUnf = (TH2D*)histoMeas -> Clone("histoUnf_clone2"); // (TH2D*) tells C++ to treat the cloned object as a TH2D
    histoUnf -> Reset(); // clear the data to keep only the bin structure
    histoUnf -> Sumw2(); // tell the empty histogram to allocate memory for errors

    // get numbers of bins
    int nBinsX = histoMeas -> GetNbinsX(); // x-axis
    int nBinsY = histoMeas -> GetNbinsY(); // y-axis

    // create TVectors to store data
    TVectorD vecMeas(nBinsY); // y-axis values for the selected x-axis bin
    TVectorD vecMeasErr(nBinsY); // errors of these values
    TVectorD vecUnf(nBinsY); // unfolded y-axis values for the selected x-axis bin

    // loop over x-axis bins (TH2Ds are 1-indexed)
    for (int ix = 1; ix <= nBinsX; ix++) { 

        // fill the vecMeas with the y-axis values and vecMeasErr with their errors
        for (int iy = 1; iy <= nBinsY; iy++) {
            vecMeas[iy-1] = histoMeas -> GetBinContent(ix, iy);
            vecMeasErr[iy-1] = histoMeas -> GetBinError(ix, iy);
        }

        // multiply the x-axis bin by the unfolding matrix
        vecUnf = (*UM) * vecMeas; // (*...) is the deference operator: since UM is just a pointer to a memory adress, we must tell C++ to actually go to the memory adress and grab the object the pointer is pointing to

        // store the unfolded x-axis bin and calculated errors into the empty histogram
        for (int iy = 1; iy <= nBinsY; iy++) {
            
            // store the value
            histoUnf -> SetBinContent(ix, iy, vecUnf[iy-1]);

            // calculate the error using error propagation (binErr_i^2 = sum_j (UM_ij * err_j)^2) and store it
            double errSquared = 0.0;
            for (int iy2 = 1; iy2 <= nBinsY; iy2++) {
                double matrixElement = (*UM)(iy-1, iy2-1);
                double binErr = vecMeasErr[iy2-1];
                errSquared += (matrixElement * binErr) * (matrixElement * binErr);
            }
            histoUnf -> SetBinError(ix, iy, std::sqrt(errSquared));
        }
    }

    // memory management
    histoUnf -> SetDirectory(nullptr);

    return histoUnf;
}