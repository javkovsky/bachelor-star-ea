#include <TMatrixD.h>

void normalizeUM(TMatrixD& UM) {
    int nRows = UM.GetNrows();
    int nCols = UM.GetNcols();
    
    // loop over columns
    for (int jCol = 0; jCol < nCols; jCol++) {
        double colSum = 0;

        // calculate the column sum
        for (int iRow = 0; iRow < nRows; iRow++) {
            colSum += UM(iRow, jCol);
        }

        // normalize the column
        if (colSum > 0.0) {
            for (int iRow = 0; iRow < nRows; iRow++) {
                UM(iRow, jCol) = UM(iRow, jCol) / colSum;
            }
        }
    }
}