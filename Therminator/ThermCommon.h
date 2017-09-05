/* ThermCommon.h */

/* Some common macros used by post-processing programs */

#ifndef THERMCOMMON_H
#define THERMCOMMON_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <cassert>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TFile.h"
#include "TPad.h"

#include "Types.h"
#include "PIDMapping.h"

using namespace std;


extern TH1D* Get1dHisto(TString FileName, TString HistoName);
extern TH2D* Get2dHisto(TString FileName, TString HistoName);
extern TH3D* Get3dHisto(TString FileName, TString HistoName);

//-----------------------------------------------------------
extern const char* tXAxisLabels_LamKchP[13];
extern const char* tXAxisLabels_ALamKchM[13];

extern const char* tXAxisLabels_LamKchM[13];
extern const char* tXAxisLabels_ALamKchP[13];

extern const char* tXAxisLabels_LamK0[13];
extern const char* tXAxisLabels_ALamK0[13];

extern void SetXAxisLabels(AnalysisType aAnType, TH1D* aHist);

//-----------------------------------------------------------

extern void PrintLambdaValues(TPad* aPad, TH1D* aHisto);
extern void DrawPairFractions(TPad* aPad, TH1D* aHisto, bool aSave=false, TString aSaveName = "");

//-----------------------------------------------------------

extern void DrawParentsMatrixBackground(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix);
extern void DrawOnlyPairsInOthers(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, double aMaxDecayLength=-1.);
extern void DrawParentsMatrix(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, bool aZoomROI=false, bool aSetLogZ=false, bool aSave=false, TString aSaveName="");
extern void DrawCondensedParentsMatrix(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, bool aSetLogZ=false, bool aSave=false, TString aSaveName="");

//-----------------------------------------------------------

#endif
