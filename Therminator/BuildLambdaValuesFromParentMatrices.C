#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"

#include "PIDMapping.h"
#include "ThermCommon.h"

//________________________________________________________________________________________________________________
TH1D* BuildPairFractions(AnalysisType aAnType, TH2D* aMatrix, double aMaxDecayLength=-1.)
{
  TString tName = TString::Format("#lambda Estimates: %s", cAnalysisRootTags[aAnType]);
  TH1D* tReturnHist = new TH1D(tName, tName, 13, 0, 13);

  vector<int> *tFatherCollection1, *tFatherCollection2;
  bool bParticleV0 = true;
  switch(aAnType) {
  case kLamKchP:
  case kALamKchM:
  case kLamKchM:
  case kALamKchP:
    tFatherCollection1 = &cAllLambdaFathers;
    tFatherCollection2 = &cAllKchFathers;
    bParticleV0 = true;
    break;

  case kLamK0:
  case kALamK0:
    tFatherCollection1 = &cAllLambdaFathers;
    tFatherCollection2 = &cAllK0ShortFathers;
    bParticleV0 = false;
    break;

  default:
    cout << "ERROR: BuildPairFractions: aAnType = " << aAnType << " is not appropriate" << endl << endl;
    assert(0);
  }

  int tPDG1=-1, tPDG2=-1;
  double tWeight = 0.;
  for(unsigned int iPar1=0; iPar1<tFatherCollection1->size(); iPar1++)
  {
    tPDG1 = (*tFatherCollection1)[iPar1];
    for(unsigned int iPar2=0; iPar2<tFatherCollection2->size(); iPar2++)
    {
      tPDG2 = (*tFatherCollection2)[iPar2];
      tWeight = aMatrix->GetBinContent(iPar1+1, iPar2+1);
      if(bParticleV0) ThermEventsCollection::MapAndFillPairFractionHistogramParticleV0(tReturnHist, tPDG1, tPDG2, aMaxDecayLength, tWeight);
      else ThermEventsCollection::MapAndFillPairFractionHistogramV0V0(tReturnHist, tPDG1, tPDG2, aMaxDecayLength, tWeight);
    }
  }
  PrintIncludeAsPrimary(aMaxDecayLength);
  SetXAxisLabels(aAnType, tReturnHist);
  return tReturnHist;
}



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  bool bSaveImages = false;
//  double tMaxDecayLength = -1.;
  double tMaxDecayLength = 5.0;

  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationPairFractions = tDirectory + "PairFractions.root";
  TString tFileLocationTransformMatrices = tDirectory + "TransformMatrices_Mix5.root";

  TString tSaveDirectory = "/home/jesse/Analysis/Presentations/AliFemto/20170913/";
//  tSaveDirectory = tDirectory;

//-----------------------------------------------------------------------------

  TH2D* tParentsMatrix_LamKchP = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamKchP");
  TH2D* tParentsMatrix_ALamKchM = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamKchM");

  TH2D* tParentsMatrix_LamKchM = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamKchM");
  TH2D* tParentsMatrix_ALamKchP = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamKchP");

  TH2D* tParentsMatrix_LamK0 = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamK0");
  TH2D* tParentsMatrix_ALamK0 = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamK0");

  //------------------------------------------------------------------------------
  TH1D* tPairFractions_LamKchP = BuildPairFractions(kLamKchP, tParentsMatrix_LamKchP, tMaxDecayLength);
  TH1D* tPairFractions_ALamKchM = BuildPairFractions(kALamKchM, tParentsMatrix_ALamKchM, tMaxDecayLength);

  TH1D* tPairFractions_LamKchM = BuildPairFractions(kLamKchM, tParentsMatrix_LamKchM, tMaxDecayLength);
  TH1D* tPairFractions_ALamKchP = BuildPairFractions(kALamKchP, tParentsMatrix_ALamKchP, tMaxDecayLength);

  TH1D* tPairFractions_LamK0 = BuildPairFractions(kLamK0, tParentsMatrix_LamK0, tMaxDecayLength);
  TH1D* tPairFractions_ALamK0 = BuildPairFractions(kALamK0, tParentsMatrix_ALamK0, tMaxDecayLength);

  //------------------------------------------------------------------------------
  TCanvas* tCan_LamKchP = new TCanvas("tCan_LamKchP", "tCan_LamKchP");
  TCanvas* tCan_ALamKchM = new TCanvas("tCan_ALamKchM", "tCan_ALamKchM");

  TCanvas* tCan_LamKchM = new TCanvas("tCan_LamKchM", "tCan_LamKchM");
  TCanvas* tCan_ALamKchP = new TCanvas("tCan_ALamKchP", "tCan_ALamKchP");

  TCanvas* tCan_LamK0 = new TCanvas("tCan_LamK0", "tCan_LamK0");
  TCanvas* tCan_ALamK = new TCanvas("tCan_ALamK0", "tCan_ALamK0");
  //------------------------------------------------------------------------------

  DrawPairFractions((TPad*)tCan_LamKchP, tPairFractions_LamKchP, bSaveImages, tSaveDirectory+TString("/Figures/LamKchP/LamValuesFromParentsMatrix_LamKchP"));
  DrawPairFractions((TPad*)tCan_ALamKchM, tPairFractions_ALamKchM, bSaveImages, tSaveDirectory+TString("/Figures/ALamKchM/LamValuesFromParentsMatrix_ALamKchM"));

  DrawPairFractions((TPad*)tCan_LamKchM, tPairFractions_LamKchM, bSaveImages, tSaveDirectory+TString("/Figures/LamKchM/LamValuesFromParentsMatrix_LamKchM"));
  DrawPairFractions((TPad*)tCan_ALamKchP, tPairFractions_ALamKchP, bSaveImages, tSaveDirectory+TString("/Figures/ALamKchP/LamValuesFromParentsMatrix_ALamKchP"));

  DrawPairFractions((TPad*)tCan_LamK0, tPairFractions_LamK0, bSaveImages, tSaveDirectory+TString("/Figures/LamK0/LamValuesFromParentsMatrix_LamK0"));
  DrawPairFractions((TPad*)tCan_ALamK, tPairFractions_ALamK0, bSaveImages, tSaveDirectory+TString("/Figures/ALamK0/LamValuesFromParentsMatrix_ALamK0"));

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
