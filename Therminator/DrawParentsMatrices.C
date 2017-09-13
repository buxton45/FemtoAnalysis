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
//  TString tSaveFileType = ".png";
  TString tSaveFileType = ".eps";
//  TString tSaveFileType = ".pdf";

  bool bZoomMatrixROI = true;
  bool bDrawCondensed = true;
  bool bSetLogZ = false;

  bool bDrawMatrixBackground = false;
  bool bDrawOnlyPairsInOthers = false;


  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationPairFractions = tDirectory + "PairFractions.root";

  TString tSaveDirectory = "/home/jesse/Analysis/Presentations/AliFemto/20170913/";
//  tSaveDirectory = tDirectory;

//-----------------------------------------------------------------------------

  TH2D* tParentsMatrix_LamKchP = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamKchP");
  TH2D* tParentsMatrix_ALamKchM = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamKchM");

  TH2D* tParentsMatrix_LamKchM = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamKchM");
  TH2D* tParentsMatrix_ALamKchP = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamKchP");

  TH2D* tParentsMatrix_LamK0 = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamK0");
  TH2D* tParentsMatrix_ALamK0 = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamK0");

  for(unsigned int i=0; i<cAllLambdaFathers.size(); i++)
  {
    tParentsMatrix_LamKchP->GetXaxis()->SetBinLabel(i+1, GetParticleName(cAllLambdaFathers[i]));
    tParentsMatrix_ALamKchM->GetXaxis()->SetBinLabel(i+1, GetParticleName(cAllLambdaFathers[i]));

    tParentsMatrix_LamKchM->GetXaxis()->SetBinLabel(i+1, GetParticleName(cAllLambdaFathers[i]));
    tParentsMatrix_ALamKchP->GetXaxis()->SetBinLabel(i+1, GetParticleName(cAllLambdaFathers[i]));

    tParentsMatrix_LamK0->GetXaxis()->SetBinLabel(i+1, GetParticleName(cAllLambdaFathers[i]));
    tParentsMatrix_ALamK0->GetXaxis()->SetBinLabel(i+1, GetParticleName(cAllLambdaFathers[i]));
  }

  for(unsigned int i=0; i<cAllKchFathers.size(); i++)
  {
    tParentsMatrix_LamKchP->GetYaxis()->SetBinLabel(i+1, GetParticleName(cAllKchFathers[i]));
    tParentsMatrix_ALamKchM->GetYaxis()->SetBinLabel(i+1, GetParticleName(cAllKchFathers[i]));

    tParentsMatrix_LamKchM->GetYaxis()->SetBinLabel(i+1, GetParticleName(cAllKchFathers[i]));
    tParentsMatrix_ALamKchP->GetYaxis()->SetBinLabel(i+1, GetParticleName(cAllKchFathers[i]));
  }

  for(unsigned int i=0; i<cAllK0ShortFathers.size(); i++)
  {
    tParentsMatrix_LamK0->GetYaxis()->SetBinLabel(i+1, GetParticleName(cAllK0ShortFathers[i]));
    tParentsMatrix_ALamK0->GetYaxis()->SetBinLabel(i+1, GetParticleName(cAllK0ShortFathers[i]));
  }


  TCanvas* tParentsCan_LamKchP;
  TCanvas* tParentsCan_ALamKchM;
  TCanvas* tParentsCan_LamKchM;
  TCanvas* tParentsCan_ALamKchP;
  TCanvas* tParentsCan_LamK0;
  TCanvas* tParentsCan_ALamK0;

  if(bZoomMatrixROI)
  {
    tParentsCan_LamKchP = new TCanvas("tParentsCan_LamKchP", "tParentsCan_LamKchP");
    tParentsCan_ALamKchM = new TCanvas("tParentsCan_ALamKchM", "tParentsCan_ALamKchM");
    tParentsCan_LamKchM = new TCanvas("tParentsCan_LamKchM", "tParentsCan_LamKchM");
    tParentsCan_ALamKchP = new TCanvas("tParentsCan_ALamKchP", "tParentsCan_ALamKchP");
    tParentsCan_LamK0 = new TCanvas("tParentsCan_LamK0", "tParentsCan_LamK0");
    tParentsCan_ALamK0 = new TCanvas("tParentsCan_ALamK0", "tParentsCan_ALamK0");
  }
  else
  {
    tParentsCan_LamKchP = new TCanvas("tParentsCan_LamKchP", "tParentsCan_LamKchP", 1000, 1500);
    tParentsCan_ALamKchM = new TCanvas("tParentsCan_ALamKchM", "tParentsCan_ALamKchM", 1000, 1500);
    tParentsCan_LamKchM = new TCanvas("tParentsCan_LamKchM", "tParentsCan_LamKchM", 1000, 1500);
    tParentsCan_ALamKchP = new TCanvas("tParentsCan_ALamKchP", "tParentsCan_ALamKchP", 1000, 1500);
    tParentsCan_LamK0 = new TCanvas("tParentsCan_LamK0", "tParentsCan_LamK0", 1000, 1500);
    tParentsCan_ALamK0 = new TCanvas("tParentsCan_ALamK0", "tParentsCan_ALamK0", 1000, 1500);
  }

  DrawParentsMatrix(kLamKchP, (TPad*)tParentsCan_LamKchP, tParentsMatrix_LamKchP, bZoomMatrixROI, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/LamKchP/ParentsLamKchP", tSaveFileType);
  DrawParentsMatrix(kALamKchM, (TPad*)tParentsCan_ALamKchM, tParentsMatrix_ALamKchM, bZoomMatrixROI, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/ALamKchM/ParentsALamKchM", tSaveFileType);
  DrawParentsMatrix(kLamKchM, (TPad*)tParentsCan_LamKchM , tParentsMatrix_LamKchM, bZoomMatrixROI, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/LamKchM/ParentsLamKchM", tSaveFileType);
  DrawParentsMatrix(kALamKchP, (TPad*)tParentsCan_ALamKchP, tParentsMatrix_ALamKchP, bZoomMatrixROI, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/ALamKchP/ParentsALamKchP", tSaveFileType);
  DrawParentsMatrix(kLamK0, (TPad*)tParentsCan_LamK0 , tParentsMatrix_LamK0, bZoomMatrixROI, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/LamK0/ParentsLamK0", tSaveFileType);
  DrawParentsMatrix(kALamK0, (TPad*)tParentsCan_ALamK0, tParentsMatrix_ALamK0, bZoomMatrixROI, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/ALamK0/ParentsALamK0", tSaveFileType);



  //-------------------------------------------------------------------------
  if(bDrawCondensed)
  {
    TCanvas* tCondensedParentsCan_LamKchP = new TCanvas("tCondensedParentsCan_LamKchP", "tCondensedParentsCan_LamKchP", 1000, 1500);
    TCanvas* tCondensedParentsCan_ALamKchM = new TCanvas("tCondensedParentsCan_ALamKchM", "tCondensedParentsCan_ALamKchM", 1000, 1500);
    TCanvas* tCondensedParentsCan_LamKchM = new TCanvas("tCondensedParentsCan_LamKchM", "tCondensedParentsCan_LamKchM", 1000, 1500);
    TCanvas* tCondensedParentsCan_ALamKchP = new TCanvas("tCondensedParentsCan_ALamKchP", "tCondensedParentsCan_ALamKchP", 1000, 1500);
    TCanvas* tCondensedParentsCan_LamK0 = new TCanvas("tCondensedParentsCan_LamK0", "tCondensedParentsCan_LamK0", 1000, 1500);
    TCanvas* tCondensedParentsCan_ALamK0 = new TCanvas("tCondensedParentsCan_ALamK0", "tCondensedParentsCan_ALamK0", 1000, 1500);

    DrawCondensedParentsMatrix(kLamKchP, (TPad*)tCondensedParentsCan_LamKchP, tParentsMatrix_LamKchP, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/LamKchP/CondensedParentsLamKchP", tSaveFileType);
    DrawCondensedParentsMatrix(kALamKchM, (TPad*)tCondensedParentsCan_ALamKchM, tParentsMatrix_ALamKchM, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/ALamKchM/CondensedParentsALamKchM", tSaveFileType);
    DrawCondensedParentsMatrix(kLamKchM, (TPad*)tCondensedParentsCan_LamKchM, tParentsMatrix_LamKchM, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/LamKchM/CondensedParentsLamKchM", tSaveFileType);
    DrawCondensedParentsMatrix(kALamKchP, (TPad*)tCondensedParentsCan_ALamKchP, tParentsMatrix_ALamKchP, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/ALamKchP/CondensedParentsALamKchP", tSaveFileType);
    DrawCondensedParentsMatrix(kLamK0, (TPad*)tCondensedParentsCan_LamK0, tParentsMatrix_LamK0, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/LamK0/CondensedParentsLamK0", tSaveFileType);
    DrawCondensedParentsMatrix(kALamK0, (TPad*)tCondensedParentsCan_ALamK0, tParentsMatrix_ALamK0, bSetLogZ, bSaveImages, tSaveDirectory+"Figures/ALamK0/CondensedParentsALamK0", tSaveFileType);
  }

  //-------------------------------------------------------------------------

  if(bDrawMatrixBackground)
  {
    TCanvas* tParentsBgdCan_LamKchP = new TCanvas("tParentsBgdCan_LamKchP", "tParentsBgdCan_LamKchP", 1000, 1500);
    TCanvas* tParentsBgdCan_ALamKchM = new TCanvas("tParentsBgdCan_ALamKchM", "tParentsBgdCan_ALamKchM", 1000, 1500);
    TCanvas* tParentsBgdCan_LamKchM = new TCanvas("tParentsBgdCan_LamKchM", "tParentsBgdCan_LamKchM", 1000, 1500);
    TCanvas* tParentsBgdCan_ALamKchP = new TCanvas("tParentsBgdCan_ALamKchP", "tParentsBgdCan_ALamKchP", 1000, 1500);
    TCanvas* tParentsBgdCan_LamK0 = new TCanvas("tParentsBgdCan_LamK0", "tParentsBgdCan_LamK0", 1000, 1500);
    TCanvas* tParentsBgdCan_ALamK0 = new TCanvas("tParentsBgdCan_ALamK0", "tParentsBgdCan_ALamK0", 1000, 1500);

    DrawParentsMatrixBackground(kLamKchP, (TPad*)tParentsBgdCan_LamKchP, (TH2D*)tParentsMatrix_LamKchP->Clone("fParentsMatrixBgdLamKchP"));
    DrawParentsMatrixBackground(kALamKchM, (TPad*)tParentsBgdCan_ALamKchM, (TH2D*)tParentsMatrix_ALamKchM->Clone("fParentsMatrixBgdALamKchM"));
    DrawParentsMatrixBackground(kLamKchM, (TPad*)tParentsBgdCan_LamKchM, (TH2D*)tParentsMatrix_LamKchM->Clone("fParentsMatrixBgdLamKchM"));
    DrawParentsMatrixBackground(kALamKchP, (TPad*)tParentsBgdCan_ALamKchP, (TH2D*)tParentsMatrix_ALamKchP->Clone("fParentsMatrixBgdALamKchP"));
    DrawParentsMatrixBackground(kLamK0, (TPad*)tParentsBgdCan_LamK0, (TH2D*)tParentsMatrix_LamK0->Clone("fParentsMatrixBgdLamK0"));
    DrawParentsMatrixBackground(kALamK0, (TPad*)tParentsBgdCan_ALamK0, (TH2D*)tParentsMatrix_ALamK0->Clone("fParentsMatrixBgdALamK0"));
  }

  //-------------------------------------------------------------------------
  if(bDrawOnlyPairsInOthers)
  {
//    double tMaxDecayLength = -1.;
    double tMaxDecayLength = 3.01;

    TCanvas* tParentsOthersCan_LamKchP = new TCanvas("tParentsOthersCan_LamKchP", "tParentsOthersCan_LamKchP", 1000, 1500);
    TCanvas* tParentsOthersCan_ALamKchM = new TCanvas("tParentsOthersCan_ALamKchM", "tParentsOthersCan_ALamKchM", 1000, 1500);
    TCanvas* tParentsOthersCan_LamKchM = new TCanvas("tParentsOthersCan_LamKchM", "tParentsOthersCan_LamKchM", 1000, 1500);
    TCanvas* tParentsOthersCan_ALamKchP = new TCanvas("tParentsOthersCan_ALamKchP", "tParentsOthersCan_ALamKchP", 1000, 1500);
    TCanvas* tParentsOthersCan_LamK0 = new TCanvas("tParentsOthersCan_LamK0", "tParentsOthersCan_LamK0", 1000, 1500);
    TCanvas* tParentsOthersCan_ALamK0 = new TCanvas("tParentsOthersCan_ALamK0", "tParentsOthersCan_ALamK0", 1000, 1500);

    DrawOnlyPairsInOthers(kLamKchP, (TPad*)tParentsOthersCan_LamKchP, (TH2D*)tParentsMatrix_LamKchP->Clone("fParentsMatrixOthersLamKchP"), tMaxDecayLength);
    DrawOnlyPairsInOthers(kALamKchM, (TPad*)tParentsOthersCan_ALamKchM, (TH2D*)tParentsMatrix_ALamKchM->Clone("fParentsMatrixOthersALamKchM"), tMaxDecayLength);
    DrawOnlyPairsInOthers(kLamKchM, (TPad*)tParentsOthersCan_LamKchM, (TH2D*)tParentsMatrix_LamKchM->Clone("fParentsMatrixOthersLamKchM"), tMaxDecayLength);
    DrawOnlyPairsInOthers(kALamKchP, (TPad*)tParentsOthersCan_ALamKchP, (TH2D*)tParentsMatrix_ALamKchP->Clone("fParentsMatrixOthersALamKchP"), tMaxDecayLength);
    DrawOnlyPairsInOthers(kLamK0, (TPad*)tParentsOthersCan_LamK0, (TH2D*)tParentsMatrix_LamK0->Clone("fParentsMatrixOthersLamK0"), tMaxDecayLength);
    DrawOnlyPairsInOthers(kALamK0, (TPad*)tParentsOthersCan_ALamK0, (TH2D*)tParentsMatrix_ALamK0->Clone("fParentsMatrixOthersALamK0"), tMaxDecayLength);
  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
