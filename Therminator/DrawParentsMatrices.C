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


//________________________________________________________________________________________________________________
TH2D* Get2dHisto(TString FileName, TString HistoName)
{
  TFile f1(FileName);
  TH2D *ReturnHisto = (TH2D*)f1.Get(HistoName);

  TH2D *ReturnHistoClone = (TH2D*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}

//________________________________________________________________________________________________________________
void DrawParentsMatrixBackground(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix)
{
  vector<int> tBinsXToSetToZero;
  vector<int> tBinsYToSetToZero;
  switch(aAnType) {
  case kLamKchP:
    tBinsXToSetToZero = vector<int>{50, 53, 55, 56, 59, 62, 63};
    tBinsYToSetToZero = vector<int>{61, 63};
    break;

  case kALamKchM:
    tBinsXToSetToZero = vector<int>{36, 37, 40, 43, 44, 46, 49};
    tBinsYToSetToZero = vector<int>{53, 55};
    break;

  case kLamKchM:
    tBinsXToSetToZero = vector<int>{50, 53, 55, 56, 59, 62, 63};
    tBinsYToSetToZero = vector<int>{53, 55};
    break;

  case kALamKchP:
    tBinsXToSetToZero = vector<int>{36, 37, 40, 43, 44, 46, 49};
    tBinsYToSetToZero = vector<int>{61, 63};
    break;

  case kLamK0:
    tBinsXToSetToZero = vector<int>{50, 53, 55, 56, 59, 62, 63};
    tBinsYToSetToZero = vector<int>{42, 45};
    break;

  case kALamK0:
    tBinsXToSetToZero = vector<int>{36, 37, 40, 43, 44, 46, 49};
    tBinsYToSetToZero = vector<int>{42, 45};
    break;

  default:
    cout << "ERROR: DrawParentsMatrixBackground: aAnType = " << aAnType << " is not appropriate" << endl << endl;
    assert(0);
  }

  //---------------------------------
  for(unsigned int i=0; i<tBinsXToSetToZero.size(); i++)
  {
    for(unsigned int j=0; j<tBinsYToSetToZero.size(); j++)
    {
      aMatrix->SetBinContent(tBinsXToSetToZero[i], tBinsYToSetToZero[j], 0.);
    }
  }

  aPad->cd();
  gStyle->SetOptStat(0);

  aMatrix->GetXaxis()->SetRange(1,100);
  aMatrix->GetXaxis()->SetLabelSize(0.01);
  aMatrix->LabelsOption("v", "X");

  aMatrix->GetYaxis()->SetRange(1,135);
  aMatrix->GetYaxis()->SetLabelSize(0.01);
  aMatrix->Draw("colz");
}


//________________________________________________________________________________________________________________
void DrawOnlyPairsInOthers(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix)
{
  vector<int> tParentCollection1, tParentCollection2;
  switch(aAnType) {
  case kLamKchP:
  case kALamKchM:
  case kLamKchM:
  case kALamKchP:
    tParentCollection1 = cAllLambdaFathers;
    tParentCollection2 = cAllKchFathers;
    break;

  case kLamK0:
  case kALamK0:
    tParentCollection1 = cAllLambdaFathers;
    tParentCollection2 = cAllK0ShortFathers;
    break;


  default:
    cout << "ERROR: DrawOnlyPairsInOthers: aAnType = " << aAnType << " is not appropriate" << endl << endl;
    assert(0);
  }

  for(unsigned int i=0; i<tParentCollection1.size(); i++)
  {
    for(unsigned int j=0; j<tParentCollection2.size(); j++)
    {
      if(!IncludeInOthers(tParentCollection1[i], tParentCollection2[j])) aMatrix->SetBinContent(i+1, j+1, 0.);
    }
  }

  aPad->cd();
  gStyle->SetOptStat(0);

  aMatrix->GetXaxis()->SetRange(1,100);
  aMatrix->GetXaxis()->SetLabelSize(0.01);
  aMatrix->LabelsOption("v", "X");

  aMatrix->GetYaxis()->SetRange(1,135);
  aMatrix->GetYaxis()->SetLabelSize(0.01);
  aMatrix->Draw("colz");
}

//________________________________________________________________________________________________________________
void DrawParentsMatrix(AnalysisType aAnType, TPad* aPad, TH2D* aMatrix, bool aZoomROI=false)
{
  aPad->cd();
  gStyle->SetOptStat(0);

//  aMatrix->GetXaxis()->SetTitle("Lambda Parent ID");
  aMatrix->GetXaxis()->SetRange(1,100);
  aMatrix->GetXaxis()->SetLabelSize(0.01);
  if(aZoomROI)
  {
    if(aAnType==kLamKchP || aAnType==kLamKchM || aAnType==kLamK0) aMatrix->GetXaxis()->SetRange(50,65);
    else if(aAnType==kALamKchP || aAnType==kALamKchM || aAnType==kALamK0) aMatrix->GetXaxis()->SetRange(35,50);
    else assert(0);
    aMatrix->GetXaxis()->SetLabelSize(0.03);
  }
  aMatrix->LabelsOption("v", "X");

//  aMatrix->GetYaxis()->SetTitle("Kch Parent ID");
  aMatrix->GetYaxis()->SetRange(1,135);
  aMatrix->GetYaxis()->SetLabelSize(0.01);
  if(aZoomROI)
  {
    if(aAnType==kLamKchP || aAnType==kALamKchP) aMatrix->GetYaxis()->SetRange(56,66);
    else if(aAnType==kLamKchM || aAnType==kALamKchM) aMatrix->GetYaxis()->SetRange(50,60);
    else if(aAnType==kLamK0 || aAnType==kALamK0) aMatrix->GetYaxis()->SetRange(38,48);
    else assert(0);
    aMatrix->GetYaxis()->SetLabelSize(0.04);
  }
  aMatrix->Draw("colz");
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
  bool bZoomMatrixROI = false;
  bool bDrawMatrixBackground = false;
  bool bDrawOnlyPairsInOthers = true;


  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationPairFractions = tDirectory + "PairFractions.root";

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

  DrawParentsMatrix(kLamKchP, (TPad*)tParentsCan_LamKchP, tParentsMatrix_LamKchP, bZoomMatrixROI);
  DrawParentsMatrix(kALamKchM, (TPad*)tParentsCan_ALamKchM, tParentsMatrix_ALamKchM, bZoomMatrixROI);
  DrawParentsMatrix(kLamKchM, (TPad*)tParentsCan_LamKchM , tParentsMatrix_LamKchM, bZoomMatrixROI);
  DrawParentsMatrix(kALamKchP, (TPad*)tParentsCan_ALamKchP, tParentsMatrix_ALamKchP, bZoomMatrixROI);
  DrawParentsMatrix(kLamK0, (TPad*)tParentsCan_LamK0 , tParentsMatrix_LamK0, bZoomMatrixROI);
  DrawParentsMatrix(kALamK0, (TPad*)tParentsCan_ALamK0, tParentsMatrix_ALamK0, bZoomMatrixROI);

  if(bSaveImages) 
  {
    if(bZoomMatrixROI)
    {
      tParentsCan_LamKchP->SaveAs(tDirectory+"Figures/ParentsLamKchP.pdf");
      tParentsCan_ALamKchM->SaveAs(tDirectory+"Figures/ParentsALamKchM.pdf");

      tParentsCan_LamKchM->SaveAs(tDirectory+"Figures/ParentsLamKchM.pdf");
      tParentsCan_ALamKchP->SaveAs(tDirectory+"Figures/ParentsALamKchP.pdf");

      tParentsCan_LamK0->SaveAs(tDirectory+"Figures/ParentsLamK0.pdf");
      tParentsCan_ALamK0->SaveAs(tDirectory+"Figures/ParentsALamK0.pdf");
    }
    else
    {
      tParentsCan_LamKchP->SaveAs(tDirectory+"Figures/ParentsLamKchP_UnZoomed.pdf");
      tParentsCan_ALamKchM->SaveAs(tDirectory+"Figures/ParentsALamKchM_UnZoomed.pdf");

      tParentsCan_LamKchM->SaveAs(tDirectory+"Figures/ParentsLamKchM_UnZoomed.pdf");
      tParentsCan_ALamKchP->SaveAs(tDirectory+"Figures/ParentsALamKchP_UnZoomed.pdf");

      tParentsCan_LamK0->SaveAs(tDirectory+"Figures/ParentsLamK0_UnZoomed.pdf");
      tParentsCan_ALamK0->SaveAs(tDirectory+"Figures/ParentsALamK0_UnZoomed.pdf");
    }
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
    TCanvas* tParentsOthersCan_LamKchP = new TCanvas("tParentsOthersCan_LamKchP", "tParentsOthersCan_LamKchP", 1000, 1500);
    TCanvas* tParentsOthersCan_ALamKchM = new TCanvas("tParentsOthersCan_ALamKchM", "tParentsOthersCan_ALamKchM", 1000, 1500);
    TCanvas* tParentsOthersCan_LamKchM = new TCanvas("tParentsOthersCan_LamKchM", "tParentsOthersCan_LamKchM", 1000, 1500);
    TCanvas* tParentsOthersCan_ALamKchP = new TCanvas("tParentsOthersCan_ALamKchP", "tParentsOthersCan_ALamKchP", 1000, 1500);
    TCanvas* tParentsOthersCan_LamK0 = new TCanvas("tParentsOthersCan_LamK0", "tParentsOthersCan_LamK0", 1000, 1500);
    TCanvas* tParentsOthersCan_ALamK0 = new TCanvas("tParentsOthersCan_ALamK0", "tParentsOthersCan_ALamK0", 1000, 1500);

    DrawOnlyPairsInOthers(kLamKchP, (TPad*)tParentsOthersCan_LamKchP, (TH2D*)tParentsMatrix_LamKchP->Clone("fParentsMatrixOthersLamKchP"));
    DrawOnlyPairsInOthers(kALamKchM, (TPad*)tParentsOthersCan_ALamKchM, (TH2D*)tParentsMatrix_ALamKchM->Clone("fParentsMatrixOthersALamKchM"));
    DrawOnlyPairsInOthers(kLamKchM, (TPad*)tParentsOthersCan_LamKchM, (TH2D*)tParentsMatrix_LamKchM->Clone("fParentsMatrixOthersLamKchM"));
    DrawOnlyPairsInOthers(kALamKchP, (TPad*)tParentsOthersCan_ALamKchP, (TH2D*)tParentsMatrix_ALamKchP->Clone("fParentsMatrixOthersALamKchP"));
    DrawOnlyPairsInOthers(kLamK0, (TPad*)tParentsOthersCan_LamK0, (TH2D*)tParentsMatrix_LamK0->Clone("fParentsMatrixOthersLamK0"));
    DrawOnlyPairsInOthers(kALamK0, (TPad*)tParentsOthersCan_ALamK0, (TH2D*)tParentsMatrix_ALamK0->Clone("fParentsMatrixOthersALamK0"));
  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
