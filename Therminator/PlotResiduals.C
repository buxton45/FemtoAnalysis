#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"

#include "PIDMapping.C"

//_________________________________________________________________________________________
TH1D* Get1dHisto(TString FileName, TString HistoName)
{
  TFile f1(FileName);
  TH1D *ReturnHisto = (TH1D*)f1.Get(HistoName);

  TH1D *ReturnHistoClone = (TH1D*)ReturnHisto->Clone();
  ReturnHistoClone->SetDirectory(0);

  return ReturnHistoClone;
}

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
void PrintLambdaValues(TPad* aPad, TH1D* aHisto)
{
  aPad->cd();
  TPaveText* returnText = new TPaveText(0.65,0.25,0.85,0.85,"NDC");
    returnText->SetFillColor(0);
    returnText->SetBorderSize(0);
    returnText->SetTextAlign(22);
    returnText->SetTextFont(63);
    returnText->SetTextSize(10);

  returnText->AddText("Estimated #lambda Values");

  double tTotal = 0.;
  for(int i=1; i<=12; i++) tTotal += aHisto->GetBinContent(i);
  for(int i=1; i<=12; i++) returnText->AddText(TString(aHisto->GetXaxis()->GetBinLabel(i)) + TString::Format(" = %0.3f", aHisto->GetBinContent(i)/tTotal));

  returnText->Draw();
}

//________________________________________________________________________________________________________________
void DrawPairFractions(TPad* aPad, TH1D* aHisto)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  double tNCounts = 0.;
  for(int i=1; i<=11; i++) tNCounts += aHisto->GetBinContent(i);
  double tNFakes = 0.05*tNCounts;
  aHisto->SetBinContent(12,tNFakes);

  aHisto->GetXaxis()->SetTitle("Parent System");
  aHisto->GetYaxis()->SetTitle("Counts");
  aHisto->Draw();

  PrintLambdaValues(aPad,aHisto);
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
    if(aAnType==kLamKchP || aAnType==kLamKchM) aMatrix->GetXaxis()->SetRange(50,65);
    else if(aAnType==kALamKchP || aAnType==kALamKchM) aMatrix->GetXaxis()->SetRange(35,50);
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
  bool bSaveImages = true;

  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationPairFractions = tDirectory + "PairFractions.root";
  TString tFileLocationTransformMatrices = tDirectory + "TransformMatrices_Mix5.root";

//-----------------------------------------------------------------------------

  TH1D* tPairFractions_LamKchP = Get1dHisto(tFileLocationPairFractions,"fPairFractionsLamKchP");
  const char* tXAxisLabels_LamKchP[12] = {cAnalysisRootTags[kLamKchP], cResidualRootTags[kSig0KchP], cResidualRootTags[kXi0KchP], cResidualRootTags[kXiCKchP], cResidualRootTags[kSigStPKchP], cResidualRootTags[kSigStMKchP], cResidualRootTags[kSigSt0KchP], cResidualRootTags[kLamKSt0], cResidualRootTags[kSig0KSt0], cResidualRootTags[kXi0KSt0], cResidualRootTags[kXiCKSt0], "Fake"};
  TH1D* tPairFractions_ALamKchM = Get1dHisto(tFileLocationPairFractions,"fPairFractionsALamKchM");
  const char* tXAxisLabels_ALamKchM[12] = {cAnalysisRootTags[kALamKchM], cResidualRootTags[kASig0KchM], cResidualRootTags[kAXi0KchM], cResidualRootTags[kAXiCKchM], cResidualRootTags[kASigStMKchM], cResidualRootTags[kASigStPKchM], cResidualRootTags[kASigSt0KchM], cResidualRootTags[kALamAKSt0], cResidualRootTags[kASig0AKSt0], cResidualRootTags[kAXi0AKSt0], cResidualRootTags[kAXiCAKSt0], "Fake"};

  TH1D* tPairFractions_LamKchM = Get1dHisto(tFileLocationPairFractions,"fPairFractionsLamKchM");
  const char* tXAxisLabels_LamKchM[12] = {cAnalysisRootTags[kLamKchM], cResidualRootTags[kSig0KchM], cResidualRootTags[kXi0KchM], cResidualRootTags[kXiCKchM], cResidualRootTags[kSigStPKchM], cResidualRootTags[kSigStMKchM], cResidualRootTags[kSigSt0KchM], cResidualRootTags[kLamAKSt0], cResidualRootTags[kSig0AKSt0], cResidualRootTags[kXi0AKSt0], cResidualRootTags[kXiCAKSt0], "Fake"};
  TH1D* tPairFractions_ALamKchP = Get1dHisto(tFileLocationPairFractions,"fPairFractionsALamKchP");
  const char* tXAxisLabels_ALamKchP[12] = {cAnalysisRootTags[kALamKchP], cResidualRootTags[kASig0KchP], cResidualRootTags[kAXi0KchP], cResidualRootTags[kAXiCKchP], cResidualRootTags[kASigStMKchP], cResidualRootTags[kASigStPKchP], cResidualRootTags[kASigSt0KchP], cResidualRootTags[kALamKSt0], cResidualRootTags[kASig0KSt0], cResidualRootTags[kAXi0KSt0], cResidualRootTags[kAXiCKSt0], "Fake"}; 

  for(int i=1; i<=12; i++)
  {
    tPairFractions_LamKchP->GetXaxis()->SetBinLabel(i,tXAxisLabels_LamKchP[i-1]);
    tPairFractions_ALamKchM->GetXaxis()->SetBinLabel(i,tXAxisLabels_ALamKchM[i-1]);

    tPairFractions_LamKchM->GetXaxis()->SetBinLabel(i,tXAxisLabels_LamKchM[i-1]);
    tPairFractions_ALamKchP->GetXaxis()->SetBinLabel(i,tXAxisLabels_ALamKchP[i-1]);
  }


  TCanvas* tPairFractionsCan = new TCanvas("tPairFractionsCan", "tPairFractionsCan");
  tPairFractionsCan->Divide(2,2);

  DrawPairFractions((TPad*)tPairFractionsCan->cd(1), tPairFractions_LamKchP);
  DrawPairFractions((TPad*)tPairFractionsCan->cd(2), tPairFractions_ALamKchM);

  DrawPairFractions((TPad*)tPairFractionsCan->cd(3), tPairFractions_LamKchM);
  DrawPairFractions((TPad*)tPairFractionsCan->cd(4), tPairFractions_ALamKchP);

  if(bSaveImages) tPairFractionsCan->SaveAs(tDirectory+"Figures/ParentsFractions.pdf");

  //------------------------------------
  bool bZoomMatrixROI = true;

  vector<int> tV0Fathers {-67719, -67718, -67001, -67000, -42212, -42112, -33122, -32212, -32124, -32112, -31214, -23224, -23214, -23114, -13324, -13314, -13226, -13224, -13222, -13216, -13214, -13212, -13124, -13116, -13114, -13112, -8901, -8900, -8118, -8117, -8116, -4228, -4128, -4028, -3334, -3322, -3312, -3228, -3226, -3224, -3218, -3216, -3214, -3212, -3124, -3122, -3118, -3116, -3114, 3114, 3116, 3118, 3122, 3124, 3212, 3214, 3216, 3218, 3224, 3226, 3228, 3312, 3322, 3334, 4028, 4128, 4228, 8116, 8117, 8118, 8900, 8901, 13112, 13114, 13116, 13124, 13212, 13214, 13216, 13222, 13224, 13226, 13314, 13324, 23114, 23214, 23224, 31214, 32112, 32124, 32212, 33122, 42112, 42212, 67000, 67001, 67718, 67719};

  vector<int> tTrackFathers {-100323, -100313, -67719, -67718, -67001, -53122, -43122, -42112, -33122, -32112, -31214, -30323, -30313, -23224, -23214, -23124, -23122, -23114, -20323, -20313, -13226, -13224, -13222, -13216, -13214, -13212, -13126, -13124, -10323, -10321, -10313, -10311, -10215, -10211, -9000, -8901, -8900, -8118, -8117, -4228, -4128, -3334, -3228, -3226, -3218, -3216, -3128, -3126, -3124, -3118, -327, -323, -321, -317, -313, -219, 115, 119, 215, 219, 313, 317, 321, 323, 327, 333, 335, 337, 3118, 3124, 3126, 3128, 3216, 3218, 3226, 3228, 3334, 4128, 8117, 8118, 8900, 8901, 9000, 10111, 10115, 10211, 10215, 10221, 10311, 10313, 10321, 10323, 10331, 13124, 13126, 13212, 13214, 13216, 13222, 13224, 13226, 13314, 13324, 20223, 20313, 20315, 20323, 20325, 20333, 23114, 23122, 23124, 23214, 23224, 30313, 30323, 31214, 32112, 33122, 42112, 43122, 53122, 67001, 67718, 67719, 100313, 100323, 100331, 100333, 9000223};

  TH2D* tParentsMatrix_LamKchP = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamKchP");
  TH2D* tParentsMatrix_ALamKchM = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamKchM");

  TH2D* tParentsMatrix_LamKchM = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamKchM");
  TH2D* tParentsMatrix_ALamKchP = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamKchP");

  for(unsigned int i=0; i<tV0Fathers.size(); i++)
  {
    tParentsMatrix_LamKchP->GetXaxis()->SetBinLabel(i+1, GetParticleName(tV0Fathers[i]));
    tParentsMatrix_ALamKchM->GetXaxis()->SetBinLabel(i+1, GetParticleName(tV0Fathers[i]));

    tParentsMatrix_LamKchM->GetXaxis()->SetBinLabel(i+1, GetParticleName(tV0Fathers[i]));
    tParentsMatrix_ALamKchP->GetXaxis()->SetBinLabel(i+1, GetParticleName(tV0Fathers[i]));
  }

  for(unsigned int i=0; i<tTrackFathers.size(); i++)
  {
    tParentsMatrix_LamKchP->GetYaxis()->SetBinLabel(i+1, GetParticleName(tTrackFathers[i]));
    tParentsMatrix_ALamKchM->GetYaxis()->SetBinLabel(i+1, GetParticleName(tTrackFathers[i]));

    tParentsMatrix_LamKchM->GetYaxis()->SetBinLabel(i+1, GetParticleName(tTrackFathers[i]));
    tParentsMatrix_ALamKchP->GetYaxis()->SetBinLabel(i+1, GetParticleName(tTrackFathers[i]));
  }

/*
  TCanvas* tParentsCan = new TCanvas("tParentsCan", "tParentsCan", 1400, 1000);
  tParentsCan->Divide(2,2);

  DrawParentsMatrix((TPad*)tParentsCan->cd(1), tParentsMatrix_LamKchP);
  DrawParentsMatrix((TPad*)tParentsCan->cd(2), tParentsMatrix_ALamKchM);
  DrawParentsMatrix((TPad*)tParentsCan->cd(3), tParentsMatrix_LamKchM);
  DrawParentsMatrix((TPad*)tParentsCan->cd(4), tParentsMatrix_ALamKchP);
*/

  TCanvas* tParentsCan_LamKchP;
  TCanvas* tParentsCan_ALamKchM;
  TCanvas* tParentsCan_LamKchM;
  TCanvas* tParentsCan_ALamKchP;

  if(bZoomMatrixROI)
  {
    tParentsCan_LamKchP = new TCanvas("tParentsCan_LamKchP", "tParentsCan_LamKchP");
    tParentsCan_ALamKchM = new TCanvas("tParentsCan_ALamKchM", "tParentsCan_ALamKchM");
    tParentsCan_LamKchM = new TCanvas("tParentsCan_LamKchM", "tParentsCan_LamKchM");
    tParentsCan_ALamKchP = new TCanvas("tParentsCan_ALamKchP", "tParentsCan_ALamKchP");
  }
  else
  {
    tParentsCan_LamKchP = new TCanvas("tParentsCan_LamKchP", "tParentsCan_LamKchP", 1000, 1500);
    tParentsCan_ALamKchM = new TCanvas("tParentsCan_ALamKchM", "tParentsCan_ALamKchM", 1000, 1500);
    tParentsCan_LamKchM = new TCanvas("tParentsCan_LamKchM", "tParentsCan_LamKchM", 1000, 1500);
    tParentsCan_ALamKchP = new TCanvas("tParentsCan_ALamKchP", "tParentsCan_ALamKchP", 1000, 1500);
  }

  DrawParentsMatrix(kLamKchP, (TPad*)tParentsCan_LamKchP, tParentsMatrix_LamKchP, bZoomMatrixROI);
  DrawParentsMatrix(kALamKchM, (TPad*)tParentsCan_ALamKchM, tParentsMatrix_ALamKchM, bZoomMatrixROI);
  DrawParentsMatrix(kLamKchM, (TPad*)tParentsCan_LamKchM , tParentsMatrix_LamKchM, bZoomMatrixROI);
  DrawParentsMatrix(kALamKchP, (TPad*)tParentsCan_ALamKchP, tParentsMatrix_ALamKchP, bZoomMatrixROI);

  if(bSaveImages) 
  {
    if(bZoomMatrixROI)
    {
      tParentsCan_LamKchP->SaveAs(tDirectory+"Figures/ParentsLamKchP.pdf");
      tParentsCan_ALamKchM->SaveAs(tDirectory+"Figures/ParentsALamKchM.pdf");

      tParentsCan_LamKchM->SaveAs(tDirectory+"Figures/ParentsLamKchM.pdf");
      tParentsCan_ALamKchP->SaveAs(tDirectory+"Figures/ParentsALamKchP.pdf");
    }
    else
    {
      tParentsCan_LamKchP->SaveAs(tDirectory+"Figures/ParentsLamKchP_UnZoomed.pdf");
      tParentsCan_ALamKchM->SaveAs(tDirectory+"Figures/ParentsALamKchM_UnZoomed.pdf");

      tParentsCan_LamKchM->SaveAs(tDirectory+"Figures/ParentsLamKchM_UnZoomed.pdf");
      tParentsCan_ALamKchP->SaveAs(tDirectory+"Figures/ParentsALamKchP_UnZoomed.pdf");
    }
  }

  //------------------------------------



//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
