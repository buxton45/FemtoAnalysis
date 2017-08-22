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

  const char* tXAxisLabels_LamKchP[13] = {cAnalysisRootTags[kLamKchP], cAnalysisRootTags[kResSig0KchP], cAnalysisRootTags[kResXi0KchP], cAnalysisRootTags[kResXiCKchP], cAnalysisRootTags[kResSigStPKchP], cAnalysisRootTags[kResSigStMKchP], cAnalysisRootTags[kResSigSt0KchP], cAnalysisRootTags[kResLamKSt0], cAnalysisRootTags[kResSig0KSt0], cAnalysisRootTags[kResXi0KSt0], cAnalysisRootTags[kResXiCKSt0], "Other", "Fake"};
  const char* tXAxisLabels_ALamKchM[13] = {cAnalysisRootTags[kALamKchM], cAnalysisRootTags[kResASig0KchM], cAnalysisRootTags[kResAXi0KchM], cAnalysisRootTags[kResAXiCKchM], cAnalysisRootTags[kResASigStMKchM], cAnalysisRootTags[kResASigStPKchM], cAnalysisRootTags[kResASigSt0KchM], cAnalysisRootTags[kResALamAKSt0], cAnalysisRootTags[kResASig0AKSt0], cAnalysisRootTags[kResAXi0AKSt0], cAnalysisRootTags[kResAXiCAKSt0], "Other", "Fake"};

  const char* tXAxisLabels_LamKchM[13] = {cAnalysisRootTags[kLamKchM], cAnalysisRootTags[kResSig0KchM], cAnalysisRootTags[kResXi0KchM], cAnalysisRootTags[kResXiCKchM], cAnalysisRootTags[kResSigStPKchM], cAnalysisRootTags[kResSigStMKchM], cAnalysisRootTags[kResSigSt0KchM], cAnalysisRootTags[kResLamAKSt0], cAnalysisRootTags[kResSig0AKSt0], cAnalysisRootTags[kResXi0AKSt0], cAnalysisRootTags[kResXiCAKSt0], "Other", "Fake"};
  const char* tXAxisLabels_ALamKchP[13] = {cAnalysisRootTags[kALamKchP], cAnalysisRootTags[kResASig0KchP], cAnalysisRootTags[kResAXi0KchP], cAnalysisRootTags[kResAXiCKchP], cAnalysisRootTags[kResASigStMKchP], cAnalysisRootTags[kResASigStPKchP], cAnalysisRootTags[kResASigSt0KchP], cAnalysisRootTags[kResALamKSt0], cAnalysisRootTags[kResASig0KSt0], cAnalysisRootTags[kResAXi0KSt0], cAnalysisRootTags[kResAXiCKSt0], "Other", "Fake"}; 

  const char* tXAxisLabels_LamK0[13] = {cAnalysisRootTags[kLamK0], cAnalysisRootTags[kResSig0K0], cAnalysisRootTags[kResXi0K0], cAnalysisRootTags[kResXiCK0], cAnalysisRootTags[kResSigStPK0], cAnalysisRootTags[kResSigStMK0], cAnalysisRootTags[kResSigSt0K0], cAnalysisRootTags[kResLamKSt0], cAnalysisRootTags[kResSig0KSt0], cAnalysisRootTags[kResXi0KSt0], cAnalysisRootTags[kResXiCKSt0], "Other", "Fake"};
  const char* tXAxisLabels_ALamK0[13] = {cAnalysisRootTags[kALamK0], cAnalysisRootTags[kResASig0K0], cAnalysisRootTags[kResAXi0K0], cAnalysisRootTags[kResAXiCK0], cAnalysisRootTags[kResASigStMK0], cAnalysisRootTags[kResASigStPK0], cAnalysisRootTags[kResASigSt0K0], cAnalysisRootTags[kResALamKSt0], cAnalysisRootTags[kResASig0KSt0], cAnalysisRootTags[kResAXi0KSt0], cAnalysisRootTags[kResAXiCKSt0], "Other", "Fake"}; 

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
void SetXAxisLabels(AnalysisType aAnType, TH1D* aHist)
{
  const char** tLabels;

  switch(aAnType) {
  case kLamKchP:
    tLabels = tXAxisLabels_LamKchP;
    break;

  case kALamKchM:
    tLabels = tXAxisLabels_ALamKchM;
    break;

  case kLamKchM:
    tLabels = tXAxisLabels_LamKchM;
    break;

  case kALamKchP:
    tLabels = tXAxisLabels_ALamKchP;
    break;

  case kLamK0:
    tLabels = tXAxisLabels_LamK0;
    break;

  case kALamK0:
    tLabels = tXAxisLabels_ALamK0;
    break;

  default:
    cout << "ERROR: SetXAxisLabels: aAnType = " << aAnType << " is not appropriate" << endl << endl;
    assert(0);
  }

  for(int i=1; i<=13; i++) aHist->GetXaxis()->SetBinLabel(i, tLabels[i-1]);
}


//________________________________________________________________________________________________________________
TH1D* BuildPairFractions(AnalysisType aAnType, TH2D* aMatrix)
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
      if(bParticleV0) ThermEventsCollection::MapAndFillPairFractionHistogramParticleV0(tReturnHist, tPDG1, tPDG2, tWeight);
      else ThermEventsCollection::MapAndFillPairFractionHistogramV0V0(tReturnHist, tPDG1, tPDG2, tWeight);
    }
  }

  SetXAxisLabels(aAnType, tReturnHist);
  return tReturnHist;
}

//________________________________________________________________________________________________________________
void PrintLambdaValues(TPad* aPad, TH1D* aHisto)
{
  aPad->cd();
  TPaveText* returnText = new TPaveText(0.50,0.25,0.70,0.85,"NDC");
    returnText->SetFillColor(0);
    returnText->SetBorderSize(0);
    returnText->SetTextAlign(22);
    returnText->SetTextFont(63);
    returnText->SetTextSize(10);

  returnText->AddText("Estimated #lambda Values");

  double tTotal = 0.;
  for(int i=1; i<=13; i++) tTotal += aHisto->GetBinContent(i);
  for(int i=1; i<=13; i++) returnText->AddText(TString(aHisto->GetXaxis()->GetBinLabel(i)) + TString::Format(" = %0.3f", aHisto->GetBinContent(i)/tTotal));

  returnText->Draw();
}

//________________________________________________________________________________________________________________
void DrawPairFractions(TPad* aPad, TH1D* aHisto)
{
  aPad->cd();
  gStyle->SetOptStat(0);

  double tNCounts = 0.;
  for(int i=1; i<=12; i++) tNCounts += aHisto->GetBinContent(i);
  double tNFakes = 0.05*tNCounts;
  aHisto->SetBinContent(13, tNFakes);

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

  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationPairFractions = tDirectory + "PairFractions.root";
  TString tFileLocationTransformMatrices = tDirectory + "TransformMatrices_Mix5.root";

//-----------------------------------------------------------------------------

  TH2D* tParentsMatrix_LamKchP = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamKchP");
  TH2D* tParentsMatrix_ALamKchM = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamKchM");

  TH2D* tParentsMatrix_LamKchM = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamKchM");
  TH2D* tParentsMatrix_ALamKchP = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamKchP");

  TH2D* tParentsMatrix_LamK0 = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixLamK0");
  TH2D* tParentsMatrix_ALamK0 = Get2dHisto(tFileLocationPairFractions, "fParentsMatrixALamK0");

  //------------------------------------------------------------------------------
  TH1D* tPairFractions_LamKchP = BuildPairFractions(kLamKchP, tParentsMatrix_LamKchP);
  TH1D* tPairFractions_ALamKchM = BuildPairFractions(kALamKchM, tParentsMatrix_ALamKchM);

  TH1D* tPairFractions_LamKchM = BuildPairFractions(kLamKchM, tParentsMatrix_LamKchM);
  TH1D* tPairFractions_ALamKchP = BuildPairFractions(kALamKchP, tParentsMatrix_ALamKchP);

  TH1D* tPairFractions_LamK0 = BuildPairFractions(kLamK0, tParentsMatrix_LamK0);
  TH1D* tPairFractions_ALamK0 = BuildPairFractions(kALamK0, tParentsMatrix_ALamK0);

  //------------------------------------------------------------------------------
  TCanvas* tPairFractionsCan = new TCanvas("tPairFractionsCan", "tPairFractionsCan", 750, 1500);
  tPairFractionsCan->Divide(2,3);

  DrawPairFractions((TPad*)tPairFractionsCan->cd(1), tPairFractions_LamKchP);
  DrawPairFractions((TPad*)tPairFractionsCan->cd(2), tPairFractions_ALamKchM);

  DrawPairFractions((TPad*)tPairFractionsCan->cd(3), tPairFractions_LamKchM);
  DrawPairFractions((TPad*)tPairFractionsCan->cd(4), tPairFractions_ALamKchP);

  DrawPairFractions((TPad*)tPairFractionsCan->cd(5), tPairFractions_LamK0);
  DrawPairFractions((TPad*)tPairFractionsCan->cd(6), tPairFractions_ALamK0);

  if(bSaveImages) tPairFractionsCan->SaveAs(tDirectory+"Figures/LambdaValuesFromParentsMatrices.pdf");

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
