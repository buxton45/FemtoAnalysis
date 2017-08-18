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
/*
  double tPrimaryFractionInOther = 0.5;
  double tAdditionalPrimary = tPrimaryFractionInOther*aHisto->GetBinContent(12);
  double tTotalPrimaryLambda = aHisto->GetBinContent(1) + tAdditionalPrimary;
  aHisto->SetBinContent(1, tTotalPrimaryLambda);
*/
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
  bool bZoomProtonParents = true;

  TString tDirectory = "~/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationPairFractions = tDirectory + "PairFractions.root";

//-----------------------------------------------------------------------------

  TH1D* tPairFractions_LamKchP = Get1dHisto(tFileLocationPairFractions,"fPairFractionsLamKchP");
  const char* tXAxisLabels_LamKchP[12] = {cAnalysisRootTags[kLamKchP], cAnalysisRootTags[kResSig0KchP], cAnalysisRootTags[kResXi0KchP], cAnalysisRootTags[kResXiCKchP], cAnalysisRootTags[kResSigStPKchP], cAnalysisRootTags[kResSigStMKchP], cAnalysisRootTags[kResSigSt0KchP], cAnalysisRootTags[kResLamKSt0], cAnalysisRootTags[kResSig0KSt0], cAnalysisRootTags[kResXi0KSt0], cAnalysisRootTags[kResXiCKSt0], "Fake"};
  TH1D* tPairFractions_ALamKchM = Get1dHisto(tFileLocationPairFractions,"fPairFractionsALamKchM");
  const char* tXAxisLabels_ALamKchM[12] = {cAnalysisRootTags[kALamKchM], cAnalysisRootTags[kResASig0KchM], cAnalysisRootTags[kResAXi0KchM], cAnalysisRootTags[kResAXiCKchM], cAnalysisRootTags[kResASigStMKchM], cAnalysisRootTags[kResASigStPKchM], cAnalysisRootTags[kResASigSt0KchM], cAnalysisRootTags[kResALamAKSt0], cAnalysisRootTags[kResASig0AKSt0], cAnalysisRootTags[kResAXi0AKSt0], cAnalysisRootTags[kResAXiCAKSt0], "Fake"};

  TH1D* tPairFractions_LamKchM = Get1dHisto(tFileLocationPairFractions,"fPairFractionsLamKchM");
  const char* tXAxisLabels_LamKchM[12] = {cAnalysisRootTags[kLamKchM], cAnalysisRootTags[kResSig0KchM], cAnalysisRootTags[kResXi0KchM], cAnalysisRootTags[kResXiCKchM], cAnalysisRootTags[kResSigStPKchM], cAnalysisRootTags[kResSigStMKchM], cAnalysisRootTags[kResSigSt0KchM], cAnalysisRootTags[kResLamAKSt0], cAnalysisRootTags[kResSig0AKSt0], cAnalysisRootTags[kResXi0AKSt0], cAnalysisRootTags[kResXiCAKSt0], "Fake"};
  TH1D* tPairFractions_ALamKchP = Get1dHisto(tFileLocationPairFractions,"fPairFractionsALamKchP");
  const char* tXAxisLabels_ALamKchP[12] = {cAnalysisRootTags[kALamKchP], cAnalysisRootTags[kResASig0KchP], cAnalysisRootTags[kResAXi0KchP], cAnalysisRootTags[kResAXiCKchP], cAnalysisRootTags[kResASigStMKchP], cAnalysisRootTags[kResASigStPKchP], cAnalysisRootTags[kResASigSt0KchP], cAnalysisRootTags[kResALamKSt0], cAnalysisRootTags[kResASig0KSt0], cAnalysisRootTags[kResAXi0KSt0], cAnalysisRootTags[kResAXiCKSt0], "Fake"}; 

  TH1D* tPairFractions_LamK0 = Get1dHisto(tFileLocationPairFractions,"fPairFractionsLamK0");
  const char* tXAxisLabels_LamK0[12] = {cAnalysisRootTags[kLamK0], cAnalysisRootTags[kResSig0K0], cAnalysisRootTags[kResXi0K0], cAnalysisRootTags[kResXiCK0], cAnalysisRootTags[kResSigStPK0], cAnalysisRootTags[kResSigStMK0], cAnalysisRootTags[kResSigSt0K0], cAnalysisRootTags[kResLamKSt0], cAnalysisRootTags[kResSig0KSt0], cAnalysisRootTags[kResXi0KSt0], cAnalysisRootTags[kResXiCKSt0], "Fake"};

  TH1D* tPairFractions_ALamK0 = Get1dHisto(tFileLocationPairFractions,"fPairFractionsALamK0");
  const char* tXAxisLabels_ALamK0[12] = {cAnalysisRootTags[kALamK0], cAnalysisRootTags[kResASig0K0], cAnalysisRootTags[kResAXi0K0], cAnalysisRootTags[kResAXiCK0], cAnalysisRootTags[kResASigStMK0], cAnalysisRootTags[kResASigStPK0], cAnalysisRootTags[kResASigSt0K0], cAnalysisRootTags[kResALamKSt0], cAnalysisRootTags[kResASig0KSt0], cAnalysisRootTags[kResAXi0KSt0], cAnalysisRootTags[kResAXiCKSt0], "Fake"}; 

  for(int i=1; i<=12; i++)
  {
    tPairFractions_LamKchP->GetXaxis()->SetBinLabel(i,tXAxisLabels_LamKchP[i-1]);
    tPairFractions_ALamKchM->GetXaxis()->SetBinLabel(i,tXAxisLabels_ALamKchM[i-1]);

    tPairFractions_LamKchM->GetXaxis()->SetBinLabel(i,tXAxisLabels_LamKchM[i-1]);
    tPairFractions_ALamKchP->GetXaxis()->SetBinLabel(i,tXAxisLabels_ALamKchP[i-1]);

    tPairFractions_LamK0->GetXaxis()->SetBinLabel(i,tXAxisLabels_LamK0[i-1]);
    tPairFractions_ALamK0->GetXaxis()->SetBinLabel(i,tXAxisLabels_ALamK0[i-1]);
  }


  TCanvas* tPairFractionsCan = new TCanvas("tPairFractionsCan", "tPairFractionsCan", 750, 1500);
  tPairFractionsCan->Divide(2,3);

  DrawPairFractions((TPad*)tPairFractionsCan->cd(1), tPairFractions_LamKchP);
  DrawPairFractions((TPad*)tPairFractionsCan->cd(2), tPairFractions_ALamKchM);

  DrawPairFractions((TPad*)tPairFractionsCan->cd(3), tPairFractions_LamKchM);
  DrawPairFractions((TPad*)tPairFractionsCan->cd(4), tPairFractions_ALamKchP);

  DrawPairFractions((TPad*)tPairFractionsCan->cd(5), tPairFractions_LamK0);
  DrawPairFractions((TPad*)tPairFractionsCan->cd(6), tPairFractions_ALamK0);

  if(bSaveImages) tPairFractionsCan->SaveAs(tDirectory+"Figures/ParentsFractions.pdf");


  //------------------------------------------------------------
  TH1D* tProtonParents = Get1dHisto(tFileLocationPairFractions, "fProtonParents");
  TH1D* tAProtonParents = Get1dHisto(tFileLocationPairFractions, "fAProtonParents");

  for(unsigned int i=0; i<cAllProtonFathers.size(); i++)
  {
    tProtonParents->GetXaxis()->SetBinLabel(i+1, GetParticleName(cAllProtonFathers[i]));
    tAProtonParents->GetXaxis()->SetBinLabel(i+1, GetParticleName(cAllProtonFathers[i]));
  }

  tProtonParents->LabelsOption("v", "X");
  tAProtonParents->LabelsOption("v", "X");

  TCanvas *tProtonParentsCan;
  TCanvas *tAProtonParentsCan;

  if(bZoomProtonParents)
  {
    tProtonParentsCan = new TCanvas("tProtonParentsCan","tProtonParentsCan");
    tAProtonParentsCan = new TCanvas("tAProtonParentsCan","tAProtonParentsCan");

    tProtonParents->GetXaxis()->SetRange(90,112);
    tAProtonParents->GetXaxis()->SetRange(56,78);

    
  }
  else
  {
    tProtonParentsCan = new TCanvas("tProtonParentsCan","tProtonParentsCan", 2100, 1000);
    tAProtonParentsCan = new TCanvas("tAProtonParentsCan","tAProtonParentsCan", 2100, 1000);

    tProtonParents->GetXaxis()->SetRange(82,168);
    tAProtonParents->GetXaxis()->SetRange(1,84);

    tProtonParents->GetXaxis()->SetLabelSize(0.025);
    tAProtonParents->GetXaxis()->SetLabelSize(0.025);
  }

  tProtonParentsCan->cd();
  tProtonParents->Draw();

  tAProtonParentsCan->cd();
  tAProtonParents->Draw();

  if(bSaveImages)
  {
    if(bZoomProtonParents)
    {
      tProtonParentsCan->SaveAs(tDirectory+"Figures/ProtonParents.pdf");
      tAProtonParentsCan->SaveAs(tDirectory+"Figures/AntiProtonParents.pdf");
    }
    else
    {
      tProtonParentsCan->SaveAs(tDirectory+"Figures/ProtonParents_UnZoomed.pdf");
      tAProtonParentsCan->SaveAs(tDirectory+"Figures/AntiProtonParents_UnZoomed.pdf");
    }
  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
