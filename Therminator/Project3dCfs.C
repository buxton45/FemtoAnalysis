#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "Therm3dCf.h"
class Therm3dCf;



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


  TString tDirectory = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/";
  TString tFileLocationCfs = tDirectory + "CorrelationFunctions_10MixedEvNum.root";

  AnalysisType tAnType = kLamKchP;

  int tRebin=1;
  Therm3dCf *t3dCf = new Therm3dCf(tAnType, tFileLocationCfs, tRebin);

//-------------------------------------------------------------------------------
  TH1D* tCfFullProject = t3dCf->GetFullCf();
    tCfFullProject->SetMarkerStyle(20);
    tCfFullProject->SetMarkerColor(1);

  TH1D* tCfFull = Get1dHisto(tFileLocationCfs, TString::Format("CfFull%s", cAnalysisBaseTags[tAnType]));
    tCfFull->SetMarkerStyle(20);
    tCfFull->SetMarkerColor(2);

  TCanvas *tCanFullA = new TCanvas("tCanFullA", "tCanFullA");
  tCanFullA->cd();
  tCfFullProject->Draw();
  tCfFull->Draw("same");

  TCanvas *tCanFullB = new TCanvas("tCanFullB", "tCanFullB");
  tCanFullB->Divide(2,1);
  tCanFullB->cd(1);
  tCfFullProject->Draw();
  tCanFullB->cd(2);
  tCfFull->Draw();
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
  TH1D* tCfPrimaryOnlyProject = t3dCf->GetPrimaryOnlyCf();
    tCfPrimaryOnlyProject->SetMarkerStyle(20);
    tCfPrimaryOnlyProject->SetMarkerColor(1);

  TH1D* tCfPrimOnly = Get1dHisto(tFileLocationCfs, TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[tAnType]));
    tCfPrimOnly->SetMarkerStyle(20);
    tCfPrimOnly->SetMarkerColor(2);

  TCanvas *tCanPrimA = new TCanvas("tCanPrimA", "tCanPrimA");
  tCanPrimA->cd();
  tCfPrimaryOnlyProject->Draw();
  tCfPrimOnly->Draw("same");

  TCanvas *tCanPrimB = new TCanvas("tCanPrimB", "tCanPrimB");
  tCanPrimB->Divide(2,1);
  tCanPrimB->cd(1);
  tCfPrimaryOnlyProject->Draw();
  tCanPrimB->cd(2);
  tCfPrimOnly->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfSecondaryOnlyProject = t3dCf->GetSecondaryOnlyCf();
    tCfSecondaryOnlyProject->SetMarkerStyle(20);
    tCfSecondaryOnlyProject->SetMarkerColor(1);

  TH1D* tCfSecondaryOnly = Get1dHisto(tFileLocationCfs, TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[tAnType]));
    tCfSecondaryOnly->SetMarkerStyle(20);
    tCfSecondaryOnly->SetMarkerColor(2);

  TCanvas *tCanSecondaryA = new TCanvas("tCanSecondaryA", "tCanSecondaryA");
  tCanSecondaryA->cd();
  tCfSecondaryOnlyProject->Draw();
  tCfSecondaryOnly->Draw("same");

  TCanvas *tCanSecondaryB = new TCanvas("tCanSecondaryB", "tCanSecondaryB");
  tCanSecondaryB->Divide(2,1);
  tCanSecondaryB->cd(1);
  tCfSecondaryOnlyProject->Draw();
  tCanSecondaryB->cd(2);
  tCfSecondaryOnly->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfWithoutSigmaStProject = t3dCf->GetWithoutSigmaStCf();
    tCfWithoutSigmaStProject->SetMarkerStyle(20);
    tCfWithoutSigmaStProject->SetMarkerColor(1);

  TH1D* tCfWithoutSigmaSt = Get1dHisto(tFileLocationCfs, TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[tAnType]));
    tCfWithoutSigmaSt->SetMarkerStyle(20);
    tCfWithoutSigmaSt->SetMarkerColor(2);

  TCanvas *tCanWithoutSigmaStA = new TCanvas("tCanWithoutSigmaStA", "tCanWithoutSigmaStA");
  tCanWithoutSigmaStA->cd();
  tCfWithoutSigmaStProject->Draw();
  tCfWithoutSigmaSt->Draw("same");

  TCanvas *tCanWithoutSigmaStB = new TCanvas("tCanWithoutSigmaStB", "tCanWithoutSigmaStB");
  tCanWithoutSigmaStB->Divide(2,1);
  tCanWithoutSigmaStB->cd(1);
  tCfWithoutSigmaStProject->Draw();
  tCanWithoutSigmaStB->cd(2);
  tCfWithoutSigmaSt->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfSigmaStOnlyProject = t3dCf->GetSigmaStOnlyCf();
    tCfSigmaStOnlyProject->SetMarkerStyle(20);
    tCfSigmaStOnlyProject->SetMarkerColor(1);

  TH1D* tCfSigmaStOnly = Get1dHisto(tFileLocationCfs, TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[tAnType]));
    tCfSigmaStOnly->SetMarkerStyle(20);
    tCfSigmaStOnly->SetMarkerColor(2);

  TCanvas *tCanSigmaStOnlyA = new TCanvas("tCanSigmaStOnlyA", "tCanSigmaStOnlyA");
  tCanSigmaStOnlyA->cd();
  tCfSigmaStOnlyProject->Draw();
  tCfSigmaStOnly->Draw("same");

  TCanvas *tCanSigmaStOnlyB = new TCanvas("tCanSigmaStOnlyB", "tCanSigmaStOnlyB");
  tCanSigmaStOnlyB->Divide(2,1);
  tCanSigmaStOnlyB->cd(1);
  tCfSigmaStOnlyProject->Draw();
  tCanSigmaStOnlyB->cd(2);
  tCfSigmaStOnly->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfPrimaryAndShortDecaysProject = t3dCf->GetPrimaryAndShortDecaysCf();
    tCfPrimaryAndShortDecaysProject->SetMarkerStyle(20);
    tCfPrimaryAndShortDecaysProject->SetMarkerColor(1);

  TH1D* tCfPrimaryAndShortDecays = Get1dHisto(tFileLocationCfs, TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[tAnType]));
    tCfPrimaryAndShortDecays->SetMarkerStyle(20);
    tCfPrimaryAndShortDecays->SetMarkerColor(2);

  TCanvas *tCanPrimaryAndShortDecaysA = new TCanvas("tCanPrimaryAndShortDecaysA", "tCanPrimaryAndShortDecaysA");
  tCanPrimaryAndShortDecaysA->cd();
  tCfPrimaryAndShortDecaysProject->Draw();
  tCfPrimaryAndShortDecays->Draw("same");

  TCanvas *tCanPrimaryAndShortDecaysB = new TCanvas("tCanPrimaryAndShortDecaysB", "tCanPrimaryAndShortDecaysB");
  tCanPrimaryAndShortDecaysB->Divide(2,1);
  tCanPrimaryAndShortDecaysB->cd(1);
  tCfPrimaryAndShortDecaysProject->Draw();
  tCanPrimaryAndShortDecaysB->cd(2);
  tCfPrimaryAndShortDecays->Draw();
//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
  TH1D* tCfAtLeastOneSecondaryInPairProject = t3dCf->GetAtLeastOneSecondaryInPairCf();
    tCfAtLeastOneSecondaryInPairProject->SetMarkerStyle(20);
    tCfAtLeastOneSecondaryInPairProject->SetMarkerColor(1);

  TCanvas *tCanAtLeastOneSecondaryInPairA = new TCanvas("tCanAtLeastOneSecondaryInPairA", "tCanAtLeastOneSecondaryInPairA");
  tCanAtLeastOneSecondaryInPairA->cd();
  tCfAtLeastOneSecondaryInPairProject->Draw();
//-------------------------------------------------------------------------------


  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}









