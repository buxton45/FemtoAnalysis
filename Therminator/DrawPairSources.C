#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "ThermEventsCollection.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"

#include "ThermCommon.h"


//________________________________________________________________________________________________________________
void DrawPairSources(TPad* aPad, TString aFileName, AnalysisType aAnType, bool aDrawLogY=true)
{
  aPad->cd();
  aPad->SetLogy(aDrawLogY);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  TString tBaseNameFull = TString::Format("PairSourceFull%s", cAnalysisBaseTags[aAnType]);
  TString tBaseNamePrimaryOnly = TString::Format("PairSourcePrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tBaseNamePrimaryAndShortDecays = TString::Format("PairSourcePrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tBaseNameWithoutSigmaSt = TString::Format("PairSourceWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tBaseNameSigmaStOnly = TString::Format("PairSourceSigmaStOnly%s", cAnalysisBaseTags[aAnType]);

  TH1D* tSourceFull = Get1dHisto(aFileName, tBaseNameFull);
    tSourceFull->SetLineColor(1);
  TH1D* tSourcePrimaryOnly = Get1dHisto(aFileName, tBaseNamePrimaryOnly);
    tSourcePrimaryOnly->SetLineColor(2);
  TH1D* tSourcePrimaryAndShortDecays = Get1dHisto(aFileName, tBaseNamePrimaryAndShortDecays);
    tSourcePrimaryAndShortDecays->SetLineColor(3);
  TH1D* tSourceWithoutSigmaSt = Get1dHisto(aFileName, tBaseNameWithoutSigmaSt);
    tSourceWithoutSigmaSt->SetLineColor(4);
  TH1D* tSourceSigmaStOnly = Get1dHisto(aFileName, tBaseNameSigmaStOnly);
    tSourceSigmaStOnly->SetLineColor(6);

  tSourceFull->GetXaxis()->SetTitle("r*");
  tSourceFull->GetYaxis()->SetTitle("dN/dr*");

  tSourceFull->GetXaxis()->SetRangeUser(0.,200.);

  tSourceFull->Draw();
  tSourcePrimaryOnly->Draw("same");
  tSourcePrimaryAndShortDecays->Draw("same");
  tSourceWithoutSigmaSt->Draw("same");
  tSourceSigmaStOnly->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.60, 0.85, 0.85);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s Pair Source", cAnalysisRootTags[aAnType]));

  tLeg->AddEntry(tSourceFull, "Full");
  tLeg->AddEntry(tSourceWithoutSigmaSt, "w/o #Sigma*");
  tLeg->AddEntry(tSourcePrimaryOnly, "Primary Only");
  tLeg->AddEntry(tSourcePrimaryAndShortDecays, "Primary and short decays");
  tLeg->AddEntry(tSourceSigmaStOnly, "#Sigma* Only");

  tLeg->Draw();
}


//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  bool bDrawLogY = true;
  bool bSaveFigures = false;

  TString tFileName = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions.root";
  TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20170928/Figures/";

  TString tSaveLocationBaseLamKchP = tSaveLocationBase + TString("LamKchP/");
  TString tSaveLocationBaseALamKchP = tSaveLocationBase + TString("ALamKchP/");
  TString tSaveLocationBaseLamKchM = tSaveLocationBase + TString("LamKchM/");
  TString tSaveLocationBaseALamKchM = tSaveLocationBase + TString("ALamKchM/");
  TString tSaveLocationBaseLamK0 = tSaveLocationBase + TString("LamK0/");
  TString tSaveLocationBaseALamK0 = tSaveLocationBase + TString("ALamK0/");

  //--------------------------------------------

  TCanvas* tCanLamKchP = new TCanvas("PairSources_LamKchP", "PairSources_LamKchP");
  tCanLamKchP->Divide(2,1);
  DrawPairSources((TPad*)tCanLamKchP->cd(1), tFileName, kLamKchP, bDrawLogY);
  DrawPairSources((TPad*)tCanLamKchP->cd(2), tFileName, kALamKchM, bDrawLogY);

  TCanvas* tCanLamKchM = new TCanvas("PairSources_LamKchM", "PairSources_LamKchM");
  tCanLamKchM->Divide(2,1);
  DrawPairSources((TPad*)tCanLamKchM->cd(1), tFileName, kLamKchM, bDrawLogY);
  DrawPairSources((TPad*)tCanLamKchM->cd(2), tFileName, kALamKchP, bDrawLogY);

  TCanvas* tCanLamK0 = new TCanvas("PairSources_LamK0", "PairSources_LamK0");
  tCanLamK0->Divide(2,1);
  DrawPairSources((TPad*)tCanLamK0->cd(1), tFileName, kLamK0, bDrawLogY);
  DrawPairSources((TPad*)tCanLamK0->cd(2), tFileName, kALamK0, bDrawLogY);

  if(bSaveFigures)
  {
    tCanLamKchP->SaveAs(tSaveLocationBaseLamKchP + TString(tCanLamKchP->GetName()) + TString(".eps"));
    tCanLamKchM->SaveAs(tSaveLocationBaseLamKchM + TString(tCanLamKchM->GetName()) + TString(".eps"));
    tCanLamK0->SaveAs(tSaveLocationBaseLamK0 + TString(tCanLamK0->GetName()) + TString(".eps"));
  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
