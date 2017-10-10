#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"

#include "Therm3dCf.h"
class Therm3dCf;

//________________________________________________________________________________________________________________
void Draw1vs2(TPad* aPad, AnalysisType aAnType, TH1D* aCf1, TH1D* aCf2, TString aDescriptor1, TString aDescriptor2, TString aOverallDescriptor)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //---------------------------------------------------------------
  aCf1->GetXaxis()->SetTitle("k* (GeV/c)");
  aCf1->GetYaxis()->SetTitle("C(k*)");

//  aCf1->GetXaxis()->SetRangeUser(0.,0.329);
  aCf1->GetYaxis()->SetRangeUser(0.86, 1.07);

  aCf1->Draw();
  aCf2->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s %s Cfs", cAnalysisRootTags[aAnType], aOverallDescriptor.Data()));

  tLeg->AddEntry(aCf1, aDescriptor1.Data());
  tLeg->AddEntry(aCf2, aDescriptor2.Data());


  tLeg->Draw();

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

  AnalysisType tAnType = kLamKchP;

  bool bSaveFigures = false;
  int tRebin=2;

  TString tFileLocationCfs1 = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions_10MixedEvNum.root";
  TString tFileLocationCfs2 = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions_10MixedEvNum_WeightParentsInteraction.root";

  TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171012/Figures/";

  vector<ParticlePDGType> tParticlesInPair = ThermPairAnalysis::GetPartTypes(tAnType);

  TString tDescriptor1 = TString::Format("Weight %s%s", GetPDGRootName(tParticlesInPair[0]), GetPDGRootName(tParticlesInPair[1]));
  TString tDescriptor2 = "Weight Parents";
  TString tOverallDescriptor = "";

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 24;



  //--------------------------------------------

  Therm3dCf *t3dCf1 = new Therm3dCf(tAnType, tFileLocationCfs1, tRebin);
    TH1D* tCfFull1 = t3dCf1->GetFullCf(tMarkerStyle1);
    TH1D* tCfPrimaryOnly1 = t3dCf1->GetPrimaryOnlyCf(tMarkerStyle1);
    TH1D* tCfPrimaryAndShortDecays1 = t3dCf1->GetPrimaryAndShortDecaysCf(tMarkerStyle1);
    TH1D* tCfWithoutSigmaSt1 = t3dCf1->GetWithoutSigmaStCf(tMarkerStyle1);
    TH1D* tCfSigmaStOnly1 = t3dCf1->GetSigmaStOnlyCf(tMarkerStyle1);
    TH1D* tCfSecondaryOnly1 = t3dCf1->GetSecondaryOnlyCf(tMarkerStyle1);
    TH1D* tCfAtLeastOneSecondaryInPair1 = t3dCf1->GetAtLeastOneSecondaryInPairCf(tMarkerStyle1);


  Therm3dCf *t3dCf2 = new Therm3dCf(tAnType, tFileLocationCfs2, tRebin);
    TH1D* tCfFull2 = t3dCf2->GetFullCf(tMarkerStyle2);
    TH1D* tCfPrimaryOnly2 = t3dCf2->GetPrimaryOnlyCf(tMarkerStyle2);
    TH1D* tCfPrimaryAndShortDecays2 = t3dCf2->GetPrimaryAndShortDecaysCf(tMarkerStyle2);
    TH1D* tCfWithoutSigmaSt2 = t3dCf2->GetWithoutSigmaStCf(tMarkerStyle2);
    TH1D* tCfSigmaStOnly2 = t3dCf2->GetSigmaStOnlyCf(tMarkerStyle2);
    TH1D* tCfSecondaryOnly2 = t3dCf2->GetSecondaryOnlyCf(tMarkerStyle2);
    TH1D* tCfAtLeastOneSecondaryInPair2 = t3dCf2->GetAtLeastOneSecondaryInPairCf(tMarkerStyle2);

//-------------------------------------------------------------------------------

  TCanvas* tCanFull = new TCanvas("CompareCfs_Full", "CompareCfs_Full");
  tOverallDescriptor = TString("Full");
  Draw1vs2((TPad*)tCanFull, tAnType, tCfFull1, tCfFull2, tDescriptor1, tDescriptor2, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanPrimaryOnly = new TCanvas("CompareCfs_PrimaryOnly", "CompareCfs_PrimaryOnly");
  tOverallDescriptor = TString("PrimaryOnly");
  Draw1vs2((TPad*)tCanPrimaryOnly, tAnType, tCfPrimaryOnly1, tCfPrimaryOnly2, tDescriptor1, tDescriptor2, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanPrimaryAndShortDecays = new TCanvas("CompareCfs_PrimaryAndShortDecays", "CompareCfs_PrimaryAndShortDecays");
  tOverallDescriptor = TString("PrimaryAndShortDecays");
  Draw1vs2((TPad*)tCanPrimaryAndShortDecays, tAnType, tCfPrimaryAndShortDecays1, tCfPrimaryAndShortDecays2, tDescriptor1, tDescriptor2, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanWithoutSigmaSt = new TCanvas("CompareCfs_WithoutSigmaSt", "CompareCfs_WithoutSigmaSt");
  tOverallDescriptor = TString("WithoutSigmaSt");
  Draw1vs2((TPad*)tCanWithoutSigmaSt, tAnType, tCfWithoutSigmaSt1, tCfWithoutSigmaSt2, tDescriptor1, tDescriptor2, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanSigmaStOnly = new TCanvas("CompareCfs_SigmaStOnly", "CompareCfs_SigmaStOnly");
  tOverallDescriptor = TString("SigmaStOnly");
  Draw1vs2((TPad*)tCanSigmaStOnly, tAnType, tCfSigmaStOnly1, tCfSigmaStOnly2, tDescriptor1, tDescriptor2, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanSecondaryOnly = new TCanvas("CompareCfs_SecondaryOnly", "CompareCfs_SecondaryOnly");
  tOverallDescriptor = TString("SecondaryOnly");
  Draw1vs2((TPad*)tCanSecondaryOnly, tAnType, tCfSecondaryOnly1, tCfSecondaryOnly2, tDescriptor1, tDescriptor2, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanAtLeastOneSecondaryInPair = new TCanvas("CompareCfs_AtLeastOneSecondaryInPair", "CompareCfs_AtLeastOneSecondaryInPair");
  tOverallDescriptor = TString("AtLeastOneSecondaryInPair");
  Draw1vs2((TPad*)tCanAtLeastOneSecondaryInPair, tAnType, tCfAtLeastOneSecondaryInPair1, tCfAtLeastOneSecondaryInPair2, tDescriptor1, tDescriptor2, tOverallDescriptor);

//-------------------------------------------------------------------------------

  if(bSaveFigures)
  {
    TString tSaveFileBase = tSaveLocationBase + TString::Format("%s/", cAnalysisBaseTags[tAnType]);
    TString tSaveNameFull, tSaveNamePrimaryOnly, tSaveNamePrimaryAndShortDecays, tSaveNameWithoutSigmaSt, tSaveNameSigmaStOnly, tSaveNameSecondaryOnly, tSaveNameAtLeastOneSecondaryInPair;

    tSaveNameFull = tSaveFileBase + TString(tCanFull->GetName());
    tSaveNamePrimaryOnly = tSaveFileBase + TString(tCanPrimaryOnly->GetName());
    tSaveNamePrimaryAndShortDecays = tSaveFileBase + TString(tCanPrimaryAndShortDecays->GetName());
    tSaveNameWithoutSigmaSt = tSaveFileBase + TString(tCanWithoutSigmaSt->GetName());
    tSaveNameSigmaStOnly = tSaveFileBase + TString(tCanSigmaStOnly->GetName());
    tSaveNameSecondaryOnly = tSaveFileBase + TString(tCanSecondaryOnly->GetName());
    tSaveNameAtLeastOneSecondaryInPair = tSaveFileBase + TString(tCanAtLeastOneSecondaryInPair->GetName());

    tSaveNameFull += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNamePrimaryOnly += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNamePrimaryAndShortDecays += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNameWithoutSigmaSt += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNameSigmaStOnly += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNameSecondaryOnly += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNameAtLeastOneSecondaryInPair += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");

    tCanFull->SaveAs(tSaveNameFull);
    tCanPrimaryOnly->SaveAs(tSaveNamePrimaryOnly);
    tCanPrimaryAndShortDecays->SaveAs(tSaveNamePrimaryAndShortDecays);
    tCanWithoutSigmaSt->SaveAs(tSaveNameWithoutSigmaSt);
    tCanSigmaStOnly->SaveAs(tSaveNameSigmaStOnly);
    tCanSecondaryOnly->SaveAs(tSaveNameSecondaryOnly);
    tCanAtLeastOneSecondaryInPair->SaveAs(tSaveNameAtLeastOneSecondaryInPair);
  }


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
