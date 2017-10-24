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
void Draw1vs2vs3(TPad* aPad, AnalysisType aAnType, TH1D* aCf1, TH1D* aCf2, TH1D* aCf3, TString aDescriptor1, TString aDescriptor2, TString aDescriptor3, TString aOverallDescriptor)
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
  aCf3->Draw("same");
  aCf1->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s %s Cfs", cAnalysisRootTags[aAnType], aOverallDescriptor.Data()));

  tLeg->AddEntry(aCf1, aDescriptor1.Data());
  tLeg->AddEntry(aCf2, aDescriptor2.Data());
  tLeg->AddEntry(aCf3, aDescriptor3.Data());


  tLeg->Draw();

  TLine* tLine = new TLine(aCf1->GetXaxis()->GetBinLowEdge(1), 1, aCf1->GetXaxis()->GetBinUpEdge(aCf1->GetNbinsX()), 1);
  tLine->SetLineColor(14);
  tLine->Draw();
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

  bool bSaveFigures = true;
  int tRebin=2;
  double tMinNorm = 1.0/*0.32*/;
  double tMaxNorm = 2.0/*0.40*/;

  TString tFileLocationCfs1 = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions.root";
  TString tFileLocationCfs2 = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions_WeightParentsInteraction.root";
  TString tFileLocationCfs3 = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions_WeightParentsInteraction_OnlyWeightLongDecayParents.root";

  TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171026/Figures/";

  vector<ParticlePDGType> tParticlesInPair = ThermPairAnalysis::GetPartTypes(tAnType);

  TString tDescriptor1 = TString::Format("Weight %s%s", GetPDGRootName(tParticlesInPair[0]), GetPDGRootName(tParticlesInPair[1]));
  TString tDescriptor2 = "Weight Parents";
  TString tDescriptor3 = "Weight Only Long Parents";
  TString tOverallDescriptor = "";

  int tMarkerStyle1 = 20;
  int tMarkerStyle2 = 24;
  int tMarkerStyle3 = 26;

  int tColor1 = 1;
  int tColor2 = 2;
  int tColor3 = 4;

  //--------------------------------------------

  Therm3dCf *t3dCf1 = new Therm3dCf(tAnType, tFileLocationCfs1, tRebin);
  t3dCf1->SetNormalizationRegion(tMinNorm, tMaxNorm);
    TH1D* tCfFull1 = t3dCf1->GetFullCf(tMarkerStyle1, tColor1);
    TH1D* tCfPrimaryOnly1 = t3dCf1->GetPrimaryOnlyCf(tMarkerStyle1, tColor1);
    TH1D* tCfPrimaryAndShortDecays1 = t3dCf1->GetPrimaryAndShortDecaysCf(tMarkerStyle1, tColor1);
    TH1D* tCfWithoutSigmaSt1 = t3dCf1->GetWithoutSigmaStCf(tMarkerStyle1, tColor1);
    TH1D* tCfSigmaStOnly1 = t3dCf1->GetSigmaStOnlyCf(tMarkerStyle1, tColor1);
    TH1D* tCfSecondaryOnly1 = t3dCf1->GetSecondaryOnlyCf(tMarkerStyle1, tColor1);
    TH1D* tCfAtLeastOneSecondaryInPair1 = t3dCf1->GetAtLeastOneSecondaryInPairCf(tMarkerStyle1, tColor1);
    TH1D* tCfLongDecays1 = t3dCf1->GetLongDecaysCf(1000, tMarkerStyle1, tColor1);


  Therm3dCf *t3dCf2 = new Therm3dCf(tAnType, tFileLocationCfs2, tRebin);
  t3dCf2->SetNormalizationRegion(tMinNorm, tMaxNorm);
    TH1D* tCfFull2 = t3dCf2->GetFullCf(tMarkerStyle2, tColor2);
    TH1D* tCfPrimaryOnly2 = t3dCf2->GetPrimaryOnlyCf(tMarkerStyle2, tColor2);
    TH1D* tCfPrimaryAndShortDecays2 = t3dCf2->GetPrimaryAndShortDecaysCf(tMarkerStyle2, tColor2);
    TH1D* tCfWithoutSigmaSt2 = t3dCf2->GetWithoutSigmaStCf(tMarkerStyle2, tColor2);
    TH1D* tCfSigmaStOnly2 = t3dCf2->GetSigmaStOnlyCf(tMarkerStyle2, tColor2);
    TH1D* tCfSecondaryOnly2 = t3dCf2->GetSecondaryOnlyCf(tMarkerStyle2, tColor2);
    TH1D* tCfAtLeastOneSecondaryInPair2 = t3dCf2->GetAtLeastOneSecondaryInPairCf(tMarkerStyle2, tColor2);
    TH1D* tCfLongDecays2 = t3dCf2->GetLongDecaysCf(1000, tMarkerStyle2, tColor2);

  Therm3dCf *t3dCf3 = new Therm3dCf(tAnType, tFileLocationCfs3, tRebin);
  t3dCf3->SetNormalizationRegion(tMinNorm, tMaxNorm);
    TH1D* tCfFull3 = t3dCf3->GetFullCf(tMarkerStyle3, tColor3);
    TH1D* tCfPrimaryOnly3 = t3dCf3->GetPrimaryOnlyCf(tMarkerStyle3, tColor3);
    TH1D* tCfPrimaryAndShortDecays3 = t3dCf3->GetPrimaryAndShortDecaysCf(tMarkerStyle3, tColor3);
    TH1D* tCfWithoutSigmaSt3 = t3dCf3->GetWithoutSigmaStCf(tMarkerStyle3, tColor3);
    TH1D* tCfSigmaStOnly3 = t3dCf3->GetSigmaStOnlyCf(tMarkerStyle3, tColor3);
    TH1D* tCfSecondaryOnly3 = t3dCf3->GetSecondaryOnlyCf(tMarkerStyle3, tColor3);
    TH1D* tCfAtLeastOneSecondaryInPair3 = t3dCf3->GetAtLeastOneSecondaryInPairCf(tMarkerStyle3, tColor3);
    TH1D* tCfLongDecays3 = t3dCf3->GetLongDecaysCf(1000, tMarkerStyle3, tColor3);

//-------------------------------------------------------------------------------

  TCanvas* tCanFull = new TCanvas("CompareCfs3Methods_Full", "CompareCfs3Methods_Full");
  tOverallDescriptor = TString("Full");
  Draw1vs2vs3((TPad*)tCanFull, tAnType, tCfFull1, tCfFull2, tCfFull3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanPrimaryOnly = new TCanvas("CompareCfs3Methods_PrimaryOnly", "CompareCfs3Methods_PrimaryOnly");
  tOverallDescriptor = TString("PrimaryOnly");
  Draw1vs2vs3((TPad*)tCanPrimaryOnly, tAnType, tCfPrimaryOnly1, tCfPrimaryOnly2, tCfPrimaryOnly3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanPrimaryAndShortDecays = new TCanvas("CompareCfs3Methods_PrimaryAndShortDecays", "CompareCfs3Methods_PrimaryAndShortDecays");
  tOverallDescriptor = TString("PrimaryAndShortDecays");
  Draw1vs2vs3((TPad*)tCanPrimaryAndShortDecays, tAnType, tCfPrimaryAndShortDecays1, tCfPrimaryAndShortDecays2, tCfPrimaryAndShortDecays3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanWithoutSigmaSt = new TCanvas("CompareCfs3Methods_WithoutSigmaSt", "CompareCfs3Methods_WithoutSigmaSt");
  tOverallDescriptor = TString("WithoutSigmaSt");
  Draw1vs2vs3((TPad*)tCanWithoutSigmaSt, tAnType, tCfWithoutSigmaSt1, tCfWithoutSigmaSt2, tCfWithoutSigmaSt3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanSigmaStOnly = new TCanvas("CompareCfs3Methods_SigmaStOnly", "CompareCfs3Methods_SigmaStOnly");
  tOverallDescriptor = TString("SigmaStOnly");
  Draw1vs2vs3((TPad*)tCanSigmaStOnly, tAnType, tCfSigmaStOnly1, tCfSigmaStOnly2, tCfSigmaStOnly3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanSecondaryOnly = new TCanvas("CompareCfs3Methods_SecondaryOnly", "CompareCfs3Methods_SecondaryOnly");
  tOverallDescriptor = TString("SecondaryOnly");
  Draw1vs2vs3((TPad*)tCanSecondaryOnly, tAnType, tCfSecondaryOnly1, tCfSecondaryOnly2, tCfSecondaryOnly3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanAtLeastOneSecondaryInPair = new TCanvas("CompareCfs3Methods_AtLeastOneSecondaryInPair", "CompareCfs3Methods_AtLeastOneSecondaryInPair");
  tOverallDescriptor = TString("AtLeastOneSecondaryInPair");
  Draw1vs2vs3((TPad*)tCanAtLeastOneSecondaryInPair, tAnType, tCfAtLeastOneSecondaryInPair1, tCfAtLeastOneSecondaryInPair2, tCfAtLeastOneSecondaryInPair3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

  //--------------------------------------------

  TCanvas* tCanLongDecays = new TCanvas("CompareCfs3Methods_LongDecays", "CompareCfs3Methods_LongDecays");
  tOverallDescriptor = TString("LongDecays");
  Draw1vs2vs3((TPad*)tCanLongDecays, tAnType, tCfLongDecays1, tCfLongDecays2, tCfLongDecays3, tDescriptor1, tDescriptor2, tDescriptor3, tOverallDescriptor);

//-------------------------------------------------------------------------------

  if(bSaveFigures)
  {
    TString tSaveFileBase = tSaveLocationBase + TString::Format("%s/", cAnalysisBaseTags[tAnType]);
    TString tSaveNameFull, tSaveNamePrimaryOnly, tSaveNamePrimaryAndShortDecays, tSaveNameWithoutSigmaSt, tSaveNameSigmaStOnly, tSaveNameSecondaryOnly, tSaveNameAtLeastOneSecondaryInPair, tSaveNameLongDecays;

    tSaveNameFull = tSaveFileBase + TString(tCanFull->GetName());
    tSaveNamePrimaryOnly = tSaveFileBase + TString(tCanPrimaryOnly->GetName());
    tSaveNamePrimaryAndShortDecays = tSaveFileBase + TString(tCanPrimaryAndShortDecays->GetName());
    tSaveNameWithoutSigmaSt = tSaveFileBase + TString(tCanWithoutSigmaSt->GetName());
    tSaveNameSigmaStOnly = tSaveFileBase + TString(tCanSigmaStOnly->GetName());
    tSaveNameSecondaryOnly = tSaveFileBase + TString(tCanSecondaryOnly->GetName());
    tSaveNameAtLeastOneSecondaryInPair = tSaveFileBase + TString(tCanAtLeastOneSecondaryInPair->GetName());
    tSaveNameLongDecays = tSaveFileBase + TString(tCanLongDecays->GetName());

    tSaveNameFull += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNamePrimaryOnly += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNamePrimaryAndShortDecays += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNameWithoutSigmaSt += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNameSigmaStOnly += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNameSecondaryOnly += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNameAtLeastOneSecondaryInPair += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");
    tSaveNameLongDecays += TString::Format("_%s", cAnalysisBaseTags[tAnType]) + TString(".eps");

    tCanFull->SaveAs(tSaveNameFull);
    tCanPrimaryOnly->SaveAs(tSaveNamePrimaryOnly);
    tCanPrimaryAndShortDecays->SaveAs(tSaveNamePrimaryAndShortDecays);
    tCanWithoutSigmaSt->SaveAs(tSaveNameWithoutSigmaSt);
    tCanSigmaStOnly->SaveAs(tSaveNameSigmaStOnly);
    tCanSecondaryOnly->SaveAs(tSaveNameSecondaryOnly);
    tCanAtLeastOneSecondaryInPair->SaveAs(tSaveNameAtLeastOneSecondaryInPair);
    tCanLongDecays->SaveAs(tSaveNameLongDecays);
  }


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
