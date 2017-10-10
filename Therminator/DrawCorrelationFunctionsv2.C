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
TH1D* CombineConjugates(TH1* aNum1, TH1* aCf1, TH1* aNum2, TH1* aCf2, TString aName)
{
  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;

  TH1D* tReturnCf = (TH1D*)aCf1->Clone(aName);
  double tScale1 = aNum1->Integral(aNum1->FindBin(tMinNorm), aNum1->FindBin(tMaxNorm));
  tReturnCf->Scale(tScale1);

  double tScale2 = aNum2->Integral(aNum2->FindBin(tMinNorm), aNum2->FindBin(tMaxNorm));
  tReturnCf->Add(aCf2, tScale2);

  double tScaleTot = tScale1 + tScale2;
  tReturnCf->Scale(1./tScaleTot);

  return tReturnCf;
}

//________________________________________________________________________________________________________________
void DrawCfsWithConj(TPad* aPad, Therm3dCf* a3dCf, Therm3dCf* aConj3dCf)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //---------------------------------------------------------------

  int tColorFull = 1;
  int tColorPrimaryOnly = 2;
  int tColorPrimaryAndShortDecays = 3;
  int tColorWithoutSigmaSt = 4;
  int tColorSigmaStOnly = 20;
  int tColorSecondaryOnly = 6;
  int tColorAtLeastOneSecondaryInPair = 28;

  int tMarkerStyleFull = 20;
  int tMarkerStylePrimaryOnly = 20;
  int tMarkerStylePrimaryAndShortDecays = 20;
  int tMarkerStyleWithoutSigmaSt = 20;
  int tMarkerStyleSigmaStOnly = 20;
  int tMarkerStyleSecondaryOnly = 20;
  int tMarkerStyleAtLeastOneSecondaryInPair = 20;

  //---------------------------------------------------------------

  TH1D* tNumFull = a3dCf->GetFullNum();
  TH1D* tCfFull = a3dCf->GetFullCf(tMarkerStyleFull, tColorFull);

  TH1D* tNumPrimaryOnly = a3dCf->GetPrimaryOnlyNum();
  TH1D* tCfPrimaryOnly = a3dCf->GetPrimaryOnlyCf(tMarkerStylePrimaryOnly, tColorPrimaryOnly);

  TH1D* tNumPrimaryAndShortDecays = a3dCf->GetPrimaryAndShortDecaysNum();
  TH1D* tCfPrimaryAndShortDecays = a3dCf->GetPrimaryAndShortDecaysCf(tMarkerStylePrimaryAndShortDecays, tColorPrimaryAndShortDecays);

  TH1D* tNumWithoutSigmaSt = a3dCf->GetWithoutSigmaStNum();
  TH1D* tCfWithoutSigmaSt = a3dCf->GetWithoutSigmaStCf(tMarkerStyleWithoutSigmaSt, tColorWithoutSigmaSt);

  TH1D* tNumSigmaStOnly = a3dCf->GetSigmaStOnlyNum();
  TH1D* tCfSigmaStOnly = a3dCf->GetSigmaStOnlyCf(tMarkerStyleSigmaStOnly, tColorSigmaStOnly);

  TH1D* tNumSecondaryOnly = a3dCf->GetSecondaryOnlyNum();
  TH1D* tCfSecondaryOnly = a3dCf->GetSecondaryOnlyCf(tMarkerStyleSecondaryOnly, tColorSecondaryOnly);

  TH1D* tNumAtLeastOneSecondaryInPair = a3dCf->GetAtLeastOneSecondaryInPairNum();
  TH1D* tCfAtLeastOneSecondaryInPair = a3dCf->GetAtLeastOneSecondaryInPairCf(tMarkerStyleAtLeastOneSecondaryInPair, tColorAtLeastOneSecondaryInPair);

  //---------------------------------------------------------------

  TH1D* tConjNumFull = aConj3dCf->GetFullNum();
  TH1D* tConjCfFull = aConj3dCf->GetFullCf(tMarkerStyleFull, tColorFull);

  TH1D* tConjNumPrimaryOnly = aConj3dCf->GetPrimaryOnlyNum();
  TH1D* tConjCfPrimaryOnly = aConj3dCf->GetPrimaryOnlyCf(tMarkerStylePrimaryOnly, tColorPrimaryOnly);

  TH1D* tConjNumPrimaryAndShortDecays = aConj3dCf->GetPrimaryAndShortDecaysNum();
  TH1D* tConjCfPrimaryAndShortDecays = aConj3dCf->GetPrimaryAndShortDecaysCf(tMarkerStylePrimaryAndShortDecays, tColorPrimaryAndShortDecays);

  TH1D* tConjNumWithoutSigmaSt = aConj3dCf->GetWithoutSigmaStNum();
  TH1D* tConjCfWithoutSigmaSt = aConj3dCf->GetWithoutSigmaStCf(tMarkerStyleWithoutSigmaSt, tColorWithoutSigmaSt);

  TH1D* tConjNumSigmaStOnly = aConj3dCf->GetSigmaStOnlyNum();
  TH1D* tConjCfSigmaStOnly = aConj3dCf->GetSigmaStOnlyCf(tMarkerStyleSigmaStOnly, tColorSigmaStOnly);

  TH1D* tConjNumSecondaryOnly = aConj3dCf->GetSecondaryOnlyNum();
  TH1D* tConjCfSecondaryOnly = aConj3dCf->GetSecondaryOnlyCf(tMarkerStyleSecondaryOnly, tColorSecondaryOnly);

  TH1D* tConjNumAtLeastOneSecondaryInPair = aConj3dCf->GetAtLeastOneSecondaryInPairNum();
  TH1D* tConjCfAtLeastOneSecondaryInPair = aConj3dCf->GetAtLeastOneSecondaryInPairCf(tMarkerStyleAtLeastOneSecondaryInPair, tColorAtLeastOneSecondaryInPair);

  //---------------------------------------------------------------

  TString tCfBaseNameFull = TString::Format("CfFull%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNamePrimaryOnly = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNamePrimaryAndShortDecays = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNameWithoutSigmaSt = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNameSigmaStOnly = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNameSecondaryOnly = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNameAtLeastOneSecondaryInPair = TString::Format("CfAtLeastOneSecondaryInPair%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);

  TString tConjCfBaseNameFull = TString::Format("CfFull%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNamePrimaryOnly = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNamePrimaryAndShortDecays = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameWithoutSigmaSt = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameSigmaStOnly = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameSecondaryOnly = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameAtLeastOneSecondaryInPair = TString::Format("CfAtLeastOneSecondaryInPair%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);

  //---------------------------------------------------------------

  TH1D* tCfFullTot = CombineConjugates(tNumFull, tCfFull, tConjNumFull, tConjCfFull, tCfBaseNameFull+tConjCfBaseNameFull);
  TH1D* tCfPrimaryOnlyTot = CombineConjugates(tNumPrimaryOnly, tCfPrimaryOnly, tConjNumPrimaryOnly, tConjCfPrimaryOnly, tCfBaseNamePrimaryOnly+tConjCfBaseNamePrimaryOnly);
  TH1D* tCfPrimaryAndShortDecaysTot = CombineConjugates(tNumPrimaryAndShortDecays, tCfPrimaryAndShortDecays, tConjNumPrimaryAndShortDecays, tConjCfPrimaryAndShortDecays, tCfBaseNamePrimaryAndShortDecays+tConjCfBaseNamePrimaryAndShortDecays);
  TH1D* tCfWithoutSigmaStTot = CombineConjugates(tNumWithoutSigmaSt, tCfWithoutSigmaSt, tConjNumWithoutSigmaSt, tConjCfWithoutSigmaSt, tCfBaseNameWithoutSigmaSt+tConjCfBaseNameWithoutSigmaSt);
  TH1D* tCfSigmaStOnlyTot = CombineConjugates(tNumSigmaStOnly, tCfSigmaStOnly, tConjNumSigmaStOnly, tConjCfSigmaStOnly, tCfBaseNameSigmaStOnly+tConjCfBaseNameSigmaStOnly);
  TH1D* tCfSecondaryOnlyTot = CombineConjugates(tNumSecondaryOnly, tCfSecondaryOnly, tConjNumSecondaryOnly, tConjCfSecondaryOnly, tCfBaseNameSecondaryOnly+tConjCfBaseNameSecondaryOnly);
  TH1D* tCfAtLeastOneSecondaryInPairTot = CombineConjugates(tNumAtLeastOneSecondaryInPair, tCfAtLeastOneSecondaryInPair, tConjNumAtLeastOneSecondaryInPair, tConjCfAtLeastOneSecondaryInPair, tCfBaseNameAtLeastOneSecondaryInPair+tConjCfBaseNameAtLeastOneSecondaryInPair);

  tCfFullTot->GetXaxis()->SetTitle("k* (GeV/c)");
  tCfFullTot->GetYaxis()->SetTitle("C(k*)");

//  tCfFullTot->GetXaxis()->SetRangeUser(0.,0.329);
  tCfFullTot->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCfFullTot->Draw();
  tCfPrimaryOnlyTot->Draw("same");
  tCfPrimaryAndShortDecaysTot->Draw("same");
  tCfWithoutSigmaStTot->Draw("same");
  tCfSigmaStOnlyTot->Draw("same");
  tCfSecondaryOnlyTot->Draw("same");
  tCfAtLeastOneSecondaryInPairTot->Draw("same");
  tCfFullTot->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s + %s Cfs", cAnalysisRootTags[a3dCf->GetAnalysisType()], cAnalysisRootTags[aConj3dCf->GetAnalysisType()]));

  tLeg->AddEntry(tCfFullTot, "Full");
  tLeg->AddEntry(tCfPrimaryOnlyTot, "Primary Only");
  tLeg->AddEntry(tCfPrimaryAndShortDecaysTot, "Primary and short decays");
  tLeg->AddEntry(tCfWithoutSigmaStTot, "w/o #Sigma*");
  tLeg->AddEntry(tCfAtLeastOneSecondaryInPairTot, "At Least One Secondary");
  tLeg->AddEntry(tCfSecondaryOnlyTot, "Secondary Only");
  tLeg->AddEntry(tCfSigmaStOnlyTot, "#Sigma* Only");

  tLeg->Draw();
}

//________________________________________________________________________________________________________________
void DrawAllSigmaStFlavorsWithConj(TPad* aPad, Therm3dCf* a3dCf, Therm3dCf* aConj3dCf)
{
  aPad->cd();
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  //---------------------------------------------------------------

  int tColorSigmaStOnly = 20;
  int tColorSigmaStPOnly = 20;
  int tColorSigmaStMOnly = 20;
  int tColorSigmaSt0Only = 20;

  int tMarkerStyleSigmaStOnly = 20;
  int tMarkerStyleSigmaStPOnly = 24;
  int tMarkerStyleSigmaStMOnly = 25;
  int tMarkerStyleSigmaSt0Only = 26;

  //---------------------------------------------------------------

  TH1D* tNumSigmaStOnly = a3dCf->GetSigmaStOnlyNum();
  TH1D* tCfSigmaStOnly = a3dCf->GetSigmaStOnlyCf(tMarkerStyleSigmaStOnly, tColorSigmaStOnly);

  TH1D* tNumSigmaStPOnly = a3dCf->GetSigmaStPOnlyNum();
  TH1D* tCfSigmaStPOnly = a3dCf->GetSigmaStPOnlyCf(tMarkerStyleSigmaStPOnly, tColorSigmaStPOnly);

  TH1D* tNumSigmaStMOnly = a3dCf->GetSigmaStMOnlyNum();
  TH1D* tCfSigmaStMOnly = a3dCf->GetSigmaStMOnlyCf(tMarkerStyleSigmaStMOnly, tColorSigmaStMOnly);

  TH1D* tNumSigmaSt0Only = a3dCf->GetSigmaSt0OnlyNum();
  TH1D* tCfSigmaSt0Only = a3dCf->GetSigmaSt0OnlyCf(tMarkerStyleSigmaSt0Only, tColorSigmaSt0Only);

  //---------------------------------------------------------------

  TH1D* tConjNumSigmaStOnly = aConj3dCf->GetSigmaStOnlyNum();
  TH1D* tConjCfSigmaStOnly = aConj3dCf->GetSigmaStOnlyCf(tMarkerStyleSigmaStOnly, tColorSigmaStOnly);

  TH1D* tConjNumSigmaStPOnly = aConj3dCf->GetSigmaStPOnlyNum();
  TH1D* tConjCfSigmaStPOnly = aConj3dCf->GetSigmaStPOnlyCf(tMarkerStyleSigmaStPOnly, tColorSigmaStPOnly);

  TH1D* tConjNumSigmaStMOnly = aConj3dCf->GetSigmaStMOnlyNum();
  TH1D* tConjCfSigmaStMOnly = aConj3dCf->GetSigmaStMOnlyCf(tMarkerStyleSigmaStMOnly, tColorSigmaStMOnly);

  TH1D* tConjNumSigmaSt0Only = aConj3dCf->GetSigmaSt0OnlyNum();
  TH1D* tConjCfSigmaSt0Only = aConj3dCf->GetSigmaSt0OnlyCf(tMarkerStyleSigmaSt0Only, tColorSigmaSt0Only);

  //---------------------------------------------------------------

  TString tCfBaseNameSigmaStOnly = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameSigmaStOnly = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);

  TString tCfBaseNameSigmaStPOnly = TString::Format("CfSigmaStPOnly%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameSigmaStPOnly = TString::Format("CfSigmaStPOnly%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);

  TString tCfBaseNameSigmaStMOnly = TString::Format("CfSigmaStMOnly%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameSigmaStMOnly = TString::Format("CfSigmaStMOnly%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);

  TString tCfBaseNameSigmaSt0Only = TString::Format("CfSigmaSt0Only%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameSigmaSt0Only = TString::Format("CfSigmaSt0Only%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);

  //---------------------------------------------------------------

  TH1D* tCfSigmaStOnlyTot = CombineConjugates(tNumSigmaStOnly, tCfSigmaStOnly, tConjNumSigmaStOnly, tConjCfSigmaStOnly, tCfBaseNameSigmaStOnly+tConjCfBaseNameSigmaStOnly);
  TH1D* tCfSigmaStPOnlyTot = CombineConjugates(tNumSigmaStPOnly, tCfSigmaStPOnly, tConjNumSigmaStPOnly, tConjCfSigmaStPOnly, tCfBaseNameSigmaStPOnly+tConjCfBaseNameSigmaStPOnly);
  TH1D* tCfSigmaStMOnlyTot = CombineConjugates(tNumSigmaStMOnly, tCfSigmaStMOnly, tConjNumSigmaStMOnly, tConjCfSigmaStMOnly, tCfBaseNameSigmaStMOnly+tConjCfBaseNameSigmaStMOnly);
  TH1D* tCfSigmaSt0OnlyTot = CombineConjugates(tNumSigmaSt0Only, tCfSigmaSt0Only, tConjNumSigmaSt0Only, tConjCfSigmaSt0Only, tCfBaseNameSigmaSt0Only+tConjCfBaseNameSigmaSt0Only);

  tCfSigmaStOnlyTot->GetXaxis()->SetTitle("k* (GeV/c)");
  tCfSigmaStOnlyTot->GetYaxis()->SetTitle("C(k*)");

//  tCfSigmaStOnlyTot->GetXaxis()->SetRangeUser(0.,0.329);
  tCfSigmaStOnlyTot->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCfSigmaStOnlyTot->Draw();
  tCfSigmaStPOnlyTot->Draw("same");
  tCfSigmaStMOnlyTot->Draw("same");
  tCfSigmaSt0OnlyTot->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s + %s Cfs", cAnalysisRootTags[a3dCf->GetAnalysisType()], cAnalysisRootTags[aConj3dCf->GetAnalysisType()]));

  tLeg->AddEntry(tCfSigmaStOnlyTot, "#Sigma* Only (Total)");
  tLeg->AddEntry(tCfSigmaStPOnlyTot, "#Sigma*^{+} (#bar{#Sigma*}^{-}) Only");
  tLeg->AddEntry(tCfSigmaStMOnlyTot, "#Sigma*^{-} (#bar{#Sigma*}^{+}) Only");
  tLeg->AddEntry(tCfSigmaSt0OnlyTot, "#Sigma*^{0} (#bar{#Sigma*}^{0}) Only");

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
  bool bCombineConjugates = false;
  bool bSaveFigures = false;

  int tRebin=2;

  TString tFileLocationCfs = "/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b2/CorrelationFunctions_10MixedEvNum";

  TString tFileNameModifier = "";
//  TString tFileNameModifier = "_WeightParentsInteraction";
//  TString tFileNameModifier = "_WeightParentsInteraction_NoCharged";

  tFileLocationCfs += tFileNameModifier;
  tFileLocationCfs += TString(".root");

  TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20171012/Figures/";

  TString tSaveLocationBaseLamKchP = tSaveLocationBase + TString("LamKchP/");
  TString tSaveLocationBaseALamKchP = tSaveLocationBase + TString("ALamKchP/");
  TString tSaveLocationBaseLamKchM = tSaveLocationBase + TString("LamKchM/");
  TString tSaveLocationBaseALamKchM = tSaveLocationBase + TString("ALamKchM/");
  TString tSaveLocationBaseLamK0 = tSaveLocationBase + TString("LamK0/");
  TString tSaveLocationBaseALamK0 = tSaveLocationBase + TString("ALamK0/");

  //--------------------------------------------

  Therm3dCf *t3dCf_LamKchP = new Therm3dCf(kLamKchP, tFileLocationCfs, tRebin);
  Therm3dCf *t3dCf_ALamKchM = new Therm3dCf(kALamKchM, tFileLocationCfs, tRebin);

  Therm3dCf *t3dCf_LamKchM = new Therm3dCf(kLamKchM, tFileLocationCfs, tRebin);
  Therm3dCf *t3dCf_ALamKchP = new Therm3dCf(kALamKchP, tFileLocationCfs, tRebin);

  Therm3dCf *t3dCf_LamK0 = new Therm3dCf(kLamK0, tFileLocationCfs, tRebin);
  Therm3dCf *t3dCf_ALamK0 = new Therm3dCf(kALamK0, tFileLocationCfs, tRebin);

  //--------------------------------------------

  if(!bCombineConjugates)
  {
    int tCommonMarkerStyle = 20;

    TCanvas* tCanLamKchP = new TCanvas("Cfs_LamKchP", "Cfs_LamKchP");
    tCanLamKchP->Divide(2,1);
    t3dCf_LamKchP->DrawAllCfs((TPad*)tCanLamKchP->cd(1), tCommonMarkerStyle);
    t3dCf_ALamKchM->DrawAllCfs((TPad*)tCanLamKchP->cd(2), tCommonMarkerStyle);

    TCanvas* tCanLamKchM = new TCanvas("Cfs_LamKchM", "Cfs_LamKchM");
    tCanLamKchM->Divide(2,1);
    t3dCf_LamKchM->DrawAllCfs((TPad*)tCanLamKchM->cd(1), tCommonMarkerStyle);
    t3dCf_ALamKchP->DrawAllCfs((TPad*)tCanLamKchM->cd(2), tCommonMarkerStyle);

    TCanvas* tCanLamK0 = new TCanvas("Cfs_LamK0", "Cfs_LamK0");
    tCanLamK0->Divide(2,1);
    t3dCf_LamK0->DrawAllCfs((TPad*)tCanLamK0->cd(1), tCommonMarkerStyle);
    t3dCf_ALamK0->DrawAllCfs((TPad*)tCanLamK0->cd(2), tCommonMarkerStyle);

    if(bSaveFigures)
    {
      tCanLamKchP->SaveAs(tSaveLocationBaseLamKchP + TString(tCanLamKchP->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamKchM->SaveAs(tSaveLocationBaseLamKchM + TString(tCanLamKchM->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamK0->SaveAs(tSaveLocationBaseLamK0 + TString(tCanLamK0->GetName()) + tFileNameModifier + TString(".eps"));
    }
    //--------------------------------------------
    TCanvas* tCanLamKchP_SigStFlavors = new TCanvas("Cfs_SigStFlavors_LamKchP", "Cfs_SigStFlavors_LamKchP");
    tCanLamKchP_SigStFlavors->Divide(2,1);
    t3dCf_LamKchP->DrawAllSigmaStFlavors((TPad*)tCanLamKchP_SigStFlavors->cd(1));
    t3dCf_ALamKchM->DrawAllSigmaStFlavors((TPad*)tCanLamKchP_SigStFlavors->cd(2));

    TCanvas* tCanLamKchM_SigStFlavors = new TCanvas("Cfs_SigStFlavors_LamKchM", "Cfs_SigStFlavors_LamKchM");
    tCanLamKchM_SigStFlavors->Divide(2,1);
    t3dCf_LamKchM->DrawAllSigmaStFlavors((TPad*)tCanLamKchM_SigStFlavors->cd(1));
    t3dCf_ALamKchP->DrawAllSigmaStFlavors((TPad*)tCanLamKchM_SigStFlavors->cd(2));

    TCanvas* tCanLamK0_SigStFlavors = new TCanvas("Cfs_SigStFlavors_LamK0", "Cfs_SigStFlavors_LamK0");
    tCanLamK0_SigStFlavors->Divide(2,1);
    t3dCf_LamK0->DrawAllSigmaStFlavors((TPad*)tCanLamK0_SigStFlavors->cd(1));
    t3dCf_ALamK0->DrawAllSigmaStFlavors((TPad*)tCanLamK0_SigStFlavors->cd(2));

    if(bSaveFigures)
    {
      tCanLamKchP_SigStFlavors->SaveAs(tSaveLocationBaseLamKchP + TString(tCanLamKchP_SigStFlavors->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamKchM_SigStFlavors->SaveAs(tSaveLocationBaseLamKchM + TString(tCanLamKchM_SigStFlavors->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamK0_SigStFlavors->SaveAs(tSaveLocationBaseLamK0 + TString(tCanLamK0_SigStFlavors->GetName()) + tFileNameModifier + TString(".eps"));
    }
  }

  //--------------------------------------------
  else
  {
    TCanvas* tCanLamKchPwConj = new TCanvas("Cfs_LamKchPwConj", "Cfs_LamKchPwConj");
    DrawCfsWithConj((TPad*)tCanLamKchPwConj, t3dCf_LamKchP, t3dCf_ALamKchM);
  
    TCanvas* tCanLamKchMwConj = new TCanvas("Cfs_LamKchMwConj", "Cfs_LamKchMwConj");
    DrawCfsWithConj((TPad*)tCanLamKchMwConj, t3dCf_LamKchM, t3dCf_ALamKchP);

    TCanvas* tCanLamK0wConj = new TCanvas("Cfs_LamK0wConj", "Cfs_LamK0wConj");
    DrawCfsWithConj((TPad*)tCanLamK0wConj, t3dCf_LamK0, t3dCf_ALamK0);

    if(bSaveFigures)
    {
      tCanLamKchPwConj->SaveAs(tSaveLocationBaseLamKchP + TString(tCanLamKchPwConj->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamKchMwConj->SaveAs(tSaveLocationBaseLamKchM + TString(tCanLamKchMwConj->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamK0wConj->SaveAs(tSaveLocationBaseLamK0 + TString(tCanLamK0wConj->GetName()) + tFileNameModifier + TString(".eps"));
    }
    //--------------------------------------------
    TCanvas* tCanLamKchPwConj_SigStFlavors = new TCanvas("Cfs_SigStFlavors_LamKchPwConj", "Cfs_SigStFlavors_LamKchPwConj");
    DrawAllSigmaStFlavorsWithConj((TPad*)tCanLamKchPwConj_SigStFlavors, t3dCf_LamKchP, t3dCf_ALamKchM);
  
    TCanvas* tCanLamKchMwConj_SigStFlavors = new TCanvas("Cfs_SigStFlavors_LamKchMwConj", "Cfs_SigStFlavors_LamKchMwConj");
    DrawAllSigmaStFlavorsWithConj((TPad*)tCanLamKchMwConj_SigStFlavors, t3dCf_LamKchM, t3dCf_ALamKchP);

    TCanvas* tCanLamK0wConj_SigStFlavors = new TCanvas("Cfs_SigStFlavors_LamK0wConj", "Cfs_SigStFlavors_LamK0wConj");
    DrawAllSigmaStFlavorsWithConj((TPad*)tCanLamK0wConj_SigStFlavors, t3dCf_LamK0, t3dCf_ALamK0);

    if(bSaveFigures)
    {
      tCanLamKchPwConj_SigStFlavors->SaveAs(tSaveLocationBaseLamKchP + TString(tCanLamKchPwConj_SigStFlavors->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamKchMwConj_SigStFlavors->SaveAs(tSaveLocationBaseLamKchM + TString(tCanLamKchMwConj_SigStFlavors->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamK0wConj_SigStFlavors->SaveAs(tSaveLocationBaseLamK0 + TString(tCanLamK0wConj_SigStFlavors->GetName()) + tFileNameModifier + TString(".eps"));
    }
  }
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
