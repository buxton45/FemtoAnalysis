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
void DrawCfs(TPad* aPad, Therm3dCf* a3dCf)
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

  int tMarkerStyleFull = 20;
  int tMarkerStylePrimaryOnly = 20;
  int tMarkerStylePrimaryAndShortDecays = 20;
  int tMarkerStyleWithoutSigmaSt = 20;
  int tMarkerStyleSigmaStOnly = 20;
  int tMarkerStyleSecondaryOnly = 20;

  //---------------------------------------------------------------

  TH1D* tCfFull = a3dCf->GetFullCf();
    tCfFull->SetLineColor(tColorFull);
    tCfFull->SetMarkerColor(tColorFull);
    tCfFull->SetMarkerStyle(tMarkerStyleFull);
  TH1D* tCfPrimaryOnly = a3dCf->GetPrimaryOnlyCf();
    tCfPrimaryOnly->SetLineColor(tColorPrimaryOnly);
    tCfPrimaryOnly->SetMarkerColor(tColorPrimaryOnly);
    tCfPrimaryOnly->SetMarkerStyle(tMarkerStylePrimaryOnly);
  TH1D* tCfPrimaryAndShortDecays = a3dCf->GetPrimaryAndShortDecaysCf();
    tCfPrimaryAndShortDecays->SetLineColor(tColorPrimaryAndShortDecays);
    tCfPrimaryAndShortDecays->SetMarkerColor(tColorPrimaryAndShortDecays);
    tCfPrimaryAndShortDecays->SetMarkerStyle(tMarkerStylePrimaryAndShortDecays);
  TH1D* tCfWithoutSigmaSt = a3dCf->GetWithoutSigmaStCf();
    tCfWithoutSigmaSt->SetLineColor(tColorWithoutSigmaSt);
    tCfWithoutSigmaSt->SetMarkerColor(tColorWithoutSigmaSt);
    tCfWithoutSigmaSt->SetMarkerStyle(tMarkerStyleWithoutSigmaSt);
  TH1D* tCfSigmaStOnly = a3dCf->GetSigmaStOnlyCf();
    tCfSigmaStOnly->SetLineColor(tColorSigmaStOnly);
    tCfSigmaStOnly->SetMarkerColor(tColorSigmaStOnly);
    tCfSigmaStOnly->SetMarkerStyle(tMarkerStyleSigmaStOnly);
  TH1D* tCfSecondaryOnly = a3dCf->GetSecondaryOnlyCf();
    tCfSecondaryOnly->SetLineColor(tColorSecondaryOnly);
    tCfSecondaryOnly->SetMarkerColor(tColorSecondaryOnly);
    tCfSecondaryOnly->SetMarkerStyle(tMarkerStyleSecondaryOnly);

  tCfFull->GetXaxis()->SetTitle("k* (GeV/c)");
  tCfFull->GetYaxis()->SetTitle("C(k*)");

//  tCfFull->GetXaxis()->SetRangeUser(0.,0.329);
  tCfFull->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCfFull->Draw();
  tCfPrimaryOnly->Draw("same");
  tCfPrimaryAndShortDecays->Draw("same");
  tCfWithoutSigmaSt->Draw("same");
  tCfSigmaStOnly->Draw("same");
  tCfSecondaryOnly->Draw("same");
  tCfFull->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s Cfs", cAnalysisRootTags[a3dCf->GetAnalysisType()]));

  tLeg->AddEntry(tCfFull, "Full");
  tLeg->AddEntry(tCfWithoutSigmaSt, "w/o #Sigma*");
  tLeg->AddEntry(tCfSecondaryOnly, "#Sigma* Only");
  tLeg->AddEntry(tCfPrimaryOnly, "Primary Only");
  tLeg->AddEntry(tCfPrimaryAndShortDecays, "Primary and short decays");
  tLeg->AddEntry(tCfSigmaStOnly, "#Sigma* Only");

  tLeg->Draw();
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

  int tMarkerStyleFull = 20;
  int tMarkerStylePrimaryOnly = 20;
  int tMarkerStylePrimaryAndShortDecays = 20;
  int tMarkerStyleWithoutSigmaSt = 20;
  int tMarkerStyleSigmaStOnly = 20;
  int tMarkerStyleSecondaryOnly = 20;

  //---------------------------------------------------------------

  TH1D* tNumFull = a3dCf->GetFullNum();
  TH1D* tCfFull = a3dCf->GetFullCf();

  TH1D* tNumPrimaryOnly = a3dCf->GetPrimaryOnlyNum();
  TH1D* tCfPrimaryOnly = a3dCf->GetPrimaryOnlyCf();

  TH1D* tNumPrimaryAndShortDecays = a3dCf->GetPrimaryAndShortDecaysNum();
  TH1D* tCfPrimaryAndShortDecays = a3dCf->GetPrimaryAndShortDecaysCf();

  TH1D* tNumWithoutSigmaSt = a3dCf->GetWithoutSigmaStNum();
  TH1D* tCfWithoutSigmaSt = a3dCf->GetWithoutSigmaStCf();

  TH1D* tNumSigmaStOnly = a3dCf->GetSigmaStOnlyNum();
  TH1D* tCfSigmaStOnly = a3dCf->GetSigmaStOnlyCf();

  TH1D* tNumSecondaryOnly = a3dCf->GetSecondaryOnlyNum();
  TH1D* tCfSecondaryOnly = a3dCf->GetSecondaryOnlyCf();


  //---------------------------------------------------------------

  TH1D* tConjNumFull = aConj3dCf->GetFullNum();
  TH1D* tConjCfFull = aConj3dCf->GetFullCf();

  TH1D* tConjNumPrimaryOnly = aConj3dCf->GetPrimaryOnlyNum();
  TH1D* tConjCfPrimaryOnly = aConj3dCf->GetPrimaryOnlyCf();

  TH1D* tConjNumPrimaryAndShortDecays = aConj3dCf->GetPrimaryAndShortDecaysNum();
  TH1D* tConjCfPrimaryAndShortDecays = aConj3dCf->GetPrimaryAndShortDecaysCf();

  TH1D* tConjNumWithoutSigmaSt = aConj3dCf->GetWithoutSigmaStNum();
  TH1D* tConjCfWithoutSigmaSt = aConj3dCf->GetWithoutSigmaStCf();

  TH1D* tConjNumSigmaStOnly = aConj3dCf->GetSigmaStOnlyNum();
  TH1D* tConjCfSigmaStOnly = aConj3dCf->GetSigmaStOnlyCf();

  TH1D* tConjNumSecondaryOnly = aConj3dCf->GetSecondaryOnlyNum();
  TH1D* tConjCfSecondaryOnly = aConj3dCf->GetSecondaryOnlyCf();

  //---------------------------------------------------------------

  TString tCfBaseNameFull = TString::Format("CfFull%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNamePrimaryOnly = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNamePrimaryAndShortDecays = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNameWithoutSigmaSt = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNameSigmaStOnly = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);
  TString tCfBaseNameSecondaryOnly = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[a3dCf->GetAnalysisType()]);

  TString tConjCfBaseNameFull = TString::Format("CfFull%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNamePrimaryOnly = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNamePrimaryAndShortDecays = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameWithoutSigmaSt = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameSigmaStOnly = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);
  TString tConjCfBaseNameSecondaryOnly = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[aConj3dCf->GetAnalysisType()]);


  //---------------------------------------------------------------

  TH1D* tCfFullTot = CombineConjugates(tNumFull, tCfFull, tConjNumFull, tConjCfFull, tCfBaseNameFull+tConjCfBaseNameFull);
    tCfFullTot->SetLineColor(tColorFull);
    tCfFullTot->SetMarkerColor(tColorFull);
    tCfFullTot->SetMarkerStyle(tMarkerStyleFull);
  TH1D* tCfPrimaryOnlyTot = CombineConjugates(tNumPrimaryOnly, tCfPrimaryOnly, tConjNumPrimaryOnly, tConjCfPrimaryOnly, tCfBaseNamePrimaryOnly+tConjCfBaseNamePrimaryOnly);
    tCfPrimaryOnlyTot->SetLineColor(tColorPrimaryOnly);
    tCfPrimaryOnlyTot->SetMarkerColor(tColorPrimaryOnly);
    tCfPrimaryOnlyTot->SetMarkerStyle(tMarkerStylePrimaryOnly);
  TH1D* tCfPrimaryAndShortDecaysTot = CombineConjugates(tNumPrimaryAndShortDecays, tCfPrimaryAndShortDecays, tConjNumPrimaryAndShortDecays, tConjCfPrimaryAndShortDecays, tCfBaseNamePrimaryAndShortDecays+tConjCfBaseNamePrimaryAndShortDecays);
    tCfPrimaryAndShortDecaysTot->SetLineColor(tColorPrimaryAndShortDecays);
    tCfPrimaryAndShortDecaysTot->SetMarkerColor(tColorPrimaryAndShortDecays);
    tCfPrimaryAndShortDecaysTot->SetMarkerStyle(tMarkerStylePrimaryAndShortDecays);
  TH1D* tCfWithoutSigmaStTot = CombineConjugates(tNumWithoutSigmaSt, tCfWithoutSigmaSt, tConjNumWithoutSigmaSt, tConjCfWithoutSigmaSt, tCfBaseNameWithoutSigmaSt+tConjCfBaseNameWithoutSigmaSt);
    tCfWithoutSigmaStTot->SetLineColor(tColorWithoutSigmaSt);
    tCfWithoutSigmaStTot->SetMarkerColor(tColorWithoutSigmaSt);
    tCfWithoutSigmaStTot->SetMarkerStyle(tMarkerStyleWithoutSigmaSt);
  TH1D* tCfSigmaStOnlyTot = CombineConjugates(tNumSigmaStOnly, tCfSigmaStOnly, tConjNumSigmaStOnly, tConjCfSigmaStOnly, tCfBaseNameSigmaStOnly+tConjCfBaseNameSigmaStOnly);
    tCfSigmaStOnlyTot->SetLineColor(tColorSigmaStOnly);
    tCfSigmaStOnlyTot->SetMarkerColor(tColorSigmaStOnly);
    tCfSigmaStOnlyTot->SetMarkerStyle(tMarkerStyleSigmaStOnly);
  TH1D* tCfSecondaryOnlyTot = CombineConjugates(tNumSecondaryOnly, tCfSecondaryOnly, tConjNumSecondaryOnly, tConjCfSecondaryOnly, tCfBaseNameSecondaryOnly+tConjCfBaseNameSecondaryOnly);
    tCfSecondaryOnlyTot->SetLineColor(tColorSecondaryOnly);
    tCfSecondaryOnlyTot->SetMarkerColor(tColorSecondaryOnly);
    tCfSecondaryOnlyTot->SetMarkerStyle(tMarkerStyleSecondaryOnly);

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
  tCfFullTot->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s + %s Cfs", cAnalysisRootTags[a3dCf->GetAnalysisType()], cAnalysisRootTags[aConj3dCf->GetAnalysisType()]));

  tLeg->AddEntry(tCfFullTot, "Full");
  tLeg->AddEntry(tCfWithoutSigmaStTot, "w/o #Sigma*");
  tLeg->AddEntry(tCfSecondaryOnlyTot, "Secondary Only");
  tLeg->AddEntry(tCfPrimaryOnlyTot, "Primary Only");
  tLeg->AddEntry(tCfPrimaryAndShortDecaysTot, "Primary and short decays");
  tLeg->AddEntry(tCfSigmaStOnlyTot, "#Sigma* Only");

  tLeg->Draw();
}

//________________________________________________________________________________________________________________
void DrawAllSigmaStFlavors(TPad* aPad, Therm3dCf* a3dCf)
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

  TH1D* tCfSigmaStOnly = a3dCf->GetSigmaStOnlyCf();
    tCfSigmaStOnly->SetLineColor(tColorSigmaStOnly);
    tCfSigmaStOnly->SetMarkerColor(tColorSigmaStOnly);
    tCfSigmaStOnly->SetMarkerStyle(tMarkerStyleSigmaStOnly);

  TH1D* tCfSigmaStPOnly = a3dCf->GetSigmaStPOnlyCf();
    tCfSigmaStPOnly->SetLineColor(tColorSigmaStPOnly);
    tCfSigmaStPOnly->SetMarkerColor(tColorSigmaStPOnly);
    tCfSigmaStPOnly->SetMarkerStyle(tMarkerStyleSigmaStPOnly);

  TH1D* tCfSigmaStMOnly = a3dCf->GetSigmaStMOnlyCf();
    tCfSigmaStMOnly->SetLineColor(tColorSigmaStMOnly);
    tCfSigmaStMOnly->SetMarkerColor(tColorSigmaStMOnly);
    tCfSigmaStMOnly->SetMarkerStyle(tMarkerStyleSigmaStMOnly);

  TH1D* tCfSigmaSt0Only = a3dCf->GetSigmaSt0OnlyCf();
    tCfSigmaSt0Only->SetLineColor(tColorSigmaSt0Only);
    tCfSigmaSt0Only->SetMarkerColor(tColorSigmaSt0Only);
    tCfSigmaSt0Only->SetMarkerStyle(tMarkerStyleSigmaSt0Only);

  tCfSigmaStOnly->GetXaxis()->SetTitle("k* (GeV/c)");
  tCfSigmaStOnly->GetYaxis()->SetTitle("C(k*)");

//  tCfSigmaStOnly->GetXaxis()->SetRangeUser(0.,0.329);
  tCfSigmaStOnly->GetYaxis()->SetRangeUser(0.86, 1.07);

  tCfSigmaStOnly->Draw();
  tCfSigmaStPOnly->Draw("same");
  tCfSigmaStMOnly->Draw("same");
  tCfSigmaSt0Only->Draw("same");

  TLegend* tLeg = new TLegend(0.60, 0.15, 0.85, 0.40);
    tLeg->SetFillColor(0);
    tLeg->SetBorderSize(0);
    tLeg->SetTextAlign(22);
  tLeg->SetHeader(TString::Format("%s Cfs", cAnalysisRootTags[a3dCf->GetAnalysisType()]));

  tLeg->AddEntry(tCfSigmaStOnly, "#Sigma* Only (Total)");
  tLeg->AddEntry(tCfSigmaStPOnly, "#Sigma*^{+} (#bar{#Sigma*}^{-}) Only");
  tLeg->AddEntry(tCfSigmaStMOnly, "#Sigma*^{-} (#bar{#Sigma*}^{+}) Only");
  tLeg->AddEntry(tCfSigmaSt0Only, "#Sigma*^{0} (#bar{#Sigma*}^{0}) Only");

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
  TH1D* tCfSigmaStOnly = a3dCf->GetSigmaStOnlyCf();

  TH1D* tNumSigmaStPOnly = a3dCf->GetSigmaStPOnlyNum();
  TH1D* tCfSigmaStPOnly = a3dCf->GetSigmaStPOnlyCf();

  TH1D* tNumSigmaStMOnly = a3dCf->GetSigmaStMOnlyNum();
  TH1D* tCfSigmaStMOnly = a3dCf->GetSigmaStMOnlyCf();

  TH1D* tNumSigmaSt0Only = a3dCf->GetSigmaSt0OnlyNum();
  TH1D* tCfSigmaSt0Only = a3dCf->GetSigmaSt0OnlyCf();

  //---------------------------------------------------------------

  TH1D* tConjNumSigmaStOnly = aConj3dCf->GetSigmaStOnlyNum();
  TH1D* tConjCfSigmaStOnly = aConj3dCf->GetSigmaStOnlyCf();

  TH1D* tConjNumSigmaStPOnly = aConj3dCf->GetSigmaStPOnlyNum();
  TH1D* tConjCfSigmaStPOnly = aConj3dCf->GetSigmaStPOnlyCf();

  TH1D* tConjNumSigmaStMOnly = aConj3dCf->GetSigmaStMOnlyNum();
  TH1D* tConjCfSigmaStMOnly = aConj3dCf->GetSigmaStMOnlyCf();

  TH1D* tConjNumSigmaSt0Only = aConj3dCf->GetSigmaSt0OnlyNum();
  TH1D* tConjCfSigmaSt0Only = aConj3dCf->GetSigmaSt0OnlyCf();

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
    tCfSigmaStOnlyTot->SetLineColor(tColorSigmaStOnly);
    tCfSigmaStOnlyTot->SetMarkerColor(tColorSigmaStOnly);
    tCfSigmaStOnlyTot->SetMarkerStyle(tMarkerStyleSigmaStOnly);

  TH1D* tCfSigmaStPOnlyTot = CombineConjugates(tNumSigmaStPOnly, tCfSigmaStPOnly, tConjNumSigmaStPOnly, tConjCfSigmaStPOnly, tCfBaseNameSigmaStPOnly+tConjCfBaseNameSigmaStPOnly);
    tCfSigmaStPOnlyTot->SetLineColor(tColorSigmaStPOnly);
    tCfSigmaStPOnlyTot->SetMarkerColor(tColorSigmaStPOnly);
    tCfSigmaStPOnlyTot->SetMarkerStyle(tMarkerStyleSigmaStPOnly);

  TH1D* tCfSigmaStMOnlyTot = CombineConjugates(tNumSigmaStMOnly, tCfSigmaStMOnly, tConjNumSigmaStMOnly, tConjCfSigmaStMOnly, tCfBaseNameSigmaStMOnly+tConjCfBaseNameSigmaStMOnly);
    tCfSigmaStMOnlyTot->SetLineColor(tColorSigmaStMOnly);
    tCfSigmaStMOnlyTot->SetMarkerColor(tColorSigmaStMOnly);
    tCfSigmaStMOnlyTot->SetMarkerStyle(tMarkerStyleSigmaStMOnly);

  TH1D* tCfSigmaSt0OnlyTot = CombineConjugates(tNumSigmaSt0Only, tCfSigmaSt0Only, tConjNumSigmaSt0Only, tConjCfSigmaSt0Only, tCfBaseNameSigmaSt0Only+tConjCfBaseNameSigmaSt0Only);
    tCfSigmaSt0OnlyTot->SetLineColor(tColorSigmaSt0Only);
    tCfSigmaSt0OnlyTot->SetMarkerColor(tColorSigmaSt0Only);
    tCfSigmaSt0OnlyTot->SetMarkerStyle(tMarkerStyleSigmaSt0Only);

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
  bool bCombineConjugates = true;
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
    TCanvas* tCanLamKchP = new TCanvas("Cfs_LamKchP", "Cfs_LamKchP");
    tCanLamKchP->Divide(2,1);
    DrawCfs((TPad*)tCanLamKchP->cd(1), t3dCf_LamKchP);
    DrawCfs((TPad*)tCanLamKchP->cd(2), t3dCf_ALamKchM);

    TCanvas* tCanLamKchM = new TCanvas("Cfs_LamKchM", "Cfs_LamKchM");
    tCanLamKchM->Divide(2,1);
    DrawCfs((TPad*)tCanLamKchM->cd(1), t3dCf_LamKchM);
    DrawCfs((TPad*)tCanLamKchM->cd(2), t3dCf_ALamKchP);

    TCanvas* tCanLamK0 = new TCanvas("Cfs_LamK0", "Cfs_LamK0");
    tCanLamK0->Divide(2,1);
    DrawCfs((TPad*)tCanLamK0->cd(1), t3dCf_LamK0);
    DrawCfs((TPad*)tCanLamK0->cd(2), t3dCf_ALamK0);

    if(bSaveFigures)
    {
      tCanLamKchP->SaveAs(tSaveLocationBaseLamKchP + TString(tCanLamKchP->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamKchM->SaveAs(tSaveLocationBaseLamKchM + TString(tCanLamKchM->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamK0->SaveAs(tSaveLocationBaseLamK0 + TString(tCanLamK0->GetName()) + tFileNameModifier + TString(".eps"));
    }
  }

  //--------------------------------------------
  else
  {
    TCanvas* tCanLamKchPwConj = new TCanvas("tCanLamKchPwConj", "tCanLamKchPwConj");
    DrawCfsWithConj((TPad*)tCanLamKchPwConj, t3dCf_LamKchP, t3dCf_ALamKchM);
  
    TCanvas* tCanLamKchMwConj = new TCanvas("tCanLamKchMwConj", "tCanLamKchMwConj");
    DrawCfsWithConj((TPad*)tCanLamKchMwConj, t3dCf_LamKchM, t3dCf_ALamKchP);

    TCanvas* tCanLamK0wConj = new TCanvas("tCanLamK0wConj", "tCanLamK0wConj");
    DrawCfsWithConj((TPad*)tCanLamK0wConj, t3dCf_LamK0, t3dCf_ALamK0);

    if(bSaveFigures)
    {
      tCanLamKchPwConj->SaveAs(tSaveLocationBaseLamKchP + TString(tCanLamKchPwConj->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamKchMwConj->SaveAs(tSaveLocationBaseLamKchM + TString(tCanLamKchMwConj->GetName()) + tFileNameModifier + TString(".eps"));
      tCanLamK0wConj->SaveAs(tSaveLocationBaseLamK0 + TString(tCanLamK0wConj->GetName()) + tFileNameModifier + TString(".eps"));
    }
  }
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
