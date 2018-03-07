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
TH1D* BuildCf(TH1* aNum, TH1* aDen, TString aName, int aRebin=1)
{
  aNum->Rebin(aRebin);
  aDen->Rebin(aRebin);

  double tMinNorm = 0.32;
  double tMaxNorm = 0.40;

  int tMinNormBin = aNum->FindBin(tMinNorm);
  int tMaxNormBin = aNum->FindBin(tMaxNorm);
  double tNumScale = aNum->Integral(tMinNormBin,tMaxNormBin);

  tMinNormBin = aDen->FindBin(tMinNorm);
  tMaxNormBin = aDen->FindBin(tMaxNorm);
  double tDenScale = aDen->Integral(tMinNormBin,tMaxNormBin);

  TH1D* tReturnCf = (TH1D*)aNum->Clone(aName);

  //-----Check to see if Sumw2 has already been called, and if not, call it
  if(!tReturnCf->GetSumw2N()) tReturnCf->Sumw2();

  tReturnCf->Divide(aDen);
  tReturnCf->Scale(tDenScale/tNumScale);
  tReturnCf->SetTitle(aName);

  if(!tReturnCf->GetSumw2N()) {tReturnCf->Sumw2();}

  return tReturnCf;
}

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
void DrawCfs(TPad* aPad, TString aFileName, AnalysisType aAnType)
{
  int aRebin=2;

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

  TString tNumBaseNameFull = TString::Format("NumFull%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNamePrimaryOnly = TString::Format("NumPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNamePrimaryAndShortDecays = TString::Format("NumPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNameWithoutSigmaSt = TString::Format("NumWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNameSigmaStOnly = TString::Format("NumSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNameSecondaryOnly = TString::Format("NumSecondaryOnly%s", cAnalysisBaseTags[aAnType]);

  TString tDenBaseNameFull = TString::Format("DenFull%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNamePrimaryOnly = TString::Format("DenPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNamePrimaryAndShortDecays = TString::Format("DenPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNameWithoutSigmaSt = TString::Format("DenWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNameSigmaStOnly = TString::Format("DenSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNameSecondaryOnly = TString::Format("DenSecondaryOnly%s", cAnalysisBaseTags[aAnType]);

  TString tCfBaseNameFull = TString::Format("CfFull%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNamePrimaryOnly = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNamePrimaryAndShortDecays = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNameWithoutSigmaSt = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNameSigmaStOnly = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNameSecondaryOnly = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[aAnType]);

  //---------------------------------------------------------------

  TH1D* tNumFull = Get1dHisto(aFileName, tNumBaseNameFull);
  TH1D* tDenFull = Get1dHisto(aFileName, tDenBaseNameFull);

  TH1D* tNumPrimaryOnly = Get1dHisto(aFileName, tNumBaseNamePrimaryOnly);
  TH1D* tDenPrimaryOnly = Get1dHisto(aFileName, tDenBaseNamePrimaryOnly);

  TH1D* tNumPrimaryAndShortDecays = Get1dHisto(aFileName, tNumBaseNamePrimaryAndShortDecays);
  TH1D* tDenPrimaryAndShortDecays = Get1dHisto(aFileName, tDenBaseNamePrimaryAndShortDecays);

  TH1D* tNumWithoutSigmaSt = Get1dHisto(aFileName, tNumBaseNameWithoutSigmaSt);
  TH1D* tDenWithoutSigmaSt = Get1dHisto(aFileName, tDenBaseNameWithoutSigmaSt);

  TH1D* tNumSigmaStOnly = Get1dHisto(aFileName, tNumBaseNameSigmaStOnly);
  TH1D* tDenSigmaStOnly = Get1dHisto(aFileName, tDenBaseNameSigmaStOnly);

  TH1D* tNumSecondaryOnly = Get1dHisto(aFileName, tNumBaseNameSecondaryOnly);
  TH1D* tDenSecondaryOnly = Get1dHisto(aFileName, tDenBaseNameSecondaryOnly);

  //---------------------------------------------------------------

  TH1D* tCfFull = BuildCf(tNumFull, tDenFull, tCfBaseNameFull, aRebin);
    tCfFull->SetLineColor(tColorFull);
    tCfFull->SetMarkerColor(tColorFull);
    tCfFull->SetMarkerStyle(tMarkerStyleFull);
  TH1D* tCfPrimaryOnly = BuildCf(tNumPrimaryOnly, tDenPrimaryOnly, tCfBaseNamePrimaryOnly, aRebin);
    tCfPrimaryOnly->SetLineColor(tColorPrimaryOnly);
    tCfPrimaryOnly->SetMarkerColor(tColorPrimaryOnly);
    tCfPrimaryOnly->SetMarkerStyle(tMarkerStylePrimaryOnly);
  TH1D* tCfPrimaryAndShortDecays = BuildCf(tNumPrimaryAndShortDecays, tDenPrimaryAndShortDecays, tCfBaseNamePrimaryAndShortDecays, aRebin);
    tCfPrimaryAndShortDecays->SetLineColor(tColorPrimaryAndShortDecays);
    tCfPrimaryAndShortDecays->SetMarkerColor(tColorPrimaryAndShortDecays);
    tCfPrimaryAndShortDecays->SetMarkerStyle(tMarkerStylePrimaryAndShortDecays);
  TH1D* tCfWithoutSigmaSt = BuildCf(tNumWithoutSigmaSt, tDenWithoutSigmaSt, tCfBaseNameWithoutSigmaSt, aRebin);
    tCfWithoutSigmaSt->SetLineColor(tColorWithoutSigmaSt);
    tCfWithoutSigmaSt->SetMarkerColor(tColorWithoutSigmaSt);
    tCfWithoutSigmaSt->SetMarkerStyle(tMarkerStyleWithoutSigmaSt);
  TH1D* tCfSigmaStOnly = BuildCf(tNumSigmaStOnly, tDenSigmaStOnly, tCfBaseNameSigmaStOnly, aRebin);
    tCfSigmaStOnly->SetLineColor(tColorSigmaStOnly);
    tCfSigmaStOnly->SetMarkerColor(tColorSigmaStOnly);
    tCfSigmaStOnly->SetMarkerStyle(tMarkerStyleSigmaStOnly);
  TH1D* tCfSecondaryOnly = BuildCf(tNumSecondaryOnly, tDenSecondaryOnly, tCfBaseNameSecondaryOnly, aRebin);
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
  tLeg->SetHeader(TString::Format("%s Cfs", cAnalysisRootTags[aAnType]));

  tLeg->AddEntry(tCfFull, "Full");
  tLeg->AddEntry(tCfWithoutSigmaSt, "w/o #Sigma*");
  tLeg->AddEntry(tCfSecondaryOnly, "Secondary Only");
  tLeg->AddEntry(tCfPrimaryOnly, "Primary Only");
  tLeg->AddEntry(tCfPrimaryAndShortDecays, "Primary and short decays");
  tLeg->AddEntry(tCfSigmaStOnly, "#Sigma* Only");

  tLeg->Draw();
}

//________________________________________________________________________________________________________________
void DrawCfsWithConj(TPad* aPad, TString aFileName, AnalysisType aAnType, AnalysisType aConjType)
{
  int aRebin=2;

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

  TString tNumBaseNameFull = TString::Format("NumFull%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNamePrimaryOnly = TString::Format("NumPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNamePrimaryAndShortDecays = TString::Format("NumPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNameWithoutSigmaSt = TString::Format("NumWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNameSigmaStOnly = TString::Format("NumSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TString tNumBaseNameSecondaryOnly = TString::Format("NumSecondaryOnly%s", cAnalysisBaseTags[aAnType]);

  TString tDenBaseNameFull = TString::Format("DenFull%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNamePrimaryOnly = TString::Format("DenPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNamePrimaryAndShortDecays = TString::Format("DenPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNameWithoutSigmaSt = TString::Format("DenWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNameSigmaStOnly = TString::Format("DenSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TString tDenBaseNameSecondaryOnly = TString::Format("DenSecondaryOnly%s", cAnalysisBaseTags[aAnType]);

  TString tCfBaseNameFull = TString::Format("CfFull%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNamePrimaryOnly = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNamePrimaryAndShortDecays = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNameWithoutSigmaSt = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNameSigmaStOnly = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TString tCfBaseNameSecondaryOnly = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[aAnType]);

  //---------

  TH1D* tNumFull = Get1dHisto(aFileName, tNumBaseNameFull);
  TH1D* tDenFull = Get1dHisto(aFileName, tDenBaseNameFull);
  TH1D* tCfFull = BuildCf(tNumFull, tDenFull, tCfBaseNameFull, aRebin);

  TH1D* tNumPrimaryOnly = Get1dHisto(aFileName, tNumBaseNamePrimaryOnly);
  TH1D* tDenPrimaryOnly = Get1dHisto(aFileName, tDenBaseNamePrimaryOnly);
  TH1D* tCfPrimaryOnly = BuildCf(tNumPrimaryOnly, tDenPrimaryOnly, tCfBaseNamePrimaryOnly, aRebin);

  TH1D* tNumPrimaryAndShortDecays = Get1dHisto(aFileName, tNumBaseNamePrimaryAndShortDecays);
  TH1D* tDenPrimaryAndShortDecays = Get1dHisto(aFileName, tDenBaseNamePrimaryAndShortDecays);
  TH1D* tCfPrimaryAndShortDecays = BuildCf(tNumPrimaryAndShortDecays, tDenPrimaryAndShortDecays, tCfBaseNamePrimaryAndShortDecays, aRebin);

  TH1D* tNumWithoutSigmaSt = Get1dHisto(aFileName, tNumBaseNameWithoutSigmaSt);
  TH1D* tDenWithoutSigmaSt = Get1dHisto(aFileName, tDenBaseNameWithoutSigmaSt);
  TH1D* tCfWithoutSigmaSt = BuildCf(tNumWithoutSigmaSt, tDenWithoutSigmaSt, tCfBaseNameWithoutSigmaSt, aRebin);

  TH1D* tNumSigmaStOnly = Get1dHisto(aFileName, tNumBaseNameSigmaStOnly);
  TH1D* tDenSigmaStOnly = Get1dHisto(aFileName, tDenBaseNameSigmaStOnly);
  TH1D* tCfSigmaStOnly = BuildCf(tNumSigmaStOnly, tDenSigmaStOnly, tCfBaseNameSigmaStOnly, aRebin);

  TH1D* tNumSecondaryOnly = Get1dHisto(aFileName, tNumBaseNameSecondaryOnly);
  TH1D* tDenSecondaryOnly = Get1dHisto(aFileName, tDenBaseNameSecondaryOnly);
  TH1D* tCfSecondaryOnly = BuildCf(tNumSecondaryOnly, tDenSecondaryOnly, tCfBaseNameSecondaryOnly, aRebin);


  //---------------------------------------------------------------

  TString tConjNumBaseNameFull = TString::Format("NumFull%s", cAnalysisBaseTags[aAnType]);
  TString tConjNumBaseNamePrimaryOnly = TString::Format("NumPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tConjNumBaseNamePrimaryAndShortDecays = TString::Format("NumPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tConjNumBaseNameWithoutSigmaSt = TString::Format("NumWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tConjNumBaseNameSigmaStOnly = TString::Format("NumSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TString tConjNumBaseNameSecondaryOnly = TString::Format("NumSecondaryOnly%s", cAnalysisBaseTags[aAnType]);

  TString tConjDenBaseNameFull = TString::Format("DenFull%s", cAnalysisBaseTags[aAnType]);
  TString tConjDenBaseNamePrimaryOnly = TString::Format("DenPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tConjDenBaseNamePrimaryAndShortDecays = TString::Format("DenPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tConjDenBaseNameWithoutSigmaSt = TString::Format("DenWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tConjDenBaseNameSigmaStOnly = TString::Format("DenSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TString tConjDenBaseNameSecondaryOnly = TString::Format("DenSecondaryOnly%s", cAnalysisBaseTags[aAnType]);

  TString tConjCfBaseNameFull = TString::Format("CfFull%s", cAnalysisBaseTags[aAnType]);
  TString tConjCfBaseNamePrimaryOnly = TString::Format("CfPrimaryOnly%s", cAnalysisBaseTags[aAnType]);
  TString tConjCfBaseNamePrimaryAndShortDecays = TString::Format("CfPrimaryAndShortDecays%s", cAnalysisBaseTags[aAnType]);
  TString tConjCfBaseNameWithoutSigmaSt = TString::Format("CfWithoutSigmaSt%s", cAnalysisBaseTags[aAnType]);
  TString tConjCfBaseNameSigmaStOnly = TString::Format("CfSigmaStOnly%s", cAnalysisBaseTags[aAnType]);
  TString tConjCfBaseNameSecondaryOnly = TString::Format("CfSecondaryOnly%s", cAnalysisBaseTags[aAnType]);

  //---------

  TH1D* tConjNumFull = Get1dHisto(aFileName, tConjNumBaseNameFull);
  TH1D* tConjDenFull = Get1dHisto(aFileName, tConjDenBaseNameFull);
  TH1D* tConjCfFull = BuildCf(tConjNumFull, tConjDenFull, tConjCfBaseNameFull, aRebin);

  TH1D* tConjNumPrimaryOnly = Get1dHisto(aFileName, tConjNumBaseNamePrimaryOnly);
  TH1D* tConjDenPrimaryOnly = Get1dHisto(aFileName, tConjDenBaseNamePrimaryOnly);
  TH1D* tConjCfPrimaryOnly = BuildCf(tConjNumPrimaryOnly, tConjDenPrimaryOnly, tConjCfBaseNamePrimaryOnly, aRebin);

  TH1D* tConjNumPrimaryAndShortDecays = Get1dHisto(aFileName, tConjNumBaseNamePrimaryAndShortDecays);
  TH1D* tConjDenPrimaryAndShortDecays = Get1dHisto(aFileName, tConjDenBaseNamePrimaryAndShortDecays);
  TH1D* tConjCfPrimaryAndShortDecays = BuildCf(tConjNumPrimaryAndShortDecays, tConjDenPrimaryAndShortDecays, tConjCfBaseNamePrimaryAndShortDecays, aRebin);

  TH1D* tConjNumWithoutSigmaSt = Get1dHisto(aFileName, tConjNumBaseNameWithoutSigmaSt);
  TH1D* tConjDenWithoutSigmaSt = Get1dHisto(aFileName, tConjDenBaseNameWithoutSigmaSt);
  TH1D* tConjCfWithoutSigmaSt = BuildCf(tConjNumWithoutSigmaSt, tConjDenWithoutSigmaSt, tConjCfBaseNameWithoutSigmaSt, aRebin);

  TH1D* tConjNumSigmaStOnly = Get1dHisto(aFileName, tConjNumBaseNameSigmaStOnly);
  TH1D* tConjDenSigmaStOnly = Get1dHisto(aFileName, tConjDenBaseNameSigmaStOnly);
  TH1D* tConjCfSigmaStOnly = BuildCf(tConjNumSigmaStOnly, tConjDenSigmaStOnly, tConjCfBaseNameSigmaStOnly, aRebin);

  TH1D* tConjNumSecondaryOnly = Get1dHisto(aFileName, tConjNumBaseNameSecondaryOnly);
  TH1D* tConjDenSecondaryOnly = Get1dHisto(aFileName, tConjDenBaseNameSecondaryOnly);
  TH1D* tConjCfSecondaryOnly = BuildCf(tConjNumSecondaryOnly, tConjDenSecondaryOnly, tConjCfBaseNameSecondaryOnly, aRebin);


  //---------------------------------------------------------------

  TH1D* tCfFullTot = CombineConjugates(tNumFull, tCfFull, tConjNumFull, tConjCfFull, tNumBaseNameFull+tConjNumBaseNameFull);
    tCfFullTot->SetLineColor(tColorFull);
    tCfFullTot->SetMarkerColor(tColorFull);
    tCfFullTot->SetMarkerStyle(tMarkerStyleFull);
  TH1D* tCfPrimaryOnlyTot = CombineConjugates(tNumPrimaryOnly, tCfPrimaryOnly, tConjNumPrimaryOnly, tConjCfPrimaryOnly, tNumBaseNamePrimaryOnly+tConjNumBaseNamePrimaryOnly);
    tCfPrimaryOnlyTot->SetLineColor(tColorPrimaryOnly);
    tCfPrimaryOnlyTot->SetMarkerColor(tColorPrimaryOnly);
    tCfPrimaryOnlyTot->SetMarkerStyle(tMarkerStylePrimaryOnly);
  TH1D* tCfPrimaryAndShortDecaysTot = CombineConjugates(tNumPrimaryAndShortDecays, tCfPrimaryAndShortDecays, tConjNumPrimaryAndShortDecays, tConjCfPrimaryAndShortDecays, tNumBaseNamePrimaryAndShortDecays+tConjNumBaseNamePrimaryAndShortDecays);
    tCfPrimaryAndShortDecaysTot->SetLineColor(tColorPrimaryAndShortDecays);
    tCfPrimaryAndShortDecaysTot->SetMarkerColor(tColorPrimaryAndShortDecays);
    tCfPrimaryAndShortDecaysTot->SetMarkerStyle(tMarkerStylePrimaryAndShortDecays);
  TH1D* tCfWithoutSigmaStTot = CombineConjugates(tNumWithoutSigmaSt, tCfWithoutSigmaSt, tConjNumWithoutSigmaSt, tConjCfWithoutSigmaSt, tNumBaseNameWithoutSigmaSt+tConjNumBaseNameWithoutSigmaSt);
    tCfWithoutSigmaStTot->SetLineColor(tColorWithoutSigmaSt);
    tCfWithoutSigmaStTot->SetMarkerColor(tColorWithoutSigmaSt);
    tCfWithoutSigmaStTot->SetMarkerStyle(tMarkerStyleWithoutSigmaSt);
  TH1D* tCfSigmaStOnlyTot = CombineConjugates(tNumSigmaStOnly, tCfSigmaStOnly, tConjNumSigmaStOnly, tConjCfSigmaStOnly, tNumBaseNameSigmaStOnly+tConjNumBaseNameSigmaStOnly);
    tCfSigmaStOnlyTot->SetLineColor(tColorSigmaStOnly);
    tCfSigmaStOnlyTot->SetMarkerColor(tColorSigmaStOnly);
    tCfSigmaStOnlyTot->SetMarkerStyle(tMarkerStyleSigmaStOnly);
  TH1D* tCfSecondaryOnlyTot = CombineConjugates(tNumSecondaryOnly, tCfSecondaryOnly, tConjNumSecondaryOnly, tConjCfSecondaryOnly, tNumBaseNameSecondaryOnly+tConjNumBaseNameSecondaryOnly);
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
  tLeg->SetHeader(TString::Format("%s Cfs", cAnalysisRootTags[aAnType]));

  tLeg->AddEntry(tCfFullTot, "Full");
  tLeg->AddEntry(tCfWithoutSigmaStTot, "w/o #Sigma*");
  tLeg->AddEntry(tCfSecondaryOnlyTot, "Secondary Only");
  tLeg->AddEntry(tCfPrimaryOnlyTot, "Primary Only");
  tLeg->AddEntry(tCfPrimaryAndShortDecaysTot, "Primary and short decays");
  tLeg->AddEntry(tCfSigmaStOnlyTot, "#Sigma* Only");

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
  bool bCombineConjugates = true;
  bool bSaveFigures = false;

  int tImpactParam = 2;

  TString tFileName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/CorrelationFunctions_10MixedEvNum.root", tImpactParam);
//  TString tFileName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/CorrelationFunctions_10MixedEvNum_WeightParentsInteraction.root", tImpactParam);
//  TString tFileName = TString::Format("/home/jesse/Analysis/ReducedTherminator2Events/lhyqid3v_LHCPbPb_2760_b%d/CorrelationFunctions_10MixedEvNum_WeightParentsInteraction_NoCharged.root", tImpactParam);

  TString tSaveLocationBase = "/home/jesse/Analysis/Presentations/GroupMeetings/20170928/Figures/";

  TString tSaveLocationBaseLamKchP = tSaveLocationBase + TString("LamKchP/");
  TString tSaveLocationBaseALamKchP = tSaveLocationBase + TString("ALamKchP/");
  TString tSaveLocationBaseLamKchM = tSaveLocationBase + TString("LamKchM/");
  TString tSaveLocationBaseALamKchM = tSaveLocationBase + TString("ALamKchM/");
  TString tSaveLocationBaseLamK0 = tSaveLocationBase + TString("LamK0/");
  TString tSaveLocationBaseALamK0 = tSaveLocationBase + TString("ALamK0/");

  //--------------------------------------------

  TCanvas* tCanLamKchP = new TCanvas("Cfs_LamKchP", "Cfs_LamKchP");
  tCanLamKchP->Divide(2,1);
  DrawCfs((TPad*)tCanLamKchP->cd(1), tFileName, kLamKchP);
  DrawCfs((TPad*)tCanLamKchP->cd(2), tFileName, kALamKchM);

  TCanvas* tCanLamKchM = new TCanvas("Cfs_LamKchM", "Cfs_LamKchM");
  tCanLamKchM->Divide(2,1);
  DrawCfs((TPad*)tCanLamKchM->cd(1), tFileName, kLamKchM);
  DrawCfs((TPad*)tCanLamKchM->cd(2), tFileName, kALamKchP);

  TCanvas* tCanLamK0 = new TCanvas("Cfs_LamK0", "Cfs_LamK0");
  tCanLamK0->Divide(2,1);
  DrawCfs((TPad*)tCanLamK0->cd(1), tFileName, kLamK0);
  DrawCfs((TPad*)tCanLamK0->cd(2), tFileName, kALamK0);

  if(bSaveFigures)
  {
    tCanLamKchP->SaveAs(tSaveLocationBaseLamKchP + TString(tCanLamKchP->GetName()) + TString(".eps"));
    tCanLamKchM->SaveAs(tSaveLocationBaseLamKchM + TString(tCanLamKchM->GetName()) + TString(".eps"));
    tCanLamK0->SaveAs(tSaveLocationBaseLamK0 + TString(tCanLamK0->GetName()) + TString(".eps"));
  }

  //--------------------------------------------
  TCanvas* tCanLamKchPwConj = new TCanvas("tCanLamKchPwConj", "tCanLamKchPwConj");
  DrawCfsWithConj((TPad*)tCanLamKchPwConj, tFileName, kLamKchP, kALamKchM);

  TCanvas* tCanLamKchMwConj = new TCanvas("tCanLamKchMwConj", "tCanLamKchMwConj");
  DrawCfsWithConj((TPad*)tCanLamKchMwConj, tFileName, kLamKchM, kALamKchP);

  TCanvas* tCanLamK0wConj = new TCanvas("tCanLamK0wConj", "tCanLamK0wConj");
  DrawCfsWithConj((TPad*)tCanLamK0wConj, tFileName, kLamK0, kALamK0);
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
