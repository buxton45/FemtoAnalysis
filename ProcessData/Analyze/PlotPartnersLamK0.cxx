///////////////////////////////////////////////////////////////////////////
// PlotPartnersLamK0:                                                   //
///////////////////////////////////////////////////////////////////////////


#include "PlotPartnersLamK0.h"

#ifdef __ROOT__
ClassImp(PlotPartnersLamK0)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

//________________________________________________________________________________________________________________
PlotPartnersLamK0::PlotPartnersLamK0(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, TString aDirNameModifier) :
  PlotPartners(aFileLocationBase,aAnalysisType,aCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier)

{

}

//________________________________________________________________________________________________________________
PlotPartnersLamK0::PlotPartnersLamK0(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, TString aDirNameModifier) :
  PlotPartners(aFileLocationBase,aFileLocationBaseMC,aAnalysisType,aCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier)

{

}



//________________________________________________________________________________________________________________
PlotPartnersLamK0::~PlotPartnersLamK0()
{
}


//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType)
{
  TH1* tHistToDraw;

  switch(aAnalysisType) {
  case kLamK0:
    tHistToDraw = fAnalysis1->GetMassAssumingK0ShortHypothesis();
    break;

  case kALamK0:
    tHistToDraw = fConjAnalysis1->GetMassAssumingK0ShortHypothesis();
    break;

  default:
    cout << "ERROR: PlotPartnersLamK0::DrawMassAssumingK0ShortHypothesis: Invalid aAnalysisType = " << aAnalysisType << endl;
  }

  //------------------------------------
  TString tCanvasName = TString("canMassAssK0Hyp_") + TString(cAnalysisBaseTags[aAnalysisType]);
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  tHistToDraw->SetMarkerColor(1);
  tHistToDraw->SetLineColor(1);
  tHistToDraw->SetMarkerStyle(20);
  tHistToDraw->SetMarkerSize(0.5);

  SetupAxis(tHistToDraw->GetXaxis(),0.29,0.58,"Mass Assuming K^{0}_{S} Hypothesis (GeV/c^{2})");
  SetupAxis(tHistToDraw->GetYaxis(),"dN/dM_{inv}");

  tHistToDraw->DrawCopy();
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType)
{
  TH1* tHistToDraw;

  switch(aAnalysisType) {
  case kLamK0:
    tHistToDraw = fAnalysis1->GetMassAssumingLambdaHypothesis();
    break;

  case kALamK0:
    tHistToDraw = fConjAnalysis1->GetMassAssumingLambdaHypothesis();
    break;

  default:
    cout << "ERROR: PlotPartnersLamK0::DrawMassAssumingLambdaHypothesis: Invalid aAnalysisType = " << aAnalysisType << endl;
  }
  //------------------------------------
  TString tCanvasName = TString("canMassAssLamHyp_") + TString(cAnalysisBaseTags[aAnalysisType]);
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  tHistToDraw->SetMarkerColor(1);
  tHistToDraw->SetLineColor(1);
  tHistToDraw->SetMarkerStyle(20);
  tHistToDraw->SetMarkerSize(0.5);

  SetupAxis(tHistToDraw->GetXaxis(),1.0,2.2,"Mass Assuming #Lambda Hypothesis (GeV/c^{2})");
  SetupAxis(tHistToDraw->GetYaxis(),"dN/dM_{inv}");

  TPad* tPadLarge = new TPad("tPadLarge","tPadLarge",0.0,0.0,1.0,1.0);
    tPadLarge->Draw();
  TPad* tPadSmall = new TPad("tPadLarge","tPadLarge",0.42,0.39,0.9,0.89);
    tPadSmall->SetMargin(0.08,0.01,0.06,0.0);
    tPadSmall->Draw();

  tPadLarge->cd();
  tHistToDraw->DrawCopy();

  tPadSmall->cd();
  tHistToDraw->SetTitle("");
  SetupAxis(tHistToDraw->GetXaxis(),1.1,1.13,"");
  SetupAxis(tHistToDraw->GetYaxis(),"");
  tHistToDraw->Draw();

  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType)
{
  TH1* tHistToDraw;

  switch(aAnalysisType) {
  case kLamK0:
    tHistToDraw = fAnalysis1->GetMassAssumingAntiLambdaHypothesis();
    break;

  case kALamK0:
    tHistToDraw = fConjAnalysis1->GetMassAssumingAntiLambdaHypothesis();
    break;

  default:
    cout << "ERROR: PlotPartnersLamK0::DrawMassAssumingAntiLambdaHypothesis: Invalid aAnalysisType = " << aAnalysisType << endl;
  }
  //------------------------------------
  TString tCanvasName = TString("canMassAssALamHyp_") + TString(cAnalysisBaseTags[aAnalysisType]);
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  tHistToDraw->SetMarkerColor(1);
  tHistToDraw->SetLineColor(1);
  tHistToDraw->SetMarkerStyle(20);
  tHistToDraw->SetMarkerSize(0.5);

  SetupAxis(tHistToDraw->GetXaxis(),1.0,2.2,"Mass Assuming #bar{#Lambda} Hypothesis (GeV/c^{2})");
  SetupAxis(tHistToDraw->GetYaxis(),"dN/dM_{inv}");

  TPad* tPadLarge = new TPad("tPadLarge","tPadLarge",0.0,0.0,1.0,1.0);
    tPadLarge->Draw();
  TPad* tPadSmall = new TPad("tPadLarge","tPadLarge",0.42,0.39,0.9,0.89);
    tPadSmall->SetMargin(0.08,0.01,0.06,0.0);
    tPadSmall->Draw();

  tPadLarge->cd();
  tHistToDraw->DrawCopy();

  tPadSmall->cd();
  tHistToDraw->SetTitle("");
  SetupAxis(tHistToDraw->GetXaxis(),1.1,1.13,"");
  SetupAxis(tHistToDraw->GetYaxis(),"");
  tHistToDraw->Draw();

  return tReturnCan;
}


