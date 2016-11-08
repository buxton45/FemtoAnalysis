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
double PlotPartnersLamK0::GetPurity(AnalysisType aAnalysisType, ParticleType aV0Type)
{
  double tReturnValue = 0.;
  if(aAnalysisType == kLamK0)
  {
    if(aV0Type != kLam && aV0Type != kK0)
    {
      cout << "ERROR: PlotPartnersLamK0::GetPurity invalid aV0Type = " << aV0Type << " with aAnalysisType = " << aAnalysisType << endl;
      assert(0);
    }
    if(fAnalysis1->GetPurityCollection().size()==0) fAnalysis1->BuildPurityCollection();
    tReturnValue = fAnalysis1->GetPurity(aV0Type);
  }

  else if(aAnalysisType == kALamK0)
  {
    if(aV0Type != kALam && aV0Type != kK0)
    {
      cout << "ERROR: PlotPartnersLamK0::GetPurity invalid aV0Type = " << aV0Type << " with aAnalysisType = " << aAnalysisType << endl;
      assert(0);
    }
    if(fConjAnalysis1->GetPurityCollection().size()==0) fConjAnalysis1->BuildPurityCollection();
    tReturnValue = fConjAnalysis1->GetPurity(aV0Type);
  }

  else
  {
    cout << "ERROR: PlotPartnersLamK0::GetPurity invalid aAnalysisType = " << aAnalysisType << endl;
    assert(0);
  }

  return tReturnValue;
}


//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawPurity(bool aSaveImage)
{
  fAnalysis1->BuildPurityCollection();
  fConjAnalysis1->BuildPurityCollection();

  TString tCanvasName = TString("canPurity") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas *tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,1);

  fAnalysis1->DrawAllPurityHistos((TPad*)tReturnCan->cd(1));
  fConjAnalysis1->DrawAllPurityHistos((TPad*)tReturnCan->cd(2));

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }

  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawKStarCfs(bool aSaveImage)
{
  fAnalysis1->BuildKStarHeavyCf();
  fConjAnalysis1->BuildKStarHeavyCf();

  TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
    leg1->SetFillColor(0);
    leg1->AddEntry(fAnalysis1->GetKStarHeavyCf()->GetHeavyCf(),fAnalysis1->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

  TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
    leg2->SetFillColor(0);
    leg2->AddEntry(fConjAnalysis1->GetKStarHeavyCf()->GetHeavyCf(),fConjAnalysis1->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

  TString tCanvasName = TString("canKStarCf") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas *tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,1);


  fAnalysis1->DrawKStarHeavyCf((TPad*)tReturnCan->cd(1),1);
  tReturnCan->cd(1);
  leg1->Draw();

  fConjAnalysis1->DrawKStarHeavyCf((TPad*)tReturnCan->cd(2),1);
  tReturnCan->cd(2);
  leg2->Draw();

  //----------------------------------

  fAnalysis1->OutputPassFailInfo();
  fConjAnalysis1->OutputPassFailInfo();

  //----------------------------------
  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }

  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawKStarTrueVsRec(KStarTrueVsRecType aType, bool aSaveImage)
{
  gStyle->SetOptTitle(0);

  fAnalysisMC1->BuildModelKStarTrueVsRecTotal(aType);
  fConjAnalysisMC1->BuildModelKStarTrueVsRecTotal(aType);

  TH2* tTrueVsRecAn1 = fAnalysisMC1->GetModelKStarTrueVsRecTotal(aType);
    SetupAxis(tTrueVsRecAn1->GetXaxis(),"k*_{true} (GeV/c)");
    SetupAxis(tTrueVsRecAn1->GetYaxis(),"k*_{rec} (GeV/c)");
  TH2* tTrueVsRecConjAn1 = fConjAnalysisMC1->GetModelKStarTrueVsRecTotal(aType);
    SetupAxis(tTrueVsRecConjAn1->GetXaxis(),"k*_{true} (GeV/c)");
    SetupAxis(tTrueVsRecConjAn1->GetYaxis(),"k*_{rec} (GeV/c)");

  TString tCanvasName = TString("canKStarTrueVsRec") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()])
                         + TString(cKStarTrueVsRecTypeTags[aType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas *tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,1);
  //gStyle->SetOptStat(0);

  tReturnCan->cd(1);
    gPad->SetLogz();
    tTrueVsRecAn1->Draw("colz");
    PrintAnalysisType((TPad*)tReturnCan->cd(1),fAnalysisMC1->GetAnalysisType(),0.05,0.85,0.15,0.10,63,20);
    PrintText((TPad*)tReturnCan->cd(1),TString(cKStarTrueVsRecTypeTags[aType]),0.05,0.75,0.15,0.10,63,10);

  tReturnCan->cd(2);
    gPad->SetLogz();
    tTrueVsRecConjAn1->Draw("colz");
    PrintAnalysisType((TPad*)tReturnCan->cd(2),fConjAnalysisMC1->GetAnalysisType(),0.05,0.85,0.15,0.10,63,20);
    PrintText((TPad*)tReturnCan->cd(2),TString(cKStarTrueVsRecTypeTags[aType]),0.05,0.75,0.15,0.10,63,10);


  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawAvgSepCfs(bool aSaveImage)
{

  fAnalysis1->BuildAllAvgSepHeavyCfs();
  fConjAnalysis1->BuildAllAvgSepHeavyCfs();

  TString tCanvasName = TString("canAvgSepCfs") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()])  + TString("wConj")
                        + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,4);

  fAnalysis1->DrawAvgSepHeavyCf(kPosPos,(TPad*)tReturnCan->cd(1));
  fAnalysis1->DrawAvgSepHeavyCf(kPosNeg,(TPad*)tReturnCan->cd(2));
  fAnalysis1->DrawAvgSepHeavyCf(kNegPos,(TPad*)tReturnCan->cd(3));
  fAnalysis1->DrawAvgSepHeavyCf(kNegNeg,(TPad*)tReturnCan->cd(4));

  fConjAnalysis1->DrawAvgSepHeavyCf(kPosPos,(TPad*)tReturnCan->cd(5));
  fConjAnalysis1->DrawAvgSepHeavyCf(kPosNeg,(TPad*)tReturnCan->cd(6));
  fConjAnalysis1->DrawAvgSepHeavyCf(kNegPos,(TPad*)tReturnCan->cd(7));
  fConjAnalysis1->DrawAvgSepHeavyCf(kPosPos,(TPad*)tReturnCan->cd(8));


//TODO
/*
  //----------------------------------
  if(bSaveFile)
  {
    LamKchP->SaveAllAvgSepHeavyCfs(mySaveFile);
    LamKchM->SaveAllAvgSepHeavyCfs(mySaveFile);
    ALamKchP->SaveAllAvgSepHeavyCfs(mySaveFile);
    ALamKchM->SaveAllAvgSepHeavyCfs(mySaveFile);
  }
*/

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawAvgSepCfs(AnalysisType aAnalysisType, bool aSaveImage)
{

  fAnalysis1->BuildAllAvgSepHeavyCfs();
  fConjAnalysis1->BuildAllAvgSepHeavyCfs();

  TString tCanvasName = TString("canAvgSepCfs_") + TString(cAnalysisBaseTags[aAnalysisType]);
  tCanvasName += TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;

  int tNx=2, tNy=2;

  double tXLow = -1.;
  double tXHigh = 19.9;

  double tYLow = -1.;
  double tYHigh = 5.;

  CanvasPartition* tCanvasPartition = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);

  switch(aAnalysisType) {
  case kLamK0:
    tCanvasPartition->AddGraph(0,0,fAnalysis1->GetAvgSepHeavyCf(kPosPos)->GetHeavyCf(),(TString)fAnalysis1->GetDaughtersHistoTitle(kPosPos));
    tCanvasPartition->AddGraph(1,0,fAnalysis1->GetAvgSepHeavyCf(kPosNeg)->GetHeavyCf(),fAnalysis1->GetDaughtersHistoTitle(kPosNeg));

    tCanvasPartition->AddGraph(0,1,fAnalysis1->GetAvgSepHeavyCf(kNegPos)->GetHeavyCf(),fAnalysis1->GetDaughtersHistoTitle(kNegPos));
    tCanvasPartition->AddGraph(1,1,fAnalysis1->GetAvgSepHeavyCf(kNegNeg)->GetHeavyCf(),fAnalysis1->GetDaughtersHistoTitle(kNegNeg));
    break;

  case kALamK0:
    tCanvasPartition->AddGraph(0,0,fConjAnalysis1->GetAvgSepHeavyCf(kPosPos)->GetHeavyCf(),fConjAnalysis1->GetDaughtersHistoTitle(kPosPos));
    tCanvasPartition->AddGraph(1,0,fConjAnalysis1->GetAvgSepHeavyCf(kPosNeg)->GetHeavyCf(),fConjAnalysis1->GetDaughtersHistoTitle(kPosNeg));

    tCanvasPartition->AddGraph(0,1,fConjAnalysis1->GetAvgSepHeavyCf(kNegPos)->GetHeavyCf(),fConjAnalysis1->GetDaughtersHistoTitle(kNegPos));
    tCanvasPartition->AddGraph(1,1,fConjAnalysis1->GetAvgSepHeavyCf(kNegNeg)->GetHeavyCf(),fConjAnalysis1->GetDaughtersHistoTitle(kNegNeg));
    break;

  default:
    cout << "ERROR: PlotPartnersLamKch::DrawAvgSepCfs: Invalid aAnalysisType = " << aAnalysisType << endl;
  }

//TODO

  //----------------------------------
//  if(bSaveFile)
//  {
//    LamKchP->SaveAllAvgSepHeavyCfs(mySaveFile);
//    LamKchM->SaveAllAvgSepHeavyCfs(mySaveFile);
//    ALamKchP->SaveAllAvgSepHeavyCfs(mySaveFile);
//    ALamKchM->SaveAllAvgSepHeavyCfs(mySaveFile);
//  }

  tCanvasPartition->SetDrawUnityLine(true);
  tCanvasPartition->DrawAll();
  tCanvasPartition->DrawXaxisTitle("Avg. Sep. (cm)");
  tCanvasPartition->DrawYaxisTitle("C(Avg. Sep)");

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanvasPartition->GetCanvas()->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }

  return tCanvasPartition->GetCanvas();
}


//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::ViewPart1MassFail(bool aDrawWideRangeToo, bool aSaveImage)
{

  TString tCanvasName = TString("canPart1MassFail") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,1);

  fAnalysis1->DrawPart1MassFail((TPad*)tReturnCan->cd(1),aDrawWideRangeToo);
  fConjAnalysis1->DrawPart1MassFail((TPad*)tReturnCan->cd(2),aDrawWideRangeToo);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TH1* PlotPartnersLamK0::GetMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aNormByNEv, int aMarkerColor, int aMarkerStyle, double aMarkerSize)
{
  TH1* tReturnHist;
  double tNEvents = 0.;

  switch(aAnalysisType) {
  case kLamK0:
    tReturnHist = fAnalysis1->GetMassAssumingK0ShortHypothesis();
    tNEvents = fAnalysis1->GetNEventsPass();
    break;

  case kALamK0:
    tReturnHist = fConjAnalysis1->GetMassAssumingK0ShortHypothesis();
    tNEvents = fConjAnalysis1->GetNEventsPass();
    break;

  default:
    cout << "ERROR: PlotPartnersLamK0::GetMassAssumingK0ShortHypothesis: Invalid aAnalysisType = " << aAnalysisType << endl;
  }

  tReturnHist->SetMarkerColor(aMarkerColor);
  tReturnHist->SetLineColor(aMarkerColor);
  tReturnHist->SetMarkerStyle(aMarkerStyle);
  tReturnHist->SetMarkerSize(aMarkerSize);

  SetupAxis(tReturnHist->GetXaxis(),0.29,0.58,"Mass Assuming K^{0}_{S} Hypothesis (GeV/c^{2})");
  SetupAxis(tReturnHist->GetYaxis(),"dN/dM_{inv}");

  if(aNormByNEv)
  {
    tReturnHist->Scale(1./tNEvents);
    SetupAxis(tReturnHist->GetYaxis(),"(1/N_{Ev})dN/dM_{inv}");
  }
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* PlotPartnersLamK0::GetMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, bool aNormByNEv, int aMarkerColor, int aMarkerStyle, double aMarkerSize)
{
  TH1* tReturnHist;
  double tNEvents = 0.;

  switch(aAnalysisType) {
  case kLamK0:
    tReturnHist = fAnalysis1->GetMassAssumingLambdaHypothesis();
    tNEvents = fAnalysis1->GetNEventsPass();
    break;

  case kALamK0:
    tReturnHist = fConjAnalysis1->GetMassAssumingLambdaHypothesis();
    tNEvents = fConjAnalysis1->GetNEventsPass();
    break;

  default:
    cout << "ERROR: PlotPartnersLamK0::GetMassAssumingLambdaHypothesis: Invalid aAnalysisType = " << aAnalysisType << endl;
  }

  tReturnHist->SetMarkerColor(aMarkerColor);
  tReturnHist->SetLineColor(aMarkerColor);
  tReturnHist->SetMarkerStyle(aMarkerStyle);
  tReturnHist->SetMarkerSize(aMarkerSize);

  SetupAxis(tReturnHist->GetXaxis(),1.0,2.2,"Mass Assuming #Lambda Hypothesis (GeV/c^{2})");
  SetupAxis(tReturnHist->GetYaxis(),"dN/dM_{inv}");

  if(aNormByNEv)
  {
    tReturnHist->Scale(1./tNEvents);
    SetupAxis(tReturnHist->GetYaxis(),"(1/N_{Ev})dN/dM_{inv}");
  }
  return tReturnHist;
}

//________________________________________________________________________________________________________________
TH1* PlotPartnersLamK0::GetMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, bool aNormByNEv, int aMarkerColor, int aMarkerStyle, double aMarkerSize)
{
  TH1* tReturnHist;
  double tNEvents = 0.;

  switch(aAnalysisType) {
  case kLamK0:
    tReturnHist = fAnalysis1->GetMassAssumingAntiLambdaHypothesis();
    tNEvents = fAnalysis1->GetNEventsPass();
    break;

  case kALamK0:
    tReturnHist = fConjAnalysis1->GetMassAssumingAntiLambdaHypothesis();
    tNEvents = fConjAnalysis1->GetNEventsPass();
    break;

  default:
    cout << "ERROR: PlotPartnersLamK0::GetMassAssumingAntiLambdaHypothesis: Invalid aAnalysisType = " << aAnalysisType << endl;
  }

  tReturnHist->SetMarkerColor(aMarkerColor);
  tReturnHist->SetLineColor(aMarkerColor);
  tReturnHist->SetMarkerStyle(aMarkerStyle);
  tReturnHist->SetMarkerSize(aMarkerSize);

  SetupAxis(tReturnHist->GetXaxis(),1.0,2.2,"Mass Assuming #bar{#Lambda} Hypothesis (GeV/c^{2})");
  SetupAxis(tReturnHist->GetYaxis(),"dN/dM_{inv}");

  if(aNormByNEv)
  {
    tReturnHist->Scale(1./tNEvents);
    SetupAxis(tReturnHist->GetYaxis(),"(1/N_{Ev})dN/dM_{inv}");
  }
  return tReturnHist;
}



//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aSaveImage, bool aNormByNEv)
{
  gStyle->SetOptTitle(0);
  TH1* tHistToDraw = GetMassAssumingK0ShortHypothesis(aAnalysisType,aNormByNEv);
  //------------------------------------
  TString tCanvasName = TString("canMassAssK0Hyp_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  tHistToDraw->DrawCopy();
  PrintAnalysisType((TPad*)tReturnCan,aAnalysisType,0.84,0.89,0.15,0.10,63,30);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, bool aSaveImage, bool aNormByNEv)
{
  gStyle->SetOptTitle(0);
  TH1* tHistToDraw = GetMassAssumingLambdaHypothesis(aAnalysisType,aNormByNEv);
  //------------------------------------
  TString tCanvasName = TString("canMassAssLamHyp_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  TPad* tPadLarge = new TPad("tPadLarge","tPadLarge",0.0,0.0,1.0,1.0);
    tPadLarge->Draw();
  TPad* tPadSmall = new TPad("tPadLarge","tPadLarge",0.42,0.39,0.9,0.89);
    tPadSmall->SetMargin(0.08,0.01,0.06,0.0);
    tPadSmall->Draw();

  tPadLarge->cd();
  tHistToDraw->DrawCopy();
  PrintAnalysisType(tPadLarge,aAnalysisType,0.85,0.05,0.15,0.10,63,30);

  tPadSmall->cd();
  double tXRangeMin = 1.1;
  double tXRangeMax = 1.13;
  tHistToDraw->SetTitle("");
  SetupAxis(tHistToDraw->GetXaxis(),tXRangeMin,tXRangeMax,"");
  SetupAxis(tHistToDraw->GetYaxis(),"");
  tHistToDraw->DrawCopy();

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, bool aSaveImage, bool aNormByNEv)
{
  gStyle->SetOptTitle(0);
  TH1* tHistToDraw = GetMassAssumingAntiLambdaHypothesis(aAnalysisType,aNormByNEv);
  //------------------------------------
  TString tCanvasName = TString("canMassAssALamHyp_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  TPad* tPadLarge = new TPad("tPadLarge","tPadLarge",0.0,0.0,1.0,1.0);
    tPadLarge->Draw();
  TPad* tPadSmall = new TPad("tPadLarge","tPadLarge",0.42,0.39,0.9,0.89);
    tPadSmall->SetMargin(0.08,0.01,0.06,0.0);
    tPadSmall->Draw();

  tPadLarge->cd();
  tHistToDraw->DrawCopy();
  PrintAnalysisType(tPadLarge,aAnalysisType,0.85,0.05,0.15,0.10,63,30);

  tPadSmall->cd();
  double tXRangeMin = 1.1;
  double tXRangeMax = 1.13;
  tHistToDraw->SetTitle("");
  SetupAxis(tHistToDraw->GetXaxis(),tXRangeMin,tXRangeMax,"");
  SetupAxis(tHistToDraw->GetYaxis(),"");
  tHistToDraw->DrawCopy();

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, TH1* aHist1, TH1* aHist2, bool aSaveImage, TString aText1, TString aText2)
{
  gStyle->SetOptTitle(0);
  TString tCanvasName = TString("canMassAssK0HypCompare_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  aHist1->DrawCopy();
  aHist2->DrawCopy("same");
  PrintAnalysisType((TPad*)tReturnCan,aAnalysisType,0.84,0.89,0.15,0.10,63,30);

  TString tLegModifier = "";
  if(aHist1->Integral() < 100. && aHist2->Integral() < 100.) tLegModifier = "/N_{Ev}";

  TLegend *tLeg = new TLegend(0.40,0.15,0.65,0.30);
  tLeg->SetFillColor(0);
  tLeg->AddEntry(aHist1,aText1,"lp");
  tLeg->AddEntry((TObject*)0, TString::Format("N_{pass}%s = %0.4e",tLegModifier.Data(),aHist1->Integral()), "");
  tLeg->AddEntry(aHist2,aText2,"lp");
  tLeg->AddEntry((TObject*)0, TString::Format("N_{pass}%s = %0.4e",tLegModifier.Data(),aHist2->Integral()), "");
  tLeg->Draw();

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, TH1* aHist1, TH1* aHist2, bool aSaveImage, TString aText1, TString aText2)
{
  gStyle->SetOptTitle(0);
  TString tCanvasName = TString("canMassAssLamHypCompare_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  TPad* tPadLarge = new TPad("tPadLarge","tPadLarge",0.0,0.0,1.0,1.0);
    tPadLarge->Draw();
  TPad* tPadSmall = new TPad("tPadLarge","tPadLarge",0.42,0.39,0.9,0.89);
    tPadSmall->SetMargin(0.08,0.01,0.06,0.0);
    tPadSmall->Draw();

  tPadLarge->cd();
  aHist1->DrawCopy();
  aHist2->DrawCopy("same");
  PrintAnalysisType(tPadLarge,aAnalysisType,0.85,0.05,0.15,0.10,63,30);

  TString tLegModifier = "";
  if(aHist1->Integral() < 100. && aHist2->Integral() < 100.) tLegModifier = "/N_{Ev}";

  TLegend *tLeg = new TLegend(0.65,0.23,0.89,0.38);
  tLeg->SetFillColor(0);
  tLeg->AddEntry(aHist1,aText1,"lp");
  tLeg->AddEntry((TObject*)0, TString::Format("N_{pass}%s = %0.4e",tLegModifier.Data(),aHist1->Integral()), "");
  tLeg->AddEntry(aHist2,aText2,"lp");
  tLeg->AddEntry((TObject*)0, TString::Format("N_{pass}%s = %0.4e",tLegModifier.Data(),aHist2->Integral()), "");
  tLeg->Draw();

  tPadSmall->cd();
  double tXRangeMin = 1.1;
  double tXRangeMax = 1.13;
  aHist1->SetTitle("");
  SetupAxis(aHist1->GetXaxis(),tXRangeMin,tXRangeMax,"");
  SetupAxis(aHist1->GetYaxis(),"");
  //Make sure y-axis goes to 0 for min.
  aHist1->GetYaxis()->SetRangeUser(0.0,aHist1->GetBinContent(aHist1->FindBin(tXRangeMax)));
  aHist1->DrawCopy();
  aHist2->DrawCopy("same");

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, TH1* aHist1, TH1* aHist2, bool aSaveImage, TString aText1, TString aText2)
{
  gStyle->SetOptTitle(0);
  TString tCanvasName = TString("canMassAssALamHypCompare_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  TPad* tPadLarge = new TPad("tPadLarge","tPadLarge",0.0,0.0,1.0,1.0);
    tPadLarge->Draw();
  TPad* tPadSmall = new TPad("tPadLarge","tPadLarge",0.42,0.39,0.9,0.89);
    tPadSmall->SetMargin(0.08,0.01,0.06,0.0);
    tPadSmall->Draw();

  tPadLarge->cd();
  aHist1->DrawCopy();
  aHist2->DrawCopy("same");
  PrintAnalysisType(tPadLarge,aAnalysisType,0.85,0.05,0.15,0.10,63,30);

  TString tLegModifier = "";
  if(aHist1->Integral() < 100. && aHist2->Integral() < 100.) tLegModifier = "/N_{Ev}";

  TLegend *tLeg = new TLegend(0.65,0.23,0.89,0.38);
  tLeg->SetFillColor(0);
  tLeg->AddEntry(aHist1,aText1,"lp");
  tLeg->AddEntry((TObject*)0, TString::Format("N_{pass}%s = %0.4e",tLegModifier.Data(),aHist1->Integral()), "");
  tLeg->AddEntry(aHist2,aText2,"lp");
  tLeg->AddEntry((TObject*)0, TString::Format("N_{pass}%s = %0.4e",tLegModifier.Data(),aHist2->Integral()), "");
  tLeg->Draw();


  tPadSmall->cd();
  double tXRangeMin = 1.1;
  double tXRangeMax = 1.13;
  aHist1->SetTitle("");
  SetupAxis(aHist1->GetXaxis(),tXRangeMin,tXRangeMax,"");
  SetupAxis(aHist1->GetYaxis(),"");
  //Make sure y-axis goes to 0 for min.
  aHist1->GetYaxis()->SetRangeUser(0.0,aHist1->GetBinContent(aHist1->FindBin(tXRangeMax)));
  aHist1->DrawCopy();
  aHist2->DrawCopy("same");

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, TObjArray* tHists, vector<TString> &tLegendEntries, vector<double> &aPurityValues, bool aSaveImage)
{
  gStyle->SetOptTitle(0);
  TString tCanvasName = TString("canMassAssK0HypCompare_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  TLegend *tLeg = new TLegend(0.35,0.15,0.60,0.55);
  tLeg->SetFillColor(0);
  tLeg->SetEntrySeparation(0.25);

  assert(tHists->GetEntries() == (int)tLegendEntries.size());
  assert(tLegendEntries.size() == aPurityValues.size());
  int tNEntries = tHists->GetEntries();

  for(int i=0; i<tNEntries; i++)
  {
    TH1* tHistToDraw = (TH1*)tHists->At(i);
    if(i==0) tHistToDraw->DrawCopy();
    else tHistToDraw->DrawCopy("same");

    tLeg->AddEntry(tHistToDraw,tLegendEntries[i],"lp");
    TString tLegModifier = "";
    if(tHistToDraw->Integral() < 100) tLegModifier = "/N_{Ev}";
    tLeg->AddEntry((TObject*)0, TString::Format("N_{pass}%s = %0.4f",tLegModifier.Data(),tHistToDraw->Integral()), "");
    tLeg->AddEntry((TObject*)0, TString::Format("Purity = %0.4f",aPurityValues[i]), "");
  }
  tLeg->Draw();
  PrintAnalysisType((TPad*)tReturnCan,aAnalysisType,0.84,0.89,0.15,0.10,63,30);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, TObjArray* tHists, vector<TString> &tLegendEntries, vector<double> &aPurityValues, bool aSaveImage)
{
  bool bDrawFirstTwice = true;

  gStyle->SetOptTitle(0);
  TString tCanvasName = TString("canMassAssLamHypCompare_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  TPad* tPadLarge = new TPad("tPadLarge","tPadLarge",0.0,0.0,1.0,1.0);
    tPadLarge->Draw();
  TPad* tPadSmall = new TPad("tPadLarge","tPadLarge",0.42,0.39,0.9,0.89);
    tPadSmall->SetMargin(0.08,0.01,0.06,0.0);
    tPadSmall->Draw();

  assert(tHists->GetEntries() == (int)tLegendEntries.size());
  assert(tLegendEntries.size() == aPurityValues.size());
  int tNEntries = tHists->GetEntries();

  //-------------------------
  tPadLarge->cd();

  double x1min,y1min,x1max,y1max;
  double width, height;

  width = 0.20;
  height = 0.36;
  
  x1min = 0.20;
  x1max = x1min + width;
  y1min = 0.52;
  y1max = y1min + height;

  TLegend* tLeg1 = new TLegend(x1min,y1min,x1max,y1max);
  tLeg1->SetFillColor(0);
  tLeg1->SetEntrySeparation(0.25);

  double tYRangeMaxLarge = 0.;
  for(int i=0; i<tNEntries; i++)
  {
    TH1* tHist = (TH1*)tHists->At(i);
    double tYMaxLarge = tHist->GetMaximum();
    if(tYMaxLarge > tYRangeMaxLarge) tYRangeMaxLarge = tYMaxLarge;
  }
  tYRangeMaxLarge*=1.05;

  for(int i=0; i<tNEntries; i++)
  {
    TH1* tHistToDraw = (TH1*)tHists->At(i);
    if(i==0)
    {
      tHistToDraw->GetYaxis()->SetRangeUser(0.0,tYRangeMaxLarge);
      tHistToDraw->DrawCopy();
    }
    else tHistToDraw->DrawCopy("same");

    tLeg1->AddEntry(tHistToDraw,tLegendEntries[i],"lp");
    TString tLegModifier = "";
    if(tHistToDraw->Integral() < 100) tLegModifier = "/N_{Ev}";
    tLeg1->AddEntry((TObject*)0, TString::Format("     N_{pass}%s = %0.4f",tLegModifier.Data(),tHistToDraw->Integral()), "");
    tLeg1->AddEntry((TObject*)0, TString::Format("     Purity = %0.4f",aPurityValues[i]), "");
//    tLeg1->AddEntry((TObject*)0, TString::Format("N_{pass}%s = %0.4f; Purity = %0.4f",tLegModifier.Data(),tHistToDraw->Integral(),aPurityValues[i]), "");
  }
  if(bDrawFirstTwice) ((TH1*)tHists->At(0))->DrawCopy("same");
  tLeg1->Draw();
  PrintAnalysisType(tPadLarge,aAnalysisType,0.85,0.05,0.15,0.10,63,30);

  //-------------------------
  tPadSmall->cd();
  double tXRangeMin = 1.1;
  double tXRangeMax = 1.13;

  double tYRangeMaxSmall = 0.;
  for(int i=0; i<tNEntries; i++)
  {
    TH1* tHist = (TH1*)tHists->At(i);
    double tYMaxSmall = tHist->GetBinContent(tHist->FindBin(tXRangeMax));
    if(tYMaxSmall > tYRangeMaxSmall) tYRangeMaxSmall = tYMaxSmall;
  }
  tYRangeMaxSmall*=1.01;

  for(int i=0; i<tNEntries; i++)
  {
    TH1* tHistToDraw = (TH1*)tHists->At(i);
    if(i==0)
    {
      tHistToDraw->SetTitle("");
      SetupAxis(tHistToDraw->GetXaxis(),tXRangeMin,tXRangeMax,"");
      SetupAxis(tHistToDraw->GetYaxis(),"");
      //Make sure y-axis goes to 0 for min.
      tHistToDraw->GetYaxis()->SetRangeUser(0.0,tYRangeMaxSmall);
      tHistToDraw->DrawCopy();
    }
    else tHistToDraw->DrawCopy("same");
  }
  if(bDrawFirstTwice) ((TH1*)tHists->At(0))->DrawCopy("same");

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, TObjArray* tHists, vector<TString> &tLegendEntries, vector<double> &aPurityValues, bool aSaveImage)
{
  bool bDrawFirstTwice = true;

  gStyle->SetOptTitle(0);
  TString tCanvasName = TString("canMassAssALamHypCompare_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  TPad* tPadLarge = new TPad("tPadLarge","tPadLarge",0.0,0.0,1.0,1.0);
    tPadLarge->Draw();
  TPad* tPadSmall = new TPad("tPadLarge","tPadLarge",0.42,0.39,0.9,0.89);
    tPadSmall->SetMargin(0.08,0.01,0.06,0.0);
    tPadSmall->Draw();

  assert(tHists->GetEntries() == (int)tLegendEntries.size());
  assert(tLegendEntries.size() == aPurityValues.size());
  int tNEntries = tHists->GetEntries();

  //-------------------------
  tPadLarge->cd();

  double x1min,y1min,x1max,y1max;
  double width, height;

  width = 0.20;
  height = 0.36;
  
  x1min = 0.20;
  x1max = x1min + width;
  y1min = 0.52;
  y1max = y1min + height;

  TLegend* tLeg1 = new TLegend(x1min,y1min,x1max,y1max);
  tLeg1->SetFillColor(0);
  tLeg1->SetEntrySeparation(0.25);

  double tYRangeMaxLarge = 0.;
  for(int i=0; i<tNEntries; i++)
  {
    TH1* tHist = (TH1*)tHists->At(i);
    double tYMaxLarge = tHist->GetMaximum();
    if(tYMaxLarge > tYRangeMaxLarge) tYRangeMaxLarge = tYMaxLarge;
  }
  tYRangeMaxLarge*=1.01;

  for(int i=0; i<tNEntries; i++)
  {
    TH1* tHistToDraw = (TH1*)tHists->At(i);
    if(i==0)
    {
      tHistToDraw->GetYaxis()->SetRangeUser(0.0,tYRangeMaxLarge);
      tHistToDraw->DrawCopy();
    }
    else tHistToDraw->DrawCopy("same");

    tLeg1->AddEntry(tHistToDraw,tLegendEntries[i],"lp");
    TString tLegModifier = "";
    if(tHistToDraw->Integral() < 100) tLegModifier = "/N_{Ev}";
    tLeg1->AddEntry((TObject*)0, TString::Format("     N_{pass}%s = %0.4e",tLegModifier.Data(),tHistToDraw->Integral()), "");
    tLeg1->AddEntry((TObject*)0, TString::Format("     Purity = %0.4f",aPurityValues[i]), "");
  }
  if(bDrawFirstTwice) ((TH1*)tHists->At(0))->DrawCopy("same");
  tLeg1->Draw();
  PrintAnalysisType(tPadLarge,aAnalysisType,0.85,0.05,0.15,0.10,63,30);

  //-------------------------
  tPadSmall->cd();
  double tXRangeMin = 1.1;
  double tXRangeMax = 1.13;

  double tYRangeMaxSmall = 0.;
  for(int i=0; i<tNEntries; i++)
  {
    TH1* tHist = (TH1*)tHists->At(i);
    double tYMaxSmall = tHist->GetBinContent(tHist->FindBin(tXRangeMax));
    if(tYMaxSmall > tYRangeMaxSmall) tYRangeMaxSmall = tYMaxSmall;
  }
  tYRangeMaxSmall*=1.01;

  for(int i=0; i<tNEntries; i++)
  {
    TH1* tHistToDraw = (TH1*)tHists->At(i);
    if(i==0)
    {
      tHistToDraw->SetTitle("");
      SetupAxis(tHistToDraw->GetXaxis(),tXRangeMin,tXRangeMax,"");
      SetupAxis(tHistToDraw->GetYaxis(),"");
      //Make sure y-axis goes to 0 for min.
      tHistToDraw->GetYaxis()->SetRangeUser(0.0,tYRangeMaxSmall);
      tHistToDraw->DrawCopy();
    }
    else tHistToDraw->DrawCopy("same");
  }
  if(bDrawFirstTwice) ((TH1*)tHists->At(0))->DrawCopy("same");

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawSumMassAssumingLambdaAndAntiLambdaHypotheses(AnalysisType aAnalysisType, bool aSaveImage)
{
  TH1* tLamHyp = GetMassAssumingLambdaHypothesis(aAnalysisType);
  TH1* tALamHyp = GetMassAssumingAntiLambdaHypothesis(aAnalysisType);
  TString tHistoName = TString("SumMassAssLamAndALamHyp_") + TString(cAnalysisBaseTags[aAnalysisType]);
  TH1* tHistToDraw = (TH1*)tLamHyp->Clone(tHistoName);
  tHistToDraw->Add(tALamHyp);

  TString tCanvasName = TString("can") + tHistoName;
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  TPad* tPadLarge = new TPad("tPadLarge","tPadLarge",0.0,0.0,1.0,1.0);
    tPadLarge->Draw();
  TPad* tPadSmall = new TPad("tPadLarge","tPadLarge",0.42,0.39,0.9,0.89);
    tPadSmall->SetMargin(0.08,0.01,0.06,0.0);
    tPadSmall->Draw();

  tPadLarge->cd();
  tHistToDraw->DrawCopy();
  PrintAnalysisType(tPadLarge,aAnalysisType,0.80,0.10,0.15,0.10,63,30);

  tPadSmall->cd();
  double tXRangeMin = 1.1;
  double tXRangeMax = 1.13;
  tHistToDraw->SetTitle("");
  SetupAxis(tHistToDraw->GetXaxis(),tXRangeMin,tXRangeMax,"");
  SetupAxis(tHistToDraw->GetYaxis(),"");
  tHistToDraw->DrawCopy();

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;

}




