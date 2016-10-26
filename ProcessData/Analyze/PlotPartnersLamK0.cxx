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
TCanvas* PlotPartnersLamK0::DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aSaveImage)
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
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
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
  PrintAnalysisType((TPad*)tReturnCan,aAnalysisType,0.80,0.85,0.15,0.10,63,30);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamK0::DrawMassAssumingLambdaHypothesis(AnalysisType aAnalysisType, bool aSaveImage)
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
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
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
  PrintAnalysisType(tPadLarge,aAnalysisType,0.80,0.10,0.15,0.10,63,30);

  tPadSmall->cd();
  tHistToDraw->SetTitle("");
  SetupAxis(tHistToDraw->GetXaxis(),1.1,1.13,"");
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
TCanvas* PlotPartnersLamK0::DrawMassAssumingAntiLambdaHypothesis(AnalysisType aAnalysisType, bool aSaveImage)
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
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
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
  PrintAnalysisType(tPadLarge,aAnalysisType,0.80,0.10,0.15,0.10,63,30);

  tPadSmall->cd();
  tHistToDraw->SetTitle("");
  SetupAxis(tHistToDraw->GetXaxis(),1.1,1.13,"");
  SetupAxis(tHistToDraw->GetYaxis(),"");
  tHistToDraw->DrawCopy();

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}


