///////////////////////////////////////////////////////////////////////////
// PlotPartnersLamKch:                                                   //
///////////////////////////////////////////////////////////////////////////


#include "PlotPartnersLamKch.h"

#ifdef __ROOT__
ClassImp(PlotPartnersLamKch)
#endif



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

//________________________________________________________________________________________________________________
PlotPartnersLamKch::PlotPartnersLamKch(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, TString aDirNameModifier) :
  PlotPartners(aFileLocationBase,aAnalysisType,aCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier)

{

}

//________________________________________________________________________________________________________________
PlotPartnersLamKch::PlotPartnersLamKch(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, AnalysisRunType aRunType, int aNPartialAnalysis, TString aDirNameModifier) :
  PlotPartners(aFileLocationBase,aFileLocationBaseMC,aAnalysisType,aCentralityType,aRunType,aNPartialAnalysis,aDirNameModifier)

{

}



//________________________________________________________________________________________________________________
PlotPartnersLamKch::~PlotPartnersLamKch()
{
}


//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawPurity(bool aSaveImage)
{
  fAnalysis1->BuildPurityCollection();
  fConjAnalysis1->BuildPurityCollection();
  fAnalysis2->BuildPurityCollection();
  fConjAnalysis2->BuildPurityCollection();

  TString tCanvasName = TString("canPurity") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  TCanvas *tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,2);

  fAnalysis1->DrawAllPurityHistos((TPad*)tReturnCan->cd(1));
  fConjAnalysis1->DrawAllPurityHistos((TPad*)tReturnCan->cd(2));
  fAnalysis2->DrawAllPurityHistos((TPad*)tReturnCan->cd(3));
  fConjAnalysis2->DrawAllPurityHistos((TPad*)tReturnCan->cd(4));

  bool tExistsSaveLocation = ExistsSaveLocationBase();
  if(aSaveImage) tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));

  return tReturnCan;
}


//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawKStarCfs(bool aSaveImage)
{
  fAnalysis1->BuildKStarHeavyCf();
  fConjAnalysis1->BuildKStarHeavyCf();
  fAnalysis2->BuildKStarHeavyCf();
  fConjAnalysis2->BuildKStarHeavyCf();

  TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
    leg1->SetFillColor(0);
    leg1->AddEntry(fAnalysis1->GetKStarHeavyCf()->GetHeavyCf(),fAnalysis1->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
    leg1->AddEntry(fAnalysis2->GetKStarHeavyCf()->GetHeavyCf(),fAnalysis2->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

  TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
    leg2->SetFillColor(0);
    leg2->AddEntry(fConjAnalysis1->GetKStarHeavyCf()->GetHeavyCf(),fConjAnalysis1->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
    leg2->AddEntry(fConjAnalysis2->GetKStarHeavyCf()->GetHeavyCf(),fConjAnalysis2->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");


  TString tNewNameAn12 = fAnalysis1->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
    tNewNameAn12 += " & " ;
    tNewNameAn12 += fAnalysis2->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
    tNewNameAn12 += TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  fAnalysis1->GetKStarHeavyCf()->GetHeavyCf()->SetTitle(tNewNameAn12);

  TString tNewNameConjAn12 = fConjAnalysis1->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
    tNewNameConjAn12 += " & " ;
    tNewNameConjAn12 += fConjAnalysis2->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
    tNewNameConjAn12 += TString(cCentralityTags[fConjAnalysis1->GetCentralityType()]);
  fConjAnalysis1->GetKStarHeavyCf()->GetHeavyCf()->SetTitle(tNewNameConjAn12);

  TString tCanvasName = TString("canKStarCf") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  TCanvas *tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,1);


  fAnalysis1->DrawKStarHeavyCf((TPad*)tReturnCan->cd(1),2);
  fAnalysis2->DrawKStarHeavyCf((TPad*)tReturnCan->cd(1),4,"same");
  tReturnCan->cd(1);
  leg1->Draw();

  fConjAnalysis1->DrawKStarHeavyCf((TPad*)tReturnCan->cd(2),2);
  fConjAnalysis2->DrawKStarHeavyCf((TPad*)tReturnCan->cd(2),4,"same");
  tReturnCan->cd(2);
  leg2->Draw();

  //----------------------------------

  fAnalysis1->OutputPassFailInfo();
  fAnalysis2->OutputPassFailInfo();
  fConjAnalysis1->OutputPassFailInfo();
  fConjAnalysis2->OutputPassFailInfo();

  //----------------------------------
  bool tExistsSaveLocation = ExistsSaveLocationBase();
  if(aSaveImage) tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));

  return tReturnCan;
}



//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawKStarTrueVsRec(KStarTrueVsRecType aType, bool aSaveImage)
{
  fAnalysisMC1->BuildModelKStarTrueVsRecTotal(aType);
  fConjAnalysisMC1->BuildModelKStarTrueVsRecTotal(aType);
  fAnalysisMC2->BuildModelKStarTrueVsRecTotal(aType);
  fConjAnalysisMC2->BuildModelKStarTrueVsRecTotal(aType);

  TH2* tTrueVsRecAn1 = fAnalysisMC1->GetModelKStarTrueVsRecTotal(aType);
    SetupAxis(tTrueVsRecAn1->GetXaxis(),"k*_{true} (GeV/c)");
    SetupAxis(tTrueVsRecAn1->GetYaxis(),"k*_{rec} (GeV/c)");
  TH2* tTrueVsRecConjAn1 = fConjAnalysisMC1->GetModelKStarTrueVsRecTotal(aType);
    SetupAxis(tTrueVsRecConjAn1->GetXaxis(),"k*_{true} (GeV/c)");
    SetupAxis(tTrueVsRecConjAn1->GetYaxis(),"k*_{rec} (GeV/c)");
  TH2* tTrueVsRecAn2 = fAnalysisMC2->GetModelKStarTrueVsRecTotal(aType);
    SetupAxis(tTrueVsRecAn2->GetXaxis(),"k*_{true} (GeV/c)");
    SetupAxis(tTrueVsRecAn2->GetYaxis(),"k*_{rec} (GeV/c)");
  TH2* tTrueVsRecConjAn2 = fConjAnalysisMC2->GetModelKStarTrueVsRecTotal(aType);
    SetupAxis(tTrueVsRecConjAn2->GetXaxis(),"k*_{true} (GeV/c)");
    SetupAxis(tTrueVsRecConjAn2->GetYaxis(),"k*_{rec} (GeV/c)");


  TString tCanvasName = TString("canKStarTrueVsRec") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()])
                         + TString(cKStarTrueVsRecTypeTags[aType]);
  TCanvas *tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,2);
  //gStyle->SetOptStat(0);

  tReturnCan->cd(1);
    gPad->SetLogz();
    tTrueVsRecAn1->Draw("colz");

  tReturnCan->cd(2);
    gPad->SetLogz();
    tTrueVsRecConjAn1->Draw("colz");

  tReturnCan->cd(3);
    gPad->SetLogz();
    tTrueVsRecAn2->Draw("colz");

  tReturnCan->cd(4);
    gPad->SetLogz();
    tTrueVsRecConjAn2->Draw("colz");

  bool tExistsSaveLocation = ExistsSaveLocationBase();
  if(aSaveImage) tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  return tReturnCan;
}





//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawAvgSepCfs(bool aSaveImage)
{

  fAnalysis1->BuildAllAvgSepHeavyCfs();
  fConjAnalysis1->BuildAllAvgSepHeavyCfs();
  fAnalysis2->BuildAllAvgSepHeavyCfs();
  fConjAnalysis2->BuildAllAvgSepHeavyCfs();

  TString tCanvasName = TString("canAvgSepCfs") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()])  + TString("wConjAnd")
                        + TString(cAnalysisBaseTags[fAnalysis2->GetAnalysisType()])  + TString("wConj")
                        + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,4);

  fAnalysis1->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(1));
  fAnalysis1->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(2));

  fConjAnalysis1->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(3));
  fConjAnalysis1->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(4));

  fAnalysis2->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(5));
  fAnalysis2->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(6));

  fConjAnalysis2->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(7));
  fConjAnalysis2->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(8));

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

  bool tExistsSaveLocation = ExistsSaveLocationBase();
  if(aSaveImage) tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  return tReturnCan;
}

/*
//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawAvgSepCfs(AnalysisType aAnalysisType, bool aDrawConj)
{

  fAnalysis1->BuildAllAvgSepHeavyCfs();
  fConjAnalysis1->BuildAllAvgSepHeavyCfs();
  fAnalysis2->BuildAllAvgSepHeavyCfs();
  fConjAnalysis2->BuildAllAvgSepHeavyCfs();

  TString tCanvasName = TString("canAvgSepCfs") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(aDrawConj) tCanvasName += TString("wConj");
  tCanvasName += TString(cCentralityTags[fAnalysis1->GetCentralityType()]);

  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  if(aDrawConj) tReturnCan->Divide(2,2);
  else tReturnCan->Divide(2,1);

  switch(aAnalysisType) {
  case kLamKchP:
    fAnalysis1->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(1));
    fAnalysis1->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(2));

    if(aDrawConj)
    {
      fConjAnalysis1->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(3));
      fConjAnalysis1->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(4));
    }
    break;

  case kALamKchM:
    fConjAnalysis1->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(1));
    fConjAnalysis1->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(2));

    if(aDrawConj)
    {
      fAnalysis1->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(3));
      fAnalysis1->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(4));
    }
    break;

  case kLamKchM:
    fAnalysis2->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(1));
    fAnalysis2->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(2));

    if(aDrawConj)
    {
      fConjAnalysis2->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(3));
      fConjAnalysis2->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(4));
    }
    break;

  case kALamKchP:
    fConjAnalysis2->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(1));
    fConjAnalysis2->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(2));

    if(aDrawConj)
    {
      fAnalysis2->DrawAvgSepHeavyCf(kTrackPos,(TPad*)tReturnCan->cd(3));
      fAnalysis2->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)tReturnCan->cd(4));
    }
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

  bool tExistsSaveLocation = ExistsSaveLocationBase();
  if(aSaveImage) tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  return tReturnCan;
}
*/

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawAvgSepCfs(AnalysisType aAnalysisType, bool aDrawConj, bool aSaveImage)
{

  fAnalysis1->BuildAllAvgSepHeavyCfs();
  fConjAnalysis1->BuildAllAvgSepHeavyCfs();
  fAnalysis2->BuildAllAvgSepHeavyCfs();
  fConjAnalysis2->BuildAllAvgSepHeavyCfs();

  TString tCanvasName = TString("canAvgSepCfs_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(aDrawConj) tCanvasName += TString("wConj");
  tCanvasName += TString(cCentralityTags[fAnalysis1->GetCentralityType()]);

  int tNx=2, tNy=1;
  if(aDrawConj) tNy=2;


  double tXLow = -1.;
  double tXHigh = 19.9;

  double tYLow = -1.;
  double tYHigh = 5.;

  CanvasPartition* tCanvasPartition = new CanvasPartition(tCanvasName,tNx,tNy,tXLow,tXHigh,tYLow,tYHigh,0.12,0.05,0.13,0.05);


  switch(aAnalysisType) {
  case kLamKchP:
    tCanvasPartition->AddGraph(0,0,fAnalysis1->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCf(),fAnalysis1->GetDaughtersHistoTitle(kTrackPos));
    tCanvasPartition->AddGraph(1,0,fAnalysis1->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCf(),fAnalysis1->GetDaughtersHistoTitle(kTrackNeg));

    if(aDrawConj)
    {
      tCanvasPartition->AddGraph(0,1,fConjAnalysis1->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCf(),fConjAnalysis1->GetDaughtersHistoTitle(kTrackPos));
      tCanvasPartition->AddGraph(1,1,fConjAnalysis1->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCf(),fConjAnalysis1->GetDaughtersHistoTitle(kTrackNeg));
    }
    break;

  case kALamKchM:
    tCanvasPartition->AddGraph(0,0,fConjAnalysis1->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCf(),fConjAnalysis1->GetDaughtersHistoTitle(kTrackPos));
    tCanvasPartition->AddGraph(1,0,fConjAnalysis1->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCf(),fConjAnalysis1->GetDaughtersHistoTitle(kTrackNeg));

    if(aDrawConj)
    {
      tCanvasPartition->AddGraph(0,1,fAnalysis1->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCf(),fAnalysis1->GetDaughtersHistoTitle(kTrackPos));
      tCanvasPartition->AddGraph(1,1,fAnalysis1->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCf(),fAnalysis1->GetDaughtersHistoTitle(kTrackNeg));
    }
    break;

  case kLamKchM:
    tCanvasPartition->AddGraph(0,0,fAnalysis2->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCf(),fAnalysis2->GetDaughtersHistoTitle(kTrackPos));
    tCanvasPartition->AddGraph(1,0,fAnalysis2->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCf(),fAnalysis2->GetDaughtersHistoTitle(kTrackNeg));

    if(aDrawConj)
    {
      tCanvasPartition->AddGraph(0,1,fConjAnalysis2->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCf(),fConjAnalysis2->GetDaughtersHistoTitle(kTrackPos));
      tCanvasPartition->AddGraph(1,1,fConjAnalysis2->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCf(),fConjAnalysis2->GetDaughtersHistoTitle(kTrackNeg));
    }
    break;

  case kALamKchP:
    tCanvasPartition->AddGraph(0,0,fConjAnalysis2->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCf(),fConjAnalysis2->GetDaughtersHistoTitle(kTrackPos));
    tCanvasPartition->AddGraph(1,0,fConjAnalysis2->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCf(),fConjAnalysis2->GetDaughtersHistoTitle(kTrackNeg));

    if(aDrawConj)
    {
      tCanvasPartition->AddGraph(0,1,fAnalysis2->GetAvgSepHeavyCf(kTrackPos)->GetHeavyCf(),fAnalysis2->GetDaughtersHistoTitle(kTrackPos));
      tCanvasPartition->AddGraph(1,1,fAnalysis2->GetAvgSepHeavyCf(kTrackNeg)->GetHeavyCf(),fAnalysis2->GetDaughtersHistoTitle(kTrackNeg));
    }
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

  bool tExistsSaveLocation = ExistsSaveLocationBase();
  if(aSaveImage) tCanvasPartition->GetCanvas()->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));

  return tCanvasPartition->GetCanvas();
}



//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::ViewPart1MassFail(bool aDrawWideRangeToo, bool aSaveImage)
{

  TString tCanvasName = TString("canPart1MassFail") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,2);

  fAnalysis1->DrawPart1MassFail((TPad*)tReturnCan->cd(1),aDrawWideRangeToo);
  fConjAnalysis1->DrawPart1MassFail((TPad*)tReturnCan->cd(2),aDrawWideRangeToo);
  fAnalysis2->DrawPart1MassFail((TPad*)tReturnCan->cd(3),aDrawWideRangeToo);
  fConjAnalysis2->DrawPart1MassFail((TPad*)tReturnCan->cd(4),aDrawWideRangeToo);

  bool tExistsSaveLocation = ExistsSaveLocationBase();
  if(aSaveImage) tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  return tReturnCan;
}


//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aSaveImage)
{
  TH1* tHistToDraw;

  switch(aAnalysisType) {
  case kLamKchP:
    tHistToDraw = fAnalysis1->GetMassAssumingK0ShortHypothesis();
    break;

  case kALamKchM:
    tHistToDraw = fConjAnalysis1->GetMassAssumingK0ShortHypothesis();
    break;

  case kLamKchM:
    tHistToDraw = fAnalysis2->GetMassAssumingK0ShortHypothesis();
    break;

  case kALamKchP:
    tHistToDraw = fConjAnalysis2->GetMassAssumingK0ShortHypothesis();
    break;

  default:
    cout << "ERROR: PlotPartnersLamKch::DrawMassAssumingK0ShortHypothesis: Invalid aAnalysisType = " << aAnalysisType << endl;
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
  bool tExistsSaveLocation = ExistsSaveLocationBase();
  if(aSaveImage) tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  return tReturnCan;
}



