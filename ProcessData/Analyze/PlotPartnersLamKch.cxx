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
PlotPartnersLamKch::PlotPartnersLamKch(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNPartialAnalysis, bool aIsTrainResults) :
  PlotPartners(aFileLocationBase,aAnalysisType,aCentralityType,aNPartialAnalysis,aIsTrainResults)

{

}

//________________________________________________________________________________________________________________
PlotPartnersLamKch::PlotPartnersLamKch(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType, CentralityType aCentralityType, int aNPartialAnalysis, bool aIsTrainResults) :
  PlotPartners(aFileLocationBase,aFileLocationBaseMC,aAnalysisType,aCentralityType,aNPartialAnalysis,aIsTrainResults)

{

}



//________________________________________________________________________________________________________________
PlotPartnersLamKch::~PlotPartnersLamKch()
{
}


//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawPurity()
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

  return tReturnCan;
}


//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawKStarCfs()
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

  return tReturnCan;
}



//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawKStarTrueVsRec(KStarTrueVsRecType aType)
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

  return tReturnCan;
}





//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawAvgSepCfs()
{

  fAnalysis1->BuildAllAvgSepHeavyCfs();
  fConjAnalysis1->BuildAllAvgSepHeavyCfs();
  fAnalysis2->BuildAllAvgSepHeavyCfs();
  fConjAnalysis2->BuildAllAvgSepHeavyCfs();

  TString tCanvasName = TString("canAvgSepCfs") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
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

  return tReturnCan;
}


//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::ViewPart1MassFail(bool aDrawWideRangeToo)
{

  TString tCanvasName = TString("canPart1MassFail") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,2);

  fAnalysis1->DrawPart1MassFail((TPad*)tReturnCan->cd(1),aDrawWideRangeToo);
  fConjAnalysis1->DrawPart1MassFail((TPad*)tReturnCan->cd(2),aDrawWideRangeToo);
  fAnalysis2->DrawPart1MassFail((TPad*)tReturnCan->cd(3),aDrawWideRangeToo);
  fConjAnalysis2->DrawPart1MassFail((TPad*)tReturnCan->cd(4),aDrawWideRangeToo);

  return tReturnCan;
}


