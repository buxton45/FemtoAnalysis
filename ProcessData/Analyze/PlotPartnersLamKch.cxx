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
double PlotPartnersLamKch::GetPurity(AnalysisType aAnalysisType, ParticleType aV0Type)
{
  double tReturnValue = 0.;
  if(aV0Type != kLam && aV0Type != kALam)
  {
    cout << "ERROR: PlotPartnersLamKch::GetPurity invalid aV0Type = " << aV0Type << endl;
    assert(0);
  }

  switch(aAnalysisType) {
  case kLamKchP:
    if(fAnalysis1->GetPurityCollection().size()==0) fAnalysis1->BuildPurityCollection();
    tReturnValue = fAnalysis1->GetPurity(aV0Type);
    break;

  case kALamKchM:
    if(fConjAnalysis1->GetPurityCollection().size()==0) fConjAnalysis1->BuildPurityCollection();
    tReturnValue = fConjAnalysis1->GetPurity(aV0Type);
    break;

  case kLamKchM:
    if(fAnalysis2->GetPurityCollection().size()==0) fAnalysis2->BuildPurityCollection();
    tReturnValue = fAnalysis2->GetPurity(aV0Type);
    break;

  case kALamKchP:
    if(fAnalysis2->GetPurityCollection().size()==0) fAnalysis2->BuildPurityCollection();
    tReturnValue = fConjAnalysis2->GetPurity(aV0Type);
    break;


  default:
    cout << "ERROR: PlotPartnersLamKch::GetPurity invalid aAnalysisType = " << aAnalysisType << endl;
    assert(0);
  }

  return tReturnValue;
}

//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawPurity(bool aSaveImage)
{
  fAnalysis1->BuildPurityCollection();
  fConjAnalysis1->BuildPurityCollection();
  fAnalysis2->BuildPurityCollection();
  fConjAnalysis2->BuildPurityCollection();

  TString tCanvasName = TString("canPurity") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas *tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,2);

  fAnalysis1->DrawAllPurityHistos((TPad*)tReturnCan->cd(1));
  fConjAnalysis1->DrawAllPurityHistos((TPad*)tReturnCan->cd(2));
  fAnalysis2->DrawAllPurityHistos((TPad*)tReturnCan->cd(3));
  fConjAnalysis2->DrawAllPurityHistos((TPad*)tReturnCan->cd(4));

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }

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
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
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
  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }

  return tReturnCan;
}



//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::DrawKStarTrueVsRec(KStarTrueVsRecType aType, bool aSaveImage)
{
  gStyle->SetOptTitle(0);

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
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas *tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,2);
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

  tReturnCan->cd(3);
    gPad->SetLogz();
    tTrueVsRecAn2->Draw("colz");
    PrintAnalysisType((TPad*)tReturnCan->cd(3),fAnalysisMC2->GetAnalysisType(),0.05,0.85,0.15,0.10,63,20);
    PrintText((TPad*)tReturnCan->cd(3),TString(cKStarTrueVsRecTypeTags[aType]),0.05,0.75,0.15,0.10,63,10);

  tReturnCan->cd(4);
    gPad->SetLogz();
    tTrueVsRecConjAn2->Draw("colz");
    PrintAnalysisType((TPad*)tReturnCan->cd(4),fConjAnalysisMC2->GetAnalysisType(),0.05,0.85,0.15,0.10,63,20);
    PrintText((TPad*)tReturnCan->cd(4),TString(cKStarTrueVsRecTypeTags[aType]),0.05,0.75,0.15,0.10,63,10);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
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
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
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

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
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

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
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
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;

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

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tCanvasPartition->GetCanvas()->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }

  return tCanvasPartition->GetCanvas();
}



//________________________________________________________________________________________________________________
TCanvas* PlotPartnersLamKch::ViewPart1MassFail(bool aDrawWideRangeToo, bool aSaveImage)
{

  TString tCanvasName = TString("canPart1MassFail") + TString(cAnalysisBaseTags[fAnalysis1->GetAnalysisType()]) + TString(cCentralityTags[fAnalysis1->GetCentralityType()]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->Divide(2,2);

  fAnalysis1->DrawPart1MassFail((TPad*)tReturnCan->cd(1),aDrawWideRangeToo);
  fConjAnalysis1->DrawPart1MassFail((TPad*)tReturnCan->cd(2),aDrawWideRangeToo);
  fAnalysis2->DrawPart1MassFail((TPad*)tReturnCan->cd(3),aDrawWideRangeToo);
  fConjAnalysis2->DrawPart1MassFail((TPad*)tReturnCan->cd(4),aDrawWideRangeToo);

  if(aSaveImage)
  {
    ExistsSaveLocationBase();
    tReturnCan->SaveAs(fSaveLocationBase+tCanvasName+TString(".pdf"));
  }
  return tReturnCan;
}

//________________________________________________________________________________________________________________
TH1* PlotPartnersLamKch::GetMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aNormByNEv, int aMarkerColor, int aMarkerStyle, double aMarkerSize)
{
  TH1* tReturnHist;
  double tNEvents = 0.;

  switch(aAnalysisType) {
  case kLamKchP:
    tReturnHist = fAnalysis1->GetMassAssumingK0ShortHypothesis();
    tNEvents = fAnalysis1->GetNEventsPass();
    break;

  case kALamKchM:
    tReturnHist = fConjAnalysis1->GetMassAssumingK0ShortHypothesis();
    tNEvents = fConjAnalysis1->GetNEventsPass();
    break;

  case kLamKchM:
    tReturnHist = fAnalysis2->GetMassAssumingK0ShortHypothesis();
    tNEvents = fAnalysis2->GetNEventsPass();
    break;

  case kALamKchP:
    tReturnHist = fConjAnalysis2->GetMassAssumingK0ShortHypothesis();
    tNEvents = fConjAnalysis2->GetNEventsPass();
    break;

  default:
    cout << "ERROR: PlotPartnersLamKch::GetMassAssumingK0ShortHypothesis: Invalid aAnalysisType = " << aAnalysisType << endl;
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
TCanvas* PlotPartnersLamKch::DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, bool aSaveImage)
{
  gStyle->SetOptTitle(0);
  TH1* tHistToDraw = GetMassAssumingK0ShortHypothesis(aAnalysisType);

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
TCanvas* PlotPartnersLamKch::DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, TH1* aHist1, TH1* aHist2, bool aSaveImage, TString aText1, TString aText2)
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
TCanvas* PlotPartnersLamKch::DrawMassAssumingK0ShortHypothesis(AnalysisType aAnalysisType, TObjArray* tHists, vector<TString> &tLegendEntries, vector<double> &aPurityValues, bool aSaveImage)
{
  gStyle->SetOptTitle(0);
  TString tCanvasName = TString("canMassAssK0HypCompare_") + TString(cAnalysisBaseTags[aAnalysisType]);
  if(!fDirNameModifier.IsNull()) tCanvasName += fDirNameModifier;
  TCanvas* tReturnCan = new TCanvas(tCanvasName,tCanvasName);
  tReturnCan->cd();
  gStyle->SetOptStat(0);

  TLegend *tLeg = new TLegend(0.375,0.15,0.625,0.45);
  tLeg->SetFillColor(0);

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
    tLeg->AddEntry((TObject*)0, TString::Format("N_{pass}%s = %0.4e",tLegModifier.Data(),tHistToDraw->Integral()), "");
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


