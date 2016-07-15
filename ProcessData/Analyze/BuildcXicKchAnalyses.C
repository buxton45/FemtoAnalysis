#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TList.h"

#include "TApplication.h"

using std::cout;
using std::endl;
using std::vector;

#include "PartialAnalysis.h"
class PartialAnalysis;

#include "Analysis.h"
class Analysis;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  //-----Data

  TString FileLocationBase = "~/Analysis/K0Lam/Results_cXicKch_20160202/Results_cXicKch_20160202";
  Analysis* XiKchP = new Analysis(FileLocationBase,kXiKchP,k0010);
  Analysis* AXiKchP = new Analysis(FileLocationBase,kAXiKchP,k0010);
  Analysis* XiKchM = new Analysis(FileLocationBase,kXiKchM,k0010);
  Analysis* AXiKchM = new Analysis(FileLocationBase,kAXiKchM,k0010);

  TString SaveFileName = "~/Analysis/K0Lam/Results_cXicKch_20160202/0010/Results_cXicKch_20160202_0010.root";

  //-----MC
/*
  TString FileLocationBaseMC = "~/Analysis/K0Lam/Results_cXicKch_20160125/Results_cXicKch_MC_20160125";
  Analysis* XiKchPMC = new Analysis(FileLocationBaseMC,kXiKchP,k0010);
  Analysis* AXiKchPMC = new Analysis(FileLocationBaseMC,kAXiKchP,k0010);
  Analysis* XiKchMMC = new Analysis(FileLocationBaseMC,kXiKchM,k0010);
  Analysis* AXiKchMMC = new Analysis(FileLocationBaseMC,kAXiKchM,k0010);

  TString FileLocationBaseMCd = "~/Analysis/K0Lam/Results_cXicKch_20160125/Results_cXicKch_MCd_20160125";
  Analysis* XiKchPMCd = new Analysis(FileLocationBaseMCd,kXiKchP,k0010);
  Analysis* AXiKchPMCd = new Analysis(FileLocationBaseMCd,kAXiKchP,k0010);
  Analysis* XiKchMMCd = new Analysis(FileLocationBaseMCd,kXiKchM,k0010);
  Analysis* AXiKchMMCd = new Analysis(FileLocationBaseMCd,kAXiKchM,k0010);

  vector<PartialAnalysis*> tXiKchPMCTotVec;
  vector<PartialAnalysis*> tTempXiKchPMCVec = XiKchPMC->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tTempXiKchPMCdVec = XiKchPMCd->GetPartialAnalysisCollection();
  assert(tTempXiKchPMCVec.size() == tTempXiKchPMCdVec.size());
  for(unsigned int i=0; i<tTempXiKchPMCVec.size(); i++)
  {
    tXiKchPMCTotVec.push_back(tTempXiKchPMCVec[i]);
    tXiKchPMCTotVec.push_back(tTempXiKchPMCdVec[i]);
  }

  vector<PartialAnalysis*> tAXiKchPMCTotVec;
  vector<PartialAnalysis*> tTempAXiKchPMCVec = AXiKchPMC->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tTempAXiKchPMCdVec = AXiKchPMCd->GetPartialAnalysisCollection();
  assert(tTempAXiKchPMCVec.size() == tTempAXiKchPMCdVec.size());
  for(unsigned int i=0; i<tTempAXiKchPMCVec.size(); i++)
  {
    tAXiKchPMCTotVec.push_back(tTempAXiKchPMCVec[i]);
    tAXiKchPMCTotVec.push_back(tTempAXiKchPMCdVec[i]);
  }

  vector<PartialAnalysis*> tXiKchMMCTotVec;
  vector<PartialAnalysis*> tTempXiKchMMCVec = XiKchMMC->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tTempXiKchMMCdVec = XiKchMMCd->GetPartialAnalysisCollection();
  assert(tTempXiKchMMCVec.size() == tTempXiKchMMCdVec.size());
  for(unsigned int i=0; i<tTempXiKchMMCVec.size(); i++)
  {
    tXiKchMMCTotVec.push_back(tTempXiKchMMCVec[i]);
    tXiKchMMCTotVec.push_back(tTempXiKchMMCdVec[i]);
  }

  vector<PartialAnalysis*> tAXiKchMMCTotVec;
  vector<PartialAnalysis*> tTempAXiKchMMCVec = AXiKchMMC->GetPartialAnalysisCollection();
  vector<PartialAnalysis*> tTempAXiKchMMCdVec = AXiKchMMCd->GetPartialAnalysisCollection();
  assert(tTempAXiKchMMCVec.size() == tTempAXiKchMMCdVec.size());
  for(unsigned int i=0; i<tTempAXiKchMMCVec.size(); i++)
  {
    tAXiKchMMCTotVec.push_back(tTempAXiKchMMCVec[i]);
    tAXiKchMMCTotVec.push_back(tTempAXiKchMMCdVec[i]);
  }

  Analysis* XiKchPMCTot = new Analysis("XiKchPMCTot_0010",tXiKchPMCTotVec);
  Analysis* AXiKchPMCTot = new Analysis("AXiKchPMCTot_0010",tAXiKchPMCTotVec);
  Analysis* XiKchMMCTot = new Analysis("XiKchMMCTot_0010",tXiKchMMCTotVec);
  Analysis* AXiKchMMCTot = new Analysis("AXiKchMMCTot_0010",tAXiKchMMCTotVec);
*/

//-----------------------------------------------------------------------------


  bool bSaveFile = false;
  TFile *mySaveFile;
  if(bSaveFile) {mySaveFile = new TFile(SaveFileName, "RECREATE");}

  bool bContainsPurity = true;
  bool bContainsKStarCfs = true;
  bool bContainsAvgSepCfs = false;

  bool bContainsKStar2dCfs = false;

  bool bContainsSepHeavyCfs = false;
  bool bContainsAvgSepCowSailCfs = false;

  bool bViewPart1MassFail = false;

  bool bDrawMC = false;

  bool bSaveFigures = true;
  TString tSaveFiguresLocation = "~/Analysis/K0Lam/Results_cXicKch_20160202/0010/";
  //-------------------------------------------------------------------

  if(bContainsKStarCfs)
  {
    XiKchP->BuildKStarHeavyCf();
    AXiKchP->BuildKStarHeavyCf();
    XiKchM->BuildKStarHeavyCf();
    AXiKchM->BuildKStarHeavyCf();

    XiKchP->GetKStarHeavyCf()->Rebin(2);
    AXiKchP->GetKStarHeavyCf()->Rebin(2);
    XiKchM->GetKStarHeavyCf()->Rebin(2);
    AXiKchM->GetKStarHeavyCf()->Rebin(2);

    TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
      leg1->SetFillColor(0);
      leg1->AddEntry(XiKchP->GetKStarHeavyCf()->GetHeavyCf(),XiKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg1->AddEntry(XiKchM->GetKStarHeavyCf()->GetHeavyCf(),XiKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

    TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
      leg2->SetFillColor(0);
      leg2->AddEntry(AXiKchP->GetKStarHeavyCf()->GetHeavyCf(),AXiKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg2->AddEntry(AXiKchM->GetKStarHeavyCf()->GetHeavyCf(),AXiKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

    TString tNewNameXiKchP = XiKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
      tNewNameXiKchP += " & " ;
      tNewNameXiKchP += XiKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
      tNewNameXiKchP += TString(cCentralityTags[XiKchP->GetCentralityType()]);
    XiKchP->GetKStarHeavyCf()->GetHeavyCf()->SetTitle(tNewNameXiKchP);

    TString tNewNameAXiKchP = AXiKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
      tNewNameAXiKchP += " & " ;
      tNewNameAXiKchP += AXiKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
      tNewNameAXiKchP += TString(cCentralityTags[AXiKchP->GetCentralityType()]);
    AXiKchP->GetKStarHeavyCf()->GetHeavyCf()->SetTitle(tNewNameAXiKchP);

    TCanvas *canKStar = new TCanvas("canKStar","canKStar");
    canKStar->Divide(2,1);


    XiKchP->DrawKStarHeavyCf((TPad*)canKStar->cd(1),2);
    XiKchM->DrawKStarHeavyCf((TPad*)canKStar->cd(1),4,"same");
    canKStar->cd(1);
    leg1->Draw();


    AXiKchP->DrawKStarHeavyCf((TPad*)canKStar->cd(2),4);
    AXiKchM->DrawKStarHeavyCf((TPad*)canKStar->cd(2),2,"same");
    canKStar->cd(2);
    leg2->Draw();

    if(bSaveFigures)
    {
      TString aName = "cXicKchKStarCfs.eps";
      canKStar->SaveAs(tSaveFiguresLocation+aName);
    }

    //----------------------------------
/*
    XiKchP->OutputPassFailInfo();
    XiKchM->OutputPassFailInfo();
    AXiKchP->OutputPassFailInfo();
    AXiKchM->OutputPassFailInfo();
*/
    //----------------------------------
    if(bSaveFile)
    {
      XiKchP->SaveAllKStarHeavyCf(mySaveFile);
      XiKchM->SaveAllKStarHeavyCf(mySaveFile);
      AXiKchP->SaveAllKStarHeavyCf(mySaveFile);
      AXiKchM->SaveAllKStarHeavyCf(mySaveFile);
    }

  }

/*
  if(bContainsKStarCfs && bDrawMC)
  {
    XiKchP->BuildKStarHeavyCf();
    AXiKchP->BuildKStarHeavyCf();
    XiKchM->BuildKStarHeavyCf();
    AXiKchM->BuildKStarHeavyCf();

    XiKchPMCTot->BuildKStarHeavyCf();
    AXiKchPMCTot->BuildKStarHeavyCf();
    XiKchMMCTot->BuildKStarHeavyCf();
    AXiKchMMCTot->BuildKStarHeavyCf();

    XiKchPMCTot->BuildKStarHeavyCfMCTrue();
    AXiKchPMCTot->BuildKStarHeavyCfMCTrue();
    XiKchMMCTot->BuildKStarHeavyCfMCTrue();
    AXiKchMMCTot->BuildKStarHeavyCfMCTrue();

    TCanvas* canKStarvMC = new TCanvas("canKStarvMC","canKStarvMC");
    canKStarvMC->Divide(2,2);

    XiKchP->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(1),2);
    XiKchPMCTot->GetKStarHeavyCf()->Rebin(2);
    XiKchPMCTot->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(1),1,"same",20);
    XiKchPMCTot->GetKStarHeavyCfMCTrue()->Rebin(2);
    XiKchPMCTot->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(1),1,"same",24);

    AXiKchM->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(2),2);
    AXiKchMMCTot->GetKStarHeavyCf()->Rebin(2);
    AXiKchMMCTot->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(2),1,"same",20);
    AXiKchMMCTot->GetKStarHeavyCfMCTrue()->Rebin(2);
    AXiKchMMCTot->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(2),1,"same",24);

    XiKchM->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(3),4);
    XiKchMMCTot->GetKStarHeavyCf()->Rebin(2);
    XiKchMMCTot->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(3),1,"same",20);
    XiKchMMCTot->GetKStarHeavyCfMCTrue()->Rebin(2);
    XiKchMMCTot->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(3),1,"same",24);

    AXiKchP->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(4),4);
    AXiKchPMCTot->GetKStarHeavyCf()->Rebin(2);
    AXiKchPMCTot->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(4),1,"same",20);
    AXiKchPMCTot->GetKStarHeavyCfMCTrue()->Rebin(2);
    AXiKchPMCTot->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(4),1,"same",24);

    //------------------------------------------------------------

    TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
      leg1->SetFillColor(0);
      leg1->AddEntry(XiKchP->GetKStarHeavyCf()->GetHeavyCf(),XiKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg1->AddEntry(XiKchPMCTot->GetKStarHeavyCf()->GetHeavyCf(),TString(XiKchPMCTot->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg1->AddEntry(XiKchPMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(XiKchPMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(1);
    leg1->Draw();

    TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
      leg2->SetFillColor(0);
      leg2->AddEntry(AXiKchM->GetKStarHeavyCf()->GetHeavyCf(),AXiKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg2->AddEntry(AXiKchMMCTot->GetKStarHeavyCf()->GetHeavyCf(),TString(AXiKchMMCTot->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg2->AddEntry(AXiKchMMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(AXiKchMMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(2);
    leg2->Draw();

    TLegend* leg3 = new TLegend(0.60,0.12,0.89,0.32);
      leg3->SetFillColor(0);
      leg3->AddEntry(XiKchM->GetKStarHeavyCf()->GetHeavyCf(),XiKchM->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg3->AddEntry(XiKchMMCTot->GetKStarHeavyCf()->GetHeavyCf(),TString(XiKchMMCTot->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg3->AddEntry(XiKchMMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(XiKchMMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(3);
    leg3->Draw();

    TLegend* leg4 = new TLegend(0.60,0.12,0.89,0.32);
      leg4->SetFillColor(0);
      leg4->AddEntry(AXiKchP->GetKStarHeavyCf()->GetHeavyCf(),AXiKchP->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg4->AddEntry(AXiKchPMCTot->GetKStarHeavyCf()->GetHeavyCf(),TString(AXiKchPMCTot->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg4->AddEntry(AXiKchPMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(AXiKchPMCTot->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(4);
    leg4->Draw();

    if(bSaveFigures)
    {
      TString aName = "cXicKchKStarvMCCfs.eps";
      canKStarvMC->SaveAs(tSaveFiguresLocation+aName);
    }

  }
*/

  if(bContainsAvgSepCfs)
  {
    XiKchP->BuildAllAvgSepHeavyCfs();
    AXiKchP->BuildAllAvgSepHeavyCfs();
    XiKchM->BuildAllAvgSepHeavyCfs();
    AXiKchM->BuildAllAvgSepHeavyCfs();

    TCanvas *canAvgSepXiKchP = new TCanvas("canAvgSepXiKchP","canAvgSepXiKchP");
    TCanvas *canAvgSepAXiKchP = new TCanvas("canAvgSepAXiKchP","canAvgSepAXiKchP");
    TCanvas *canAvgSepXiKchM = new TCanvas("canAvgSepXiKchM","canAvgSepXiKchM");
    TCanvas *canAvgSepAXiKchM = new TCanvas("canAvgSepAXiKchM","canAvgSepAXiKchM");

    canAvgSepXiKchP->Divide(1,2);
    canAvgSepAXiKchP->Divide(1,2);
    canAvgSepXiKchM->Divide(1,2);
    canAvgSepAXiKchM->Divide(1,2);

    XiKchP->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepXiKchP->cd(1));
    XiKchP->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepXiKchP->cd(2));

    AXiKchP->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepAXiKchP->cd(1));
    AXiKchP->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepAXiKchP->cd(2));

    XiKchM->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepXiKchM->cd(1));
    XiKchM->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepXiKchM->cd(2));

    AXiKchM->DrawAvgSepHeavyCf(kTrackPos,(TPad*)canAvgSepAXiKchM->cd(1));
    AXiKchM->DrawAvgSepHeavyCf(kTrackNeg,(TPad*)canAvgSepAXiKchM->cd(2));

    //----------------------------------
    if(bSaveFile)
    {
      XiKchP->SaveAllAvgSepHeavyCfs(mySaveFile);
      XiKchM->SaveAllAvgSepHeavyCfs(mySaveFile);
      AXiKchP->SaveAllAvgSepHeavyCfs(mySaveFile);
      AXiKchM->SaveAllAvgSepHeavyCfs(mySaveFile);
    }


  }




  if(bContainsKStar2dCfs)
  {
    XiKchP->BuildKStar2dHeavyCfs();
    AXiKchP->BuildKStar2dHeavyCfs();
    XiKchM->BuildKStar2dHeavyCfs();
    AXiKchM->BuildKStar2dHeavyCfs();

    TCanvas *canKStarRatios = new TCanvas("canKStarRatios","canKStarRatios");
    canKStarRatios->Divide(2,2);

    XiKchP->RebinKStar2dHeavyCfs(2);
    AXiKchM->RebinKStar2dHeavyCfs(2);
    XiKchM->RebinKStar2dHeavyCfs(2);
    AXiKchP->RebinKStar2dHeavyCfs(2);

    XiKchP->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(1));
    AXiKchM->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(2));
    XiKchM->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(3));
    AXiKchP->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(4));

    if(bSaveFigures)
    {
      TString aName = "cXicKchKStarCfRatios.eps";
      canKStarRatios->SaveAs(tSaveFiguresLocation+aName);
    }

  }







  if(bContainsSepHeavyCfs)
  {
    XiKchP->BuildAllSepHeavyCfs();
    AXiKchP->BuildAllSepHeavyCfs();
    XiKchM->BuildAllSepHeavyCfs();
    AXiKchM->BuildAllSepHeavyCfs();

    TCanvas *canSepCfsXiKchP = new TCanvas("canSepCfsXiKchP","canSepCfsXiKchP");
    TCanvas *canSepCfsAXiKchP = new TCanvas("canSepCfsAXiKchP","canSepCfsAXiKchP");
    TCanvas *canSepCfsXiKchM = new TCanvas("canSepCfsXiKchM","canSepCfsXiKchM");
    TCanvas *canSepCfsAXiKchM = new TCanvas("canSepCfsAXiKchM","canSepCfsAXiKchM");

    canSepCfsXiKchP->Divide(1,2);
    canSepCfsAXiKchP->Divide(1,2);
    canSepCfsXiKchM->Divide(1,2);
    canSepCfsAXiKchM->Divide(1,2);


    XiKchP->DrawSepHeavyCfs(kTrackPos,(TPad*)canSepCfsXiKchP->cd(1));
    XiKchP->DrawSepHeavyCfs(kTrackNeg,(TPad*)canSepCfsXiKchP->cd(2));

    AXiKchP->DrawSepHeavyCfs(kTrackPos,(TPad*)canSepCfsAXiKchP->cd(1));
    AXiKchP->DrawSepHeavyCfs(kTrackNeg,(TPad*)canSepCfsAXiKchP->cd(2));

    XiKchM->DrawSepHeavyCfs(kTrackPos,(TPad*)canSepCfsXiKchM->cd(1));
    XiKchM->DrawSepHeavyCfs(kTrackNeg,(TPad*)canSepCfsXiKchM->cd(2));

    AXiKchM->DrawSepHeavyCfs(kTrackPos,(TPad*)canSepCfsAXiKchM->cd(1));
    AXiKchM->DrawSepHeavyCfs(kTrackNeg,(TPad*)canSepCfsAXiKchM->cd(2));

  }



  if(bContainsAvgSepCowSailCfs)
  {
    XiKchP->BuildAllAvgSepCowSailHeavyCfs();
    AXiKchP->BuildAllAvgSepCowSailHeavyCfs();
    XiKchM->BuildAllAvgSepCowSailHeavyCfs();
    AXiKchM->BuildAllAvgSepCowSailHeavyCfs();

    TCanvas *canAvgSepCowSailXiKchP = new TCanvas("canAvgSepCowSailXiKchP","canAvgSepCowSailXiKchP");
    TCanvas *canAvgSepCowSailAXiKchP = new TCanvas("canAvgSepCowSailAXiKchP","canAvgSepCowSailAXiKchP");
    TCanvas *canAvgSepCowSailXiKchM = new TCanvas("canAvgSepCowSailXiKchM","canAvgSepCowSailXiKchM");
    TCanvas *canAvgSepCowSailAXiKchM = new TCanvas("canAvgSepCowSailAXiKchM","canAvgSepCowSailAXiKchM");

    canAvgSepCowSailXiKchP->Divide(1,2);
    canAvgSepCowSailAXiKchP->Divide(1,2);
    canAvgSepCowSailXiKchM->Divide(1,2);
    canAvgSepCowSailAXiKchM->Divide(1,2);

    XiKchP->DrawAvgSepCowSailHeavyCfs(kTrackPos,(TPad*)canAvgSepCowSailXiKchP->cd(1));
    XiKchP->DrawAvgSepCowSailHeavyCfs(kTrackNeg,(TPad*)canAvgSepCowSailXiKchP->cd(2));

    AXiKchP->DrawAvgSepCowSailHeavyCfs(kTrackPos,(TPad*)canAvgSepCowSailAXiKchP->cd(1));
    AXiKchP->DrawAvgSepCowSailHeavyCfs(kTrackNeg,(TPad*)canAvgSepCowSailAXiKchP->cd(2));

    XiKchM->DrawAvgSepCowSailHeavyCfs(kTrackPos,(TPad*)canAvgSepCowSailXiKchM->cd(1));
    XiKchM->DrawAvgSepCowSailHeavyCfs(kTrackNeg,(TPad*)canAvgSepCowSailXiKchM->cd(2));

    AXiKchM->DrawAvgSepCowSailHeavyCfs(kTrackPos,(TPad*)canAvgSepCowSailAXiKchM->cd(1));
    AXiKchM->DrawAvgSepCowSailHeavyCfs(kTrackNeg,(TPad*)canAvgSepCowSailAXiKchM->cd(2));

  }

  if(bViewPart1MassFail)
  {
    bool tDrawWideRangeToo = true;

    TCanvas* canPart1MassFail = new TCanvas("canPart1MassFail","canPart1MassFail");
    canPart1MassFail->Divide(2,2);

    XiKchP->DrawPart1MassFail((TPad*)canPart1MassFail->cd(1),tDrawWideRangeToo);
    AXiKchM->DrawPart1MassFail((TPad*)canPart1MassFail->cd(2),tDrawWideRangeToo);
    XiKchM->DrawPart1MassFail((TPad*)canPart1MassFail->cd(3),tDrawWideRangeToo);
    AXiKchP->DrawPart1MassFail((TPad*)canPart1MassFail->cd(4),tDrawWideRangeToo);

    if(bSaveFigures)
    {
      TString aName = "cXicKchPart1MassFail.eps";
      canPart1MassFail->SaveAs(tSaveFiguresLocation+aName);
    }
  }




  if(bContainsPurity)
  {
    XiKchP->BuildPurityCollection();
    AXiKchP->BuildPurityCollection();
    XiKchM->BuildPurityCollection();
    AXiKchM->BuildPurityCollection();

    TCanvas* canPurity = new TCanvas("canPurity","canPurity");
    canPurity->Divide(2,2);

    XiKchP->DrawAllPurityHistos((TPad*)canPurity->cd(1));
    XiKchM->DrawAllPurityHistos((TPad*)canPurity->cd(2));
    AXiKchP->DrawAllPurityHistos((TPad*)canPurity->cd(3));
    AXiKchM->DrawAllPurityHistos((TPad*)canPurity->cd(4));

    if(bSaveFigures)
    {
      TString aName = "cXicKchPurity.eps";
      canPurity->SaveAs(tSaveFiguresLocation+aName);

      TString aName2 = "XiPurity_XiKchP.eps";
      canPurity->cd(1)->SaveAs(tSaveFiguresLocation+aName2);
    }

  }
/*
  if(bContainsPurity && bDrawMC)
  {
    XiKchPMCTot->GetMCKchPurity(true);
    XiKchPMCTot->GetMCKchPurity(false);

    XiKchMMCTot->GetMCKchPurity(true);
    XiKchMMCTot->GetMCKchPurity(false);

    AXiKchPMCTot->GetMCKchPurity(true);
    AXiKchPMCTot->GetMCKchPurity(false);

    AXiKchMMCTot->GetMCKchPurity(true);
    AXiKchMMCTot->GetMCKchPurity(false);
  }
*/


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  if(bSaveFile) {mySaveFile->Close();}

  return 0;
}
