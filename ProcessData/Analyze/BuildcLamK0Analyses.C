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

//  TString tResultsDate = "20161027";
  TString tResultsDate = "20171227";

  AnalysisType tAnType = kLamK0;
  AnalysisType tConjAnType = kALamK0;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = k0010;  //TODO

//-----------------------------------------------------------------------------
  bool bSaveFigures = false;
  bool bSaveFile = false;

  bool bContainsPurity = true;
  bool bContainsKStarCfs = false;
  bool bContainsAvgSepCfs = false;

  bool bContainsKStar2dCfs = false;

  bool bContainsSepHeavyCfs = false;
  bool bContainsAvgSepCowSailCfs = false;

  bool bViewPart1MassFail = false;  //NOTE: kTrainSys do not include fail cut monitors

  bool bDrawMC = false;

//-----------------------------------------------------------------------------

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  //TODO PreTrain results with MCd?

  TString tSaveDirectoryBase = "/home/jesse/Analysis/Presentations/PWGCF/LamKPaperProposal/ALICE_MiniWeek_20180117/Figures/";
//  TString tSaveDirectoryBase = tDirectoryBase;

  TFile *mySaveFile;
  TString tSaveFileName = TString::Format("%sResults_%s_%s.root", tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  if(bSaveFile) {mySaveFile = new TFile(tSaveFileName, "RECREATE");}

  //-----Data
  Analysis* LamK0 = new Analysis(tFileLocationBase,tAnType,tCentType);
  Analysis* ALamK0 = new Analysis(tFileLocationBase,tConjAnType,tCentType);

  //-----MC
  Analysis* LamK0MC = new Analysis(tFileLocationBaseMC,tAnType,tCentType);
  Analysis* ALamK0MC = new Analysis(tFileLocationBaseMC,tConjAnType,tCentType);

  //-------------------------------------------------------------------

  if(bContainsKStarCfs)
  {
    LamK0->BuildKStarHeavyCf();
    ALamK0->BuildKStarHeavyCf();

    TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
      leg1->SetFillColor(0);
      leg1->AddEntry(LamK0->GetKStarHeavyCf()->GetHeavyCf(),LamK0->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg1->AddEntry(ALamK0->GetKStarHeavyCf()->GetHeavyCf(),ALamK0->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");

    TString tNewNameLamK0 = LamK0->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
      tNewNameLamK0 += " & " ;
      tNewNameLamK0 += ALamK0->GetKStarHeavyCf()->GetHeavyCf()->GetTitle();
    LamK0->GetKStarHeavyCf()->GetHeavyCf()->SetTitle(tNewNameLamK0);

    TCanvas *canKStar = new TCanvas("canKStar","canKStar");
    LamK0->DrawKStarHeavyCf(canKStar,2);
    ALamK0->DrawKStarHeavyCf(canKStar,4,"same");
    leg1->Draw();

    if(bSaveFigures)
    {
      TString aName = "cLamK0KStarCfs.eps";
      canKStar->SaveAs(tSaveDirectoryBase+aName);
    }


    //----------------------------------
    LamK0->OutputPassFailInfo();
    ALamK0->OutputPassFailInfo();
    //----------------------------------
    if(bSaveFile)
    {
      LamK0->SaveAllKStarHeavyCf(mySaveFile);
      ALamK0->SaveAllKStarHeavyCf(mySaveFile);
    }

  }

  if(bContainsKStarCfs && bDrawMC)
  {
    LamK0->BuildKStarHeavyCf();
    ALamK0->BuildKStarHeavyCf();

    LamK0MC->BuildKStarHeavyCf();
    ALamK0MC->BuildKStarHeavyCf();

    LamK0MC->BuildKStarHeavyCfMCTrue();
    ALamK0MC->BuildKStarHeavyCfMCTrue();

    TCanvas* canKStarvMC = new TCanvas("canKStarvMC","canKStarvMC");
    canKStarvMC->Divide(1,2);

    LamK0->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(1),2);
    LamK0MC->GetKStarHeavyCf()->Rebin(4);
    LamK0MC->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(1),1,"same",20);
    LamK0MC->GetKStarHeavyCfMCTrue()->Rebin(4);
    LamK0MC->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(1),1,"same",24);

    ALamK0->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(2),4);
    ALamK0MC->GetKStarHeavyCf()->Rebin(4);
    ALamK0MC->DrawKStarHeavyCf((TPad*)canKStarvMC->cd(2),1,"same",20);
    ALamK0MC->GetKStarHeavyCfMCTrue()->Rebin(4);
    ALamK0MC->DrawKStarHeavyCfMCTrue((TPad*)canKStarvMC->cd(2),1,"same",24);

    //------------------------------------------------------------

    TLegend* leg1 = new TLegend(0.60,0.12,0.89,0.32);
      leg1->SetFillColor(0);
      leg1->AddEntry(LamK0->GetKStarHeavyCf()->GetHeavyCf(),LamK0->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg1->AddEntry(LamK0MC->GetKStarHeavyCf()->GetHeavyCf(),TString(LamK0MC->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg1->AddEntry(LamK0MC->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(LamK0MC->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(1);
    leg1->Draw();

    TLegend* leg2 = new TLegend(0.60,0.12,0.89,0.32);
      leg2->SetFillColor(0);
      leg2->AddEntry(ALamK0->GetKStarHeavyCf()->GetHeavyCf(),ALamK0->GetKStarHeavyCf()->GetHeavyCf()->GetTitle(),"lp");
      leg2->AddEntry(ALamK0MC->GetKStarHeavyCf()->GetHeavyCf(),TString(ALamK0MC->GetKStarHeavyCf()->GetHeavyCf()->GetTitle())+"_{MC}","lp");
      leg2->AddEntry(ALamK0MC->GetKStarHeavyCfMCTrue()->GetHeavyCf(),TString(ALamK0MC->GetKStarHeavyCfMCTrue()->GetHeavyCf()->GetTitle())+"_{MCTrue}","lp");
    canKStarvMC->cd(2);
    leg2->Draw();

    if(bSaveFigures)
    {
      TString aName = "cLamK0KStarvMCCfs.eps";
      canKStarvMC->SaveAs(tSaveDirectoryBase+aName);
    }

  }



  if(bContainsAvgSepCfs)
  {
    LamK0->BuildAllAvgSepHeavyCfs();
    ALamK0->BuildAllAvgSepHeavyCfs();

    TCanvas *canAvgSepLamK0 = new TCanvas("canAvgSepLamK0","canAvgSepLamK0");
    TCanvas *canAvgSepALamK0 = new TCanvas("canAvgSepALamK0","canAvgSepALamK0");

    canAvgSepLamK0->Divide(2,2);
    canAvgSepALamK0->Divide(2,2);


    LamK0->DrawAvgSepHeavyCf(kPosPos,(TPad*)canAvgSepLamK0->cd(1));
    LamK0->DrawAvgSepHeavyCf(kPosNeg,(TPad*)canAvgSepLamK0->cd(2));
    LamK0->DrawAvgSepHeavyCf(kNegPos,(TPad*)canAvgSepLamK0->cd(3));
    LamK0->DrawAvgSepHeavyCf(kNegNeg,(TPad*)canAvgSepLamK0->cd(4));

    ALamK0->DrawAvgSepHeavyCf(kPosPos,(TPad*)canAvgSepALamK0->cd(1));
    ALamK0->DrawAvgSepHeavyCf(kPosNeg,(TPad*)canAvgSepALamK0->cd(2));
    ALamK0->DrawAvgSepHeavyCf(kNegPos,(TPad*)canAvgSepALamK0->cd(3));
    ALamK0->DrawAvgSepHeavyCf(kNegNeg,(TPad*)canAvgSepALamK0->cd(4));

    //----------------------------------
    if(bSaveFile)
    {
      LamK0->SaveAllAvgSepHeavyCfs(mySaveFile);
      ALamK0->SaveAllAvgSepHeavyCfs(mySaveFile);
    }

  }



  if(bContainsKStar2dCfs)
  {
    LamK0->BuildKStar2dHeavyCfs();
    ALamK0->BuildKStar2dHeavyCfs();

    TCanvas *canKStarRatios = new TCanvas("canKStarRatios","canKStarRatios");
    canKStarRatios->Divide(1,2);

    LamK0->RebinKStar2dHeavyCfs(4);
    ALamK0->RebinKStar2dHeavyCfs(4);

    LamK0->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(1));
    ALamK0->DrawKStar2dHeavyCfRatios((TPad*)canKStarRatios->cd(2));


    if(bSaveFigures)
    {
      TString aName = "cLamK0KStarCfRatios.eps";
      canKStarRatios->SaveAs(tSaveDirectoryBase+aName);
    }
  }






  if(bContainsSepHeavyCfs)
  {
    LamK0->BuildAllSepHeavyCfs();
    ALamK0->BuildAllSepHeavyCfs();

    TCanvas *canSepCfsLamK0 = new TCanvas("canSepCfsLamK0","canSepCfsLamK0");
    TCanvas *canSepCfsALamK0 = new TCanvas("canSepCfsALamK0","canSepCfsALamK0");

    canSepCfsLamK0->Divide(2,2);
    canSepCfsALamK0->Divide(2,2);

    LamK0->DrawSepHeavyCfs(kPosPos,(TPad*)canSepCfsLamK0->cd(1));
    LamK0->DrawSepHeavyCfs(kPosNeg,(TPad*)canSepCfsLamK0->cd(2));
    LamK0->DrawSepHeavyCfs(kNegPos,(TPad*)canSepCfsLamK0->cd(3));
    LamK0->DrawSepHeavyCfs(kNegNeg,(TPad*)canSepCfsLamK0->cd(4));

    ALamK0->DrawSepHeavyCfs(kPosPos,(TPad*)canSepCfsALamK0->cd(1));
    ALamK0->DrawSepHeavyCfs(kPosNeg,(TPad*)canSepCfsALamK0->cd(2));
    ALamK0->DrawSepHeavyCfs(kNegPos,(TPad*)canSepCfsALamK0->cd(3));
    ALamK0->DrawSepHeavyCfs(kNegNeg,(TPad*)canSepCfsALamK0->cd(4));



  }


  if(bContainsAvgSepCowSailCfs)
  {
    LamK0->BuildAllAvgSepCowSailHeavyCfs();
    ALamK0->BuildAllAvgSepCowSailHeavyCfs();

    TCanvas *canAvgSepCowSailLamK0 = new TCanvas("canAvgSepCowSailLamK0","canAvgSepCowSailLamK0");
    TCanvas *canAvgSepCowSailALamK0 = new TCanvas("canAvgSepCowSailALamK0","canAvgSepCowSailALamK0");

    canAvgSepCowSailLamK0->Divide(2,2);
    canAvgSepCowSailALamK0->Divide(2,2);

    LamK0->DrawAvgSepCowSailHeavyCfs(kPosPos,(TPad*)canAvgSepCowSailLamK0->cd(1));
    LamK0->DrawAvgSepCowSailHeavyCfs(kPosNeg,(TPad*)canAvgSepCowSailLamK0->cd(2));
    LamK0->DrawAvgSepCowSailHeavyCfs(kNegPos,(TPad*)canAvgSepCowSailLamK0->cd(3));
    LamK0->DrawAvgSepCowSailHeavyCfs(kNegNeg,(TPad*)canAvgSepCowSailLamK0->cd(4));

    ALamK0->DrawAvgSepCowSailHeavyCfs(kPosPos,(TPad*)canAvgSepCowSailALamK0->cd(1));
    ALamK0->DrawAvgSepCowSailHeavyCfs(kPosNeg,(TPad*)canAvgSepCowSailALamK0->cd(2));
    ALamK0->DrawAvgSepCowSailHeavyCfs(kNegPos,(TPad*)canAvgSepCowSailALamK0->cd(3));
    ALamK0->DrawAvgSepCowSailHeavyCfs(kNegNeg,(TPad*)canAvgSepCowSailALamK0->cd(4));

  }


  if(bViewPart1MassFail)
  {
    bool tDrawWideRangeToo = true;

    TCanvas* canPart1MassFail = new TCanvas("canPart1MassFail","canPart1MassFail");
    canPart1MassFail->Divide(1,2);

    LamK0->DrawPart1MassFail((TPad*)canPart1MassFail->cd(1),tDrawWideRangeToo);
    ALamK0->DrawPart1MassFail((TPad*)canPart1MassFail->cd(2),tDrawWideRangeToo);

    if(bSaveFigures)
    {
      TString aName = "cLamK0Part1MassFail.eps";
      canPart1MassFail->SaveAs(tSaveDirectoryBase+aName);
    }
  }




  if(bContainsPurity)
  {
    LamK0->BuildPurityCollection();
    ALamK0->BuildPurityCollection();

    TCanvas* canPurity = new TCanvas("canPurity","canPurity");
    canPurity->Divide(2,1);

    LamK0->DrawAllPurityHistos((TPad*)canPurity->cd(1));
    ALamK0->DrawAllPurityHistos((TPad*)canPurity->cd(2));

    if(bSaveFigures)
    {
      TString aName = "cLamK0Purity.eps";
      canPurity->SaveAs(tSaveDirectoryBase+aName);

      TString aName2 = "LamPurity_LamK0.eps";
      canPurity->cd(1)->cd(1)->SaveAs(tSaveDirectoryBase+aName2);

      TString aName3 = "K0Purity_LamK0.eps";
      canPurity->cd(1)->cd(2)->SaveAs(tSaveDirectoryBase+aName3);
    }

  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  if(bSaveFile) {mySaveFile->Close();}

  return 0;
}
