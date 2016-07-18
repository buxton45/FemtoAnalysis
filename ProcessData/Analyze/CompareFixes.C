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

#include "DataAndModel.h"
class DataAndModel;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  //-----Data
  TString FileLocationBaseNoFix = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_20151007/Results_cLamcKch_AsRc_20151007";
    Analysis* LamKchPNoFix = new Analysis(FileLocationBaseNoFix,kLamKchP,k0010);
    Analysis* ALamKchPNoFix = new Analysis(FileLocationBaseNoFix,kALamKchP,k0010);
    Analysis* LamKchMNoFix = new Analysis(FileLocationBaseNoFix,kLamKchM,k0010);
    Analysis* ALamKchMNoFix = new Analysis(FileLocationBaseNoFix,kALamKchM,k0010);

  TString FileLocationBaseFix1 = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_KchAndLamFix1_20160229/Results_cLamcKch_AsRc_KchAndLamFix1_20160229";
    Analysis* LamKchPFix1 = new Analysis(FileLocationBaseFix1,kLamKchP,k0010);
    Analysis* ALamKchPFix1 = new Analysis(FileLocationBaseFix1,kALamKchP,k0010);
    Analysis* LamKchMFix1 = new Analysis(FileLocationBaseFix1,kLamKchM,k0010);
    Analysis* ALamKchMFix1 = new Analysis(FileLocationBaseFix1,kALamKchM,k0010);

  TString FileLocationBaseFix2 = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_KchAndLamFix2_20160229/Results_cLamcKch_AsRc_KchAndLamFix2_20160229";
    Analysis* LamKchPFix2 = new Analysis(FileLocationBaseFix2,kLamKchP,k0010);
    Analysis* ALamKchPFix2 = new Analysis(FileLocationBaseFix2,kALamKchP,k0010);
    Analysis* LamKchMFix2 = new Analysis(FileLocationBaseFix2,kLamKchM,k0010);
    Analysis* ALamKchMFix2 = new Analysis(FileLocationBaseFix2,kALamKchM,k0010);

//-----------------------------------------------------------------------------

  bool bSaveFigures = true;
  TString tSaveFiguresLocation = "~/Analysis/Presentations/Group Meetings/20160303/";
  //-------------------------------------------------------------------

  double tColorNoFix = 1;
  double tMarkerStyleNoFix = 20;
  double tMarkerStyleoNoFix = 24;

  double tColorFix1 = 2;
  double tMarkerStyleFix1 = 21;
  double tMarkerStyleoFix1 = 25;

  double tColorFix2 = 4;
  double tMarkerStyleFix2 = 22;
  double tMarkerStyleoFix2 = 26;


  //----------------


  LamKchPNoFix->BuildKStarHeavyCf();
  ALamKchPNoFix->BuildKStarHeavyCf();
  LamKchMNoFix->BuildKStarHeavyCf();
  ALamKchMNoFix->BuildKStarHeavyCf();

  LamKchPFix1->BuildKStarHeavyCf();
  ALamKchPFix1->BuildKStarHeavyCf();
  LamKchMFix1->BuildKStarHeavyCf();
  ALamKchMFix1->BuildKStarHeavyCf();

  LamKchPFix2->BuildKStarHeavyCf();
  ALamKchPFix2->BuildKStarHeavyCf();
  LamKchMFix2->BuildKStarHeavyCf();
  ALamKchMFix2->BuildKStarHeavyCf();

  //-----------------------------------

  TLegend* legLamKchP = new TLegend(0.60,0.12,0.89,0.32);
    legLamKchP->SetFillColor(0);
    legLamKchP->AddEntry(LamKchPNoFix->GetKStarHeavyCf()->GetHeavyCf(),"NoFix","lp");
    legLamKchP->AddEntry(LamKchPFix1->GetKStarHeavyCf()->GetHeavyCf(),"Fix1","lp");
    legLamKchP->AddEntry(LamKchPFix2->GetKStarHeavyCf()->GetHeavyCf(),"Fix2","lp");

  TLegend* legALamKchP = new TLegend(0.60,0.12,0.89,0.32);
    legALamKchP->SetFillColor(0);
    legALamKchP->AddEntry(ALamKchPNoFix->GetKStarHeavyCf()->GetHeavyCf(),"NoFix","lp");
    legALamKchP->AddEntry(ALamKchPFix1->GetKStarHeavyCf()->GetHeavyCf(),"Fix1","lp");
    legALamKchP->AddEntry(ALamKchPFix2->GetKStarHeavyCf()->GetHeavyCf(),"Fix2","lp");

  TLegend* legLamKchM = new TLegend(0.60,0.12,0.89,0.32);
    legLamKchM->SetFillColor(0);
    legLamKchM->AddEntry(LamKchMNoFix->GetKStarHeavyCf()->GetHeavyCf(),"NoFix","lp");
    legLamKchM->AddEntry(LamKchMFix1->GetKStarHeavyCf()->GetHeavyCf(),"Fix1","lp");
    legLamKchM->AddEntry(LamKchMFix2->GetKStarHeavyCf()->GetHeavyCf(),"Fix2","lp");

  TLegend* legALamKchM = new TLegend(0.60,0.12,0.89,0.32);
    legALamKchM->SetFillColor(0);
    legALamKchM->AddEntry(ALamKchMNoFix->GetKStarHeavyCf()->GetHeavyCf(),"NoFix","lp");
    legALamKchM->AddEntry(ALamKchMFix1->GetKStarHeavyCf()->GetHeavyCf(),"Fix1","lp");
    legALamKchM->AddEntry(ALamKchMFix2->GetKStarHeavyCf()->GetHeavyCf(),"Fix2","lp");


  //-----------------------------
  TCanvas *canKStar = new TCanvas("canKStar","canKStar");
  canKStar->Divide(2,2);

  LamKchPNoFix->DrawKStarHeavyCf((TPad*)canKStar->cd(1),tColorNoFix,"",tMarkerStyleNoFix);
  LamKchPFix1->DrawKStarHeavyCf((TPad*)canKStar->cd(1),tColorFix1,"same",tMarkerStyleFix1);
  LamKchPFix2->DrawKStarHeavyCf((TPad*)canKStar->cd(1),tColorFix2,"same",tMarkerStyleFix2);
  canKStar->cd(1);
  legLamKchP->Draw();

  ALamKchMNoFix->DrawKStarHeavyCf((TPad*)canKStar->cd(2),tColorNoFix,"",tMarkerStyleNoFix);
  ALamKchMFix1->DrawKStarHeavyCf((TPad*)canKStar->cd(2),tColorFix1,"same",tMarkerStyleFix1);
  ALamKchMFix2->DrawKStarHeavyCf((TPad*)canKStar->cd(2),tColorFix2,"same",tMarkerStyleFix2);
  canKStar->cd(2);
  legALamKchM->Draw();

  LamKchMNoFix->DrawKStarHeavyCf((TPad*)canKStar->cd(3),tColorNoFix,"",tMarkerStyleNoFix);
  LamKchMFix1->DrawKStarHeavyCf((TPad*)canKStar->cd(3),tColorFix1,"same",tMarkerStyleFix1);
  LamKchMFix2->DrawKStarHeavyCf((TPad*)canKStar->cd(3),tColorFix2,"same",tMarkerStyleFix2);
  canKStar->cd(3);
  legLamKchM->Draw();

  ALamKchPNoFix->DrawKStarHeavyCf((TPad*)canKStar->cd(4),tColorNoFix,"",tMarkerStyleNoFix);
  ALamKchPFix1->DrawKStarHeavyCf((TPad*)canKStar->cd(4),tColorFix1,"same",tMarkerStyleFix1);
  ALamKchPFix2->DrawKStarHeavyCf((TPad*)canKStar->cd(4),tColorFix2,"same",tMarkerStyleFix2);
  canKStar->cd(4);
  legALamKchP->Draw();




  if(bSaveFigures)
  {
    TString aName = "cLamcKchKStarCfs.eps";
    canKStar->SaveAs(tSaveFiguresLocation+aName);
  }


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
