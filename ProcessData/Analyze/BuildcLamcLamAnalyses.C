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


  TString FileLocationBase = "~/Analysis/K0Lam/Results_cLamcLam_AsRc_20150923/Results_cLamcLam_AsRc_20150923";

  Analysis* LamLam = new Analysis(FileLocationBase,kLamLam,k0010);
  Analysis* ALamALam = new Analysis(FileLocationBase,kALamALam,k0010);
  Analysis* LamALam = new Analysis(FileLocationBase,kLamALam,k0010);


  //--------Jai's Plots-----------------------------------------------------------------------------
  TString JaiLamLamLocation = "~/Analysis/K0Lam/Results_cLamcLam_AsRc_20150923/JaicfsLamLam.root";
  TFile tFileJaiLamLam(JaiLamLamLocation);
  TH1D* tJaiLamLam0010 = (TH1D*)tFileJaiLamLam.Get("LamLam0-10");
  tJaiLamLam0010->SetDirectory(0);
  tFileJaiLamLam.Close();
    tJaiLamLam0010->SetMarkerStyle(20);
    tJaiLamLam0010->SetMarkerColor(2);
    tJaiLamLam0010->SetLineColor(2);

  TString JaiALamALamLocation = "~/Analysis/K0Lam/Results_cLamcLam_AsRc_20150923/JaicfsALamALam.root";
  TFile tFileJaiALamALam(JaiALamALamLocation);
  TH1D* tJaiALamALam0010 = (TH1D*)tFileJaiALamALam.Get("ALamALam0-10");
  tJaiALamALam0010->SetDirectory(0);
  tFileJaiALamALam.Close();
    tJaiALamALam0010->SetMarkerStyle(20);
    tJaiALamALam0010->SetMarkerColor(2);
    tJaiALamALam0010->SetLineColor(2);

  TString JaiLamALamLocation = "~/Analysis/K0Lam/Results_cLamcLam_AsRc_20150923/JaicfsLamALamKstarMomCorrected.root";
  TFile tFileJaiLamALam(JaiLamALamLocation);
  TH1D* tJaiLamALam0010 = (TH1D*)tFileJaiLamALam.Get("LamALam0-10centrality_varBin5BothFieldsKstarMomCorrected");
  tJaiLamALam0010->SetDirectory(0);
  tFileJaiLamALam.Close();
    tJaiLamALam0010->SetMarkerStyle(20);
    tJaiLamALam0010->SetMarkerColor(2);
    tJaiLamALam0010->SetLineColor(2);
  //------------------------------------------------------------------------------------------------

  //bool bContainsPurity = true;
  bool bContainsKStarCfs = true;
/*
  bool bContainsAvgSepCfs = true;
  bool bContainsKStar2dCfs = true;
  bool bContainsSepHeavyCfs = false;
  bool bContainsAvgSepCowSailCfs = false;
*/

  //-------------------------------------------------------------------
  if(bContainsKStarCfs)
  {
    LamLam->BuildKStarHeavyCf(0.3,0.5);
    ALamALam->BuildKStarHeavyCf(0.3,0.5);
    LamALam->BuildKStarHeavyCf(0.3,0.5);

    LamLam->GetKStarHeavyCf()->Rebin(2);
    ALamALam->GetKStarHeavyCf()->Rebin(2);
    LamALam->GetKStarHeavyCf()->Rebin(2);

    TCanvas *canKStar = new TCanvas("canKStar","canKStar");
    canKStar->Divide(1,3);


    LamLam->DrawKStarHeavyCf((TPad*)canKStar->cd(1));
    tJaiLamLam0010->Draw("same");    

    ALamALam->DrawKStarHeavyCf((TPad*)canKStar->cd(2));
    tJaiALamALam0010->Draw("same");

    LamALam->DrawKStarHeavyCf((TPad*)canKStar->cd(3));
    tJaiLamALam0010->Draw("same");

    //----------------------------------

    LamLam->OutputPassFailInfo();
    ALamALam->OutputPassFailInfo();
    LamALam->OutputPassFailInfo();



  }

  TCanvas* canPart1MassFail = new TCanvas("canPart1MassFail","canPart1MassFail");
  LamLam->DrawPart1MassFail(canPart1MassFail,true);


//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
  return 0;
}
