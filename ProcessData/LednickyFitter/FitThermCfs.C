#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TApplication.h"
#include "TStyle.h"
#include "TPaveText.h"
#include "TLegend.h"

#include "ThermCf.h"
#include "FitPartialAnalysis.h"



//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  AnalysisType tAnType = kLamKchP;

  bool bCombineConjugates = true;
  bool bSaveFigures = false;

  int tRebin=2;
  double tMinNorm = /*0.80*//*0.80*/0.32;
  double tMaxNorm = /*0.99*//*0.99*/0.40;

  int tImpactParam = 3;


  TString tFilaNameBase = "CorrelationFunctions";
  TString tFileNameModifier = "";
//  TString tFileNameModifier = "_WeightParentsInteraction";
//  TString tFileNameModifier = "_WeightParentsInteraction_OnlyWeightLongDecayParents";
//  TString tFileNameModifier = "_WeightParentsInteraction_NoCharged";


  TString tFileName = TString::Format("%s%s.root", tFilaNameBase.Data(), tFileNameModifier.Data());



  //--------------------------------------------

  TH1* tThermCf = ThermCf::GetThermCf(tFileName, "PrimaryOnly", tAnType, 3, true, kMe, tRebin, tMinNorm, tMaxNorm);

  TCanvas* tCan = new TCanvas("tCan", "tCan");
  tCan->cd();

  tThermCf->Draw();

  //--------------------------------------------
  double tRef0, tImf0, td0;
  if(tAnType == kLamKchP)
  {
    tRef0 = -1.16;
    tImf0 = 0.51;
    td0 = 1.08;
  }
  else if(tAnType == kLamKchM)
  {
    tRef0 = 0.41;
    tImf0 = 0.47;
    td0 = -4.89;
  }
  else if(tAnType == kLamK0)
  {
    tRef0 = -0.41;
    tImf0 = 0.20;
    td0 = 2.08;
  }
  else assert(0);

  int tNFitParams = 5;
  TString tFitName = "tFitFcn";
  TF1* tFitFcn = new TF1(tFitName, FitPartialAnalysis::LednickyEqWithNorm,0.,0.5,tNFitParams+1);

    tFitFcn->SetParameter(0, 1.);
    tFitFcn->SetParameter(1, 5.);

    tFitFcn->FixParameter(2, tRef0);
    tFitFcn->FixParameter(3, tImf0);
    tFitFcn->FixParameter(4, td0);

    tFitFcn->SetParameter(5, 1.);

  tThermCf->Fit(tFitName, "0", "", 0.0, 0.3);

  tFitFcn->Draw("same");

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
