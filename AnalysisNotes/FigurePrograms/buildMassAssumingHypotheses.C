#include "PlotPartnersLamKch.h"
class PlotPartnersLamKch;

#include "PlotPartnersLamK0.h"
class PlotPartnersLamK0;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------

  bool bDrawNoCutComparision = false;

  //-----Data
  TString tResultsDate_LamKch = "20161027";
  TString tDirectoryBase_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch.Data());
  TString tFileLocationBase_LamKch = tDirectoryBase_LamKch+"Results_cLamcKch_"+tResultsDate_LamKch;
  TString tFileLocationBaseMC_LamKch = tDirectoryBase_LamKch+"Results_cLamcKchMC_"+tResultsDate_LamKch;
  PlotPartnersLamKch* tLamKch0010 = new PlotPartnersLamKch(tFileLocationBase_LamKch,tFileLocationBaseMC_LamKch,kLamKchP,k0010,kTrain,2);

  if(bDrawNoCutComparision)
  {
    TString tResultsDate_LamKch_NoRm = "20161028";
    TString tDirectoryBase_LamKch_NoRm = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch_NoRm.Data());
    TString tFileLocationBase_LamKch_NoRm = tDirectoryBase_LamKch_NoRm+"Results_cLamcKch_"+tResultsDate_LamKch_NoRm;
    TString tFileLocationBaseMC_LamKch_NoRm = tDirectoryBase_LamKch_NoRm+"Results_cLamcKchMC_"+tResultsDate_LamKch_NoRm;
    PlotPartnersLamKch* tLamKch0010_NoRm = new PlotPartnersLamKch(tFileLocationBase_LamKch_NoRm,tFileLocationBaseMC_LamKch_NoRm,kLamKchP,k0010,kTrain,2);

    TH1* tMassAssK0Short_LamKchP = tLamKch0010->GetMassAssumingK0ShortHypothesis(kLamKchP);
    TH1* tMassAssK0Short_NoRm_LamKchP = tLamKch0010_NoRm->GetMassAssumingK0ShortHypothesis(kLamKchP,2);
    TCanvas* tCanMassAssK0_LamKchP = tLamKch0010->DrawMassAssumingK0ShortHypothesis(kLamKchP,tMassAssK0Short_NoRm_LamKchP,tMassAssK0Short_LamKchP,false);
  }
  else TCanvas* tCanMassAssK0_LamKchP = tLamKch0010->DrawMassAssumingK0ShortHypothesis(kLamKchP,false);

//-------------------------------------------------------------------------------

  TString tResultsDate_LamK0 = "20161027";
  TString tDirectoryBase_LamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_" + tResultsDate_LamK0 + "/";
  TString tFileLocationBase_LamK0 = tDirectoryBase_LamK0+"Results_cLamK0_" + tResultsDate_LamK0;
  TString tFileLocationBaseMC_LamK0 = tDirectoryBase_LamK0+"Results_cLamK0MC_" + tResultsDate_LamK0;
  PlotPartnersLamK0* tLamK00010 = new PlotPartnersLamK0(tFileLocationBase_LamK0,tFileLocationBaseMC_LamK0,kLamK0,k0010,kTrain,2);

  if(bDrawNoCutComparision)
  {
    TString tResultsDate_LamK0_NoRm = "20161028";
    TString tDirectoryBase_LamK0_NoRm = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_" + tResultsDate_LamK0_NoRm + "/";
    TString tFileLocationBase_LamK0_NoRm = tDirectoryBase_LamK0_NoRm+"Results_cLamK0_" + tResultsDate_LamK0_NoRm;
    TString tFileLocationBaseMC_LamK0_NoRm = tDirectoryBase_LamK0_NoRm+"Results_cLamK0MC_" + tResultsDate_LamK0_NoRm;
    PlotPartnersLamK0* tLamK00010_NoRm = new PlotPartnersLamK0(tFileLocationBase_LamK0_NoRm,tFileLocationBaseMC_LamK0_NoRm,kLamK0,k0010,kTrain,2);

    TH1* tMassAssK0Short_LamK0 = tLamK00010->GetMassAssumingK0ShortHypothesis(kLamK0);
    TH1* tMassAssK0Short_NoRm_LamK0 = tLamK00010_NoRm->GetMassAssumingK0ShortHypothesis(kLamK0,2);
    TCanvas* tCanMassAssK0_LamK0 = tLamK00010->DrawMassAssumingK0ShortHypothesis(kLamK0,tMassAssK0Short_NoRm_LamK0,tMassAssK0Short_LamK0,false);

    TH1* tMassAssLam_LamK0 = tLamK00010->GetMassAssumingLambdaHypothesis(kLamK0);
    TH1* tMassAssLam_NoRm_LamK0 = tLamK00010_NoRm->GetMassAssumingLambdaHypothesis(kLamK0,2);
    TCanvas* tCanMassAssLam_LamK0 = tLamK00010->DrawMassAssumingLambdaHypothesis(kLamK0,tMassAssLam_NoRm_LamK0,tMassAssLam_LamK0,false);

    TH1* tMassAssALam_LamK0 = tLamK00010->GetMassAssumingAntiLambdaHypothesis(kLamK0);
    TH1* tMassAssALam_NoRm_LamK0 = tLamK00010_NoRm->GetMassAssumingAntiLambdaHypothesis(kLamK0,2);
    TCanvas* tCanMassAssALam_LamK0 = tLamK00010->DrawMassAssumingAntiLambdaHypothesis(kLamK0,tMassAssALam_NoRm_LamK0,tMassAssALam_LamK0,false);
  }
  else
  {
    TCanvas* tCanMassAssK0_LamK0 = tLamK00010->DrawMassAssumingK0ShortHypothesis(kLamK0,false);
    TCanvas* tCanMassAssLam_LamK0 = tLamK00010->DrawMassAssumingLambdaHypothesis(kLamK0,false);
    TCanvas* tCanMassAssALam_LamK0 = tLamK00010->DrawMassAssumingAntiLambdaHypothesis(kLamK0,false);
  }

  tLamK00010->DrawSumMassAssumingLambdaAndAntiLambdaHypotheses(kLamK0,false);
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
