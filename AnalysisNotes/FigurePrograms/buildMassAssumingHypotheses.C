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
  bool bDrawSum = false;
  TString tSaveLocationBase = "/home/jesse/Analysis/FemtoAnalysis/AnalysisNotes/3_DataSelection/Figures/MassAssHypotheses/";
  bool bSaveImages = true;

  AnalysisType tAnTypeLamKch = kLamKchP;
  AnalysisType tAnTypeLamK0 = kLamK0;

  //-----Data
  TString tResultsDate_LamKch = "20161027";
  TString tDirectoryBase_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch.Data());
  TString tFileLocationBase_LamKch = tDirectoryBase_LamKch+"Results_cLamcKch_"+tResultsDate_LamKch;
  TString tFileLocationBaseMC_LamKch = tDirectoryBase_LamKch+"Results_cLamcKchMC_"+tResultsDate_LamKch;
  PlotPartnersLamKch* tLamKch0010 = new PlotPartnersLamKch(tFileLocationBase_LamKch,tFileLocationBaseMC_LamKch,tAnTypeLamKch,k0010,kTrain,2);
  tLamKch0010->SetSaveLocationBase(tSaveLocationBase);

  if(bDrawNoCutComparision)
  {
    TString tResultsDate_LamKch_NoRm = "20161028";
    TString tDirectoryBase_LamKch_NoRm = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch_NoRm.Data());
    TString tFileLocationBase_LamKch_NoRm = tDirectoryBase_LamKch_NoRm+"Results_cLamcKch_"+tResultsDate_LamKch_NoRm;
    TString tFileLocationBaseMC_LamKch_NoRm = tDirectoryBase_LamKch_NoRm+"Results_cLamcKchMC_"+tResultsDate_LamKch_NoRm;
    PlotPartnersLamKch* tLamKch0010_NoRm = new PlotPartnersLamKch(tFileLocationBase_LamKch_NoRm,tFileLocationBaseMC_LamKch_NoRm,tAnTypeLamKch,k0010,kTrain,2);

    TH1* tMassAssK0Short_LamKchP = tLamKch0010->GetMassAssumingK0ShortHypothesis(tAnTypeLamKch);
    TH1* tMassAssK0Short_NoRm_LamKchP = tLamKch0010_NoRm->GetMassAssumingK0ShortHypothesis(tAnTypeLamKch,2);
    TCanvas* tCanMassAssK0_LamKchP = tLamKch0010->DrawMassAssumingK0ShortHypothesis(tAnTypeLamKch,tMassAssK0Short_NoRm_LamKchP,tMassAssK0Short_LamKchP,bSaveImages);
  }
  else TCanvas* tCanMassAssK0_LamKchP = tLamKch0010->DrawMassAssumingK0ShortHypothesis(tAnTypeLamKch,bSaveImages);

//-------------------------------------------------------------------------------

  TString tResultsDate_LamK0 = "20161027";
  TString tDirectoryBase_LamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_" + tResultsDate_LamK0 + "/";
  TString tFileLocationBase_LamK0 = tDirectoryBase_LamK0+"Results_cLamK0_" + tResultsDate_LamK0;
  TString tFileLocationBaseMC_LamK0 = tDirectoryBase_LamK0+"Results_cLamK0MC_" + tResultsDate_LamK0;
  PlotPartnersLamK0* tLamK00010 = new PlotPartnersLamK0(tFileLocationBase_LamK0,tFileLocationBaseMC_LamK0,tAnTypeLamK0,k0010,kTrain,2);
  tLamK00010->SetSaveLocationBase(tSaveLocationBase);

  if(bDrawNoCutComparision)
  {
    TString tResultsDate_LamK0_NoRm = "20161028";
    TString tDirectoryBase_LamK0_NoRm = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_" + tResultsDate_LamK0_NoRm + "/";
    TString tFileLocationBase_LamK0_NoRm = tDirectoryBase_LamK0_NoRm+"Results_cLamK0_" + tResultsDate_LamK0_NoRm;
    TString tFileLocationBaseMC_LamK0_NoRm = tDirectoryBase_LamK0_NoRm+"Results_cLamK0MC_" + tResultsDate_LamK0_NoRm;
    PlotPartnersLamK0* tLamK00010_NoRm = new PlotPartnersLamK0(tFileLocationBase_LamK0_NoRm,tFileLocationBaseMC_LamK0_NoRm,tAnTypeLamK0,k0010,kTrain,2);

    TH1* tMassAssK0Short_LamK0 = tLamK00010->GetMassAssumingK0ShortHypothesis(tAnTypeLamK0);
    TH1* tMassAssK0Short_NoRm_LamK0 = tLamK00010_NoRm->GetMassAssumingK0ShortHypothesis(tAnTypeLamK0,2);
    TCanvas* tCanMassAssK0_LamK0 = tLamK00010->DrawMassAssumingK0ShortHypothesis(tAnTypeLamK0,tMassAssK0Short_NoRm_LamK0,tMassAssK0Short_LamK0,bSaveImages);

    TH1* tMassAssLam_LamK0 = tLamK00010->GetMassAssumingLambdaHypothesis(tAnTypeLamK0);
    TH1* tMassAssLam_NoRm_LamK0 = tLamK00010_NoRm->GetMassAssumingLambdaHypothesis(tAnTypeLamK0,2);
    TCanvas* tCanMassAssLam_LamK0 = tLamK00010->DrawMassAssumingLambdaHypothesis(tAnTypeLamK0,tMassAssLam_NoRm_LamK0,tMassAssLam_LamK0,bSaveImages);

    TH1* tMassAssALam_LamK0 = tLamK00010->GetMassAssumingAntiLambdaHypothesis(tAnTypeLamK0);
    TH1* tMassAssALam_NoRm_LamK0 = tLamK00010_NoRm->GetMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,2);
    TCanvas* tCanMassAssALam_LamK0 = tLamK00010->DrawMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,tMassAssALam_NoRm_LamK0,tMassAssALam_LamK0,bSaveImages);
  }
  else
  {
    TCanvas* tCanMassAssK0_LamK0 = tLamK00010->DrawMassAssumingK0ShortHypothesis(tAnTypeLamK0,bSaveImages);
    TCanvas* tCanMassAssLam_LamK0 = tLamK00010->DrawMassAssumingLambdaHypothesis(tAnTypeLamK0,bSaveImages);
    TCanvas* tCanMassAssALam_LamK0 = tLamK00010->DrawMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,bSaveImages);
  }

  if(bDrawSum) tLamK00010->DrawSumMassAssumingLambdaAndAntiLambdaHypotheses(tAnTypeLamK0,bSaveImages);
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
