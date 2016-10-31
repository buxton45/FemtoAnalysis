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

  //-----Data
/*
  TString FileLocationBase = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRc_KchAndLamFix2_20160229/Results_cLamcKch_AsRc_KchAndLamFix2_20160229";
  TString FileLocationBaseMC = "~/Analysis/FemtoAnalysis/Results/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229/Results_cLamcKch_AsRcMC_KchAndLamFix2_20160229";
  PlotPartnersLamKch* tLamKch0010 = new PlotPartnersLamKch(FileLocationBase,FileLocationBaseMC,kLamKchP,k0010);
*/

  TString tResultsDate_LamKch = "20161022";
  TString tDirectoryBase_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch.Data());

  //TString tDirectoryBase_LamKch = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_" + tResultsDate_LamKch + "/";
  TString tFileLocationBase_LamKch = tDirectoryBase_LamKch+"Results_cLamcKch_"+tResultsDate_LamKch;
  TString tFileLocationBaseMC_LamKch = tDirectoryBase_LamKch+"Results_cLamcKchMC_"+tResultsDate_LamKch;

  PlotPartnersLamKch* tLamKch0010 = new PlotPartnersLamKch(tFileLocationBase_LamKch,tFileLocationBaseMC_LamKch,kLamKchP,k0010,kTrain,2);
  tLamKch0010->SetSaveLocationBase(tDirectoryBase_LamKch);
  bool SaveImages_LamKch = true;

  TCanvas* tCanPur_LamKch = tLamKch0010->DrawPurity(SaveImages_LamKch);
  TCanvas* tCanKStarCf_LamKch = tLamKch0010->DrawKStarCfs(SaveImages_LamKch);
  TCanvas* tCanKStarTrueVsRec_LamKch = tLamKch0010->DrawKStarTrueVsRec(kSame,SaveImages_LamKch);

  TCanvas* tCanAvgSepCfs_LamKch = tLamKch0010->DrawAvgSepCfs(SaveImages_LamKch);
  TCanvas* tCanAvgSepCfsLamKchP_LamKch = tLamKch0010->DrawAvgSepCfs(kLamKchP,true,SaveImages_LamKch);
  TCanvas* tCanAvgSepCfsLamKchM_LamKch = tLamKch0010->DrawAvgSepCfs(kLamKchM,true,SaveImages_LamKch);

//  TCanvas* tCanPart1MassFail_LamKch = tLamKch0010->ViewPart1MassFail(true,SaveImages_LamKch);

  TCanvas* tCanMassAssK0_LamKchP = tLamKch0010->DrawMassAssumingK0ShortHypothesis(kLamKchP,SaveImages_LamKch);

//-------------------------------------------------------------------------------

  TString tResultsDate_LamK0 = "20161022";
  TString tDirectoryBase_LamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_" + tResultsDate_LamK0 + "/";
  TString tFileLocationBase_LamK0 = tDirectoryBase_LamK0+"Results_cLamK0_" + tResultsDate_LamK0;
  TString tFileLocationBaseMC_LamK0 = tDirectoryBase_LamK0+"Results_cLamK0MC_" + tResultsDate_LamK0;

  PlotPartnersLamK0* tLamK00010 = new PlotPartnersLamK0(tFileLocationBase_LamK0,tFileLocationBaseMC_LamK0,kLamK0,k0010,kTrain,2);
  tLamK00010->SetSaveLocationBase(tDirectoryBase_LamK0);
  bool SaveImages_LamK0 = true;

  TCanvas* tCanPur_LamK0 = tLamK00010->DrawPurity(SaveImages_LamK0);
  TCanvas* tCanKStarCf_LamK0 = tLamK00010->DrawKStarCfs(SaveImages_LamK0);
  TCanvas* tCanKStarTrueVsRec_LamK0 = tLamK00010->DrawKStarTrueVsRec(kMixed,SaveImages_LamK0);

  TCanvas* tCanAvgSepCfs_LamK0 = tLamK00010->DrawAvgSepCfs(SaveImages_LamK0);
  TCanvas* tCanAvgSepCfsLamK0 = tLamK00010->DrawAvgSepCfs(kLamK0,SaveImages_LamK0);
  TCanvas* tCanAvgSepCfsALamK0 = tLamK00010->DrawAvgSepCfs(kALamK0,SaveImages_LamK0);

//  TCanvas* tCanPart1MassFail_LamK0 = tLamK00010->ViewPart1MassFail(false,SaveImages_LamK0);

  TCanvas* tCanMassAssK0_LamK0 = tLamK00010->DrawMassAssumingK0ShortHypothesis(kLamK0,SaveImages_LamK0);
  TCanvas* tCanMassAssLam_LamK0 = tLamK00010->DrawMassAssumingLambdaHypothesis(kLamK0,SaveImages_LamK0);
  TCanvas* tCanMassAssALam_LamK0 = tLamK00010->DrawMassAssumingAntiLambdaHypothesis(kLamK0,SaveImages_LamK0);







//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
