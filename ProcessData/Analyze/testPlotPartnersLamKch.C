#include "PlotPartnersLamKch.h"
class PlotPartnersLamKch;

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

  TString FileLocationBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161012/Results_cLamcKch_20161012";
  TString FileLocationBaseMC = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161012/Results_cLamcKchMC_20161012";
  PlotPartnersLamKch* tLamKch0010 = new PlotPartnersLamKch(FileLocationBase,FileLocationBaseMC,kLamKchP,k0010,2,true);


  TCanvas* tCanPur = tLamKch0010->DrawPurity();
  TCanvas* tCanKStarCf = tLamKch0010->DrawKStarCfs();
  TCanvas* tCanKStarTrueVsRec = tLamKch0010->DrawKStarTrueVsRec(kMixed);
  TCanvas* tCanAvgSepCfs = tLamKch0010->DrawAvgSepCfs();
  TCanvas* tCanPart1MassFail = tLamKch0010->ViewPart1MassFail(false);

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
