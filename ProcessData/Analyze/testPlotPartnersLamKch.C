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

  TString DirectoryBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_20161022/";
  TString FileLocationBase = DirectoryBase+"Results_cLamcKch_20161022";
  TString FileLocationBaseMC = DirectoryBase+"/Results_cLamcKchMC_20161022";
  PlotPartnersLamKch* tLamKch0010 = new PlotPartnersLamKch(FileLocationBase,FileLocationBaseMC,kLamKchP,k0010,kTrain,2);
  tLamKch0010->SetSaveLocationBase(DirectoryBase);
  bool SaveImages = true;

  TCanvas* tCanPur = tLamKch0010->DrawPurity(SaveImages);
  TCanvas* tCanKStarCf = tLamKch0010->DrawKStarCfs(SaveImages);
  TCanvas* tCanKStarTrueVsRec = tLamKch0010->DrawKStarTrueVsRec(kMixed,SaveImages);

  TCanvas* tCanAvgSepCfs = tLamKch0010->DrawAvgSepCfs(SaveImages);
  TCanvas* tCanAvgSepCfsLamKchP = tLamKch0010->DrawAvgSepCfs(kLamKchP,true,SaveImages);
  TCanvas* tCanAvgSepCfsLamKchM = tLamKch0010->DrawAvgSepCfs(kLamKchM,true,SaveImages);

//  TCanvas* tCanPart1MassFail = tLamKch0010->ViewPart1MassFail(false,SaveImages);

  TCanvas* tCanMassAssK0_LamKchP = tLamKch0010->DrawMassAssumingK0ShortHypothesis(kLamKchP,SaveImages);

//-------------------------------------------------------------------------------


  TString FileLocationBase_LamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_20161022/Results_cLamK0_20161022";
  TString FileLocationBaseMC_LamK0 = "/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_20161022/Results_cLamK0MC_20161022";
  PlotPartnersLamK0* tLamK00010 = new PlotPartnersLamK0(FileLocationBase_LamK0,FileLocationBaseMC_LamK0,kLamK0,k0010,kTrain,2);


  TCanvas* tCanMassAssK0_LamK0 = tLamK00010->DrawMassAssumingK0ShortHypothesis(kLamK0);
  TCanvas* tCanMassAssLam_LamK0 = tLamK00010->DrawMassAssumingLambdaHypothesis(kLamK0);
  TCanvas* tCanMassAssALam_LamK0 = tLamK00010->DrawMassAssumingAntiLambdaHypothesis(kLamK0);







//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
