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

  TString tDirectoryBase = "/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_cLamcKch_20161025/";
  TString tFileLocationBase = tDirectoryBase+"Results_cLamcKch_Systematics_20161025";
  TString tFileLocationBaseMC = tDirectoryBase+"Results_cLamcKchMC_Systematics_20161025";

/*
  TString tDirNameModifierBase1 = "_ALLV0S_maxDcaV0Daughters_";
  vector<double> tModifierValues1 = {0.30,0.40,0.50};
*/

  TString tDirNameModifierBase1 = "_ALLV0S_minInvMassReject_";
  vector<double> tModifierValues1 = {0.494614, 0.492614, 0.488614, 0.482614};

  TString tDirNameModifierBase2 = "_ALLV0S_maxInvMassReject_";
  vector<double> tModifierValues2 = {0.500614, 0.502614, 0.506614, 0.512614};

  bool SaveImages = true;

  for(unsigned int iVal=0; iVal<tModifierValues1.size(); iVal++)
  {
    TString tDirNameModifier = tDirNameModifierBase1 + TString::Format("%0.6f",tModifierValues1[iVal]);
    if(!tDirNameModifierBase2.IsNull() && tModifierValues2.size()==tModifierValues1.size())
    {
      tDirNameModifier += tDirNameModifierBase2 + TString::Format("%0.6f",tModifierValues2[iVal]);
    }

    PlotPartnersLamKch* tLamKch0010 = new PlotPartnersLamKch(tFileLocationBase,tFileLocationBaseMC,kLamKchP,k0010,kTrainSys,2,tDirNameModifier);
    tLamKch0010->SetSaveLocationBase(tDirectoryBase);


    TCanvas* tCanPur = tLamKch0010->DrawPurity(SaveImages);

    TCanvas* tCanKStarCf = tLamKch0010->DrawKStarCfs(SaveImages);
    TCanvas* tCanKStarTrueVsRec = tLamKch0010->DrawKStarTrueVsRec(kMixed,SaveImages);

/*
    TCanvas* tCanAvgSepCfs = tLamKch0010->DrawAvgSepCfs(SaveImages);
    TCanvas* tCanAvgSepCfsLamKchP = tLamKch0010->DrawAvgSepCfs(kLamKchP,true,SaveImages);
    TCanvas* tCanAvgSepCfsLamKchM = tLamKch0010->DrawAvgSepCfs(kLamKchM,true,SaveImages);
*/

//    TCanvas* tCanPart1MassFail = tLamKch0010->ViewPart1MassFail(false,SaveImages);

    TCanvas* tCanMassAssK0_LamKchP = tLamKch0010->DrawMassAssumingK0ShortHypothesis(kLamKchP,SaveImages);
  }

cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
