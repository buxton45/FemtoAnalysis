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

  TString tResultsDate = "20161026";

/*
  TString tDirNameModifierBase1 = "_ALLV0S_minInvMassReject_";
  vector<double> tModifierValues1 = {0.494614, 0.492614, 0.488614, 0.482614};

  TString tDirNameModifierBase2 = "_ALLV0S_maxInvMassReject_";
  vector<double> tModifierValues2 = {0.500614, 0.502614, 0.506614, 0.512614};
*/

  TString tDirNameModifierBase1 = "_ALLV0S_maxDcaV0Daughters_";
  vector<double> tModifierValues1 = {0.30,0.40,0.50};

  TString tDirNameModifierBase2 = "";
  vector<double> tModifierValues2 = {};


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_cLamcKch_Systematics%s",tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tDirectoryBase.Remove(TString::kTrailing,'_');
    tDirectoryBase += tDirNameModifierBase2;
  }
  tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

  TString tFileLocationBase = tDirectoryBase + TString::Format("Results_cLamcKch_Systematics%s",tDirNameModifierBase1.Data());
  TString tFileLocationBaseMC = tDirectoryBase + TString::Format("Results_cLamcKchMC_Systematics%s",tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tFileLocationBase.Remove(TString::kTrailing,'_');
    tFileLocationBaseMC.Remove(TString::kTrailing,'_');

    tFileLocationBase += tDirNameModifierBase2;
    tFileLocationBaseMC += tDirNameModifierBase2;
  }
  tFileLocationBase += tResultsDate;
  tFileLocationBaseMC += tResultsDate;


  bool SaveImages = false;

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

//    TCanvas* tCanAvgSepCfs = tLamKch0010->DrawAvgSepCfs(SaveImages);
//    TCanvas* tCanAvgSepCfsLamKchP = tLamKch0010->DrawAvgSepCfs(kLamKchP,true,SaveImages);
//    TCanvas* tCanAvgSepCfsLamKchM = tLamKch0010->DrawAvgSepCfs(kLamKchM,true,SaveImages);

//    TCanvas* tCanPart1MassFail = tLamKch0010->ViewPart1MassFail(false,SaveImages);

    TCanvas* tCanMassAssK0_LamKchP = tLamKch0010->DrawMassAssumingK0ShortHypothesis(kLamKchP,SaveImages);
  }


/*
  TString tResultsDate = "20161025";

  TString tDirNameModifierBase1 = "_K0s_minInvMassReject_";
  vector<double> tModifierValues1 = {1.112683, 1.110683, 1.106683, 1.100683};

  TString tDirNameModifierBase2 = "_K0s_maxInvMassReject_";
  vector<double> tModifierValues2 = {1.118683, 1.120683, 1.124683, 1.130683};

  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Systematics/Results_cLamK0_Systematics%s",tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tDirectoryBase.Remove(TString::kTrailing,'_');
    tDirectoryBase += tDirNameModifierBase2;
  }
  tDirectoryBase += TString::Format("%s/",tResultsDate.Data());

  TString tFileLocationBase = tDirectoryBase + TString::Format("Results_cLamK0_Systematics%s",tDirNameModifierBase1.Data());
  TString tFileLocationBaseMC = tDirectoryBase + TString::Format("Results_cLamK0MC_Systematics%s",tDirNameModifierBase1.Data());
  if(!tDirNameModifierBase2.IsNull())
  {
    tFileLocationBase.Remove(TString::kTrailing,'_');
    tFileLocationBaseMC.Remove(TString::kTrailing,'_');

    tFileLocationBase += tDirNameModifierBase2;
    tFileLocationBaseMC += tDirNameModifierBase2;
  }
  tFileLocationBase += tResultsDate;
  tFileLocationBaseMC += tResultsDate;



  bool SaveImages = true;

  for(unsigned int iVal=0; iVal<tModifierValues1.size(); iVal++)
  {
    TString tDirNameModifier = tDirNameModifierBase1 + TString::Format("%0.6f",tModifierValues1[iVal]);
    if(!tDirNameModifierBase2.IsNull() && tModifierValues2.size()==tModifierValues1.size())
    {
      tDirNameModifier += tDirNameModifierBase2 + TString::Format("%0.6f",tModifierValues2[iVal]);
    }

    PlotPartnersLamK0* tLamK00010 = new PlotPartnersLamK0(tFileLocationBase,tFileLocationBaseMC,kLamK0,k0010,kTrainSys,2,tDirNameModifier);
    tLamK00010->SetSaveLocationBase(tDirectoryBase);


    TCanvas* tCanPur = tLamK00010->DrawPurity(SaveImages);

    TCanvas* tCanKStarCf = tLamK00010->DrawKStarCfs(SaveImages);
    TCanvas* tCanKStarTrueVsRec = tLamK00010->DrawKStarTrueVsRec(kMixed,SaveImages);

    TCanvas* tCanMassAssK0_LamK0 = tLamK00010->DrawMassAssumingK0ShortHypothesis(kLamK0,SaveImages);
    TCanvas* tCanMassAssLam_LamK0 = tLamK00010->DrawMassAssumingLambdaHypothesis(kLamK0,SaveImages);
    TCanvas* tCanMassAssALam_LamK0 = tLamK00010->DrawMassAssumingAntiLambdaHypothesis(kLamK0,SaveImages);

    TCanvas* tCanMassAssK0_ALamK0 = tLamK00010->DrawMassAssumingK0ShortHypothesis(kALamK0,SaveImages);
    TCanvas* tCanMassAssLam_ALamK0 = tLamK00010->DrawMassAssumingLambdaHypothesis(kALamK0,SaveImages);
    TCanvas* tCanMassAssALam_ALamK0 = tLamK00010->DrawMassAssumingAntiLambdaHypothesis(kALamK0,SaveImages);
  }

*/


cout << "DONE" << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
