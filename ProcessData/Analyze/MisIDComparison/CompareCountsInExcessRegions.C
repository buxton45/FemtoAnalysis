#include "PlotPartnersLamKch.h"
class PlotPartnersLamKch;

#include "PlotPartnersLamK0.h"
class PlotPartnersLamK0;

#include "Types.h"

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything
//-----------------------------------------------------------------------------
  TString tSaveLocationBase = "/home/jesse/Analysis/FemtoAnalysis/ProcessData/Analyze/MisIDComparison/";
  bool bSaveImages = false;
  bool bNormByEvent = false;

  AnalysisType tAnTypeLamK0 = kLamK0;
  ParticleType tPurityLamTypeLamK0;

//-----------------------------------------------------------------------------
  if(tAnTypeLamK0 == kLamK0) tPurityLamTypeLamK0 = kLam;
  else if(tAnTypeLamK0 == kALamK0) tPurityLamTypeLamK0 = kALam;
  else
  {
    cout << "ERROR: CompareCountsInExcessRegions.C invalue tAnTypeLamK0 = " << tAnTypeLamK0 << endl;
    assert(0);
  }
//-------------------------------------------------------------------------------

//  TString tResultsDate_LamK0_NoMisIDCut = "20161028";
  TString tResultsDate_LamK0_NoMisIDCut = "20161108";
  TString tLegendEntry_LamK0_NoMisIDCut = "NoMisID";
  TString tDirectoryBase_LamK0_NoMisIDCut = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_LamK0_NoMisIDCut.Data());
  TString tFileLocationBase_LamK0_NoMisIDCut = tDirectoryBase_LamK0_NoMisIDCut+"Results_cLamK0_"+tResultsDate_LamK0_NoMisIDCut;
  TString tFileLocationBaseMC_LamK0_NoMisIDCut = tDirectoryBase_LamK0_NoMisIDCut+"Results_cLamK0MC_"+tResultsDate_LamK0_NoMisIDCut;
  PlotPartnersLamK0* tLamK0_NoMisIDCut = new PlotPartnersLamK0(tFileLocationBase_LamK0_NoMisIDCut,tFileLocationBaseMC_LamK0_NoMisIDCut,tAnTypeLamK0,k0010,kTrain,2);
    int tColor_LamK0_NoMisIDCut = 1;
    int tMarkerStyle_LamK0_NoMisIDCut = 22;
    TH1* tMassAssLam_LamK0_NoMisIDCut = tLamK0_NoMisIDCut->GetMassAssumingLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_NoMisIDCut,tMarkerStyle_LamK0_NoMisIDCut);
    TH1* tMassAssALam_LamK0_NoMisIDCut = tLamK0_NoMisIDCut->GetMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_NoMisIDCut,tMarkerStyle_LamK0_NoMisIDCut);


  TString tResultsDate_LamK0_MisIDCut_MinvCut_MinvComp = "20161027";
  TString tLegendEntry_LamK0_MisIDCut_MinvCut_MinvComp = "MisID_M_{inv}Cut_M_{inv}Comp";
  TString tDirectoryBase_LamK0_MisIDCut_MinvCut_MinvComp = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_LamK0_MisIDCut_MinvCut_MinvComp.Data());
  TString tFileLocationBase_LamK0_MisIDCut_MinvCut_MinvComp = tDirectoryBase_LamK0_MisIDCut_MinvCut_MinvComp+"Results_cLamK0_"+tResultsDate_LamK0_MisIDCut_MinvCut_MinvComp;
  TString tFileLocationBaseMC_LamK0_MisIDCut_MinvCut_MinvComp = tDirectoryBase_LamK0_MisIDCut_MinvCut_MinvComp+"Results_cLamK0MC_"+tResultsDate_LamK0_MisIDCut_MinvCut_MinvComp;
  PlotPartnersLamK0* tLamK0_MisIDCut_MinvCut_MinvComp = new PlotPartnersLamK0(tFileLocationBase_LamK0_MisIDCut_MinvCut_MinvComp,tFileLocationBaseMC_LamK0_MisIDCut_MinvCut_MinvComp,tAnTypeLamK0,k0010,kTrain,2);
    int tColor_LamK0_MisIDCut_MinvCut_MinvComp = 2;
    int tMarkerStyle_LamK0_MisIDCut_MinvCut_MinvComp = 20;
    TH1* tMassAssLam_LamK0_MisIDCut_MinvCut_MinvComp = tLamK0_MisIDCut_MinvCut_MinvComp->GetMassAssumingLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_MinvCut_MinvComp,tMarkerStyle_LamK0_MisIDCut_MinvCut_MinvComp);
    TH1* tMassAssALam_LamK0_MisIDCut_MinvCut_MinvComp = tLamK0_MisIDCut_MinvCut_MinvComp->GetMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_MinvCut_MinvComp,tMarkerStyle_LamK0_MisIDCut_MinvCut_MinvComp);
//-------------------------------------------------------------------------------
  double tXLow1, tXHigh1;
  double tXLow2, tXHigh2;

  tXLow1 = 1.025683;
  tXHigh1 = 1.205683;

  tXLow2 = 1.6;
  tXHigh2 = 2.4;

  double tNPeak1MassAssLam_NoMisIDCut = tMassAssLam_LamK0_NoMisIDCut->Integral(tMassAssLam_LamK0_NoMisIDCut->FindBin(tXLow1),tMassAssLam_LamK0_NoMisIDCut->FindBin(tXHigh1));
  double tNPeak2MassAssLam_NoMisIDCut = tMassAssLam_LamK0_NoMisIDCut->Integral(tMassAssLam_LamK0_NoMisIDCut->FindBin(tXLow2),tMassAssLam_LamK0_NoMisIDCut->FindBin(tXHigh2));

  double tNPeak1MassAssLam_MisIDCut = tMassAssLam_LamK0_MisIDCut_MinvCut_MinvComp->Integral(tMassAssLam_LamK0_MisIDCut_MinvCut_MinvComp->FindBin(tXLow1),tMassAssLam_LamK0_MisIDCut_MinvCut_MinvComp->FindBin(tXHigh1));
  double tNPeak2MassAssLam_MisIDCut = tMassAssLam_LamK0_MisIDCut_MinvCut_MinvComp->Integral(tMassAssLam_LamK0_MisIDCut_MinvCut_MinvComp->FindBin(tXLow2),tMassAssLam_LamK0_MisIDCut_MinvCut_MinvComp->FindBin(tXHigh2));

  double tNPeak1MassAssALam_NoMisIDCut = tMassAssALam_LamK0_NoMisIDCut->Integral(tMassAssALam_LamK0_NoMisIDCut->FindBin(tXLow1),tMassAssALam_LamK0_NoMisIDCut->FindBin(tXHigh1));
  double tNPeak2MassAssALam_NoMisIDCut = tMassAssALam_LamK0_NoMisIDCut->Integral(tMassAssALam_LamK0_NoMisIDCut->FindBin(tXLow2),tMassAssALam_LamK0_NoMisIDCut->FindBin(tXHigh2));

  double tNPeak1MassAssALam_MisIDCut = tMassAssALam_LamK0_MisIDCut_MinvCut_MinvComp->Integral(tMassAssALam_LamK0_MisIDCut_MinvCut_MinvComp->FindBin(tXLow1),tMassAssALam_LamK0_MisIDCut_MinvCut_MinvComp->FindBin(tXHigh1));
  double tNPeak2MassAssALam_MisIDCut = tMassAssALam_LamK0_MisIDCut_MinvCut_MinvComp->Integral(tMassAssALam_LamK0_MisIDCut_MinvCut_MinvComp->FindBin(tXLow2),tMassAssALam_LamK0_MisIDCut_MinvCut_MinvComp->FindBin(tXHigh2));

//-------------------------------------------------------------------------------

  double tNLam_MassAssLam = tNPeak1MassAssLam_NoMisIDCut - tNPeak1MassAssLam_MisIDCut;
  double tNALam_MassAssLam = tNPeak2MassAssLam_NoMisIDCut - tNPeak2MassAssLam_MisIDCut;

  double tNALam_MassAssALam = tNPeak1MassAssALam_NoMisIDCut - tNPeak1MassAssALam_MisIDCut;
  double tNLam_MassAssALam = tNPeak2MassAssALam_NoMisIDCut - tNPeak2MassAssALam_MisIDCut;

  cout << "tNLam_MassAssLam  = " << tNLam_MassAssLam << endl;
  cout << "tNLam_MassAssALam = " << tNLam_MassAssALam << endl << endl;

  cout << "tNALam_MassAssALam = " << tNALam_MassAssALam << endl;
  cout << "tNALam_MassAssLam  = " << tNALam_MassAssLam << endl;
//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
