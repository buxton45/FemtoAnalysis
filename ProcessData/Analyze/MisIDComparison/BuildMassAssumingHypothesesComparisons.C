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
  bool bSaveImages = true;
  bool bNormByEvent = true;
  bool bDrawNoMisIDCut = true;


  AnalysisType tAnTypeLamKch = kLamKchP;
  ParticleType tPurityLamTypeLamKch;

  AnalysisType tAnTypeLamK0 = kLamK0;
  ParticleType tPurityLamTypeLamK0;

//-----------------------------------------------------------------------------
  if(tAnTypeLamKch == kLamKchP || tAnTypeLamKch == kLamKchM) tPurityLamTypeLamKch = kLam;
  else if(tAnTypeLamKch == kALamKchP || tAnTypeLamKch == kALamKchM) tPurityLamTypeLamKch = kALam;
  else
  {
    cout << "ERROR: BuildMassAssumingHypothesesComparisons.C invalue tAnTypeLamKch = " << tAnTypeLamKch << endl;
    assert(0);
  }

  if(tAnTypeLamK0 == kLamK0) tPurityLamTypeLamK0 = kLam;
  else if(tAnTypeLamK0 == kALamK0) tPurityLamTypeLamK0 = kALam;
  else
  {
    cout << "ERROR: BuildMassAssumingHypothesesComparisons.C invalue tAnTypeLamK0 = " << tAnTypeLamK0 << endl;
    assert(0);
  }

//-----------------------------------------------------------------------------

  TString tResultsDate_LamKch_NoMisIDCut = "20161028";
  TString tLegendEntry_LamKch_NoMisIDCut = "NoMisID";
  TString tDirectoryBase_LamKch_NoMisIDCut = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch_NoMisIDCut.Data());
  TString tFileLocationBase_LamKch_NoMisIDCut = tDirectoryBase_LamKch_NoMisIDCut+"Results_cLamcKch_"+tResultsDate_LamKch_NoMisIDCut;
  TString tFileLocationBaseMC_LamKch_NoMisIDCut = tDirectoryBase_LamKch_NoMisIDCut+"Results_cLamcKchMC_"+tResultsDate_LamKch_NoMisIDCut;
  PlotPartnersLamKch* tLamKch_NoMisIDCut = new PlotPartnersLamKch(tFileLocationBase_LamKch_NoMisIDCut,tFileLocationBaseMC_LamKch_NoMisIDCut,tAnTypeLamKch,k0010,kTrain,2);
  int tColor_LamKch_NoMisIDCut = 1;
  int tMarkerStyle_LamKch_NoMisIDCut = 22;
  TH1* tMassAssK0Short_LamKch_NoMisIDCut = tLamKch_NoMisIDCut->GetMassAssumingK0ShortHypothesis(tAnTypeLamKch,bNormByEvent,tColor_LamKch_NoMisIDCut,tMarkerStyle_LamKch_NoMisIDCut);
  double tPurity_LamKch_NoMisIDCut = tLamKch_NoMisIDCut->GetPurity(tAnTypeLamKch,tPurityLamTypeLamKch);

  TString tResultsDate_LamKch_MisIDCut_MinvCut_NoMinvComp = "20161025";
  TString tLegendEntry_LamKch_MisIDCut_MinvCut_NoMinvComp = "MisID_M_{inv}Cut_NoM_{inv}Comp";
  TString tDirectoryBase_LamKch_MisIDCut_MinvCut_NoMinvComp = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch_MisIDCut_MinvCut_NoMinvComp.Data());
  TString tFileLocationBase_LamKch_MisIDCut_MinvCut_NoMinvComp = tDirectoryBase_LamKch_MisIDCut_MinvCut_NoMinvComp+"Results_cLamcKch_"+tResultsDate_LamKch_MisIDCut_MinvCut_NoMinvComp;
  TString tFileLocationBaseMC_LamKch_MisIDCut_MinvCut_NoMinvComp = tDirectoryBase_LamKch_MisIDCut_MinvCut_NoMinvComp+"Results_cLamcKchMC_"+tResultsDate_LamKch_MisIDCut_MinvCut_NoMinvComp;
  PlotPartnersLamKch* tLamKch_MisIDCut_MinvCut_NoMinvComp = new PlotPartnersLamKch(tFileLocationBase_LamKch_MisIDCut_MinvCut_NoMinvComp,tFileLocationBaseMC_LamKch_MisIDCut_MinvCut_NoMinvComp,tAnTypeLamKch,k0010,kTrain,2);
  int tColor_LamKch_MisIDCut_MinvCut_NoMinvComp = 3;
  int tMarkerStyle_LamKch_MisIDCut_MinvCut_NoMinvComp = 21;
  TH1* tMassAssK0Short_LamKch_MisIDCut_MinvCut_NoMinvComp = tLamKch_MisIDCut_MinvCut_NoMinvComp->GetMassAssumingK0ShortHypothesis(tAnTypeLamKch,bNormByEvent,tColor_LamKch_MisIDCut_MinvCut_NoMinvComp,tMarkerStyle_LamKch_MisIDCut_MinvCut_NoMinvComp);
  double tPurity_LamKch_MisIDCut_MinvCut_NoMinvComp = tLamKch_MisIDCut_MinvCut_NoMinvComp->GetPurity(tAnTypeLamKch,tPurityLamTypeLamKch);

  TString tResultsDate_LamKch_MisIDCut_MinvCut_MinvComp = "20161027";
  TString tLegendEntry_LamKch_MisIDCut_MinvCut_MinvComp = "MisID_M_{inv}Cut_M_{inv}Comp";
  TString tDirectoryBase_LamKch_MisIDCut_MinvCut_MinvComp = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch_MisIDCut_MinvCut_MinvComp.Data());
  TString tFileLocationBase_LamKch_MisIDCut_MinvCut_MinvComp = tDirectoryBase_LamKch_MisIDCut_MinvCut_MinvComp+"Results_cLamcKch_"+tResultsDate_LamKch_MisIDCut_MinvCut_MinvComp;
  TString tFileLocationBaseMC_LamKch_MisIDCut_MinvCut_MinvComp = tDirectoryBase_LamKch_MisIDCut_MinvCut_MinvComp+"Results_cLamcKchMC_"+tResultsDate_LamKch_MisIDCut_MinvCut_MinvComp;
  PlotPartnersLamKch* tLamKch_MisIDCut_MinvCut_MinvComp = new PlotPartnersLamKch(tFileLocationBase_LamKch_MisIDCut_MinvCut_MinvComp,tFileLocationBaseMC_LamKch_MisIDCut_MinvCut_MinvComp,tAnTypeLamKch,k0010,kTrain,2);
  int tColor_LamKch_MisIDCut_MinvCut_MinvComp = 2;
  int tMarkerStyle_LamKch_MisIDCut_MinvCut_MinvComp = 20;
  TH1* tMassAssK0Short_LamKch_MisIDCut_MinvCut_MinvComp = tLamKch_MisIDCut_MinvCut_MinvComp->GetMassAssumingK0ShortHypothesis(tAnTypeLamKch,bNormByEvent,tColor_LamKch_MisIDCut_MinvCut_MinvComp,tMarkerStyle_LamKch_MisIDCut_MinvCut_MinvComp);
  double tPurity_LamKch_MisIDCut_MinvCut_MinvComp = tLamKch_MisIDCut_MinvCut_MinvComp->GetPurity(tAnTypeLamKch,tPurityLamTypeLamKch);

  TString tResultsDate_LamKch_MisIDCut_NoMinvCut_MinvComp = "20161031";
  TString tLegendEntry_LamKch_MisIDCut_NoMinvCut_MinvComp = "MisID_NoMinvCut_MinvComp";
  TString tDirectoryBase_LamKch_MisIDCut_NoMinvCut_MinvComp = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch_MisIDCut_NoMinvCut_MinvComp.Data());
  TString tFileLocationBase_LamKch_MisIDCut_NoMinvCut_MinvComp = tDirectoryBase_LamKch_MisIDCut_NoMinvCut_MinvComp+"Results_cLamcKch_"+tResultsDate_LamKch_MisIDCut_NoMinvCut_MinvComp;
  TString tFileLocationBaseMC_LamKch_MisIDCut_NoMinvCut_MinvComp = tDirectoryBase_LamKch_MisIDCut_NoMinvCut_MinvComp+"Results_cLamcKchMC_"+tResultsDate_LamKch_MisIDCut_NoMinvCut_MinvComp;
  PlotPartnersLamKch* tLamKch_MisIDCut_NoMinvCut_MinvComp = new PlotPartnersLamKch(tFileLocationBase_LamKch_MisIDCut_NoMinvCut_MinvComp,tFileLocationBaseMC_LamKch_MisIDCut_NoMinvCut_MinvComp,tAnTypeLamKch,k0010,kTrain,2);
  int tColor_LamKch_MisIDCut_NoMinvCut_MinvComp = 4;
  int tMarkerStyle_LamKch_MisIDCut_NoMinvCut_MinvComp = 29;
  TH1* tMassAssK0Short_LamKch_MisIDCut_NoMinvCut_MinvComp = tLamKch_MisIDCut_NoMinvCut_MinvComp->GetMassAssumingK0ShortHypothesis(tAnTypeLamKch,bNormByEvent,tColor_LamKch_MisIDCut_NoMinvCut_MinvComp,tMarkerStyle_LamKch_MisIDCut_NoMinvCut_MinvComp);
  double tPurity_LamKch_MisIDCut_NoMinvCut_MinvComp = tLamKch_MisIDCut_NoMinvCut_MinvComp->GetPurity(tAnTypeLamKch,tPurityLamTypeLamKch);

  TString tResultsDate_LamKch_SimpleMisIDCut = "20161102";
  TString tLegendEntry_LamKch_SimpleMisIDCut = "SimpleMisID";
  TString tDirectoryBase_LamKch_SimpleMisIDCut = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate_LamKch_SimpleMisIDCut.Data());
  TString tFileLocationBase_LamKch_SimpleMisIDCut = tDirectoryBase_LamKch_SimpleMisIDCut+"Results_cLamcKch_"+tResultsDate_LamKch_SimpleMisIDCut;
  TString tFileLocationBaseMC_LamKch_SimpleMisIDCut = tDirectoryBase_LamKch_SimpleMisIDCut+"Results_cLamcKchMC_"+tResultsDate_LamKch_SimpleMisIDCut;
  PlotPartnersLamKch* tLamKch_SimpleMisIDCut = new PlotPartnersLamKch(tFileLocationBase_LamKch_SimpleMisIDCut,tFileLocationBaseMC_LamKch_SimpleMisIDCut,tAnTypeLamKch,k0010,kTrain,2);
  int tColor_LamKch_SimpleMisIDCut = 6;
  int tMarkerStyle_LamKch_SimpleMisIDCut = 25;
  TH1* tMassAssK0Short_LamKch_SimpleMisIDCut = tLamKch_SimpleMisIDCut->GetMassAssumingK0ShortHypothesis(tAnTypeLamKch,bNormByEvent,tColor_LamKch_SimpleMisIDCut,tMarkerStyle_LamKch_SimpleMisIDCut);
  double tPurity_LamKch_SimpleMisIDCut = tLamKch_SimpleMisIDCut->GetPurity(tAnTypeLamKch,tPurityLamTypeLamKch);


  TObjArray* tHists_LamKch = new TObjArray();
  vector<TString> tLegendEntries_LamKch(0);
  vector<double> tPurities_LamKch(0);

  tHists_LamKch->Add(tMassAssK0Short_LamKch_MisIDCut_MinvCut_MinvComp);
  tLegendEntries_LamKch.push_back(tLegendEntry_LamKch_MisIDCut_MinvCut_MinvComp);
  tPurities_LamKch.push_back(tPurity_LamKch_MisIDCut_MinvCut_MinvComp);

  tHists_LamKch->Add(tMassAssK0Short_LamKch_MisIDCut_MinvCut_NoMinvComp);
  tLegendEntries_LamKch.push_back(tLegendEntry_LamKch_MisIDCut_MinvCut_NoMinvComp);
  tPurities_LamKch.push_back(tPurity_LamKch_MisIDCut_MinvCut_NoMinvComp);

  tHists_LamKch->Add(tMassAssK0Short_LamKch_MisIDCut_NoMinvCut_MinvComp);
  tLegendEntries_LamKch.push_back(tLegendEntry_LamKch_MisIDCut_NoMinvCut_MinvComp);
  tPurities_LamKch.push_back(tPurity_LamKch_MisIDCut_NoMinvCut_MinvComp);

  tHists_LamKch->Add(tMassAssK0Short_LamKch_SimpleMisIDCut);
  tLegendEntries_LamKch.push_back(tLegendEntry_LamKch_SimpleMisIDCut);
  tPurities_LamKch.push_back(tPurity_LamKch_SimpleMisIDCut);

  if(bDrawNoMisIDCut)
  {
    tHists_LamKch->Add(tMassAssK0Short_LamKch_NoMisIDCut);
    tLegendEntries_LamKch.push_back(tLegendEntry_LamKch_NoMisIDCut);
    tPurities_LamKch.push_back(tPurity_LamKch_NoMisIDCut);
  }


  TCanvas* tCanMassAssK0_LamKch = tLamKch_MisIDCut_MinvCut_MinvComp->DrawMassAssumingK0ShortHypothesis(tAnTypeLamKch,tHists_LamKch,tLegendEntries_LamKch,tPurities_LamKch,false);

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
  TH1* tMassAssK0Short_LamK0_NoMisIDCut = tLamK0_NoMisIDCut->GetMassAssumingK0ShortHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_NoMisIDCut,tMarkerStyle_LamK0_NoMisIDCut);
  TH1* tMassAssLam_LamK0_NoMisIDCut = tLamK0_NoMisIDCut->GetMassAssumingLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_NoMisIDCut,tMarkerStyle_LamK0_NoMisIDCut);
  TH1* tMassAssALam_LamK0_NoMisIDCut = tLamK0_NoMisIDCut->GetMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_NoMisIDCut,tMarkerStyle_LamK0_NoMisIDCut);
  double tPurityLam_LamK0_NoMisIDCut = tLamK0_NoMisIDCut->GetPurity(tAnTypeLamK0,tPurityLamTypeLamK0);
  double tPurityK0_LamK0_NoMisIDCut = tLamK0_NoMisIDCut->GetPurity(tAnTypeLamK0,kK0);


  TString tResultsDate_LamK0_MisIDCut_MinvCut_NoMinvComp = "20161025";
  TString tLegendEntry_LamK0_MisIDCut_MinvCut_NoMinvComp = "MisID_M_{inv}Cut_NoM_{inv}Comp";
  TString tDirectoryBase_LamK0_MisIDCut_MinvCut_NoMinvComp = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_LamK0_MisIDCut_MinvCut_NoMinvComp.Data());
  TString tFileLocationBase_LamK0_MisIDCut_MinvCut_NoMinvComp = tDirectoryBase_LamK0_MisIDCut_MinvCut_NoMinvComp+"Results_cLamK0_"+tResultsDate_LamK0_MisIDCut_MinvCut_NoMinvComp;
  TString tFileLocationBaseMC_LamK0_MisIDCut_MinvCut_NoMinvComp = tDirectoryBase_LamK0_MisIDCut_MinvCut_NoMinvComp+"Results_cLamK0MC_"+tResultsDate_LamK0_MisIDCut_MinvCut_NoMinvComp;
  PlotPartnersLamK0* tLamK0_MisIDCut_MinvCut_NoMinvComp = new PlotPartnersLamK0(tFileLocationBase_LamK0_MisIDCut_MinvCut_NoMinvComp,tFileLocationBaseMC_LamK0_MisIDCut_MinvCut_NoMinvComp,tAnTypeLamK0,k0010,kTrain,2);
  int tColor_LamK0_MisIDCut_MinvCut_NoMinvComp = 3;
  int tMarkerStyle_LamK0_MisIDCut_MinvCut_NoMinvComp = 21;
  TH1* tMassAssK0Short_LamK0_MisIDCut_MinvCut_NoMinvComp = tLamK0_MisIDCut_MinvCut_NoMinvComp->GetMassAssumingK0ShortHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_MinvCut_NoMinvComp,tMarkerStyle_LamK0_MisIDCut_MinvCut_NoMinvComp);
  TH1* tMassAssLam_LamK0_MisIDCut_MinvCut_NoMinvComp = tLamK0_MisIDCut_MinvCut_NoMinvComp->GetMassAssumingLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_MinvCut_NoMinvComp,tMarkerStyle_LamK0_MisIDCut_MinvCut_NoMinvComp);
  TH1* tMassAssALam_LamK0_MisIDCut_MinvCut_NoMinvComp = tLamK0_MisIDCut_MinvCut_NoMinvComp->GetMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_MinvCut_NoMinvComp,tMarkerStyle_LamK0_MisIDCut_MinvCut_NoMinvComp);
  double tPurityLam_LamK0_MisIDCut_MinvCut_NoMinvComp = tLamK0_MisIDCut_MinvCut_NoMinvComp->GetPurity(tAnTypeLamK0,tPurityLamTypeLamK0);
  double tPurityK0_LamK0_MisIDCut_MinvCut_NoMinvComp = tLamK0_MisIDCut_MinvCut_NoMinvComp->GetPurity(tAnTypeLamK0,kK0);

  TString tResultsDate_LamK0_MisIDCut_MinvCut_MinvComp = "20161027";
  TString tLegendEntry_LamK0_MisIDCut_MinvCut_MinvComp = "MisID_M_{inv}Cut_M_{inv}Comp";
  TString tDirectoryBase_LamK0_MisIDCut_MinvCut_MinvComp = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_LamK0_MisIDCut_MinvCut_MinvComp.Data());
  TString tFileLocationBase_LamK0_MisIDCut_MinvCut_MinvComp = tDirectoryBase_LamK0_MisIDCut_MinvCut_MinvComp+"Results_cLamK0_"+tResultsDate_LamK0_MisIDCut_MinvCut_MinvComp;
  TString tFileLocationBaseMC_LamK0_MisIDCut_MinvCut_MinvComp = tDirectoryBase_LamK0_MisIDCut_MinvCut_MinvComp+"Results_cLamK0MC_"+tResultsDate_LamK0_MisIDCut_MinvCut_MinvComp;
  PlotPartnersLamK0* tLamK0_MisIDCut_MinvCut_MinvComp = new PlotPartnersLamK0(tFileLocationBase_LamK0_MisIDCut_MinvCut_MinvComp,tFileLocationBaseMC_LamK0_MisIDCut_MinvCut_MinvComp,tAnTypeLamK0,k0010,kTrain,2);
  int tColor_LamK0_MisIDCut_MinvCut_MinvComp = 2;
  int tMarkerStyle_LamK0_MisIDCut_MinvCut_MinvComp = 20;
  TH1* tMassAssK0Short_LamK0_MisIDCut_MinvCut_MinvComp = tLamK0_MisIDCut_MinvCut_MinvComp->GetMassAssumingK0ShortHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_MinvCut_MinvComp,tMarkerStyle_LamK0_MisIDCut_MinvCut_MinvComp);
  TH1* tMassAssLam_LamK0_MisIDCut_MinvCut_MinvComp = tLamK0_MisIDCut_MinvCut_MinvComp->GetMassAssumingLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_MinvCut_MinvComp,tMarkerStyle_LamK0_MisIDCut_MinvCut_MinvComp);
  TH1* tMassAssALam_LamK0_MisIDCut_MinvCut_MinvComp = tLamK0_MisIDCut_MinvCut_MinvComp->GetMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_MinvCut_MinvComp,tMarkerStyle_LamK0_MisIDCut_MinvCut_MinvComp);
  double tPurityLam_LamK0_MisIDCut_MinvCut_MinvComp = tLamK0_MisIDCut_MinvCut_MinvComp->GetPurity(tAnTypeLamK0,tPurityLamTypeLamK0);
  double tPurityK0_LamK0_MisIDCut_MinvCut_MinvComp = tLamK0_MisIDCut_MinvCut_MinvComp->GetPurity(tAnTypeLamK0,kK0);

  TString tResultsDate_LamK0_MisIDCut_NoMinvCut_MinvComp = "20161031";
  TString tLegendEntry_LamK0_MisIDCut_NoMinvCut_MinvComp = "MisID_NoM_{inv}Cut_M_{inv}Comp";
  TString tDirectoryBase_LamK0_MisIDCut_NoMinvCut_MinvComp = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_LamK0_MisIDCut_NoMinvCut_MinvComp.Data());
  TString tFileLocationBase_LamK0_MisIDCut_NoMinvCut_MinvComp = tDirectoryBase_LamK0_MisIDCut_NoMinvCut_MinvComp+"Results_cLamK0_"+tResultsDate_LamK0_MisIDCut_NoMinvCut_MinvComp;
  TString tFileLocationBaseMC_LamK0_MisIDCut_NoMinvCut_MinvComp = tDirectoryBase_LamK0_MisIDCut_NoMinvCut_MinvComp+"Results_cLamK0MC_"+tResultsDate_LamK0_MisIDCut_NoMinvCut_MinvComp;
  PlotPartnersLamK0* tLamK0_MisIDCut_NoMinvCut_MinvComp = new PlotPartnersLamK0(tFileLocationBase_LamK0_MisIDCut_NoMinvCut_MinvComp,tFileLocationBaseMC_LamK0_MisIDCut_NoMinvCut_MinvComp,tAnTypeLamK0,k0010,kTrain,2);
  int tColor_LamK0_MisIDCut_NoMinvCut_MinvComp = 4;
  int tMarkerStyle_LamK0_MisIDCut_NoMinvCut_MinvComp = 29;
  TH1* tMassAssK0Short_LamK0_MisIDCut_NoMinvCut_MinvComp = tLamK0_MisIDCut_NoMinvCut_MinvComp->GetMassAssumingK0ShortHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_NoMinvCut_MinvComp,tMarkerStyle_LamK0_MisIDCut_NoMinvCut_MinvComp);
  TH1* tMassAssLam_LamK0_MisIDCut_NoMinvCut_MinvComp = tLamK0_MisIDCut_NoMinvCut_MinvComp->GetMassAssumingLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_NoMinvCut_MinvComp,tMarkerStyle_LamK0_MisIDCut_NoMinvCut_MinvComp);
  TH1* tMassAssALam_LamK0_MisIDCut_NoMinvCut_MinvComp = tLamK0_MisIDCut_NoMinvCut_MinvComp->GetMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_MisIDCut_NoMinvCut_MinvComp,tMarkerStyle_LamK0_MisIDCut_NoMinvCut_MinvComp);
  double tPurityLam_LamK0_MisIDCut_NoMinvCut_MinvComp = tLamK0_MisIDCut_NoMinvCut_MinvComp->GetPurity(tAnTypeLamK0,tPurityLamTypeLamK0);
  double tPurityK0_LamK0_MisIDCut_NoMinvCut_MinvComp = tLamK0_MisIDCut_NoMinvCut_MinvComp->GetPurity(tAnTypeLamK0,kK0);

  TString tResultsDate_LamK0_SimpleMisIDCut = "20161102";
  TString tLegendEntry_LamK0_SimpleMisIDCut = "SimpleMisID";
  TString tDirectoryBase_LamK0_SimpleMisIDCut = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate_LamK0_SimpleMisIDCut.Data());
  TString tFileLocationBase_LamK0_SimpleMisIDCut = tDirectoryBase_LamK0_SimpleMisIDCut+"Results_cLamK0_"+tResultsDate_LamK0_SimpleMisIDCut;
  TString tFileLocationBaseMC_LamK0_SimpleMisIDCut = tDirectoryBase_LamK0_SimpleMisIDCut+"Results_cLamK0MC_"+tResultsDate_LamK0_SimpleMisIDCut;
  PlotPartnersLamK0* tLamK0_SimpleMisIDCut = new PlotPartnersLamK0(tFileLocationBase_LamK0_SimpleMisIDCut,tFileLocationBaseMC_LamK0_SimpleMisIDCut,tAnTypeLamK0,k0010,kTrain,2);
  int tColor_LamK0_SimpleMisIDCut = 6;
  int tMarkerStyle_LamK0_SimpleMisIDCut = 25;
  TH1* tMassAssK0Short_LamK0_SimpleMisIDCut = tLamK0_SimpleMisIDCut->GetMassAssumingK0ShortHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_SimpleMisIDCut,tMarkerStyle_LamK0_SimpleMisIDCut);
  TH1* tMassAssLam_LamK0_SimpleMisIDCut = tLamK0_SimpleMisIDCut->GetMassAssumingLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_SimpleMisIDCut,tMarkerStyle_LamK0_SimpleMisIDCut);
  TH1* tMassAssALam_LamK0_SimpleMisIDCut = tLamK0_SimpleMisIDCut->GetMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,bNormByEvent,tColor_LamK0_SimpleMisIDCut,tMarkerStyle_LamK0_SimpleMisIDCut);
  double tPurityLam_LamK0_SimpleMisIDCut = tLamK0_SimpleMisIDCut->GetPurity(tAnTypeLamK0,tPurityLamTypeLamK0);
  double tPurityK0_LamK0_SimpleMisIDCut = tLamK0_SimpleMisIDCut->GetPurity(tAnTypeLamK0,kK0);


  TObjArray* tHists_LamK0_MassAssK0 = new TObjArray();
  vector<TString> tLegendEntries_LamK0_MassAssK0(0);
  vector<double> tPurities_LamK0_MassAssK0(0);

  TObjArray* tHists_LamK0_MassAssLam = new TObjArray();
  vector<TString> tLegendEntries_LamK0_MassAssLam(0);
  vector<double> tPurities_LamK0_MassAssLam(0);

  TObjArray* tHists_LamK0_MassAssALam = new TObjArray();
  vector<TString> tLegendEntries_LamK0_MassAssALam(0);
  vector<double> tPurities_LamK0_MassAssALam(0);

  tHists_LamK0_MassAssK0->Add(tMassAssK0Short_LamK0_MisIDCut_MinvCut_MinvComp);
    tLegendEntries_LamK0_MassAssK0.push_back(tLegendEntry_LamK0_MisIDCut_MinvCut_MinvComp);
    tPurities_LamK0_MassAssK0.push_back(tPurityLam_LamK0_MisIDCut_MinvCut_MinvComp);
  tHists_LamK0_MassAssLam->Add(tMassAssLam_LamK0_MisIDCut_MinvCut_MinvComp);
    tLegendEntries_LamK0_MassAssLam.push_back(tLegendEntry_LamK0_MisIDCut_MinvCut_MinvComp);
    tPurities_LamK0_MassAssLam.push_back(tPurityK0_LamK0_MisIDCut_MinvCut_MinvComp);
  tHists_LamK0_MassAssALam->Add(tMassAssALam_LamK0_MisIDCut_MinvCut_MinvComp);
    tLegendEntries_LamK0_MassAssALam.push_back(tLegendEntry_LamK0_MisIDCut_MinvCut_MinvComp);
    tPurities_LamK0_MassAssALam.push_back(tPurityK0_LamK0_MisIDCut_MinvCut_MinvComp);



  tHists_LamK0_MassAssK0->Add(tMassAssK0Short_LamK0_MisIDCut_MinvCut_NoMinvComp);
    tLegendEntries_LamK0_MassAssK0.push_back(tLegendEntry_LamK0_MisIDCut_MinvCut_NoMinvComp);
    tPurities_LamK0_MassAssK0.push_back(tPurityLam_LamK0_MisIDCut_MinvCut_NoMinvComp);
  tHists_LamK0_MassAssLam->Add(tMassAssLam_LamK0_MisIDCut_MinvCut_NoMinvComp);
    tLegendEntries_LamK0_MassAssLam.push_back(tLegendEntry_LamK0_MisIDCut_MinvCut_NoMinvComp);
    tPurities_LamK0_MassAssLam.push_back(tPurityK0_LamK0_MisIDCut_MinvCut_NoMinvComp);
  tHists_LamK0_MassAssALam->Add(tMassAssALam_LamK0_MisIDCut_MinvCut_NoMinvComp);
    tLegendEntries_LamK0_MassAssALam.push_back(tLegendEntry_LamK0_MisIDCut_MinvCut_NoMinvComp);
    tPurities_LamK0_MassAssALam.push_back(tPurityK0_LamK0_MisIDCut_MinvCut_NoMinvComp);

  tHists_LamK0_MassAssK0->Add(tMassAssK0Short_LamK0_MisIDCut_NoMinvCut_MinvComp);
    tLegendEntries_LamK0_MassAssK0.push_back(tLegendEntry_LamK0_MisIDCut_NoMinvCut_MinvComp);
    tPurities_LamK0_MassAssK0.push_back(tPurityLam_LamK0_MisIDCut_NoMinvCut_MinvComp);
  tHists_LamK0_MassAssLam->Add(tMassAssLam_LamK0_MisIDCut_NoMinvCut_MinvComp);
    tLegendEntries_LamK0_MassAssLam.push_back(tLegendEntry_LamK0_MisIDCut_NoMinvCut_MinvComp);
    tPurities_LamK0_MassAssLam.push_back(tPurityK0_LamK0_MisIDCut_NoMinvCut_MinvComp);
  tHists_LamK0_MassAssALam->Add(tMassAssALam_LamK0_MisIDCut_NoMinvCut_MinvComp);
    tLegendEntries_LamK0_MassAssALam.push_back(tLegendEntry_LamK0_MisIDCut_NoMinvCut_MinvComp);
    tPurities_LamK0_MassAssALam.push_back(tPurityK0_LamK0_MisIDCut_NoMinvCut_MinvComp);

  tHists_LamK0_MassAssK0->Add(tMassAssK0Short_LamK0_SimpleMisIDCut);
    tLegendEntries_LamK0_MassAssK0.push_back(tLegendEntry_LamK0_SimpleMisIDCut);
    tPurities_LamK0_MassAssK0.push_back(tPurityLam_LamK0_SimpleMisIDCut);
  tHists_LamK0_MassAssLam->Add(tMassAssLam_LamK0_SimpleMisIDCut);
    tLegendEntries_LamK0_MassAssLam.push_back(tLegendEntry_LamK0_SimpleMisIDCut);
    tPurities_LamK0_MassAssLam.push_back(tPurityK0_LamK0_SimpleMisIDCut);
  tHists_LamK0_MassAssALam->Add(tMassAssALam_LamK0_SimpleMisIDCut);
    tLegendEntries_LamK0_MassAssALam.push_back(tLegendEntry_LamK0_SimpleMisIDCut);
    tPurities_LamK0_MassAssALam.push_back(tPurityK0_LamK0_SimpleMisIDCut);

  if(bDrawNoMisIDCut)
  {
    tHists_LamK0_MassAssK0->Add(tMassAssK0Short_LamK0_NoMisIDCut);
      tLegendEntries_LamK0_MassAssK0.push_back(tLegendEntry_LamK0_NoMisIDCut);
      tPurities_LamK0_MassAssK0.push_back(tPurityLam_LamK0_NoMisIDCut);

    tHists_LamK0_MassAssLam->Add(tMassAssLam_LamK0_NoMisIDCut);
      tLegendEntries_LamK0_MassAssLam.push_back(tLegendEntry_LamK0_NoMisIDCut);
      tPurities_LamK0_MassAssLam.push_back(tPurityK0_LamK0_NoMisIDCut);

    tHists_LamK0_MassAssALam->Add(tMassAssALam_LamK0_NoMisIDCut);
      tLegendEntries_LamK0_MassAssALam.push_back(tLegendEntry_LamK0_NoMisIDCut);
      tPurities_LamK0_MassAssALam.push_back(tPurityK0_LamK0_NoMisIDCut);
  }



  TCanvas* tCanMassAssK0_LamK0 = tLamK0_MisIDCut_MinvCut_MinvComp->DrawMassAssumingK0ShortHypothesis(tAnTypeLamK0,tHists_LamK0_MassAssK0,tLegendEntries_LamK0_MassAssK0,tPurities_LamK0_MassAssK0,false);
  TCanvas* tCanMassAssLam_LamK0 = tLamK0_MisIDCut_MinvCut_MinvComp->DrawMassAssumingLambdaHypothesis(tAnTypeLamK0,tHists_LamK0_MassAssLam,tLegendEntries_LamK0_MassAssLam,tPurities_LamK0_MassAssLam,false);
  TCanvas* tCanMassAssALam_LamK0 = tLamK0_MisIDCut_MinvCut_MinvComp->DrawMassAssumingAntiLambdaHypothesis(tAnTypeLamK0,tHists_LamK0_MassAssALam,tLegendEntries_LamK0_MassAssALam,tPurities_LamK0_MassAssALam,false);

//-------------------------------------------------------------------------------
  if(bSaveImages)
  {
    TString tNoMisIDModifier = "";
    if(bDrawNoMisIDCut) tNoMisIDModifier = "_wNoMisID";

    tCanMassAssK0_LamKch->SaveAs(tSaveLocationBase+TString::Format("%s%s.pdf",tCanMassAssK0_LamKch->GetTitle(),tNoMisIDModifier.Data()));

    tCanMassAssK0_LamK0->SaveAs(tSaveLocationBase+TString::Format("%s%s.pdf",tCanMassAssK0_LamK0->GetTitle(),tNoMisIDModifier.Data()));
    tCanMassAssLam_LamK0->SaveAs(tSaveLocationBase+TString::Format("%s%s.pdf",tCanMassAssLam_LamK0->GetTitle(),tNoMisIDModifier.Data()));
    tCanMassAssALam_LamK0->SaveAs(tSaveLocationBase+TString::Format("%s%s.pdf",tCanMassAssALam_LamK0->GetTitle(),tNoMisIDModifier.Data()));
  }

//-------------------------------------------------------------------------------
  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.

  return 0;
}
