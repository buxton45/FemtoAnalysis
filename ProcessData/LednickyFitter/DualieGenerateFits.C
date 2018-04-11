#include "DualieFitGenerator.h"
class DualieFitGenerator;

void SetAttributes();

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
//  TString tResultsDate = "20161027";
//  TString tResultsDate = "20171220_onFlyStatusFalse";
  TString tResultsDate = "20171227";
//  TString tResultsDate = "20171227_LHC10h";
//  TString tResultsDate = "20180104_useIsProbableElectronMethodTrue";
//  TString tResultsDate = "20180104_useIsProbableElectronMethodFalse";

  bool bDoFit = true;

  double tMaxFitKStar=0.3;

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
//  CentralityType tCentType = k0010;  //TODO
  CentralityType tCentType = kMB;  //TODO
  FitGeneratorType tGenType = kPairwConj;
  FitType tFitType = kChi2PML;
  bool tShareLambdaParams = false;
  bool tAllShareSingleLambdaParam = false;


  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;

  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp;
  ResPrimMaxDecayType tResPrimMaxDecayType = k4fm;

  bool UnboundLambda = true;
  bool FixAllLambdaTo1 = false;
  if(FixAllLambdaTo1) tAllShareSingleLambdaParam = true;

  double aLambdaMin=0., aLambdaMax=1.;
  if(UnboundLambda) aLambdaMax=0.;
//-----------------------------------------------------------------------------

  if(tAnType==kLamK0)
  {
    tAllShareSingleLambdaParam = true;
    UnboundLambda = false;
    aLambdaMin = 0.4;  //TODO currently, if tIncludeResidualsType = kIncludeNoResiduals, this does nothing
    aLambdaMax = 0.6;  //TODO "                                                                          "

    if(tIncludeResidualsType != kIncludeNoResiduals)
    {
      aLambdaMin = 0.6;
      aLambdaMax = 1.5;
    }
  }

//-----------------------------------------------------------------------------

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

  DualieFitGenerator* tLamKchP = new DualieFitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType, tCentType,tAnRunType,tNPartialAnalysis,tGenType,tShareLambdaParams,tAllShareSingleLambdaParam);

  tLamKchP->SetFitType(tFitType);
  tLamKchP->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tLamKchP->SetNonFlatBgdFitType(tNonFlatBgdFitType);
  tLamKchP->SetApplyMomResCorrection(ApplyMomResCorrection);
  tLamKchP->SetIncludeResidualCorrelationsType(tIncludeResidualsType, aLambdaMin, aLambdaMax);  //TODO fix this in FitGenerator
  if(!UnboundLambda) tLamKchP->SetAllLambdaParamLimits(aLambdaMin, aLambdaMax);
  tLamKchP->SetChargedResidualsType(tChargedResidualsType);
  tLamKchP->SetResPrimMaxDecayType(tResPrimMaxDecayType);



  if(FixAllLambdaTo1) tLamKchP->SetLambdaParamStartValue(1.0, false, kMB, true);


  if(bDoFit) tLamKchP->DoFit(true, true, tMaxFitKStar);

//-------------------------------------------------------------------------------

  tFullTimer.Stop();
  cout << "Finished program: ";
  tFullTimer.PrintInterval();

  theApp->Run(kTRUE); //Run the TApp to pause the code.
  // Select "Exit ROOT" from Canvas "File" menu to exit
  // and execute the next statements.
cout << "DONE" << endl;
  return 0;
}
