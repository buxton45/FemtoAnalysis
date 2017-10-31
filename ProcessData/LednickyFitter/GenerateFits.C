#include "FitGenerator.h"
class FitGenerator;

int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  TString tResultsDate = "20161027";

  AnalysisType tAnType = kLamKchP;
  AnalysisRunType tAnRunType = kTrain;
  int tNPartialAnalysis = 2;
//  CentralityType tCentType = k0010;  //TODO
  CentralityType tCentType = kMB;  //TODO
  FitGeneratorType tGenType = kPairwConj;
  FitType tFitType = kChi2PML;
  bool tShareLambdaParams = false;
  bool tAllShareSingleLambdaParam = false;

  bool SaveImages = false;
  bool SaveImagesInRootFile = false;
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  NonFlatBgdFitType tNonFlatBgdFitType = kLinear;

  bool IncludeResiduals = true;

  bool UseAll10Residuals = true;
  bool UnboundLambda = true;
  bool UseCoulombOnlyInterpCfsForChargedResiduals = true;
  bool UseCoulombOnlyInterpCfsForXiKResiduals = false;

  bool FixRadii = false;
  bool FixD0 = false;

  double aLambdaMin=0., aLambdaMax=1.;
  if(UnboundLambda) aLambdaMax=0.;

  bool bDrawResiduals = false;

  TString tGeneralAnTypeName;
  if(tAnType==kLamK0 || tAnType==kALamK0) tGeneralAnTypeName = "cLamK0";
  else if(tAnType==kLamKchP || tAnType==kALamKchM || tAnType==kLamKchM || tAnType==kALamKchP) tGeneralAnTypeName = "cLamcKch";
  else assert(0);


  TString tDirectoryBase = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_%s_%s/",tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBase = TString::Format("%sResults_%s_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC = TString::Format("%sResults_%sMC_%s",tDirectoryBase.Data(),tGeneralAnTypeName.Data(),tResultsDate.Data());

  TString tSaveDirectoryBase = TString::Format("/home/jesse/Analysis/Presentations/GroupMeetings/20171102/Figures/%s/", cAnalysisBaseTags[tAnType]);
/*
  TString tSaveDirectoryBase = TString::Format("/home/jesse/Analysis/Presentations/AliFemto/20170913/Figures/%s/", cAnalysisBaseTags[tAnType]);
    if(UseAll10Residuals) tSaveDirectoryBase += TString("10Residuals/");
    else tSaveDirectoryBase += TString("3Residuals/");

    if(UnboundLambda) tSaveDirectoryBase += TString("NoLimits/");
    else tSaveDirectoryBase += TString("LimitedLambda/");

    if(UseCoulombOnlyInterpCfsForChargedResiduals) tSaveDirectoryBase += TString("UsingCoulombOnlyInterpCfs/");
    else tSaveDirectoryBase += TString("UsingXiKData/");
*/
//  tSaveDirectoryBase = tDirectoryBase;

  TString tSaveNameModifier = "";
  if(ApplyMomResCorrection) tSaveNameModifier += TString("_MomResCrctn");
  if(ApplyNonFlatBackgroundCorrection) tSaveNameModifier += TString("_NonFlatBgdCrctn");
  if(tAllShareSingleLambdaParam) tSaveNameModifier += TString("_SingleLamParam");
  if(IncludeResiduals && UseAll10Residuals) tSaveNameModifier += TString("_10ResidualsIncluded");
  if(IncludeResiduals && !UseAll10Residuals) tSaveNameModifier += TString("_3ResidualsIncluded");
  if(FixRadii) tSaveNameModifier += TString("_FixedRadii");
  if(FixD0) tSaveNameModifier += TString("_FixedD0");

  if(UseCoulombOnlyInterpCfsForXiKResiduals && UseCoulombOnlyInterpCfsForChargedResiduals) tSaveNameModifier += TString("_UsingCoulombOnlyInterpCfsForAll");
  else if(UseCoulombOnlyInterpCfsForChargedResiduals) tSaveNameModifier += TString("_UsingCoulombOnlyInterpCfs");

  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType, tCentType,tAnRunType,tNPartialAnalysis,tGenType,tShareLambdaParams,tAllShareSingleLambdaParam);
//  FitGenerator* tLamKchP = new FitGenerator(tFileLocationBase,tFileLocationBaseMC,tAnType,{k0010,k1030},tAnRunType,tNPartialAnalysis,tGenType,tShareLambdaParams,tAllShareSingleLambdaParam);
//  tLamKchP->SetRadiusStartValues({3.0,4.0,5.0});
//  tLamKchP->SetRadiusLimits({{0.,10.},{0.,10.},{0.,10.}});
  tLamKchP->SetSaveLocationBase(tSaveDirectoryBase,tSaveNameModifier);
  //tLamKchP->SetFitType(kChi2);


//  TCanvas* tKStarCan = tLamKchP->DrawKStarCfs();

  //tLamKchP->SetDefaultSharedParameters();

//TODO!!!!  If I want to apply mom res correction to full fit, I need to give non-central analyses ability to grab
//           the matrix from the central analyses
  tLamKchP->SetFitType(tFitType);
  tLamKchP->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tLamKchP->SetNonFlatBgdFitType(tNonFlatBgdFitType);
  tLamKchP->SetApplyMomResCorrection(ApplyMomResCorrection);
  tLamKchP->SetIncludeResidualCorrelations(IncludeResiduals, aLambdaMin, aLambdaMax);
  tLamKchP->SetUseCoulombOnlyInterpCfsForChargedResiduals(UseCoulombOnlyInterpCfsForChargedResiduals);
  tLamKchP->SetUseCoulombOnlyInterpCfsForXiKResiduals(UseCoulombOnlyInterpCfsForXiKResiduals);
  if(FixRadii) 
  {
    if(tAnType==kLamK0 || tAnType==kALamK0) tLamKchP->SetRadiusLimits({{3.25, 3.25}, {2.75, 2.75}, {2.25, 2.25}});
    else tLamKchP->SetRadiusLimits({{3.5, 3.5}, {3.25, 3.25}, {2.5, 2.5}});
  }
  if(FixD0) tLamKchP->SetScattParamStartValue(0., kd0, true);

  tLamKchP->DoFit();
  TCanvas* tKStarwFitsCan = tLamKchP->DrawKStarCfswFits(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages);
//  TCanvas* tKStarCfs = tLamKchP->DrawKStarCfs(SaveImages);
//  TCanvas* tModelKStarCfs = tLamKchP->DrawModelKStarCfs(SaveImages);
//  tLamKchP->FindGoodInitialValues(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection);

//-------------------------------------------------------------------------------
  TObjArray* tAllCanLamKchP;
  TCanvas* tCanPrimwFitsAndResidual;
  TObjArray* tAllResWithTransMatrices;

  if(IncludeResiduals && bDrawResiduals)
  {
    TCanvas* tCanLamKchP = tLamKchP->DrawResiduals(0,k0010,cAnalysisBaseTags[tAnType]);

    tAllCanLamKchP = tLamKchP->DrawAllResiduals(SaveImages);

//    TCanvas* tCanPrimWithRes = tLamKchP->DrawPrimaryWithResiduals(0,k0010,TString("PrimaryWithResidual_")+TString(cAnalysisBaseTags[tAnType]));
    tCanPrimwFitsAndResidual = tLamKchP->DrawKStarCfswFitsAndResiduals(ApplyMomResCorrection,ApplyNonFlatBackgroundCorrection,tNonFlatBgdFitType,SaveImages);

    tAllResWithTransMatrices = tLamKchP->DrawAllResidualsWithTransformMatrices(SaveImages);
  }

//-------------------------------------------------------------------------------
  if(SaveImagesInRootFile)
  {
    TFile *tFile = new TFile(tLamKchP->GetSaveLocationBase() + TString(cAnalysisBaseTags[tAnType]) + TString("Plots") + tLamKchP->GetSaveNameModifier() + TString(".root"), "RECREATE");
    tKStarwFitsCan->Write();
    if(IncludeResiduals && bDrawResiduals)
    {
      for(int i=0; i<tAllCanLamKchP->GetEntries(); i++) (TCanvas*)tAllCanLamKchP->At(i)->Write();
      tCanPrimwFitsAndResidual->Write();
    }
    tFile->Close();
  }

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
