#include "TripleFitGenerator.h"
class TripleFitGenerator;

//________________________________________________________________________________________________________________
td2dVec GetParamValuesAndErrors(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType)
{
  td1dVec tParVals(6), tParStatErr(6), tParSystErr(6);
  td2dVec tReturnVec;

  ParameterType tParamType;
  FitParameter* tFitParam;
  for(unsigned int iParam=0; iParam<5; iParam++)
  {
    tParamType = static_cast<ParameterType>(iParam);
    tFitParam = FitValuesWriterwSysErrs::GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, tParamType);

    tParVals[iParam] = tFitParam->GetFitValue();
    tParStatErr[iParam] = tFitParam->GetFitValueError();
    tParSystErr[iParam] = tFitParam->GetFitValueSysError();
  }
  //Normalization
  tParVals[5] = 1.;
  tParStatErr[5] = 0.;
  tParSystErr[5] = 0.;

  tReturnVec.push_back(tParVals);
  tReturnVec.push_back(tParStatErr);
  tReturnVec.push_back(tParSystErr);

  return tReturnVec;
}


//________________________________________________________________________________________________________________
void SetParamValues(TripleFitGenerator* aTriple, 
                    TString aLocationMasterFitResults_LamKch, TString aLocationMasterFitResults_LamK0,
                    TString aSystematicsFileLocation, TString aFitInfoTString)
{ 
  td2dVec tParsAndErrs_LamKchP_0010 = GetParamValuesAndErrors(aLocationMasterFitResults_LamKch, aSystematicsFileLocation, aFitInfoTString, kLamKchP, k0010);
  td2dVec tParsAndErrs_LamKchP_1030 = GetParamValuesAndErrors(aLocationMasterFitResults_LamKch, aSystematicsFileLocation, aFitInfoTString, kLamKchP, k1030);
  td2dVec tParsAndErrs_LamKchP_3050 = GetParamValuesAndErrors(aLocationMasterFitResults_LamKch, aSystematicsFileLocation, aFitInfoTString, kLamKchP, k3050);
  
  td2dVec tParsAndErrs_LamKchM_0010 = GetParamValuesAndErrors(aLocationMasterFitResults_LamKch, aSystematicsFileLocation, aFitInfoTString, kLamKchM, k0010);
  td2dVec tParsAndErrs_LamK0_0010 = GetParamValuesAndErrors(aLocationMasterFitResults_LamK0, aSystematicsFileLocation, aFitInfoTString, kLamK0, k0010);

  
  //For above tParsAndErrs_LamKchP_0010[0] = vector of values
  //          tParsAndErrs_LamKchP_0010[1] = vector of stat. errors
  //          tParsAndErrs_LamKchP_0010[2] = vector of syst. errors
  //---------------------------
  
  //-----lambda params
  aTriple->SetLambdaParamStartValue(tParsAndErrs_LamKchP_0010[0][0], false, k0010, true);
  aTriple->SetLambdaParamStartValue(tParsAndErrs_LamKchP_1030[0][0], false, k1030, true);
  aTriple->SetLambdaParamStartValue(tParsAndErrs_LamKchP_3050[0][0], false, k3050, true);
  

                                                       
  //-----radii 
  aTriple->SetRadiusStartValues(vector<double>{tParsAndErrs_LamKchP_0010[0][1], 
                                               tParsAndErrs_LamKchP_1030[0][1], 
                                               tParsAndErrs_LamKchP_3050[0][1]}, true);
                                               
  //-----scattering parameters
  aTriple->SetScattParamStartValues(kLamKchP, tParsAndErrs_LamKchP_0010[0][2], tParsAndErrs_LamKchP_0010[0][3], tParsAndErrs_LamKchP_0010[0][4], true);
  aTriple->SetScattParamStartValues(kLamKchM, tParsAndErrs_LamKchM_0010[0][2], tParsAndErrs_LamKchM_0010[0][3], tParsAndErrs_LamKchM_0010[0][4], true);
  aTriple->SetScattParamStartValues(kLamK0, tParsAndErrs_LamK0_0010[0][2], tParsAndErrs_LamK0_0010[0][3], tParsAndErrs_LamK0_0010[0][4], true);    

}


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();
//-----------------------------------------------------------------------------
  
  //TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
  //All normalizations are set to 1!!!!!!!!!!!
  //In reality, differ slightly from 1!!!!!!!!!!
  //This should only be used for quick and dirty plots!!!!!!!!!!
  //Full fit should be performed for actual results!!!!!!!!!!!!!

  //--Rarely change---------------------
  AnalysisRunType tAnRunType_LamKch = kTrain, tAnRunType_LamK0 = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB/*k0010*/;
  FitGeneratorType tGenType = kPairwConj;
  FitType tFitType = kChi2PML;
  //------------------------------------

  //*****************************************
  bool bDoFit = true;
  bool bGenerateContours = false;

//  TString tResultsDate = "20180505";
  TString tResultsDate = "20190319";

  double tMaxFitKStar=0.3;

  bool bUseStavCf=false;
  //*****************************************

  //--Save options
  bool SaveImages = false;
  TString tSaveFileType = "pdf";

  //--Sharing lambda
  bool tShareLambdaParams = true;          //If true, only share lambda parameters across like-centralities
  bool tAllShareSingleLambdaParam = false;

  //--Triple sharing options
  bool tTripleShareLambda = true;
  bool tTripleShareRadii = true;

  //--Corrections
  bool ApplyMomResCorrection = true;
  bool ApplyNonFlatBackgroundCorrection = true;
  if(bUseStavCf) ApplyNonFlatBackgroundCorrection = false;
  NonFlatBgdFitType tNonFlatBgdFitType_LamKch = kPolynomial;
  NonFlatBgdFitType tNonFlatBgdFitType_LamK0  = kPolynomial;
  vector<NonFlatBgdFitType> tNonFlatBgdFitTypes{tNonFlatBgdFitType_LamK0, tNonFlatBgdFitType_LamK0, 
                                                tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamKch};

  bool UseNewBgdTreatment = false;  //TODO For now, and maybe forever, should always be FALSE!!!!!
    if(UseNewBgdTreatment) tMaxFitKStar = 0.5;

  //--Residuals
  IncludeResidualsType tIncludeResidualsType = kInclude3Residuals; 
  ChargedResidualsType tChargedResidualsType = kUseXiDataAndCoulombOnlyInterp/*kUseCoulombOnlyInterpForAll*/;
  ResPrimMaxDecayType tResPrimMaxDecayType = k10fm;

  //--Bound lambda
  bool UnboundLambda = true;
  double aLambdaMin=0.6, aLambdaMax=1.1;
  if(UnboundLambda) {aLambdaMin=0.; aLambdaMax=0.;}

  //--Fix parameters
  bool FixRadii = false;
  bool FixD0 = false;
  bool FixAllScattParams = false;
  bool FixAllLambdaTo1 = false;
  if(FixAllLambdaTo1) tAllShareSingleLambdaParam = true;
  bool FixAllNormTo1 = false;

  //--mT scaling
  bool UsemTScalingOfResidualRadii = false;
  double mTScalingPowerOfResidualRadii = -0.5;

  //--Plotting options
  bool bZoomROP = false;
  bool bDrawResiduals = false;
  bool bDrawResv2 = true;
  bool bDrawPartAn = false;

  bool bDrawSysErrs = true;

  bool tSuppressFitInfoOutput=false;
  bool tLabelLines=true;
  if(tLabelLines) tSuppressFitInfoOutput=true;
  bool bDrawNewSQM = false;

//-----------------------------------------------------------------------------
  if(tShareLambdaParams==false && tTripleShareLambda==true)
  {
    cout << "!!!!!!!!!!!!!!! tShareLambdaParams==false && tTripleShareLambda==true !!!!!!!!!!!!!!!" << endl;
    cout << "This means Conjugate pairs in each centrality WILL NOT share a lambda, but the different FitGenerators WILL" << endl;
    cout << "2 lambda parameters per centrality (lam1:LamKchP & LamKchM and lam2:ALamKchM & ALamKchP)" << endl;
    cout << "6 lambda parameters in total for MB analysis" << endl;
    cout << "Is this really what you want? (0=No  1=Yes)" << endl;

    int tResponse;
    cin >> tResponse;
    assert(tResponse);
  }

//-----------------------------------------------------------------------------

  TString tDirectoryBase_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/",tResultsDate.Data());
  TString tFileLocationBase_LamKch = TString::Format("%sResults_cLamcKch_%s",tDirectoryBase_LamKch.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_LamKch = TString::Format("%sResults_cLamcKchMC_%s",tDirectoryBase_LamKch.Data(),tResultsDate.Data());

  TString tSaveDirectoryBase_LamKch = tDirectoryBase_LamKch;

  TString tLocationMasterFitResults_LamKch = TString::Format("%sMasterFitResults_%s.txt", tDirectoryBase_LamKch.Data(), tResultsDate.Data());

  //-----

  TString tDirectoryBase_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/",tResultsDate.Data());
  TString tFileLocationBase_LamK0 = TString::Format("%sResults_cLamK0_%s",tDirectoryBase_LamK0.Data(),tResultsDate.Data());
  TString tFileLocationBaseMC_LamK0 = TString::Format("%sResults_cLamK0MC_%s",tDirectoryBase_LamK0.Data(),tResultsDate.Data());

  TString tSaveDirectoryBase_LamK0 = tDirectoryBase_LamK0;

  TString tLocationMasterFitResults_LamK0 = TString::Format("%sMasterFitResults_%s.txt", tDirectoryBase_LamK0.Data(), tResultsDate.Data());

//-----------------------------------------------------------------------------

  TString tSaveNameModifier = LednickyFitter::BuildSaveNameModifier(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, tIncludeResidualsType, tResPrimMaxDecayType, tChargedResidualsType, FixD0, bUseStavCf, FixAllLambdaTo1, FixAllNormTo1, FixRadii, FixAllScattParams, tShareLambdaParams, tAllShareSingleLambdaParam, UsemTScalingOfResidualRadii, true, tTripleShareLambda, tTripleShareRadii);

  //For Triple analyses, systematics only currently stored in LamKch directory
  TString tSystematicsFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Systematics/FinalFitSystematics_wFitRangeSys%s.txt", tResultsDate.Data(), tSaveNameModifier.Data(), tSaveNameModifier.Data());


//-----------------------------------------------------------------------------
  bool bExistsCurrentSysFile;
  ifstream tFileIn;
  tFileIn.open(tSystematicsFileLocation);
  if(tFileIn) bExistsCurrentSysFile=true;
  else bExistsCurrentSysFile = false;
  tFileIn.close();
  if(!bExistsCurrentSysFile) cout << "WARNING!!!!!!!!!!!!!!!!!!!!!" << endl << "!bExistsCurrentSysFile, so syst. errs. on fit parameters not precisely accurate" << endl << endl;

//-----------------------------------------------------------------------------

  TripleFitGenerator* tTriple = new TripleFitGenerator(tFileLocationBase_LamKch, tFileLocationBaseMC_LamKch, tFileLocationBase_LamK0, tFileLocationBaseMC_LamK0, tCentType, tAnRunType_LamKch, tAnRunType_LamK0, tNPartialAnalysis, tGenType, tShareLambdaParams, tAllShareSingleLambdaParam, "", "", bUseStavCf);
  tTriple->SetSaveLocationBase(tSaveDirectoryBase_LamKch, tSaveDirectoryBase_LamK0,tSaveNameModifier);
  tTriple->SetSaveFileType(tSaveFileType);

  tTriple->SetFitType(tFitType);
  tTriple->SetApplyNonFlatBackgroundCorrection(ApplyNonFlatBackgroundCorrection);
  tTriple->SetNonFlatBgdFitTypes(tNonFlatBgdFitType_LamKch, tNonFlatBgdFitType_LamK0);
  tTriple->SetUseNewBgdTreatment(UseNewBgdTreatment);
  tTriple->SetApplyMomResCorrection(ApplyMomResCorrection);
  tTriple->SetIncludeResidualCorrelationsType(tIncludeResidualsType, aLambdaMin, aLambdaMax);  //TODO fix this in FitGenerator
  if(!UnboundLambda) tTriple->SetAllLambdaParamLimits(aLambdaMin, aLambdaMax);
  tTriple->SetChargedResidualsType(tChargedResidualsType);
  tTriple->SetResPrimMaxDecayType(tResPrimMaxDecayType);
  tTriple->SetMasterFileLocation(tLocationMasterFitResults_LamKch, tLocationMasterFitResults_LamK0);
  if(bExistsCurrentSysFile) tTriple->SetSystematicsFileLocation(tSystematicsFileLocation, tSystematicsFileLocation);

  if(FixRadii) 
  {
    tTriple->SetRadiusLimits({{4.13, 4.13}, {3.34, 3.34}, {2.59, 2.59}});
  }

  if(FixAllLambdaTo1) tTriple->SetLambdaParamStartValue(1.0, false, kMB, true);
  if(FixAllNormTo1) tTriple->SetFixNormParams(FixAllNormTo1);
  if(UsemTScalingOfResidualRadii) tTriple->SetUsemTScalingOfResidualRadii(UsemTScalingOfResidualRadii, mTScalingPowerOfResidualRadii);

  if(ApplyNonFlatBackgroundCorrection && tNonFlatBgdFitType_LamKch == kPolynomial && tNonFlatBgdFitType_LamK0 == kPolynomial) 
  {
    tTriple->SetMinMaxBgdFit(kLamKchP, 0.32, 0.80);
    tTriple->SetMinMaxBgdFit(kLamK0, 0.32, 0.80);
  }
  

//-------------------------------------------------------------------------------

  SetParamValues(tTriple, 
                 tLocationMasterFitResults_LamKch, tLocationMasterFitResults_LamK0,
                 tSystematicsFileLocation, tSaveNameModifier);

  tTriple->InitializeGenerator(tTripleShareLambda, tTripleShareRadii, tMaxFitKStar);
  tTriple->GetMasterLednickyFitter()->InitializeFitter();
  
  int tNMinuitParams = tTriple->GetMasterSharedAn()->GetNMinuitParams();
  assert(tNMinuitParams==51);

  double tParVal=0., tParErr=0.;
  double tChi2=0;
  double *tParams = new double[tNMinuitParams];
  for(int i=0; i<tNMinuitParams; i++)
  {
    tTriple->GetMasterSharedAn()->GetMinuitObject()->GetParameter(i, tParVal, tParErr);
    tParams[i]=tParVal;
  }
  tTriple->GetMasterLednickyFitter()->CalculateFitFunction(tNMinuitParams, tChi2, tParams);
  tTriple->GetMasterLednickyFitter()->Finalize();
  tTriple->ReturnNecessaryInfoToFitGenerators();


//-------------------------------------------------------------------------------

  TCanvas* tKStarwFitsCan_CombineConj_AllAn_Zoom = tTriple->DrawKStarCfswFits_CombineConj_AllAn(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bDrawSysErrs, true, tSuppressFitInfoOutput, tLabelLines);
  TCanvas* tKStarwFitsCan_CombineConj_AllAn_UnZoom = tTriple->DrawKStarCfswFits_CombineConj_AllAn(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bDrawSysErrs, false, tSuppressFitInfoOutput, tLabelLines);

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
