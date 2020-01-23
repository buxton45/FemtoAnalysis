#include "TripleFitGenerator.h"
class TripleFitGenerator;


//TODO TODO TODO TODO TODO Final versions of plots should still be generated with complete fit, to make sure
//TODO TODO TODO TODO TODO normalizations etc. are correct!


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
td3dVec GetAllParamValuesAndErrors(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType)
{
  td3dVec tReturnVec(0);
  for(unsigned int i=0; i<kMB; i++) tReturnVec.push_back(GetParamValuesAndErrors(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, static_cast<CentralityType>(i)));
  return tReturnVec;
}


//-----------------------------------------------------------------------------------------------------------------
//TODO TODO TODO TODO TODO Final versions of plots should still be generated with complete fit, to make sure
//TODO TODO TODO TODO TODO normalizations etc. are correct!
//-----------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) 
{
  TApplication* theApp = new TApplication("App", &argc, argv);
  //The TApplication object allows the execution of the code to pause.
  //This allows the user a chance to look at and manipulate a TBrowser before
  //the program ends and closes everything

  ChronoTimer tFullTimer(kSec);
  tFullTimer.Start();

  //--Rarely change---------------------
  AnalysisRunType tAnRunType_LamKch = kTrain, tAnRunType_LamK0 = kTrain;
  int tNPartialAnalysis = 2;
  CentralityType tCentType = kMB/*k0010*/;
  FitGeneratorType tGenType = kPairwConj;
  FitType tFitType = kChi2PML;
  //------------------------------------

  //*****************************************
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
  TString tFitInfoTString = FitValuesWriter::BuildFitInfoTString(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, 
                                                                 tIncludeResidualsType, tResPrimMaxDecayType, 
                                                                 tChargedResidualsType, FixD0, 
                                                                 bUseStavCf, FixAllLambdaTo1, FixAllNormTo1, FixRadii, FixAllScattParams, 
                                                                 tShareLambdaParams, tAllShareSingleLambdaParam, UsemTScalingOfResidualRadii, true, 
                                                                 tTripleShareLambda, tTripleShareRadii);
                                                                 
  //For Triple analyses, systematics only currently stored in LamKch directory
  TString tSystematicsFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/%s/Systematics/FinalFitSystematics_wFitRangeSys%s.txt", tResultsDate.Data(), tFitInfoTString.Data(), tFitInfoTString.Data());    

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
  tTriple->SetSaveLocationBase(tSaveDirectoryBase_LamKch, tSaveDirectoryBase_LamK0, tFitInfoTString);
  tTriple->SetSaveFileType(tSaveFileType);
 
//-----------------------------------------------------------------------------    
  
  td3dVec tParsAndErrs_LamK0 =  GetAllParamValuesAndErrors(tLocationMasterFitResults_LamK0, tSystematicsFileLocation, tFitInfoTString, kLamK0);
  td3dVec tParsAndErrs_ALamK0 = GetAllParamValuesAndErrors(tLocationMasterFitResults_LamK0, tSystematicsFileLocation, tFitInfoTString, kALamK0);

  td3dVec tParsAndErrs_LamKchP =  GetAllParamValuesAndErrors(tLocationMasterFitResults_LamKch, tSystematicsFileLocation, tFitInfoTString, kLamKchP);
  td3dVec tParsAndErrs_ALamKchM = GetAllParamValuesAndErrors(tLocationMasterFitResults_LamKch, tSystematicsFileLocation, tFitInfoTString, kALamKchM);

  td3dVec tParsAndErrs_LamKchM =  GetAllParamValuesAndErrors(tLocationMasterFitResults_LamKch, tSystematicsFileLocation, tFitInfoTString, kLamKchM);
  td3dVec tParsAndErrs_ALamKchP = GetAllParamValuesAndErrors(tLocationMasterFitResults_LamKch, tSystematicsFileLocation, tFitInfoTString, kALamKchP);  
  
  
  tTriple->SetRadiusStartValues(vector<double>{tParsAndErrs_LamKchP[0][0][1], 
                                               tParsAndErrs_LamKchP[1][0][1],
                                               tParsAndErrs_LamKchP[2][0][1]},
                                               true);
                                              
  tTriple->SetScattParamStartValues(kLamK0,   tParsAndErrs_LamK0[0][0][2],   tParsAndErrs_LamK0[0][0][3],   tParsAndErrs_LamK0[0][0][4], true);   
  tTriple->SetScattParamStartValues(kLamKchP, tParsAndErrs_LamKchP[0][0][2], tParsAndErrs_LamKchP[0][0][3], tParsAndErrs_LamKchP[0][0][4], true);   
  tTriple->SetScattParamStartValues(kLamKchM, tParsAndErrs_LamKchM[0][0][2], tParsAndErrs_LamKchM[0][0][3], tParsAndErrs_LamKchM[0][0][4], true);       
  
  tTriple->SetLambdaParamStartValue(tParsAndErrs_LamKchP[k0010][0][0], true, k0010, true); 
  tTriple->SetLambdaParamStartValue(tParsAndErrs_LamKchP[k1030][0][0], true, k1030, true);                                           
  tTriple->SetLambdaParamStartValue(tParsAndErrs_LamKchP[k3050][0][0], true, k3050, true);                                                                                          
  
//-----------------------------------------------------------------------------  

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

  tTriple->DoFitOnce(tTripleShareLambda, tTripleShareRadii, tMaxFitKStar);


  TCanvas* tKStarwFitsCan_CombineConj_AllAn_Zoom = tTriple->DrawKStarCfswFits_CombineConj_AllAn(ApplyMomResCorrection, ApplyNonFlatBackgroundCorrection, tNonFlatBgdFitTypes, SaveImages, bDrawSysErrs, true, tSuppressFitInfoOutput, tLabelLines);

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
