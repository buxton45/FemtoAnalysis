///////////////////////////////////////////////////////////////////////////
// FitSystematicAnalysis:                                                //
///////////////////////////////////////////////////////////////////////////


#include "FitSystematicAnalysis.h"

#ifdef __ROOT__
ClassImp(FitSystematicAnalysis)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitSystematicAnalysis::FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                       TString aDirNameModifierBase2, vector<double> &aModifierValues2,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fConjAnalysisType(kALamK0),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType(kLinear),
  fApplyMomResCorrection(false),

  fIncludeResidualsType(kIncludeNoResiduals),
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fFixD0(false),

  fSaveDirectory(""),

  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(aDirNameModifierBase2),
  fModifierValues1(aModifierValues1),
  fModifierValues2(aModifierValues2)

{
  if(!fDirNameModifierBase2.IsNull()) assert(fModifierValues1.size() == fModifierValues2.size());
  SetConjAnalysisType();
}


//________________________________________________________________________________________________________________
FitSystematicAnalysis::FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fConjAnalysisType(kALamK0),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType(kLinear),
  fApplyMomResCorrection(false),

  fIncludeResidualsType(kIncludeNoResiduals),
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fFixD0(false),

  fSaveDirectory(""),

  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(0),
  fModifierValues1(aModifierValues1),
  fModifierValues2(0)

{
  fDirNameModifierBase2 = "";
  fModifierValues2 = vector<double> (0);

  *this = FitSystematicAnalysis(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, aDirNameModifierBase1, aModifierValues1, fDirNameModifierBase2, fModifierValues2, aCentralityType, aGeneratorType, aShareLambdaParams, aAllShareSingleLambdaParam);
}

//________________________________________________________________________________________________________________
FitSystematicAnalysis::FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, bool aShareLambdaParams, bool aAllShareSingleLambdaParam) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fConjAnalysisType(kALamK0),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType(kLinear),
  fApplyMomResCorrection(false),

  fIncludeResidualsType(kIncludeNoResiduals),
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fFixD0(false),

  fSaveDirectory(""),

  fDirNameModifierBase1(0),
  fDirNameModifierBase2(0),
  fModifierValues1(0),
  fModifierValues2(0)

{
  fDirNameModifierBase1 = "";
  fModifierValues1 = vector<double> (0);

  fDirNameModifierBase2 = "";
  fModifierValues2 = vector<double> (0);

  *this = FitSystematicAnalysis(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, fDirNameModifierBase1, fModifierValues1, fDirNameModifierBase2, fModifierValues2, aCentralityType, aGeneratorType, aShareLambdaParams, aAllShareSingleLambdaParam);
}


//________________________________________________________________________________________________________________
FitSystematicAnalysis::~FitSystematicAnalysis()
{
/*no-op*/
}

//________________________________________________________________________________________________________________
void FitSystematicAnalysis::SetConjAnalysisType()
{
  assert(fAnalysisType==kLamK0 || fAnalysisType==kLamKchP || fAnalysisType==kLamKchM);

  if(fAnalysisType==kLamK0) fConjAnalysisType=kALamK0;
  else if(fAnalysisType==kLamKchP) fConjAnalysisType=kALamKchM;
  else if(fAnalysisType==kLamKchM) fConjAnalysisType=kALamKchP;
  else assert(0);
}

//________________________________________________________________________________________________________________
TString FitSystematicAnalysis::GetCutValues(int aIndex)
{
  TString tCutVal1a, tCutVal1b, tCutVal1Tot;

  tCutVal1a = fDirNameModifierBase1;
    tCutVal1a.Remove(TString::kBoth,'_');
    tCutVal1a += TString::Format(" = %0.6f",fModifierValues1[aIndex]);

  tCutVal1Tot = tCutVal1a;

  if(!fDirNameModifierBase2.IsNull())
  {
    tCutVal1b = fDirNameModifierBase2;
      tCutVal1b.Remove(TString::kBoth,'_');
      tCutVal1b += TString::Format(" = %0.6f",fModifierValues2[aIndex]);

    tCutVal1Tot += TString::Format(" and %s",tCutVal1b.Data());
  }
  return tCutVal1Tot;
}


//________________________________________________________________________________________________________________
void FitSystematicAnalysis::OutputCutValues(int aIndex, ostream &aOut)
{
  aOut << "______________________________________________________________________________" << endl;
  aOut << GetCutValues(aIndex) << endl;
}

//________________________________________________________________________________________________________________
double FitSystematicAnalysis::ExtractParamValue(TString aString)
{
  TString tString = TString(aString);

  int tBeg = tString.First(":");
  tString.Remove(0,tBeg+1);

  int tEnd = tString.First("+");
  tString.Remove(tEnd,tString.Length()-tEnd);
  tString.Strip(TString::kBoth, ' ');

  return tString.Atof();  
}

//________________________________________________________________________________________________________________
void FitSystematicAnalysis::AppendDifference(vector<vector<TString> > &a2dVec, int aCut, int aLineNumber)
{
  TString tValString = a2dVec[aCut][aLineNumber];
  TString tDefaultValString = a2dVec[1][aLineNumber];

  double tVal = ExtractParamValue(tValString);
  double tDefaultVal = ExtractParamValue(tDefaultValString);

  double tDiff = (tVal-tDefaultVal)/tDefaultVal;
  tDiff *= 100;

  a2dVec[aCut][aLineNumber] += TString::Format(" (%f%%) ",tDiff);
}


//________________________________________________________________________________________________________________
void FitSystematicAnalysis::PrintText2dVec(vector<vector<TString> > &a2dVec, ostream &aOut, int aNCuts)
{
  int tNCuts = (int)a2dVec.size();
  assert(tNCuts==aNCuts);
  for(unsigned int i=1; i<a2dVec.size(); i++) assert(a2dVec[i-1].size() == a2dVec[i].size());
  int tSize = a2dVec[0].size();
  for(int iLineNumber=0; iLineNumber<tSize; iLineNumber++)
  {
    for(int iCut=0; iCut<tNCuts; iCut++)
    {
      if(a2dVec[iCut][iLineNumber].Contains("+-")) AppendDifference(a2dVec, iCut, iLineNumber);
      aOut << std::setw(50) << TString(a2dVec[iCut][iLineNumber]) << " | ";
    }
    aOut << endl;
  }
}

//________________________________________________________________________________________________________________
void FitSystematicAnalysis::AppendFitInfo(TString &aSaveName)
{
//  LednickyFitter::AppendFitInfo(aSaveName, fApplyMomResCorrection, fApplyNonFlatBackgroundCorrection, fIncludeResidualsType, fResPrimMaxDecayType, fChargedResidualsType, fFixD0);

  TString tModifier = LednickyFitter::BuildSaveNameModifier(fApplyMomResCorrection, fApplyNonFlatBackgroundCorrection, fNonFlatBgdFitType, 
                                                            fIncludeResidualsType, fResPrimMaxDecayType, 
                                                            fChargedResidualsType, fFixD0,
                                                            false, false, false, false, false,
                                                            fShareLambdaParams, fAllShareSingleLambdaParam, false, false,
                                                            false, false);
  aSaveName += tModifier;
}

//________________________________________________________________________________________________________________
void FitSystematicAnalysis::SetRadiusStartValues(FitGeneratorAndDraw* aFitGen)
{
  TString tFitInfoTString = "";
  AppendFitInfo(tFitInfoTString);

  TString tParentResultsDate = "";
  if     (fSaveDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(fSaveDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else assert(0);

  TString tMasterFileLocation = "";
  if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)
  {
    tMasterFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());
  }
  else if(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM || fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) 
  {
    tMasterFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());
  }
  else assert(0);  //Not currently set up for XiK analysis


  td1dVec tRStartValues(3);  //Currently only set up for typical case of 0-10, 10-30, 30-50 together
  tRStartValues[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k0010, kRadius)->GetFitValue();
  tRStartValues[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k1030, kRadius)->GetFitValue();
  tRStartValues[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k3050, kRadius)->GetFitValue();

  aFitGen->SetRadiusStartValues(tRStartValues);
}



//________________________________________________________________________________________________________________
void FitSystematicAnalysis::SetLambdaStartValues(FitGeneratorAndDraw* aFitGen)
{
  int tNLamParams = 0;
  if(fAllShareSingleLambdaParam) tNLamParams = 1;
  else if(fShareLambdaParams) tNLamParams = 3;
  else tNLamParams = 6;


  TString tFitInfoTString = "";
  AppendFitInfo(tFitInfoTString);

  TString tParentResultsDate = "";
  if     (fSaveDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(fSaveDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else assert(0);

  TString tMasterFileLocation = "";
  if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)
  {
    tMasterFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());
  }
  else if(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM || fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) 
  {
    tMasterFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());
  }
  else assert(0);  //Not currently set up for XiK analysis



  if     (tNLamParams==1) 
  {
    aFitGen->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k0010, kLambda)->GetFitValue(), 
                                      false, k0010, false);
  }
  else if(tNLamParams==3)
  {
    aFitGen->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k0010, kLambda)->GetFitValue(),
                                      false, k0010, false);
    aFitGen->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k1030, kLambda)->GetFitValue(),
                                      false, k1030, false);
    aFitGen->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k3050, kLambda)->GetFitValue(),
                                      false, k3050, false);
  }
  else if(tNLamParams==6)
  {
    td1dVec tLamStartValues(tNLamParams);

    tLamStartValues[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k0010, kLambda)->GetFitValue();
    tLamStartValues[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fConjAnalysisType, k0010, kLambda)->GetFitValue();

    tLamStartValues[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k1030, kLambda)->GetFitValue();
    tLamStartValues[3] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fConjAnalysisType, k1030, kLambda)->GetFitValue();

    tLamStartValues[4] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k3050, kLambda)->GetFitValue();
    tLamStartValues[5] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fConjAnalysisType, k3050, kLambda)->GetFitValue();

    aFitGen->SetAllLambdaParamStartValues(tLamStartValues, false);
  }
  else assert(0);

  if((fAnalysisType==kLamK0 || fAnalysisType==kALamK0) && fIncludeResidualsType == kIncludeNoResiduals) aFitGen->SetAllLambdaParamLimits(0.4, 0.6);
  if((fAnalysisType==kLamK0 || fAnalysisType==kALamK0) && fIncludeResidualsType != kIncludeNoResiduals) aFitGen->SetAllLambdaParamLimits(0.6, 1.5);
}

//________________________________________________________________________________________________________________
void FitSystematicAnalysis::SetScattParamStartValues(FitGeneratorAndDraw* aFitGen)
{
  TString tFitInfoTString = "";
  AppendFitInfo(tFitInfoTString);

  TString tParentResultsDate = "";
  if     (fSaveDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(fSaveDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else assert(0);

  TString tMasterFileLocation = "";
  if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)
  {
    tMasterFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());
  }
  else if(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM || fAnalysisType==kLamKchM || fAnalysisType==kALamKchP) 
  {
    tMasterFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());
  }
  else assert(0);  //Not currently set up for XiK analysis


  aFitGen->SetScattParamStartValues(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k0010, kRef0)->GetFitValue(),
                                    FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k0010, kImf0)->GetFitValue(),
                                    FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, fAnalysisType, k0010, kd0)->GetFitValue(), 
                                    false);
  if(fFixD0) aFitGen->SetScattParamStartValue(0., kd0, true);
}


//________________________________________________________________________________________________________________
FitGeneratorAndDraw* FitSystematicAnalysis::BuildFitGenerator(AnalysisRunType aRunType, TString aDirNameModifier, NonFlatBgdFitType aNonFlatBgdFitType)
{
  //For now, it appears only needed parameters here are AnalysisRunType aRunType and TString aDirNameModifier (for FitGenerator constructor)
  // and NonFlatBgdFitType aNonFlatBgdFitType for assigning attributes
  // Otherwise, default members used

  FitGeneratorAndDraw* tFitGenerator = new FitGeneratorAndDraw(fFileLocationBase, fFileLocationBaseMC, fAnalysisType, fCentralityType, aRunType, 2, fFitGeneratorType, fShareLambdaParams, fAllShareSingleLambdaParam, aDirNameModifier);

  tFitGenerator->SetApplyNonFlatBackgroundCorrection(fApplyNonFlatBackgroundCorrection);
  tFitGenerator->SetNonFlatBgdFitType(aNonFlatBgdFitType);
  tFitGenerator->SetApplyMomResCorrection(fApplyMomResCorrection);
  if(fIncludeResidualsType != kIncludeNoResiduals)
  {
    if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0) tFitGenerator->SetIncludeResidualCorrelationsType(fIncludeResidualsType, 0.60, 1.50);  //TODO This is overridden below... is this what I want?
    else tFitGenerator->SetIncludeResidualCorrelationsType(fIncludeResidualsType, 0., 0.);
  }
  tFitGenerator->SetChargedResidualsType(fChargedResidualsType);
  tFitGenerator->SetResPrimMaxDecayType(fResPrimMaxDecayType);

  //----- Set appropriate parameter start values, and limits, to keep fitter from accidentally doing something crazy
  assert(fCentralityType==kMB);  //This will fail otherwise

//  SetRadiusStartValues(tFitGenerator);
//  SetLambdaStartValues(tFitGenerator);  
//  SetScattParamStartValues(tFitGenerator);
  //----------------------------------------------------------------------------------------------------------------

  return tFitGenerator;
}



//________________________________________________________________________________________________________________
void FitSystematicAnalysis::RunAllFits(bool aSaveImages, bool aWriteToTxtFile)
{
  vector<vector<TString> > tText2dVector(0);

  TString tSpecificSaveDirectory;
  if(aSaveImages || aWriteToTxtFile)
  {
    tSpecificSaveDirectory = fSaveDirectory;
    AppendFitInfo(tSpecificSaveDirectory);
    tSpecificSaveDirectory += TString("/");
    gSystem->mkdir(tSpecificSaveDirectory);
  }

  for(unsigned int i=0; i<fModifierValues1.size(); i++)
  {
    TString tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
    if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

    FitGeneratorAndDraw* tFitGenerator = BuildFitGenerator(kTrainSys, tDirNameModifier, fNonFlatBgdFitType);
    tFitGenerator->DoFit();

//    OutputCutValues(i,aOut);
//    tFitGenerator->WriteAllFitParameters(aOut);
    TString tCutValue = GetCutValues(i);
    vector<TString> tFitParamsVec = tFitGenerator->GetAllFitParametersTStringVector();
    tFitParamsVec.insert(tFitParamsVec.begin(),tCutValue);
    tText2dVector.push_back(tFitParamsVec);

    TCanvas* tKStarwFitsCan = tFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,fNonFlatBgdFitType,false,false);
    if(aSaveImages)
    {
      TString tImageSaveName = tSpecificSaveDirectory;
      tImageSaveName += tKStarwFitsCan->GetTitle();
      TString tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
      if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

      tImageSaveName += tDirNameModifier;
      AppendFitInfo(tImageSaveName);

      tImageSaveName += TString(".pdf");
      tKStarwFitsCan->SaveAs(tImageSaveName);
    }
    delete tFitGenerator;
  }

  if(!aWriteToTxtFile) PrintText2dVec(tText2dVector);
  else
  {
    TString tOutputFileName = TString::Format("%sCfFitValues_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[fAnalysisType], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName);
    tOutputFileName += TString(".txt");
    std::ofstream tOutputFile;
    tOutputFile.open(tOutputFileName);

    PrintText2dVec(tText2dVector,tOutputFile);

    tOutputFile.close();
  }
}


//________________________________________________________________________________________________________________
void FitSystematicAnalysis::RunVaryFitRange(bool aSaveImages, bool aWriteToTxtFile, double aMaxKStar1, double aMaxKStar2, double aMaxKStar3)
{
  assert(fModifierValues1.size()==0);  //this is not intended for use with various modifier values, but for the final analysis
  int tNRangeValues = 3;
  vector<double> tRangeVec = {aMaxKStar1,aMaxKStar2,aMaxKStar3};

  vector<vector<TString> > tText2dVector(0);

  TString tSpecificSaveDirectory;
  if(aSaveImages || aWriteToTxtFile)
  {
    tSpecificSaveDirectory = fSaveDirectory;
    AppendFitInfo(tSpecificSaveDirectory);
    tSpecificSaveDirectory += TString("/Systematics/");
    gSystem->mkdir(tSpecificSaveDirectory, true);
  }

  for(int i=0; i<tNRangeValues; i++)
  {
    FitGeneratorAndDraw* tFitGenerator = BuildFitGenerator(kTrain, "", fNonFlatBgdFitType);
    //TODO are these limits necessary?
    tFitGenerator->SetAllRadiiLimits(1., 10.);
    if(fIncludeResidualsType == kIncludeNoResiduals) tFitGenerator->SetAllLambdaParamLimits(0.1, 1.);
    else tFitGenerator->SetAllLambdaParamLimits(0.1, 2.);

    tFitGenerator->DoFit(tRangeVec[i]);

    TString tRangeValue = TString::Format("Max KStar for Fit = %0.4f",tRangeVec[i]);
    vector<TString> tFitParamsVec = tFitGenerator->GetAllFitParametersTStringVector();
    tFitParamsVec.insert(tFitParamsVec.begin(),tRangeValue);
    tText2dVector.push_back(tFitParamsVec);

    TCanvas* tKStarwFitsCan = tFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,fNonFlatBgdFitType,false,false);
    if(aSaveImages)
    {
      TString tImageSaveName = tSpecificSaveDirectory;
      tImageSaveName += tKStarwFitsCan->GetTitle();
      tImageSaveName += TString::Format("_MaxFitKStar_%0.4f",tRangeVec[i]);
      AppendFitInfo(tImageSaveName);

      tImageSaveName += TString(".pdf");
      tKStarwFitsCan->SaveAs(tImageSaveName);
    }
    delete tFitGenerator;
  }

  if(!aWriteToTxtFile) PrintText2dVec(tText2dVector);
  else
  {
    TString tOutputFileName = TString::Format("%sCfFitValues_VaryMaxFitKStar_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[fAnalysisType], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName);
    tOutputFileName += TString(".txt");
    std::ofstream tOutputFile;
    tOutputFile.open(tOutputFileName);

    PrintText2dVec(tText2dVector,tOutputFile);

    tOutputFile.close();
  }
}


//________________________________________________________________________________________________________________
void FitSystematicAnalysis::RunVaryNonFlatBackgroundFit(bool aSaveImages, bool aWriteToTxtFile)
{
  assert(fModifierValues1.size()==0);  //this is not intended for use with various modifier values, but for the final analysis
  assert(fApplyNonFlatBackgroundCorrection);  //This better be true, since I'm varying to NonFlatBgd method here!
  int tNFitTypeValues = 4;
  vector<int> tFitTypeVec = {0,1,2,3};

  vector<vector<TString> > tText2dVector(0);

  TString tSpecificSaveDirectory;
  if(aSaveImages || aWriteToTxtFile)
  {
    tSpecificSaveDirectory = fSaveDirectory;
    AppendFitInfo(tSpecificSaveDirectory);
    tSpecificSaveDirectory += TString("/Systematics/");
    gSystem->mkdir(tSpecificSaveDirectory, true);
  }

  bool tZoomROP=false;
  for(int i=0; i<tNFitTypeValues; i++)
  {
    FitGeneratorAndDraw* tFitGenerator = BuildFitGenerator(kTrain, "", static_cast<NonFlatBgdFitType>(tFitTypeVec[i]));

    //--------------------------------
    //For a few cases, fitting with kQuadratic and kGaussian can be difficult, so these restrictions are put in place to combat that
    // NOTE: Changing the MinMaxBgdFit range from [0.6, 0.9] (in typical analysis) to [0.45, 0.95] (here) does not significantly change
    //       the results for kLinear, and helps the other cases to build a more stable background fit
    tFitGenerator->SetMinMaxBgdFit(0.45, 0.95);

    // NOTE: Limits for lambda in (A)LamK0 already in place in FitSystematicAnalysis::BuildFitGenerator, which is why here
    //       I have && !(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)
    if(fIncludeResidualsType == kIncludeNoResiduals && !(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)) tFitGenerator->SetAllLambdaParamLimits(0.1, 1.0);

    //Don't seem to need lambda limits with residuals.  In fact, when limits in place, the fit doesn't converge
    // Without limits, the fit converges (with lambda values within limits!)
    //if(fIncludeResidualsType != kIncludeNoResiduals && !(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)) tFitGenerator->SetAllLambdaParamLimits(0.1, 2.0);

    tFitGenerator->SetAllRadiiLimits(1., 10.);  //NOTE: This does nothing when fIncludeResidualsType != kIncludeNoResiduals && fChargedResidualsType != kUseXiDataForAll,
                                                //      as, in that case, FitGenerator hardwires limits [1., 12] to stay within interpolation regime
    //--------------------------------

    tFitGenerator->DoFit();

    TString tRangeValue = TString::Format("Fit Type = %d",tFitTypeVec[i]);
    vector<TString> tFitParamsVec = tFitGenerator->GetAllFitParametersTStringVector();
    tFitParamsVec.insert(tFitParamsVec.begin(),tRangeValue);
    tText2dVector.push_back(tFitParamsVec);

    TCanvas* tKStarwFitsCan = tFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection, fApplyNonFlatBackgroundCorrection, static_cast<NonFlatBgdFitType>(tFitTypeVec[i]), false, false, tZoomROP);
    if(aSaveImages)
    {
      TString tImageSaveName = tSpecificSaveDirectory;
      tImageSaveName += tKStarwFitsCan->GetTitle();
      tImageSaveName += TString::Format("_FitType_%d",tFitTypeVec[i]);
      AppendFitInfo(tImageSaveName);

      tImageSaveName += TString(".pdf");
      tKStarwFitsCan->SaveAs(tImageSaveName);
    }
    delete tFitGenerator;
  }

  if(!aWriteToTxtFile) PrintText2dVec(tText2dVector, std::cout, tNFitTypeValues);
  else
  {
    TString tOutputFileName = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[fAnalysisType], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName);
    tOutputFileName += TString(".txt");
    std::ofstream tOutputFile;
    tOutputFile.open(tOutputFileName);

    PrintText2dVec(tText2dVector, tOutputFile, tNFitTypeValues);

    tOutputFile.close();
  }
}

