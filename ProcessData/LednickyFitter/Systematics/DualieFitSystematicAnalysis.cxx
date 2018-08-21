///////////////////////////////////////////////////////////////////////////
// DualieFitSystematicAnalysis:                                                //
///////////////////////////////////////////////////////////////////////////


#include "DualieFitSystematicAnalysis.h"

#ifdef __ROOT__
ClassImp(DualieFitSystematicAnalysis)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
DualieFitSystematicAnalysis::DualieFitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                       TString aDirNameModifierBase2, vector<double> &aModifierValues2,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, 
                                       bool aShareLambdaParams, bool aAllShareSingleLambdaParam, bool aDualieShareLambda, bool aDualieShareRadii) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fConjAnalysisType(kALamK0),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fDualieShareLambda(aDualieShareLambda),
  fDualieShareRadii(aDualieShareRadii), 
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
DualieFitSystematicAnalysis::DualieFitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, 
                                       bool aShareLambdaParams, bool aAllShareSingleLambdaParam, bool aDualieShareLambda, bool aDualieShareRadii) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fConjAnalysisType(kALamK0),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fDualieShareLambda(aDualieShareLambda),
  fDualieShareRadii(aDualieShareRadii), 
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

  *this = DualieFitSystematicAnalysis(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, aDirNameModifierBase1, aModifierValues1, fDirNameModifierBase2, fModifierValues2, aCentralityType, aGeneratorType, aShareLambdaParams, aAllShareSingleLambdaParam);
}

//________________________________________________________________________________________________________________
DualieFitSystematicAnalysis::DualieFitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, 
                                       bool aShareLambdaParams, bool aAllShareSingleLambdaParam, bool aDualieShareLambda, bool aDualieShareRadii) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fConjAnalysisType(kALamK0),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fDualieShareLambda(aDualieShareLambda),
  fDualieShareRadii(aDualieShareRadii), 
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

  *this = DualieFitSystematicAnalysis(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, fDirNameModifierBase1, fModifierValues1, fDirNameModifierBase2, fModifierValues2, aCentralityType, aGeneratorType, aShareLambdaParams, aAllShareSingleLambdaParam);
}


//________________________________________________________________________________________________________________
DualieFitSystematicAnalysis::~DualieFitSystematicAnalysis()
{
/*no-op*/
}

//________________________________________________________________________________________________________________
void DualieFitSystematicAnalysis::SetConjAnalysisType()
{
  assert(fAnalysisType==kLamK0 || fAnalysisType==kLamKchP || fAnalysisType==kLamKchM);

  if(fAnalysisType==kLamK0) fConjAnalysisType=kALamK0;
  else if(fAnalysisType==kLamKchP) fConjAnalysisType=kALamKchM;
  else if(fAnalysisType==kLamKchM) fConjAnalysisType=kALamKchP;
  else assert(0);
}

//________________________________________________________________________________________________________________
TString DualieFitSystematicAnalysis::GetCutValues(int aIndex)
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
void DualieFitSystematicAnalysis::OutputCutValues(int aIndex, ostream &aOut)
{
  aOut << "______________________________________________________________________________" << endl;
  aOut << GetCutValues(aIndex) << endl;
}

//________________________________________________________________________________________________________________
double DualieFitSystematicAnalysis::ExtractParamValue(TString aString)
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
void DualieFitSystematicAnalysis::AppendDifference(vector<vector<TString> > &a2dVec, int aCut, int aLineNumber)
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
void DualieFitSystematicAnalysis::PrintText2dVec(vector<vector<TString> > &a2dVec, ostream &aOut, int aNCuts)
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
void DualieFitSystematicAnalysis::AppendFitInfo(TString &aSaveName)
{
//  LednickyFitter::AppendFitInfo(aSaveName, fApplyMomResCorrection, fApplyNonFlatBackgroundCorrection, fIncludeResidualsType, fResPrimMaxDecayType, fChargedResidualsType, fFixD0);

  TString tModifier = LednickyFitter::BuildSaveNameModifier(fApplyMomResCorrection, fApplyNonFlatBackgroundCorrection, fNonFlatBgdFitType, 
                                                            fIncludeResidualsType, fResPrimMaxDecayType, 
                                                            fChargedResidualsType, fFixD0,
                                                            false, false, false, false, false,
                                                            fShareLambdaParams, fAllShareSingleLambdaParam, false, true,
                                                            fDualieShareLambda, fDualieShareRadii);
  aSaveName += tModifier;
}

//________________________________________________________________________________________________________________
void DualieFitSystematicAnalysis::SetRadiusStartValues(DualieFitGenerator* aFitGen)
{
  TString tFitInfoTString = "";
  AppendFitInfo(tFitInfoTString);

  TString tParentResultsDate = "";
  if     (fSaveDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(fSaveDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else assert(0);

  TString tMasterFileLocation = "";
  assert(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM || fAnalysisType==kLamKchM || fAnalysisType==kALamKchP);
  tMasterFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());

  if(fDualieShareRadii)
  {
    td1dVec tRStartValues(3);  //Currently only set up for typical case of 0-10, 10-30, 30-50 together
    tRStartValues[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kRadius)->GetFitValue();
    tRStartValues[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k1030, kRadius)->GetFitValue();
    tRStartValues[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k3050, kRadius)->GetFitValue();

    aFitGen->SetRadiusStartValues(tRStartValues);
  }
  else
  {
    td1dVec tRStartValues_LamKchP(3);  //Currently only set up for typical case of 0-10, 10-30, 30-50 together
    tRStartValues_LamKchP[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kRadius)->GetFitValue();
    tRStartValues_LamKchP[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k1030, kRadius)->GetFitValue();
    tRStartValues_LamKchP[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k3050, kRadius)->GetFitValue();
    aFitGen->GetFitGen1()->SetRadiusStartValues(tRStartValues_LamKchP);

    td1dVec tRStartValues_LamKchM(3);  //Currently only set up for typical case of 0-10, 10-30, 30-50 together
    tRStartValues_LamKchM[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k0010, kRadius)->GetFitValue();
    tRStartValues_LamKchM[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k1030, kRadius)->GetFitValue();
    tRStartValues_LamKchM[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k3050, kRadius)->GetFitValue();
    aFitGen->GetFitGen1()->SetRadiusStartValues(tRStartValues_LamKchM);
  }
}



//________________________________________________________________________________________________________________
void DualieFitSystematicAnalysis::SetLambdaStartValues(DualieFitGenerator* aFitGen)
{
  assert(!(fShareLambdaParams==false && fDualieShareLambda==true));  //See DualieGenerateFits.C for explanation why this setting does not make much sense

  int tNLamParams = 0;
  if(fAllShareSingleLambdaParam) tNLamParams = 1;
  else if(fShareLambdaParams==true && fDualieShareLambda==true) tNLamParams = 3;
  else if(fShareLambdaParams==true && fDualieShareLambda==false) tNLamParams = 6;
  else if(fShareLambdaParams==false && fDualieShareLambda==false) tNLamParams = 12;
  else assert(0);


  TString tFitInfoTString = "";
  AppendFitInfo(tFitInfoTString);

  TString tParentResultsDate = "";
  if     (fSaveDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(fSaveDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else assert(0);

  TString tMasterFileLocation = "";
  assert(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM || fAnalysisType==kLamKchM || fAnalysisType==kALamKchP);
  tMasterFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());

  if     (tNLamParams==1) 
  {
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue(), 
                                                    false, k0010, false);
  }
  else if(tNLamParams==3)
  {
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue(),
                                                    false, k0010, false);
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k1030, kLambda)->GetFitValue(),
                                                    false, k1030, false);
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k3050, kLambda)->GetFitValue(),
                                                    false, k3050, false);
  }
  else if(tNLamParams==6)
  {
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue(),
                                                    false, k0010, false);
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k1030, kLambda)->GetFitValue(),
                                                    false, k1030, false);
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k3050, kLambda)->GetFitValue(),
                                                    false, k3050, false);
    //----------------------------
    aFitGen->GetFitGen2()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k0010, kLambda)->GetFitValue(),
                                                    false, k0010, false);
    aFitGen->GetFitGen2()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k1030, kLambda)->GetFitValue(),
                                                    false, k1030, false);
    aFitGen->GetFitGen2()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k3050, kLambda)->GetFitValue(),
                                                    false, k3050, false);
  }
  else if(tNLamParams==12)
  {
    td1dVec tLamStartValues_LamKchP(6);

    tLamStartValues_LamKchP[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue();
    tLamStartValues_LamKchP[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kALamKchM, k0010, kLambda)->GetFitValue();

    tLamStartValues_LamKchP[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k1030, kLambda)->GetFitValue();
    tLamStartValues_LamKchP[3] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kALamKchM, k1030, kLambda)->GetFitValue();

    tLamStartValues_LamKchP[4] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k3050, kLambda)->GetFitValue();
    tLamStartValues_LamKchP[5] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kALamKchM, k3050, kLambda)->GetFitValue();

    aFitGen->GetFitGen1()->SetAllLambdaParamStartValues(tLamStartValues_LamKchP, false);
    //----------------------------
    td1dVec tLamStartValues_LamKchM(6);

    tLamStartValues_LamKchM[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue();
    tLamStartValues_LamKchM[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kALamKchP, k0010, kLambda)->GetFitValue();

    tLamStartValues_LamKchM[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k1030, kLambda)->GetFitValue();
    tLamStartValues_LamKchM[3] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kALamKchP, k1030, kLambda)->GetFitValue();

    tLamStartValues_LamKchM[4] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k3050, kLambda)->GetFitValue();
    tLamStartValues_LamKchM[5] = FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kALamKchP, k3050, kLambda)->GetFitValue();

    aFitGen->GetFitGen2()->SetAllLambdaParamStartValues(tLamStartValues_LamKchM, false);

  }
  else assert(0);

//  if((fAnalysisType==kLamK0 || fAnalysisType==kALamK0) && fIncludeResidualsType == kIncludeNoResiduals) aFitGen->SetAllLambdaParamLimits(0.4, 0.6);
//  if((fAnalysisType==kLamK0 || fAnalysisType==kALamK0) && fIncludeResidualsType != kIncludeNoResiduals) aFitGen->SetAllLambdaParamLimits(0.6, 1.5);
}

//________________________________________________________________________________________________________________
void DualieFitSystematicAnalysis::SetScattParamStartValues(DualieFitGenerator* aFitGen)
{
  TString tFitInfoTString = "";
  AppendFitInfo(tFitInfoTString);

  TString tParentResultsDate = "";
  if     (fSaveDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(fSaveDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else assert(0);

  TString tMasterFileLocation = "";
  assert(fAnalysisType==kLamKchP || fAnalysisType==kALamKchM || fAnalysisType==kLamKchM || fAnalysisType==kALamKchP);
  tMasterFileLocation = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());

  aFitGen->GetFitGen1()->SetScattParamStartValues(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kRef0)->GetFitValue(),
                                               FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kImf0)->GetFitValue(),
                                               FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchP, k0010, kd0)->GetFitValue(), 
                                               false);
  if(fFixD0) aFitGen->GetFitGen1()->SetScattParamStartValue(0., kd0, true);

  aFitGen->GetFitGen2()->SetScattParamStartValues(FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k0010, kRef0)->GetFitValue(),
                                               FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k0010, kImf0)->GetFitValue(),
                                               FitValuesWriter::GetFitParameter(tMasterFileLocation, tFitInfoTString, kLamKchM, k0010, kd0)->GetFitValue(), 
                                               false);
  if(fFixD0) aFitGen->GetFitGen2()->SetScattParamStartValue(0., kd0, true);
}


//________________________________________________________________________________________________________________
DualieFitGenerator* DualieFitSystematicAnalysis::BuildDualieFitGenerator(AnalysisRunType aRunType, TString aDirNameModifier, NonFlatBgdFitType aNonFlatBgdFitType)
{
  //For now, it appears only needed parameters here are AnalysisRunType aRunType and TString aDirNameModifier (for FitGenerator constructor)
  // and NonFlatBgdFitType aNonFlatBgdFitType for assigning attributes
  // Otherwise, default members used

  DualieFitGenerator* tDualieFitGenerator = new DualieFitGenerator(fFileLocationBase, fFileLocationBaseMC, fAnalysisType, fCentralityType, aRunType, 2, fFitGeneratorType, fShareLambdaParams, fAllShareSingleLambdaParam, aDirNameModifier);

  tDualieFitGenerator->SetApplyNonFlatBackgroundCorrection(fApplyNonFlatBackgroundCorrection);
  tDualieFitGenerator->SetNonFlatBgdFitType(aNonFlatBgdFitType);
  tDualieFitGenerator->SetApplyMomResCorrection(fApplyMomResCorrection);
  if(fIncludeResidualsType != kIncludeNoResiduals)
  {
    if(fAnalysisType==kLamK0 || fAnalysisType==kALamK0) tDualieFitGenerator->SetIncludeResidualCorrelationsType(fIncludeResidualsType, 0.60, 1.50);  //TODO This is overridden below... is this what I want?
    else tDualieFitGenerator->SetIncludeResidualCorrelationsType(fIncludeResidualsType, 0., 0.);
  }
  tDualieFitGenerator->SetChargedResidualsType(fChargedResidualsType);
  tDualieFitGenerator->SetResPrimMaxDecayType(fResPrimMaxDecayType);

  //----- Set appropriate parameter start values, and limits, to keep fitter from accidentally doing something crazy
  assert(fCentralityType==kMB);  //This will fail otherwise

  SetRadiusStartValues(tDualieFitGenerator);
  SetLambdaStartValues(tDualieFitGenerator);  
  SetScattParamStartValues(tDualieFitGenerator);
  //----------------------------------------------------------------------------------------------------------------

  return tDualieFitGenerator;
}



//________________________________________________________________________________________________________________
void DualieFitSystematicAnalysis::RunAllFits(bool aSaveImages, bool aWriteToTxtFile)
{
  vector<vector<TString> > tText2dVector1(0);
  vector<vector<TString> > tText2dVector2(0);

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

    DualieFitGenerator* tDualieFitGenerator = BuildDualieFitGenerator(kTrainSys, tDirNameModifier, fNonFlatBgdFitType);
    tDualieFitGenerator->DoFit(fDualieShareLambda, fDualieShareRadii);

//    OutputCutValues(i,aOut);
//    tDualieFitGenerator->WriteAllFitParameters(aOut);
    TString tCutValue = GetCutValues(i);

    vector<TString> tFitParamsVec1 = tDualieFitGenerator->GetFitGen1()->GetAllFitParametersTStringVector();
    tFitParamsVec1.insert(tFitParamsVec1.begin(),tCutValue);
    tText2dVector1.push_back(tFitParamsVec1);

    vector<TString> tFitParamsVec2 = tDualieFitGenerator->GetFitGen2()->GetAllFitParametersTStringVector();
    tFitParamsVec2.insert(tFitParamsVec2.begin(),tCutValue);
    tText2dVector2.push_back(tFitParamsVec2);

    TObjArray* tKStarwFitsCans = tDualieFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,fNonFlatBgdFitType,false,false);
    if(aSaveImages)
    {
      assert(tKStarwFitsCans->GetEntries()==2);

      TString tImageSaveName1 = tSpecificSaveDirectory;
      tImageSaveName1 += tKStarwFitsCans->At(0)->GetTitle();

      TString tImageSaveName2 = tSpecificSaveDirectory;
      tImageSaveName2 += tKStarwFitsCans->At(1)->GetTitle();

      TString tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
      if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

      tImageSaveName1 += tDirNameModifier;
      AppendFitInfo(tImageSaveName1);

      tImageSaveName2 += tDirNameModifier;
      AppendFitInfo(tImageSaveName2);

      tImageSaveName1 += TString(".pdf");
      tImageSaveName2 += TString(".pdf");

      tKStarwFitsCans->At(0)->SaveAs(tImageSaveName1);
      tKStarwFitsCans->At(1)->SaveAs(tImageSaveName2);
    }
    delete tDualieFitGenerator;
  }

  if(!aWriteToTxtFile) 
  {
    PrintText2dVec(tText2dVector1);
    PrintText2dVec(tText2dVector2);
  }
  else
  {
    AnalysisType tAnTyp1 = kLamKchP;
    TString tOutputFileName1 = TString::Format("%sCfFitValues_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp1], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName1);
    tOutputFileName1 += TString(".txt");
    std::ofstream tOutputFile1;
    tOutputFile1.open(tOutputFileName1);

    PrintText2dVec(tText2dVector1,tOutputFile1);

    tOutputFile1.close();

    //----------

    AnalysisType tAnTyp2 = kLamKchM;
    TString tOutputFileName2 = TString::Format("%sCfFitValues_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp2], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName2);
    tOutputFileName2 += TString(".txt");
    std::ofstream tOutputFile2;
    tOutputFile2.open(tOutputFileName2);

    PrintText2dVec(tText2dVector2,tOutputFile2);

    tOutputFile2.close();
  }
}


//________________________________________________________________________________________________________________
void DualieFitSystematicAnalysis::RunVaryFitRange(bool aSaveImages, bool aWriteToTxtFile, double aMaxKStar1, double aMaxKStar2, double aMaxKStar3)
{
  assert(fModifierValues1.size()==0);  //this is not intended for use with various modifier values, but for the final analysis
  int tNRangeValues = 3;
  vector<double> tRangeVec = {aMaxKStar1,aMaxKStar2,aMaxKStar3};

  vector<vector<TString> > tText2dVector1(0);
  vector<vector<TString> > tText2dVector2(0);

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
    DualieFitGenerator* tDualieFitGenerator = BuildDualieFitGenerator(kTrain, "", fNonFlatBgdFitType);
    //TODO are these limits necessary?
    tDualieFitGenerator->SetAllRadiiLimits(1., 10.);
    if(fIncludeResidualsType == kIncludeNoResiduals) tDualieFitGenerator->SetAllLambdaParamLimits(0.1, 1.);
    else tDualieFitGenerator->SetAllLambdaParamLimits(0.1, 2.);

    tDualieFitGenerator->DoFit(fDualieShareLambda, fDualieShareRadii, tRangeVec[i]);

    TString tRangeValue = TString::Format("Max KStar for Fit = %0.4f",tRangeVec[i]);

    vector<TString> tFitParamsVec1 = tDualieFitGenerator->GetFitGen1()->GetAllFitParametersTStringVector();
    tFitParamsVec1.insert(tFitParamsVec1.begin(),tRangeValue);
    tText2dVector1.push_back(tFitParamsVec1);

    vector<TString> tFitParamsVec2 = tDualieFitGenerator->GetFitGen2()->GetAllFitParametersTStringVector();
    tFitParamsVec2.insert(tFitParamsVec2.begin(),tRangeValue);
    tText2dVector2.push_back(tFitParamsVec2);


    TObjArray* tKStarwFitsCans = tDualieFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,fNonFlatBgdFitType,false,false);
    if(aSaveImages)
    {
      assert(tKStarwFitsCans->GetEntries()==2);

      TString tImageSaveName1 = tSpecificSaveDirectory;
      tImageSaveName1 += tKStarwFitsCans->At(0)->GetTitle();
      tImageSaveName1 += TString::Format("_MaxFitKStar_%0.4f",tRangeVec[i]);
      AppendFitInfo(tImageSaveName1);

      TString tImageSaveName2 = tSpecificSaveDirectory;
      tImageSaveName2 += tKStarwFitsCans->At(1)->GetTitle();
      tImageSaveName2 += TString::Format("_MaxFitKStar_%0.4f",tRangeVec[i]);
      AppendFitInfo(tImageSaveName2);

      tImageSaveName1 += TString(".pdf");
      tImageSaveName2 += TString(".pdf");

      tKStarwFitsCans->At(0)->SaveAs(tImageSaveName1);
      tKStarwFitsCans->At(1)->SaveAs(tImageSaveName2);
    }
    delete tDualieFitGenerator;
  }

  if(!aWriteToTxtFile) 
  {
    PrintText2dVec(tText2dVector1);
    PrintText2dVec(tText2dVector2);
  }
  else
  {
    AnalysisType tAnTyp1 = kLamKchP;
    TString tOutputFileName1 = TString::Format("%sCfFitValues_VaryMaxFitKStar_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp1], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName1);
    tOutputFileName1 += TString(".txt");
    std::ofstream tOutputFile1;
    tOutputFile1.open(tOutputFileName1);

    PrintText2dVec(tText2dVector1,tOutputFile1);

    tOutputFile1.close();

    //----------

    AnalysisType tAnTyp2 = kLamKchM;
    TString tOutputFileName2 = TString::Format("%sCfFitValues_VaryMaxFitKStar_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp2], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName2);
    tOutputFileName2 += TString(".txt");
    std::ofstream tOutputFile2;
    tOutputFile2.open(tOutputFileName2);

    PrintText2dVec(tText2dVector2,tOutputFile2);

    tOutputFile2.close();
  }
}


//________________________________________________________________________________________________________________
void DualieFitSystematicAnalysis::RunVaryNonFlatBackgroundFit(bool aSaveImages, bool aWriteToTxtFile)
{
  assert(fModifierValues1.size()==0);  //this is not intended for use with various modifier values, but for the final analysis
  assert(fApplyNonFlatBackgroundCorrection);  //This better be true, since I'm varying to NonFlatBgd method here!
  int tNFitTypeValues = 4;
  vector<int> tFitTypeVec = {0,1,2,3};

  vector<vector<TString> > tText2dVector1(0);
  vector<vector<TString> > tText2dVector2(0);

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
    DualieFitGenerator* tDualieFitGenerator = BuildDualieFitGenerator(kTrain, "", static_cast<NonFlatBgdFitType>(tFitTypeVec[i]));

    //--------------------------------
    //For a few cases, fitting with kQuadratic and kGaussian can be difficult, so these restrictions are put in place to combat that
    // NOTE: Changing the MinMaxBgdFit range from [0.6, 0.9] (in typical analysis) to [0.45, 0.95] (here) does not significantly change
    //       the results for kLinear, and helps the other cases to build a more stable background fit
    tDualieFitGenerator->SetMinMaxBgdFit(0.45, 0.95);

    // NOTE: Limits for lambda in (A)LamK0 already in place in DualieFitSystematicAnalysis::BuildDualieFitGenerator, which is why here
    //       I have && !(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)
    if(fIncludeResidualsType == kIncludeNoResiduals && !(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)) tDualieFitGenerator->SetAllLambdaParamLimits(0.1, 1.0);

    //Don't seem to need lambda limits with residuals.  In fact, when limits in place, the fit doesn't converge
    // Without limits, the fit converges (with lambda values within limits!)
    //if(fIncludeResidualsType != kIncludeNoResiduals && !(fAnalysisType==kLamK0 || fAnalysisType==kALamK0)) tDualieFitGenerator->SetAllLambdaParamLimits(0.1, 2.0);

    tDualieFitGenerator->SetAllRadiiLimits(1., 10.);  //NOTE: This does nothing when fIncludeResidualsType != kIncludeNoResiduals && fChargedResidualsType != kUseXiDataForAll,
                                                //      as, in that case, FitGenerator hardwires limits [1., 12] to stay within interpolation regime
    //--------------------------------

    tDualieFitGenerator->DoFit(fDualieShareLambda, fDualieShareRadii);

    TString tRangeValue = TString::Format("Fit Type = %d",tFitTypeVec[i]);

    vector<TString> tFitParamsVec1 = tDualieFitGenerator->GetFitGen1()->GetAllFitParametersTStringVector();
    tFitParamsVec1.insert(tFitParamsVec1.begin(),tRangeValue);
    tText2dVector1.push_back(tFitParamsVec1);

    vector<TString> tFitParamsVec2 = tDualieFitGenerator->GetFitGen2()->GetAllFitParametersTStringVector();
    tFitParamsVec2.insert(tFitParamsVec2.begin(),tRangeValue);
    tText2dVector2.push_back(tFitParamsVec2);

    TObjArray* tKStarwFitsCans = tDualieFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,static_cast<NonFlatBgdFitType>(tFitTypeVec[i]),false,false, tZoomROP);
    if(aSaveImages)
    {
      assert(tKStarwFitsCans->GetEntries()==2);

      TString tImageSaveName1 = tSpecificSaveDirectory;
      tImageSaveName1 += tKStarwFitsCans->At(0)->GetTitle();
      tImageSaveName1 += TString::Format("_FitType_%d",tFitTypeVec[i]);
      AppendFitInfo(tImageSaveName1);

      TString tImageSaveName2 = tSpecificSaveDirectory;
      tImageSaveName2 += tKStarwFitsCans->At(1)->GetTitle();
      tImageSaveName2 += TString::Format("_FitType_%d",tFitTypeVec[i]);
      AppendFitInfo(tImageSaveName2);

      tImageSaveName1 += TString(".pdf");
      tImageSaveName2 += TString(".pdf");

      tKStarwFitsCans->At(0)->SaveAs(tImageSaveName1);
      tKStarwFitsCans->At(1)->SaveAs(tImageSaveName2);
    }
    delete tDualieFitGenerator;
  }

  if(!aWriteToTxtFile) 
  {
    PrintText2dVec(tText2dVector1, std::cout, tNFitTypeValues);
    PrintText2dVec(tText2dVector2, std::cout, tNFitTypeValues);
  }
  else
  {
    AnalysisType tAnTyp1 = kLamKchP;
    TString tOutputFileName1 = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp1], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName1);
    tOutputFileName1 += TString(".txt");
    std::ofstream tOutputFile1;
    tOutputFile1.open(tOutputFileName1);

    PrintText2dVec(tText2dVector1, tOutputFile1, tNFitTypeValues);

    tOutputFile1.close();

    //----------

    AnalysisType tAnTyp2 = kLamKchM;
    TString tOutputFileName2 = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp2], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName2);
    tOutputFileName2 += TString(".txt");
    std::ofstream tOutputFile2;
    tOutputFile2.open(tOutputFileName2);

    PrintText2dVec(tText2dVector2, tOutputFile2, tNFitTypeValues);

    tOutputFile2.close();
  }
}

