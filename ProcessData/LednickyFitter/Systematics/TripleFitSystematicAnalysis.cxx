///////////////////////////////////////////////////////////////////////////
// TripleFitSystematicAnalysis:                                                //
///////////////////////////////////////////////////////////////////////////


#include "TripleFitSystematicAnalysis.h"

#ifdef __ROOT__
ClassImp(TripleFitSystematicAnalysis)
#endif


//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
TripleFitSystematicAnalysis::TripleFitSystematicAnalysis(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, 
                                                         TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0,
                                                         TString aGeneralAnTypeModified,
                                                         TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                                         TString aDirNameModifierBase2, vector<double> &aModifierValues2,
                                                         CentralityType aCentralityType, FitGeneratorType aGeneratorType, 
                                                         bool aShareLambdaParams, bool aAllShareSingleLambdaParam, bool aDualieShareLambda, bool aDualieShareRadii) :
  fFileLocationBase_LamKch(aFileLocationBase_LamKch),
  fFileLocationBaseMC_LamKch(aFileLocationBaseMC_LamKch),
  fFileLocationBase_LamK0(aFileLocationBase_LamK0),
  fFileLocationBaseMC_LamK0(aFileLocationBaseMC_LamK0),

  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),

  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fDualieShareLambda(aDualieShareLambda),
  fDualieShareRadii(aDualieShareRadii), 

  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType_LamKch(kLinear),
  fNonFlatBgdFitType_LamK0(kLinear),
  fNonFlatBgdFitTypes(0),
  fApplyMomResCorrection(false),

  fIncludeResidualsType(kIncludeNoResiduals),
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fFixD0(false),

  fSaveDirectory(""),

  fGeneralAnTypeModified(aGeneralAnTypeModified),
  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(aDirNameModifierBase2),

  fModifierValues1(aModifierValues1),
  fModifierValues2(aModifierValues2)

{
  if(!fDirNameModifierBase2.IsNull()) assert(fModifierValues1.size() == fModifierValues2.size());
  fNonFlatBgdFitTypes = vector<NonFlatBgdFitType>{fNonFlatBgdFitType_LamK0, fNonFlatBgdFitType_LamK0, 
                                                  fNonFlatBgdFitType_LamKch, fNonFlatBgdFitType_LamKch, fNonFlatBgdFitType_LamKch, fNonFlatBgdFitType_LamKch};
  assert(fGeneralAnTypeModified.EqualTo("cLamcKch") || fGeneralAnTypeModified.EqualTo("cLamK0") || fGeneralAnTypeModified.IsNull());
}


//________________________________________________________________________________________________________________
TripleFitSystematicAnalysis::TripleFitSystematicAnalysis(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, 
                                                         TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0,
                                                         TString aGeneralAnTypeModified,
                                                         TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                                         CentralityType aCentralityType, FitGeneratorType aGeneratorType, 
                                                         bool aShareLambdaParams, bool aAllShareSingleLambdaParam, bool aDualieShareLambda, bool aDualieShareRadii) :
  fFileLocationBase_LamKch(aFileLocationBase_LamKch),
  fFileLocationBaseMC_LamKch(aFileLocationBaseMC_LamKch),
  fFileLocationBase_LamK0(aFileLocationBase_LamK0),
  fFileLocationBaseMC_LamK0(aFileLocationBaseMC_LamK0),

  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),

  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fDualieShareLambda(aDualieShareLambda),
  fDualieShareRadii(aDualieShareRadii), 

  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType_LamKch(kLinear),
  fNonFlatBgdFitType_LamK0(kLinear),
  fNonFlatBgdFitTypes(0),
  fApplyMomResCorrection(false),

  fIncludeResidualsType(kIncludeNoResiduals),
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fFixD0(false),

  fSaveDirectory(""),

  fGeneralAnTypeModified(aGeneralAnTypeModified),
  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(0),

  fModifierValues1(aModifierValues1),
  fModifierValues2(0)

{
  fDirNameModifierBase2 = "";
  fModifierValues2 = vector<double> (0);

  *this = TripleFitSystematicAnalysis(aFileLocationBase_LamKch, aFileLocationBaseMC_LamKch, 
                                      aFileLocationBase_LamK0, aFileLocationBaseMC_LamK0,
                                      aGeneralAnTypeModified, 
                                      aDirNameModifierBase1, aModifierValues1, 
                                      fDirNameModifierBase2, fModifierValues2, 
                                      aCentralityType, aGeneratorType, 
                                      aShareLambdaParams, aAllShareSingleLambdaParam, aDualieShareLambda, aDualieShareRadii);
}

//________________________________________________________________________________________________________________
TripleFitSystematicAnalysis::TripleFitSystematicAnalysis(TString aFileLocationBase_LamKch, TString aFileLocationBaseMC_LamKch, 
                                                         TString aFileLocationBase_LamK0, TString aFileLocationBaseMC_LamK0,
                                                         TString aGeneralAnTypeModified,
                                                         CentralityType aCentralityType, FitGeneratorType aGeneratorType, 
                                                         bool aShareLambdaParams, bool aAllShareSingleLambdaParam, bool aDualieShareLambda, bool aDualieShareRadii) :
  fFileLocationBase_LamKch(aFileLocationBase_LamKch),
  fFileLocationBaseMC_LamKch(aFileLocationBaseMC_LamKch),
  fFileLocationBase_LamK0(aFileLocationBase_LamK0),
  fFileLocationBaseMC_LamK0(aFileLocationBaseMC_LamK0),

  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),

  fShareLambdaParams(aShareLambdaParams),
  fAllShareSingleLambdaParam(aAllShareSingleLambdaParam),
  fDualieShareLambda(aDualieShareLambda),
  fDualieShareRadii(aDualieShareRadii), 

  fApplyNonFlatBackgroundCorrection(false),
  fNonFlatBgdFitType_LamKch(kLinear),
  fNonFlatBgdFitType_LamK0(kLinear),
  fNonFlatBgdFitTypes(0),
  fApplyMomResCorrection(false),

  fIncludeResidualsType(kIncludeNoResiduals),
  fChargedResidualsType(kUseXiDataAndCoulombOnlyInterp),
  fResPrimMaxDecayType(k5fm),

  fFixD0(false),

  fSaveDirectory(""),

  fGeneralAnTypeModified(aGeneralAnTypeModified),
  fDirNameModifierBase1(0),
  fDirNameModifierBase2(0),

  fModifierValues1(0),
  fModifierValues2(0)

{
  fDirNameModifierBase1 = "";
  fDirNameModifierBase2 = "";

  fModifierValues1 = vector<double> (0);
  fModifierValues2 = vector<double> (0);

  *this = TripleFitSystematicAnalysis(aFileLocationBase_LamKch, aFileLocationBaseMC_LamKch, 
                                      aFileLocationBase_LamK0, aFileLocationBaseMC_LamK0,
                                      aGeneralAnTypeModified, 
                                      fDirNameModifierBase1, fModifierValues1, 
                                      fDirNameModifierBase2, fModifierValues2, 
                                      aCentralityType, aGeneratorType, 
                                      aShareLambdaParams, aAllShareSingleLambdaParam, aDualieShareLambda, aDualieShareRadii);
}


//________________________________________________________________________________________________________________
TripleFitSystematicAnalysis::~TripleFitSystematicAnalysis()
{
/*no-op*/
}

//________________________________________________________________________________________________________________
TString TripleFitSystematicAnalysis::GetCutValues(int aIndex)
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
void TripleFitSystematicAnalysis::OutputCutValues(int aIndex, ostream &aOut)
{
  aOut << "______________________________________________________________________________" << endl;
  aOut << GetCutValues(aIndex) << endl;
}

//________________________________________________________________________________________________________________
double TripleFitSystematicAnalysis::ExtractParamValue(TString aString)
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
void TripleFitSystematicAnalysis::AppendDifference(vector<vector<TString> > &a2dVec, int aCut, int aLineNumber)
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
void TripleFitSystematicAnalysis::PrintText2dVec(vector<vector<TString> > &a2dVec, ostream &aOut, int aNCuts)
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
void TripleFitSystematicAnalysis::AppendFitInfo(TString &aSaveName)
{
  TString tModifier = LednickyFitter::BuildSaveNameModifier(fApplyMomResCorrection, fApplyNonFlatBackgroundCorrection, fNonFlatBgdFitTypes, 
                                                            fIncludeResidualsType, fResPrimMaxDecayType, 
                                                            fChargedResidualsType, fFixD0,
                                                            false, false, false, false, false,
                                                            fShareLambdaParams, fAllShareSingleLambdaParam, false, true,
                                                            fDualieShareLambda, fDualieShareRadii);
  aSaveName += tModifier;
}

//________________________________________________________________________________________________________________
void TripleFitSystematicAnalysis::SetRadiusStartValues(TripleFitGenerator* aFitGen)
{
  TString tFitInfoTString = "";
  AppendFitInfo(tFitInfoTString);

  TString tParentResultsDate = "";
  if     (fSaveDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(fSaveDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else if(fSaveDirectory.Contains("20190319")) tParentResultsDate = TString("20190319");
  else assert(0);

  TString tMasterFileLocation_LamKch = "";
  tMasterFileLocation_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());

  assert(fDualieShareRadii);

  td1dVec tRStartValues(3);  //Currently only set up for typical case of 0-10, 10-30, 30-50 together
  tRStartValues[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k0010, kRadius)->GetFitValue();
  tRStartValues[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k1030, kRadius)->GetFitValue();
  tRStartValues[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k3050, kRadius)->GetFitValue();

  aFitGen->SetRadiusStartValues(tRStartValues);
}



//________________________________________________________________________________________________________________
void TripleFitSystematicAnalysis::SetLambdaStartValues(TripleFitGenerator* aFitGen)
{
  assert(!(fShareLambdaParams==false && fDualieShareLambda==true));  //See DualieGenerateFits.C for explanation why this setting does not make much sense

  int tNLamParams = 0;
  if(fAllShareSingleLambdaParam) tNLamParams = 1;
  else if(fShareLambdaParams==true && fDualieShareLambda==true) tNLamParams = 3;
  else if(fShareLambdaParams==true && fDualieShareLambda==false) tNLamParams = 9;
  else if(fShareLambdaParams==false && fDualieShareLambda==false) tNLamParams = 18;
  else assert(0);


  TString tFitInfoTString = "";
  AppendFitInfo(tFitInfoTString);

  TString tParentResultsDate = "";
  if     (fSaveDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(fSaveDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else if(fSaveDirectory.Contains("20190319")) tParentResultsDate = TString("20190319");
  else assert(0);

  TString tMasterFileLocation_LamKch = "", tMasterFileLocation_LamK0 = "";
  tMasterFileLocation_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());
  tMasterFileLocation_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());

  if     (tNLamParams==1) 
  {
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue(), 
                                                    false, k0010, false);
  }
  else if(tNLamParams==3)
  {
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue(),
                                                    false, k0010, false);
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k1030, kLambda)->GetFitValue(),
                                                    false, k1030, false);
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k3050, kLambda)->GetFitValue(),
                                                    false, k3050, false);
  }
  else if(tNLamParams==9)
  {
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue(),
                                                    false, k0010, false);
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k1030, kLambda)->GetFitValue(),
                                                    false, k1030, false);
    aFitGen->GetFitGen1()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k3050, kLambda)->GetFitValue(),
                                                    false, k3050, false);
    //----------------------------
    aFitGen->GetFitGen2()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchM, k0010, kLambda)->GetFitValue(),
                                                    false, k0010, false);
    aFitGen->GetFitGen2()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchM, k1030, kLambda)->GetFitValue(),
                                                    false, k1030, false);
    aFitGen->GetFitGen2()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchM, k3050, kLambda)->GetFitValue(),
                                                    false, k3050, false);
    //----------------------------
    aFitGen->GetFitGen3()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kLamK0, k0010, kLambda)->GetFitValue(),
                                                    false, k0010, false);
    aFitGen->GetFitGen3()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kLamK0, k1030, kLambda)->GetFitValue(),
                                                    false, k1030, false);
    aFitGen->GetFitGen3()->SetLambdaParamStartValue(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kLamK0, k3050, kLambda)->GetFitValue(),
                                                    false, k3050, false);
  }
  else if(tNLamParams==18)
  {
    td1dVec tLamStartValues_LamKchP(6);

    tLamStartValues_LamKchP[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue();
    tLamStartValues_LamKchP[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kALamKchM, k0010, kLambda)->GetFitValue();

    tLamStartValues_LamKchP[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k1030, kLambda)->GetFitValue();
    tLamStartValues_LamKchP[3] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kALamKchM, k1030, kLambda)->GetFitValue();

    tLamStartValues_LamKchP[4] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k3050, kLambda)->GetFitValue();
    tLamStartValues_LamKchP[5] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kALamKchM, k3050, kLambda)->GetFitValue();

    aFitGen->GetFitGen1()->SetAllLambdaParamStartValues(tLamStartValues_LamKchP, false);
    //----------------------------
    td1dVec tLamStartValues_LamKchM(6);

    tLamStartValues_LamKchM[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k0010, kLambda)->GetFitValue();
    tLamStartValues_LamKchM[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kALamKchP, k0010, kLambda)->GetFitValue();

    tLamStartValues_LamKchM[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchM, k1030, kLambda)->GetFitValue();
    tLamStartValues_LamKchM[3] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kALamKchP, k1030, kLambda)->GetFitValue();

    tLamStartValues_LamKchM[4] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchM, k3050, kLambda)->GetFitValue();
    tLamStartValues_LamKchM[5] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kALamKchP, k3050, kLambda)->GetFitValue();

    aFitGen->GetFitGen2()->SetAllLambdaParamStartValues(tLamStartValues_LamKchM, false);
    //----------------------------
    td1dVec tLamStartValues_LamK0(6);

    tLamStartValues_LamK0[0] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kLamK0, k0010, kLambda)->GetFitValue();
    tLamStartValues_LamK0[1] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kALamK0, k0010, kLambda)->GetFitValue();

    tLamStartValues_LamK0[2] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kLamK0, k1030, kLambda)->GetFitValue();
    tLamStartValues_LamK0[3] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kALamK0, k1030, kLambda)->GetFitValue();

    tLamStartValues_LamK0[4] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kLamK0, k3050, kLambda)->GetFitValue();
    tLamStartValues_LamK0[5] = FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kALamK0, k3050, kLambda)->GetFitValue();

    aFitGen->GetFitGen3()->SetAllLambdaParamStartValues(tLamStartValues_LamK0, false);

  }
  else assert(0);
}

//________________________________________________________________________________________________________________
void TripleFitSystematicAnalysis::SetScattParamStartValues(TripleFitGenerator* aFitGen)
{
  TString tFitInfoTString = "";
  AppendFitInfo(tFitInfoTString);

  TString tParentResultsDate = "";
  if     (fSaveDirectory.Contains("20161027")) tParentResultsDate = TString("20161027");
  else if(fSaveDirectory.Contains("20180505")) tParentResultsDate = TString("20180505");
  else if(fSaveDirectory.Contains("20190319")) tParentResultsDate = TString("20190319");
  else assert(0);

  TString tMasterFileLocation_LamKch = "", tMasterFileLocation_LamK0 = "";
  tMasterFileLocation_LamKch = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamcKch_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());
  tMasterFileLocation_LamK0 = TString::Format("/home/jesse/Analysis/FemtoAnalysis/Results/Results_cLamK0_%s/MasterFitResults_%s.txt", tParentResultsDate.Data(), tParentResultsDate.Data());

  aFitGen->GetFitGen1()->SetScattParamStartValues(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k0010, kRef0)->GetFitValue(),
                                                  FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k0010, kImf0)->GetFitValue(),
                                                  FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchP, k0010, kd0)->GetFitValue(), 
                                                  false);
  if(fFixD0) aFitGen->GetFitGen1()->SetScattParamStartValue(0., kd0, true);

  aFitGen->GetFitGen2()->SetScattParamStartValues(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchM, k0010, kRef0)->GetFitValue(),
                                                  FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchM, k0010, kImf0)->GetFitValue(),
                                                  FitValuesWriter::GetFitParameter(tMasterFileLocation_LamKch, tFitInfoTString, kLamKchM, k0010, kd0)->GetFitValue(), 
                                                  false);
  if(fFixD0) aFitGen->GetFitGen2()->SetScattParamStartValue(0., kd0, true);

  aFitGen->GetFitGen3()->SetScattParamStartValues(FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kLamK0, k0010, kRef0)->GetFitValue(),
                                                  FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kLamK0, k0010, kImf0)->GetFitValue(),
                                                  FitValuesWriter::GetFitParameter(tMasterFileLocation_LamK0, tFitInfoTString, kLamK0, k0010, kd0)->GetFitValue(), 
                                                  false);
  if(fFixD0) aFitGen->GetFitGen3()->SetScattParamStartValue(0., kd0, true);
}


//________________________________________________________________________________________________________________
TripleFitGenerator* TripleFitSystematicAnalysis::BuildTripleFitGenerator(AnalysisRunType aRunType, TString aDirNameModifier, vector<NonFlatBgdFitType> &aNonFlatBgdFitTypes)
{
  TString tDirNameModifier_LamKch="", tDirNameModifier_LamK0="";
  if     (fGeneralAnTypeModified.EqualTo("cLamcKch")) tDirNameModifier_LamKch=aDirNameModifier;
  else if(fGeneralAnTypeModified.EqualTo("cLamK0")) tDirNameModifier_LamK0=aDirNameModifier;

  AnalysisRunType aRunType_LamKch=aRunType, aRunType_LamK0=aRunType;
  if(tDirNameModifier_LamKch.IsNull()) aRunType_LamKch = kTrain;
  if(tDirNameModifier_LamK0.IsNull()) aRunType_LamK0 = kTrain;

  TripleFitGenerator* tTripleFitGenerator = new TripleFitGenerator(fFileLocationBase_LamKch, fFileLocationBaseMC_LamKch,
                                                                   fFileLocationBase_LamK0, fFileLocationBaseMC_LamK0, 
                                                                   fCentralityType, aRunType_LamKch, aRunType_LamK0, 2, fFitGeneratorType, fShareLambdaParams, fAllShareSingleLambdaParam, 
                                                                   tDirNameModifier_LamKch, tDirNameModifier_LamK0);

  tTripleFitGenerator->SetApplyNonFlatBackgroundCorrection(fApplyNonFlatBackgroundCorrection);
  tTripleFitGenerator->SetNonFlatBgdFitTypes(aNonFlatBgdFitTypes[kLamKchP], aNonFlatBgdFitTypes[kLamK0]);
  tTripleFitGenerator->SetApplyMomResCorrection(fApplyMomResCorrection);
  if(fIncludeResidualsType != kIncludeNoResiduals) tTripleFitGenerator->SetIncludeResidualCorrelationsType(fIncludeResidualsType, 0.60, 1.5);
  else tTripleFitGenerator->SetIncludeResidualCorrelationsType(fIncludeResidualsType, 0.1, 1.0);

  tTripleFitGenerator->SetChargedResidualsType(fChargedResidualsType);
  tTripleFitGenerator->SetResPrimMaxDecayType(fResPrimMaxDecayType);

  //----- Set appropriate parameter start values, and limits, to keep fitter from accidentally doing something crazy
  assert(fCentralityType==kMB);  //This will fail otherwise

  SetRadiusStartValues(tTripleFitGenerator);
  SetLambdaStartValues(tTripleFitGenerator);  
  SetScattParamStartValues(tTripleFitGenerator);

  if(aNonFlatBgdFitTypes[kLamKchP]==kPolynomial) tTripleFitGenerator->SetMinMaxBgdFit(kLamKchP, 0.32, 0.80);
  if(aNonFlatBgdFitTypes[kLamK0]  ==kPolynomial) tTripleFitGenerator->SetMinMaxBgdFit(kLamK0, 0.32, 0.80);
  //----------------------------------------------------------------------------------------------------------------

  return tTripleFitGenerator;
}



//________________________________________________________________________________________________________________
void TripleFitSystematicAnalysis::RunAllFits(bool aSaveImages, bool aWriteToTxtFile)
{
  // 1=LamKchP;  2=LamKchM;  3=LamK0
  
  vector<vector<TString> > tText2dVector1(0);
  vector<vector<TString> > tText2dVector2(0);
  vector<vector<TString> > tText2dVector3(0);

  TString tSpecificSaveDirectory;
  if(aSaveImages || aWriteToTxtFile)
  {
    tSpecificSaveDirectory = fSaveDirectory;
    AppendFitInfo(tSpecificSaveDirectory);
    tSpecificSaveDirectory += TString("/");
    gSystem->mkdir(tSpecificSaveDirectory, true);
  }

  TString tDirNameModifier;
  for(unsigned int i=0; i<fModifierValues1.size(); i++)
  {
    tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
    if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

    TripleFitGenerator* tTripleFitGenerator = BuildTripleFitGenerator(kTrainSys, tDirNameModifier, fNonFlatBgdFitTypes);
    tTripleFitGenerator->DoFit(fDualieShareLambda, fDualieShareRadii);

    TString tCutValue = GetCutValues(i);

    vector<TString> tFitParamsVec1 = tTripleFitGenerator->GetFitGen1()->GetAllFitParametersTStringVector();
    tFitParamsVec1.insert(tFitParamsVec1.begin(),tCutValue);
    tText2dVector1.push_back(tFitParamsVec1);

    vector<TString> tFitParamsVec2 = tTripleFitGenerator->GetFitGen2()->GetAllFitParametersTStringVector();
    tFitParamsVec2.insert(tFitParamsVec2.begin(),tCutValue);
    tText2dVector2.push_back(tFitParamsVec2);

    vector<TString> tFitParamsVec3 = tTripleFitGenerator->GetFitGen3()->GetAllFitParametersTStringVector();
    tFitParamsVec3.insert(tFitParamsVec3.begin(),tCutValue);
    tText2dVector3.push_back(tFitParamsVec3);

    TObjArray* tKStarwFitsCans = tTripleFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,fNonFlatBgdFitTypes,false,false);
    if(aSaveImages)
    {
      assert(tKStarwFitsCans->GetEntries()==3);

      TString tImageSaveName1 = TString::Format("%s%s_%sVaried%s.pdf", 
                                                tSpecificSaveDirectory.Data(), tKStarwFitsCans->At(0)->GetTitle(), 
                                                fGeneralAnTypeModified.Data(), tDirNameModifier.Data());

      TString tImageSaveName2 = TString::Format("%s%s_%sVaried%s.pdf", 
                                                tSpecificSaveDirectory.Data(), tKStarwFitsCans->At(1)->GetTitle(), 
                                                fGeneralAnTypeModified.Data(), tDirNameModifier.Data());

      TString tImageSaveName3 = TString::Format("%s%s_%sVaried%s.pdf", 
                                                tSpecificSaveDirectory.Data(), tKStarwFitsCans->At(2)->GetTitle(), 
                                                fGeneralAnTypeModified.Data(), tDirNameModifier.Data());

      tKStarwFitsCans->At(0)->SaveAs(tImageSaveName1);
      tKStarwFitsCans->At(1)->SaveAs(tImageSaveName2);
      tKStarwFitsCans->At(2)->SaveAs(tImageSaveName3);
    }
    delete tTripleFitGenerator;
  }

  if(!aWriteToTxtFile) 
  {
    PrintText2dVec(tText2dVector1);
    PrintText2dVec(tText2dVector2);
    PrintText2dVec(tText2dVector3);
  }
  else
  {
    AnalysisType tAnTyp1 = kLamKchP;
    TString tOutputFileName1 = TString::Format("%sCfFitValues_%s%s.txt", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp1], cCentralityTags[fCentralityType]);
    std::ofstream tOutputFile1;
    tOutputFile1.open(tOutputFileName1);

    PrintText2dVec(tText2dVector1,tOutputFile1);

    tOutputFile1.close();

    //----------

    AnalysisType tAnTyp2 = kLamKchM;
    TString tOutputFileName2 = TString::Format("%sCfFitValues_%s%s.txt", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp2], cCentralityTags[fCentralityType]);
    std::ofstream tOutputFile2;
    tOutputFile2.open(tOutputFileName2);

    PrintText2dVec(tText2dVector2,tOutputFile2);

    tOutputFile2.close();

    //----------

    AnalysisType tAnTyp3 = kLamK0;
    TString tOutputFileName3 = TString::Format("%sCfFitValues_%s%s.txt", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp3], cCentralityTags[fCentralityType]);
    std::ofstream tOutputFile3;
    tOutputFile3.open(tOutputFileName3);

    PrintText2dVec(tText2dVector3,tOutputFile3);

    tOutputFile3.close();
  }
}


//________________________________________________________________________________________________________________
void TripleFitSystematicAnalysis::RunVaryFitRange(bool aSaveImages, bool aWriteToTxtFile, double aMaxKStar1, double aMaxKStar2, double aMaxKStar3)
{
  // 1=LamKchP;  2=LamKchM;  3=LamK0

  assert(fModifierValues1.size()==0);  //this is not intended for use with various modifier values, but for the final analysis
  int tNRangeValues = 3;
  vector<double> tRangeVec = {aMaxKStar1,aMaxKStar2,aMaxKStar3};

  vector<vector<TString> > tText2dVector1(0);
  vector<vector<TString> > tText2dVector2(0);
  vector<vector<TString> > tText2dVector3(0);

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
    TripleFitGenerator* tTripleFitGenerator = BuildTripleFitGenerator(kTrain, "", fNonFlatBgdFitTypes);
    //TODO are these limits necessary?
    tTripleFitGenerator->SetAllRadiiLimits(1., 10.);
    //TODO these should already be set in BuildTripleFitGenerator
    if(fIncludeResidualsType == kIncludeNoResiduals) tTripleFitGenerator->SetAllLambdaParamLimits(0.1, 1.);
    else tTripleFitGenerator->SetAllLambdaParamLimits(0.6, 1.5);

    tTripleFitGenerator->DoFit(fDualieShareLambda, fDualieShareRadii, tRangeVec[i]);

    TString tRangeValue = TString::Format("Max KStar for Fit = %0.4f",tRangeVec[i]);

    vector<TString> tFitParamsVec1 = tTripleFitGenerator->GetFitGen1()->GetAllFitParametersTStringVector();
    tFitParamsVec1.insert(tFitParamsVec1.begin(),tRangeValue);
    tText2dVector1.push_back(tFitParamsVec1);

    vector<TString> tFitParamsVec2 = tTripleFitGenerator->GetFitGen2()->GetAllFitParametersTStringVector();
    tFitParamsVec2.insert(tFitParamsVec2.begin(),tRangeValue);
    tText2dVector2.push_back(tFitParamsVec2);

    vector<TString> tFitParamsVec3 = tTripleFitGenerator->GetFitGen3()->GetAllFitParametersTStringVector();
    tFitParamsVec3.insert(tFitParamsVec3.begin(),tRangeValue);
    tText2dVector3.push_back(tFitParamsVec3);


    TObjArray* tKStarwFitsCans = tTripleFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,fNonFlatBgdFitTypes,false,false);
    if(aSaveImages)
    {
      assert(tKStarwFitsCans->GetEntries()==3);

      TString tImageSaveName1 = TString::Format("%s%s_MaxFitKStar_%0.4f.pdf", tSpecificSaveDirectory.Data(), tKStarwFitsCans->At(0)->GetTitle(), tRangeVec[i]);
      TString tImageSaveName2 = TString::Format("%s%s_MaxFitKStar_%0.4f.pdf", tSpecificSaveDirectory.Data(), tKStarwFitsCans->At(1)->GetTitle(), tRangeVec[i]);
      TString tImageSaveName3 = TString::Format("%s%s_MaxFitKStar_%0.4f.pdf", tSpecificSaveDirectory.Data(), tKStarwFitsCans->At(2)->GetTitle(), tRangeVec[i]);

      tKStarwFitsCans->At(0)->SaveAs(tImageSaveName1);
      tKStarwFitsCans->At(1)->SaveAs(tImageSaveName2);
      tKStarwFitsCans->At(2)->SaveAs(tImageSaveName3);
    }
    delete tTripleFitGenerator;
  }

  if(!aWriteToTxtFile) 
  {
    PrintText2dVec(tText2dVector1);
    PrintText2dVec(tText2dVector2);
    PrintText2dVec(tText2dVector3);
  }
  else
  {
    AnalysisType tAnTyp1 = kLamKchP;
    TString tOutputFileName1 = TString::Format("%sCfFitValues_VaryMaxFitKStar_%s%s.txt", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp1], cCentralityTags[fCentralityType]);
    std::ofstream tOutputFile1;
    tOutputFile1.open(tOutputFileName1);

    PrintText2dVec(tText2dVector1,tOutputFile1);

    tOutputFile1.close();

    //----------

    AnalysisType tAnTyp2 = kLamKchM;
    TString tOutputFileName2 = TString::Format("%sCfFitValues_VaryMaxFitKStar_%s%s.txt", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp2], cCentralityTags[fCentralityType]);
    std::ofstream tOutputFile2;
    tOutputFile2.open(tOutputFileName2);

    PrintText2dVec(tText2dVector2,tOutputFile2);

    tOutputFile2.close();

    //----------

    AnalysisType tAnTyp3 = kLamK0;
    TString tOutputFileName3 = TString::Format("%sCfFitValues_VaryMaxFitKStar_%s%s.txt", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp3], cCentralityTags[fCentralityType]);
    std::ofstream tOutputFile3;
    tOutputFile3.open(tOutputFileName3);

    PrintText2dVec(tText2dVector3,tOutputFile3);

    tOutputFile3.close();
  }
}


//________________________________________________________________________________________________________________
void TripleFitSystematicAnalysis::RunVaryNonFlatBackgroundFit(bool aSaveImages, bool aWriteToTxtFile)
{
  // 1=LamKchP;  2=LamKchM;  3=LamK0

  assert(fModifierValues1.size()==0);  //this is not intended for use with various modifier values, but for the final analysis
  assert(fApplyNonFlatBackgroundCorrection);  //This better be true, since I'm varying to NonFlatBgd method here!
  int tNFitTypeValues = 4;
  vector<int> tFitTypeVec = {0,1,2,3};

  vector<vector<TString> > tText2dVector1(0);
  vector<vector<TString> > tText2dVector2(0);
  vector<vector<TString> > tText2dVector3(0);

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
    SetNonFlatBgdFitTypes(static_cast<NonFlatBgdFitType>(tFitTypeVec[i]), static_cast<NonFlatBgdFitType>(tFitTypeVec[i]));
    TripleFitGenerator* tTripleFitGenerator = BuildTripleFitGenerator(kTrain, "", fNonFlatBgdFitTypes);

    //--------------------------------
    //For a few cases, fitting with kQuadratic and kGaussian can be difficult, so these restrictions are put in place to combat that
    // NOTE: Changing the MinMaxBgdFit range from [0.6, 0.9] (in typical analysis) to [0.45, 0.95] (here) does not significantly change
    //       the results for kLinear, and helps the other cases to build a more stable background fit
    tTripleFitGenerator->SetMinMaxBgdFit(kLamKchP, 0.45, 0.95);
    tTripleFitGenerator->SetMinMaxBgdFit(kLamK0, 0.45, 0.95);

    // TODO already in place in TripleFitSystematicAnalysis::BuildTripleFitGenerator?
    if(fIncludeResidualsType == kIncludeNoResiduals) tTripleFitGenerator->SetAllLambdaParamLimits(0.1, 1.0);
    else tTripleFitGenerator->SetAllLambdaParamLimits(0.6, 1.5);

    tTripleFitGenerator->SetAllRadiiLimits(1., 10.);  //NOTE: This does nothing when fIncludeResidualsType != kIncludeNoResiduals && fChargedResidualsType != kUseXiDataForAll,
                                                //      as, in that case, FitGenerator hardwires limits [1., 12] to stay within interpolation regime
    //--------------------------------

    tTripleFitGenerator->DoFit(fDualieShareLambda, fDualieShareRadii);

    TString tRangeValue = TString::Format("Fit Type = %d",tFitTypeVec[i]);

    vector<TString> tFitParamsVec1 = tTripleFitGenerator->GetFitGen1()->GetAllFitParametersTStringVector();
    tFitParamsVec1.insert(tFitParamsVec1.begin(),tRangeValue);
    tText2dVector1.push_back(tFitParamsVec1);

    vector<TString> tFitParamsVec2 = tTripleFitGenerator->GetFitGen2()->GetAllFitParametersTStringVector();
    tFitParamsVec2.insert(tFitParamsVec2.begin(),tRangeValue);
    tText2dVector2.push_back(tFitParamsVec2);

    vector<TString> tFitParamsVec3 = tTripleFitGenerator->GetFitGen3()->GetAllFitParametersTStringVector();
    tFitParamsVec3.insert(tFitParamsVec3.begin(),tRangeValue);
    tText2dVector3.push_back(tFitParamsVec3);

    TObjArray* tKStarwFitsCans = tTripleFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,fNonFlatBgdFitTypes,false,false, tZoomROP);
    if(aSaveImages)
    {
      assert(tKStarwFitsCans->GetEntries()==3);

      TString tImageSaveName1 = TString::Format("%s%s_FitType_%d.pdf", tSpecificSaveDirectory.Data(), tKStarwFitsCans->At(0)->GetTitle(), tFitTypeVec[i]);
      TString tImageSaveName2 = TString::Format("%s%s_FitType_%d.pdf", tSpecificSaveDirectory.Data(), tKStarwFitsCans->At(1)->GetTitle(), tFitTypeVec[i]);
      TString tImageSaveName3 = TString::Format("%s%s_FitType_%d.pdf", tSpecificSaveDirectory.Data(), tKStarwFitsCans->At(2)->GetTitle(), tFitTypeVec[i]);


      tKStarwFitsCans->At(0)->SaveAs(tImageSaveName1);
      tKStarwFitsCans->At(1)->SaveAs(tImageSaveName2);
      tKStarwFitsCans->At(2)->SaveAs(tImageSaveName3);
    }
    delete tTripleFitGenerator;
  }

  if(!aWriteToTxtFile) 
  {
    PrintText2dVec(tText2dVector1, std::cout, tNFitTypeValues);
    PrintText2dVec(tText2dVector2, std::cout, tNFitTypeValues);
    PrintText2dVec(tText2dVector3, std::cout, tNFitTypeValues);
  }
  else
  {
    AnalysisType tAnTyp1 = kLamKchP;
    TString tOutputFileName1 = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s%s.txt", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp1], cCentralityTags[fCentralityType]);
    std::ofstream tOutputFile1;
    tOutputFile1.open(tOutputFileName1);

    PrintText2dVec(tText2dVector1, tOutputFile1, tNFitTypeValues);

    tOutputFile1.close();

    //----------

    AnalysisType tAnTyp2 = kLamKchM;
    TString tOutputFileName2 = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s%s.txt", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp2], cCentralityTags[fCentralityType]);
    std::ofstream tOutputFile2;
    tOutputFile2.open(tOutputFileName2);

    PrintText2dVec(tText2dVector2, tOutputFile2, tNFitTypeValues);

    tOutputFile2.close();

    //----------

    AnalysisType tAnTyp3 = kLamK0;
    TString tOutputFileName3 = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s%s.txt", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[tAnTyp3], cCentralityTags[fCentralityType]);
    std::ofstream tOutputFile3;
    tOutputFile3.open(tOutputFileName3);

    PrintText2dVec(tText2dVector3, tOutputFile3, tNFitTypeValues);

    tOutputFile3.close();
  }
}

