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
void FitSystematicAnalysis::PrintText2dVec(vector<vector<TString> > &a2dVec, ostream &aOut)
{
  int tNCuts = (int)a2dVec.size();
  assert(tNCuts==3);
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
void FitSystematicAnalysis::AppendFitInfo(TString &aSaveName, bool aApplyMomResCorrection, bool aApplyNonFlatBackgroundCorrection, 
                                          IncludeResidualsType aIncludeResidualsType, ResPrimMaxDecayType aResPrimMaxDecayType, ChargedResidualsType aChargedResidualsType)
{
  if(aApplyMomResCorrection) aSaveName += TString("_MomResCrctn");
  if(aApplyNonFlatBackgroundCorrection) aSaveName += TString("_NonFlatBgdCrctn");

  aSaveName += cIncludeResidualsTypeTags[aIncludeResidualsType];
  if(aIncludeResidualsType != kIncludeNoResiduals)
  {
    aSaveName += cResPrimMaxDecayTypeTags[aResPrimMaxDecayType];
    aSaveName += cChargedResidualsTypeTags[aChargedResidualsType];
  }
}


//________________________________________________________________________________________________________________
void FitSystematicAnalysis::AppendFitInfo(TString &aSaveName)
{
  AppendFitInfo(aSaveName, fApplyMomResCorrection, fApplyNonFlatBackgroundCorrection, fIncludeResidualsType, fResPrimMaxDecayType, fChargedResidualsType);
}


//________________________________________________________________________________________________________________
FitGenerator* FitSystematicAnalysis::BuildFitGenerator(AnalysisRunType aRunType, TString aDirNameModifier, NonFlatBgdFitType aNonFlatBgdFitType)
{
  //For now, it appears only needed parameters here are AnalysisRunType aRunType and TString aDirNameModifier (for FitGenerator constructor)
  // and NonFlatBgdFitType aNonFlatBgdFitType for assigning attributes
  // Otherwise, default members used

  FitGenerator* tFitGenerator = new FitGenerator(fFileLocationBase, fFileLocationBaseMC, fAnalysisType, fCentralityType, aRunType, 2, fFitGeneratorType, fShareLambdaParams, fAllShareSingleLambdaParam, aDirNameModifier);

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

  tFitGenerator->SetRadiusStartValues({cFitParamValues[fIncludeResidualsType][fAnalysisType][k0010][kRadius], 
                                       cFitParamValues[fIncludeResidualsType][fAnalysisType][k1030][kRadius], 
                                       cFitParamValues[fIncludeResidualsType][fAnalysisType][k3050][kRadius]});

  if(fAllShareSingleLambdaParam) tFitGenerator->SetLambdaParamStartValue(cFitParamValues[fIncludeResidualsType][fAnalysisType][k0010][kLambda]);
  else
  {
    tFitGenerator->SetAllLambdaParamStartValues({cFitParamValues[fIncludeResidualsType][fAnalysisType][k0010][kLambda], 
                                                 cFitParamValues[fIncludeResidualsType][fConjAnalysisType][k0010][kLambda], 
                                                 cFitParamValues[fIncludeResidualsType][fAnalysisType][k1030][kLambda], 
                                                 cFitParamValues[fIncludeResidualsType][fConjAnalysisType][k1030][kLambda], 
                                                 cFitParamValues[fIncludeResidualsType][fAnalysisType][k3050][kLambda], 
                                                 cFitParamValues[fIncludeResidualsType][fConjAnalysisType][k3050][kLambda]});
  }

  if((fAnalysisType==kLamK0 || fAnalysisType==kALamK0) && fIncludeResidualsType == kIncludeNoResiduals) tFitGenerator->SetAllLambdaParamLimits(0.4, 0.6);
  if((fAnalysisType==kLamK0 || fAnalysisType==kALamK0) && fIncludeResidualsType != kIncludeNoResiduals) tFitGenerator->SetAllLambdaParamLimits(0.6, 1.5);

  tFitGenerator->SetScattParamStartValues(cFitParamValues[fIncludeResidualsType][fAnalysisType][k0010][kRef0], 
                                          cFitParamValues[fIncludeResidualsType][fAnalysisType][k0010][kImf0],
                                          cFitParamValues[fIncludeResidualsType][fAnalysisType][k0010][kd0]);

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

    FitGenerator* tFitGenerator = BuildFitGenerator(kTrainSys, tDirNameModifier, fNonFlatBgdFitType);
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
    tSpecificSaveDirectory = TString::Format("%sSystematics/", fSaveDirectory.Data());
    AppendFitInfo(tSpecificSaveDirectory);
    tSpecificSaveDirectory += TString("/");
    gSystem->mkdir(tSpecificSaveDirectory, true);
  }

  for(int i=0; i<tNRangeValues; i++)
  {
    FitGenerator* tFitGenerator = BuildFitGenerator(kTrain, "", fNonFlatBgdFitType);
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
  int tNFitTypeValues = 3;
  vector<int> tFitTypeVec = {0,1,2};

  vector<vector<TString> > tText2dVector(0);

  TString tSpecificSaveDirectory;
  if(aSaveImages || aWriteToTxtFile)
  {
    tSpecificSaveDirectory = TString::Format("%sSystematics/", fSaveDirectory.Data());
    AppendFitInfo(tSpecificSaveDirectory);
    tSpecificSaveDirectory += TString("/");
    gSystem->mkdir(tSpecificSaveDirectory, true);
  }

  for(int i=0; i<tNFitTypeValues; i++)
  {
    FitGenerator* tFitGenerator = BuildFitGenerator(kTrain, "", static_cast<NonFlatBgdFitType>(tFitTypeVec[i]));
    tFitGenerator->DoFit();

    TString tRangeValue = TString::Format("Fit Type = %d",tFitTypeVec[i]);
    vector<TString> tFitParamsVec = tFitGenerator->GetAllFitParametersTStringVector();
    tFitParamsVec.insert(tFitParamsVec.begin(),tRangeValue);
    tText2dVector.push_back(tFitParamsVec);

    TCanvas* tKStarwFitsCan = tFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,static_cast<NonFlatBgdFitType>(tFitTypeVec[i]),false,false);
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

  if(!aWriteToTxtFile) PrintText2dVec(tText2dVector);
  else
  {
    TString tOutputFileName = TString::Format("%sCfFitValues_VaryNonFlatBgdFitType_%s%s", tSpecificSaveDirectory.Data(), cAnalysisBaseTags[fAnalysisType], cCentralityTags[fCentralityType]);
    AppendFitInfo(tOutputFileName);
    tOutputFileName += TString(".txt");
    std::ofstream tOutputFile;
    tOutputFile.open(tOutputFileName);

    PrintText2dVec(tText2dVector,tOutputFile);

    tOutputFile.close();
  }
}

