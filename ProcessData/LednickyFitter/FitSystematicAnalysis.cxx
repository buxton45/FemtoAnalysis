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
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, bool aShareLambdaParams) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fApplyNonFlatBackgroundCorrection(false),
  fApplyMomResCorrection(false),
  fSaveDirectory(""),

  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(aDirNameModifierBase2),
  fModifierValues1(aModifierValues1),
  fModifierValues2(aModifierValues2)

{
  if(!fDirNameModifierBase2.IsNull()) assert(fModifierValues1.size() == fModifierValues2.size());
}


//________________________________________________________________________________________________________________
FitSystematicAnalysis::FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, bool aShareLambdaParams) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fApplyNonFlatBackgroundCorrection(false),
  fApplyMomResCorrection(false),
  fSaveDirectory(""),

  fDirNameModifierBase1(aDirNameModifierBase1),
  fDirNameModifierBase2(0),
  fModifierValues1(aModifierValues1),
  fModifierValues2(0)

{
  fDirNameModifierBase2 = "";
  fModifierValues2 = vector<double> (0);

  FitSystematicAnalysis(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, aDirNameModifierBase1, aModifierValues1, fDirNameModifierBase2, fModifierValues2, aCentralityType, aGeneratorType, aShareLambdaParams);
}

//________________________________________________________________________________________________________________
FitSystematicAnalysis::FitSystematicAnalysis(TString aFileLocationBase, TString aFileLocationBaseMC, AnalysisType aAnalysisType,
                                       CentralityType aCentralityType, FitGeneratorType aGeneratorType, bool aShareLambdaParams) :
  fFileLocationBase(aFileLocationBase),
  fFileLocationBaseMC(aFileLocationBaseMC),
  fAnalysisType(aAnalysisType),
  fCentralityType(aCentralityType),
  fFitGeneratorType(aGeneratorType),
  fShareLambdaParams(aShareLambdaParams),
  fApplyNonFlatBackgroundCorrection(false),
  fApplyMomResCorrection(false),
  fSaveDirectory(""),

  fDirNameModifierBase1(0),
  fDirNameModifierBase2(0),
  fModifierValues1(0),
  fModifierValues2(0)

{
  fDirNameModifierBase2 = "";
  fModifierValues2 = vector<double> (0);

  FitSystematicAnalysis(aFileLocationBase, aFileLocationBaseMC, aAnalysisType, fDirNameModifierBase1, fModifierValues1, fDirNameModifierBase2, fModifierValues2, aCentralityType, aGeneratorType, aShareLambdaParams);
}


//________________________________________________________________________________________________________________
FitSystematicAnalysis::~FitSystematicAnalysis()
{
/*no-op*/
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
void FitSystematicAnalysis::RunAllFits(bool aSave, ostream &aOut)
{
  vector<vector<TString> > tText2dVector(0);

  for(unsigned int i=0; i<fModifierValues1.size(); i++)
  {
    TString tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
    if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

    FitGenerator* tFitGenerator = new FitGenerator(fFileLocationBase, fFileLocationBaseMC, fAnalysisType, kTrainSys, 2, fCentralityType, fFitGeneratorType, fShareLambdaParams, tDirNameModifier);

    tFitGenerator->DoFit(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection);

//    OutputCutValues(i,aOut);
//    tFitGenerator->WriteAllFitParameters(aOut);
    TString tCutValue = GetCutValues(i);
    vector<TString> tFitParamsVec = tFitGenerator->GetAllFitParametersTStringVector();
    tFitParamsVec.insert(tFitParamsVec.begin(),tCutValue);
    tText2dVector.push_back(tFitParamsVec);

    TCanvas* tKStarwFitsCan = tFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,false);
    if(aSave)
    {
      TString tSaveName = fSaveDirectory;
      tSaveName += tKStarwFitsCan->GetTitle();
      TString tDirNameModifier = fDirNameModifierBase1 + TString::Format("%0.6f",fModifierValues1[i]);
      if(!fDirNameModifierBase2.IsNull()) tDirNameModifier += fDirNameModifierBase2 + TString::Format("%0.6f",fModifierValues2[i]);

      tSaveName += tDirNameModifier;
      if(fApplyMomResCorrection) tSaveName += TString("_MomResCrctn");
      if(fApplyNonFlatBackgroundCorrection) tSaveName += TString("_NonFlatBgdCrctn");
      tSaveName += TString(".pdf");
      tKStarwFitsCan->SaveAs(tSaveName);
    }
    delete tFitGenerator;
  }

  PrintText2dVec(tText2dVector,aOut);
}


//________________________________________________________________________________________________________________
void FitSystematicAnalysis::RunVaryFitRange(bool aSave, ostream &aOut, double aMaxKStar1, double aMaxKStar2, double aMaxKStar3)
{
  assert(fModifierValues1.size()==0);  //this is not intended for use with various modifier values, but for the final analysis
  int tNRangeValues = 3;
  vector<double> tRangeVec = {aMaxKStar1,aMaxKStar2,aMaxKStar3};

  vector<vector<TString> > tText2dVector(0);

  for(int i=0; i<tNRangeValues; i++)
  {
    FitGenerator* tFitGenerator = new FitGenerator(fFileLocationBase, fFileLocationBaseMC, fAnalysisType, kTrain, 2, fCentralityType, fFitGeneratorType, fShareLambdaParams);
    tFitGenerator->DoFit(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,tRangeVec[i]);

    TString tRangeValue = TString::Format("Max KStar for Fit = %0.4f",tRangeVec[i]);
    vector<TString> tFitParamsVec = tFitGenerator->GetAllFitParametersTStringVector();
    tFitParamsVec.insert(tFitParamsVec.begin(),tRangeValue);
    tText2dVector.push_back(tFitParamsVec);

    TCanvas* tKStarwFitsCan = tFitGenerator->DrawKStarCfswFits(fApplyMomResCorrection,fApplyNonFlatBackgroundCorrection,false);
    if(aSave)
    {
      TString tSaveName = fSaveDirectory;
      tSaveName += tKStarwFitsCan->GetTitle();
      tSaveName += TString::Format("_MaxFitKStar_%0.4f",tRangeVec[i]);
      if(fApplyMomResCorrection) tSaveName += TString("_MomResCrctn");
      if(fApplyNonFlatBackgroundCorrection) tSaveName += TString("_NonFlatBgdCrctn");
      tSaveName += TString(".pdf");
      tKStarwFitsCan->SaveAs(tSaveName);
    }
    delete tFitGenerator;
  }
  PrintText2dVec(tText2dVector,aOut);
}



