/* FitValuesWriterwSysErrs.cxx */

#include "FitValuesWriterwSysErrs.h"

#ifdef __ROOT__
ClassImp(FitValuesWriterwSysErrs)
#endif




//________________________________________________________________________________________________________________
FitValuesWriterwSysErrs::FitValuesWriterwSysErrs() : 
  FitValuesWriter(),
  fResType(kInclude3Residuals)
{

}





//________________________________________________________________________________________________________________
FitValuesWriterwSysErrs::~FitValuesWriterwSysErrs()
{
  //no-op
}


//________________________________________________________________________________________________________________
AnalysisType FitValuesWriterwSysErrs::GetAnalysisType(TString aLine)
{
  int tBeg = aLine.First("AnalysisType");
  TString tSmallLine = aLine.Remove(0,tBeg);

  int tEnd = tSmallLine.First('|');
  tSmallLine = tSmallLine.Remove(tEnd,tSmallLine.Length());

  TObjArray* tContents = tSmallLine.Tokenize("=");
  TString tAnName = ((TObjString*)tContents->At(1))->String().Strip(TString::kBoth, ' ');

  AnalysisType tAnType;
  if(tAnName.EqualTo("LamKchP")) tAnType = kLamKchP;
  else if(tAnName.EqualTo("ALamKchM")) tAnType = kALamKchM;

  else if(tAnName.EqualTo("LamKchM")) tAnType = kLamKchM;
  else if(tAnName.EqualTo("ALamKchP")) tAnType = kALamKchP;

  else if(tAnName.EqualTo("LamK0")) tAnType = kLamK0;
  else if(tAnName.EqualTo("ALamK0")) tAnType = kALamK0;

  else assert(0);

  return tAnType;
}

//________________________________________________________________________________________________________________
td1dVec FitValuesWriterwSysErrs::ReadParameterValue(TString aLine)
{
  td1dVec tReturnVec(0);

  assert(aLine.Contains("Error") && !aLine.Contains("%"));

  TObjArray* tCuts = aLine.Tokenize('|');
  assert(tCuts->GetEntries()==4);

  TString tErr0010 = ((TObjString*)tCuts->At(0))->String().Strip(TString::kBoth, ' ');
  TString tErr1030 = ((TObjString*)tCuts->At(1))->String().Strip(TString::kBoth, ' ');
  TString tErr3050 = ((TObjString*)tCuts->At(2))->String().Strip(TString::kBoth, ' ');

  int tBeg1 = tErr0010.First('=');
  tErr0010.Remove(0,tBeg1+1);
  tErr0010.Strip(TString::kBoth, ' ');

  int tBeg2 = tErr1030.First('=');
  tErr1030.Remove(0,tBeg2+1);
  tErr1030.Strip(TString::kBoth, ' ');

  int tBeg3 = tErr3050.First('=');
  tErr3050.Remove(0,tBeg3+1);
  tErr3050.Strip(TString::kBoth, ' ');

  tReturnVec.push_back(tErr0010.Atof());
  tReturnVec.push_back(tErr1030.Atof());
  tReturnVec.push_back(tErr3050.Atof());

  return tReturnVec;
}


//________________________________________________________________________________________________________________
vector<vector<FitParameter*> > FitValuesWriterwSysErrs::ReadAllParameters(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType)
{
  vector<vector<FitParameter*> > tReturn2dVec(0);
  //----- First, use FitValuesWriter::GetFitResults to build tReturn2dVec without systematic errors
  for(int iCent=0; iCent<(int)kMB; iCent++)
  {
    tReturn2dVec.push_back(GetFitResults(aMasterFileLocation, aFitInfoTString, aAnType, static_cast<CentralityType>(iCent)));
  }

  //----- Now, add systematic errors
  ifstream tSysFile(aSystematicsFileLocation);
  if(!tSysFile.is_open()) cout << "FAILURE - FILE NOT OPEN: " << aSystematicsFileLocation << endl;
  assert(tSysFile.is_open());

  AnalysisType tCurrentAnType;
  ParameterType tCurrentParameterType;
  td1dVec tSysErrVals(0);

  std::string tStdString;
  TString tLine;
  while(getline(tSysFile, tStdString))
  {
    tLine = TString(tStdString);

    if(tLine.Contains("AnalysisType")) tCurrentAnType = GetAnalysisType(tLine);
    if(tLine.Contains("Lambda")) tCurrentParameterType = kLambda;
    if(tLine.Contains("Radius")) tCurrentParameterType = kRadius;
    if(tLine.Contains("Ref0")) tCurrentParameterType = kRef0;
    if(tLine.Contains("Imf0")) tCurrentParameterType = kImf0;
    if(tLine.Contains("d0")) tCurrentParameterType = kd0;
    if(tLine.Contains("Error") && !tLine.Contains("%") && tCurrentAnType==aAnType)
    {
      tSysErrVals = ReadParameterValue(tLine);
      tReturn2dVec[k0010][tCurrentParameterType]->SetFitValueSysError(tSysErrVals[0]);
      tReturn2dVec[k1030][tCurrentParameterType]->SetFitValueSysError(tSysErrVals[1]);
      tReturn2dVec[k3050][tCurrentParameterType]->SetFitValueSysError(tSysErrVals[2]);
    }
  }
  tSysFile.close();
/*
  cout << "AnalysisType = " << cAnalysisBaseTags[aAnType] << endl;
  for(int iCent=0; iCent<tReturn2dVec.size(); iCent++)
  {
    cout << "CentralityType = " << cCentralityTags[iCent] << endl;
    for(unsigned int iParam=0; iParam<tReturn2dVec[iCent].size(); iParam++)
    {
      cout << TString::Format("%s = %0.3f +- %0.3f +- %0.3f", cParameterNames[tReturn2dVec[iCent][iParam]->GetType()], 
                                                              tReturn2dVec[iCent][iParam]->GetFitValue(), 
                                                              tReturn2dVec[iCent][iParam]->GetFitValueError(), 
                                                              tReturn2dVec[iCent][iParam]->GetFitValueSysError()) ;
      cout << endl;
    }
    cout << endl;
  }
*/
  return tReturn2dVec;
}

//________________________________________________________________________________________________________________
FitParameter* FitValuesWriterwSysErrs::GetFitParameterSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, ParameterType aParamType)
{
  vector<vector<FitParameter*> > tReturn2dVec = ReadAllParameters(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType);
  FitParameter* tFitParam = tReturn2dVec[aCentType][aParamType];
  return tFitParam;
}


//________________________________________________________________________________________________________________
TGraphAsymmErrors* FitValuesWriterwSysErrs::GetYvsXGraphSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, ParameterType aParamTypeY, ParameterType aParamTypeX)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);

  FitParameter* tFitParamY = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, aParamTypeY);
  FitParameter* tFitParamX = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, aParamTypeX);

  tReturnGr->SetPoint(0, tFitParamX->GetFitValue(), tFitParamY->GetFitValue());
  tReturnGr->SetPointError(0, tFitParamX->GetFitValueSysError(), tFitParamX->GetFitValueSysError(), tFitParamY->GetFitValueSysError(), tFitParamY->GetFitValueSysError());

  return tReturnGr;
}

//________________________________________________________________________________________________________________
void FitValuesWriterwSysErrs::DrawYvsXGraphSys(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, ParameterType aParamTypeY, ParameterType aParamTypeX, int aMarkerColor, int aMarkerStyle, double aMarkerSize, TString aDrawOption)
{
  aPad->cd();

  TGraphAsymmErrors* tGraphToDraw = GetYvsXGraphSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, aParamTypeY, aParamTypeX);
  tGraphToDraw->SetMarkerColor(aMarkerColor);
  tGraphToDraw->SetMarkerStyle(aMarkerStyle);
  tGraphToDraw->SetMarkerSize(aMarkerSize);
  tGraphToDraw->SetFillColor(aMarkerColor);
  tGraphToDraw->SetFillStyle(1000);
  tGraphToDraw->SetLineColor(aMarkerColor);

  tGraphToDraw->Draw(aDrawOption);
}



//________________________________________________________________________________________________________________
TGraphAsymmErrors* FitValuesWriterwSysErrs::GetImF0vsReF0GraphSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType)
{
  return GetYvsXGraphSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, kImf0, kRef0);
}

//________________________________________________________________________________________________________________
void FitValuesWriterwSysErrs::DrawImF0vsReF0GraphSys(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize, TString aDrawOption)
{
  DrawYvsXGraphSys(aPad, aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, kImf0, kRef0, aMarkerColor, aMarkerStyle, aMarkerSize, aDrawOption);
}


//________________________________________________________________________________________________________________
TGraphAsymmErrors* FitValuesWriterwSysErrs::GetLambdavsRadiusGraphSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType)
{
  return GetYvsXGraphSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, kLambda, kRadius);
}

//________________________________________________________________________________________________________________
void FitValuesWriterwSysErrs::DrawLambdavsRadiusGraphSys(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize, TString aDrawOption)
{
  DrawYvsXGraphSys(aPad, aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, kLambda, kRadius, aMarkerColor, aMarkerStyle, aMarkerSize, aDrawOption);
}

//________________________________________________________________________________________________________________
TGraphAsymmErrors* FitValuesWriterwSysErrs::GetD0GraphSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, double aXOffset)
{
  TGraphAsymmErrors* tReturnGr = new TGraphAsymmErrors(1);
  FitParameter* tFitParamD0 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, kd0);

  tReturnGr->SetPoint(0, aXOffset, tFitParamD0->GetFitValue());
  tReturnGr->SetPointError(0, 0.05, 0.05, tFitParamD0->GetFitValueSysError(), tFitParamD0->GetFitValueSysError());
  return tReturnGr;
}

//________________________________________________________________________________________________________________
void FitValuesWriterwSysErrs::DrawD0GraphSys(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, double aXOffset, int aMarkerColor, int aMarkerStyle, double aMarkerSize, TString aDrawOption)
{
  aPad->cd();

  TGraphAsymmErrors* tGraphToDraw = GetD0GraphSys(aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, aXOffset);
  tGraphToDraw->SetMarkerColor(aMarkerColor);
  tGraphToDraw->SetMarkerStyle(aMarkerStyle);
  tGraphToDraw->SetMarkerSize(aMarkerSize);
  tGraphToDraw->SetFillColor(aMarkerColor);
  tGraphToDraw->SetFillStyle(1000);
  tGraphToDraw->SetLineColor(aMarkerColor);

  tGraphToDraw->Draw(aDrawOption);
}









//________________________________________________________________________________________________________________
void FitValuesWriterwSysErrs::DrawImF0vsReF0Graph(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize, TString aDrawOptionStat, TString aDrawOptionSys, bool aDrawStatOnly)
{
  int tColorSys = TColor::GetColorTransparent(aMarkerColor, 0.3);
  if(!aDrawStatOnly) DrawImF0vsReF0GraphSys(aPad, aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, tColorSys, aMarkerStyle, aMarkerSize, aDrawOptionSys);
  DrawImF0vsReF0GraphStat(aPad, aMasterFileLocation, aFitInfoTString, aAnType, aCentType, aMarkerColor, aMarkerStyle, aMarkerSize, aDrawOptionStat);
}

//________________________________________________________________________________________________________________
void FitValuesWriterwSysErrs::DrawLambdavsRadiusGraph(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize, TString aDrawOptionStat, TString aDrawOptionSys, bool aDrawStatOnly)
{
  int tColorSys = TColor::GetColorTransparent(aMarkerColor, 0.3);
  if(!aDrawStatOnly) DrawLambdavsRadiusGraphSys(aPad, aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, tColorSys, aMarkerStyle, aMarkerSize, aDrawOptionSys);
  DrawLambdavsRadiusGraphStat(aPad, aMasterFileLocation, aFitInfoTString, aAnType, aCentType, aMarkerColor, aMarkerStyle, aMarkerSize, aDrawOptionStat);
}

//________________________________________________________________________________________________________________
void FitValuesWriterwSysErrs::DrawD0Graph(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, double aXOffset, int aMarkerColor, int aMarkerStyle, double aMarkerSize, TString aDrawOptionStat, TString aDrawOptionSys, bool aDrawStatOnly)
{
  int tColorSys = TColor::GetColorTransparent(aMarkerColor, 0.3);
  if(!aDrawStatOnly) DrawD0GraphSys(aPad, aMasterFileLocation, aSystematicsFileLocation, aFitInfoTString, aAnType, aCentType, aXOffset, tColorSys, aMarkerStyle, aMarkerSize, aDrawOptionSys);
  DrawD0GraphStat(aPad, aMasterFileLocation, aFitInfoTString, aAnType, aCentType, aXOffset, aMarkerColor, aMarkerStyle, aMarkerSize, aDrawOptionStat);
}



