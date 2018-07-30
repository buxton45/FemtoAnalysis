/* FitValuesWriterwSysErrs.h */


#ifndef FITVALUESWRITERWSYSERRS_H
#define FITVALUESWRITERWSYSERRS_H

#include "FitValuesWriter.h"

class FitValuesWriterwSysErrs : public FitValuesWriter {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitValuesWriterwSysErrs();
  virtual ~FitValuesWriterwSysErrs();


  static AnalysisType GetAnalysisType(TString aLine);
  static td1dVec ReadParameterValue(TString aLine);
  static vector<vector<FitParameter*> > ReadAllParameters(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType);

  static FitParameter* GetFitParameterSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, ParameterType aParamType);


  static TGraphAsymmErrors* GetYvsXGraphSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, ParameterType aParamTypeY, ParameterType aParamTypeX);
  static void DrawYvsXGraphSys(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, ParameterType aParamTypeY, ParameterType aParamTypeX, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOption = "e2same");

  static TGraphAsymmErrors* GetImF0vsReF0GraphSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType);
  static void DrawImF0vsReF0GraphSys(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOption = "e2same");

  static TGraphAsymmErrors* GetLambdavsRadiusGraphSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType);
  static void DrawLambdavsRadiusGraphSys(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOption = "e2same");

  static TGraphAsymmErrors* GetD0GraphSys(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, double aXOffset=0.5);
  static void DrawD0GraphSys(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, double aXOffset, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOption = "e2same");

  //-----------------------------------
  static void DrawImF0vsReF0Graph(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOptionStat = "epsame", TString aDrawOptionSys = "e2same");

  static void DrawLambdavsRadiusGraph(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOptionStat = "epsame", TString aDrawOptionSys = "e2same");

  static void DrawD0Graph(TPad* aPad, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString, AnalysisType aAnType, CentralityType aCentType, double aXOffset, int aMarkerColor, int aMarkerStyle, double aMarkerSize=0.75, TString aDrawOptionStat = "epsame", TString aDrawOptionSys = "e2same");

  //inline (i.e. simple) functions


private:
  TString fMasterFileLocation;
  TString fSystematicsFileLocation;
  TString fFitInfoTString;

  IncludeResidualsType fResType;

#ifdef __ROOT__
  ClassDef(FitValuesWriterwSysErrs, 1)
#endif
};


//inline


#endif
