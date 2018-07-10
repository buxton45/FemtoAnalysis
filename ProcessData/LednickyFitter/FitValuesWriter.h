/* FitValuesWriter.h */

#ifndef FITVALUESWRITER_H
#define FITVALUESWRITER_H

//includes and any constant variable declarations
#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TSystem.h"
#include "TObjString.h"
#include "TObjArray.h"

using std::cout;
using std::endl;
using std::vector;

#include "Types.h"
#include "Types_LambdaValues.h"
#include "Types_ThermBgdParams.h"

#include "AnalysisInfo.h"
class AnalysisInfo;

#include "FitParameter.h"

class FitValuesWriter {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitValuesWriter(TString aMasterFileLocation, TString aResultsDate, AnalysisType aAnType);
  virtual ~FitValuesWriter();

  static TString GetFitInfoTString(TString aLine);
  static AnalysisType GetAnalysisType(TString aLine);
  static CentralityType GetCentralityType(TString aLine);
  static ParameterType GetParamTypeFromName(TString aName);
  static td1dVec ReadParameterValue(TString aLine);
  static vector<vector<FitParameter*> > InterpretFitParamsTStringVec(vector<TString> &aTStringVec);


  static vector<vector<TString> > ConvertMasterTo2dVec(TString aFileLocation);
  static void WriteToMaster(TString aFileLocation, vector<TString> &aFitParamsTStringVec, TString &aFitInfoTString, TString aSaveNameModifier="");

  static vector<vector<FitParameter*> > GetAllFitResults(TString aFileLocation, TString aFitInfoTString, TString aSaveNameModifier="");
//  static td1dVec GetFitResults(TString aFileLocation, TString &aFitInfoTString, AnalysisType aAnType, CentralityType aCentType);

  //inline (i.e. simple) functions


private:
  TString fMasterFileLocation;
  TString fResultsDate;
  AnalysisType fAnalysisType;

  vector<TString> fFitParamsTStringVec;
  TString fFitInfoTString;

#ifdef __ROOT__
  ClassDef(FitValuesWriter, 1)
#endif
};


//inline


#endif
