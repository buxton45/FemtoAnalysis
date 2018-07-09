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

class FitValuesWriter {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitValuesWriter(vector<TString> &aFitParamsTStringVec, TString &aFitInfoTString);
  virtual ~FitValuesWriter();

  TString GetFitInfoTString(TString aLine);
  AnalysisType GetAnalysisType(TString aLine);
  CentralityType GetCentralityType(TString aLine);
  ParameterType GetParamTypeFromName(TString aName);
  td1dVec ReadParameterValue(TString aLine);
  void InterpretFitParamsTStringVec();


  vector<vector<TString> > ConvertMasterTo2dVec(TString aFileLocation);
  void WriteToMaster(TString aFileLocation);


  //inline (i.e. simple) functions


private:

  vector<TString> fFitParamsTStringVec;
  TString fFitInfoTString;

#ifdef __ROOT__
  ClassDef(FitValuesWriter, 1)
#endif
};


//inline


#endif
