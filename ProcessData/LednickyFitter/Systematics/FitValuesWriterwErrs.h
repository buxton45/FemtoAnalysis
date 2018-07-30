/* FitValuesWriterwErrs.h */


#ifndef FITVALUESWRITERWERRS_H
#define FITVALUESWRITERWERRS_H

#include "FitValuesWriter.h"

class FitValuesWriterwErrs : public FitValuesWriter {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitValuesWriterwErrs(TString aMasterFileLocation, TString aSystematicsFileLocation, TString aFitInfoTString);
  virtual ~FitValuesWriterwErrs();


  AnalysisType GetAnalysisType(TString aLine);
  td1dVec ReadParameterValue(TString aLine);
  vector<vector<FitParameter*> > ReadAllParameters(AnalysisType aAnType);

  //inline (i.e. simple) functions


private:
  TString fMasterFileLocation;
  TString fSystematicsFileLocation;
  TString fFitInfoTString;

  IncludeResidualsType fResType;

#ifdef __ROOT__
  ClassDef(FitValuesWriterwErrs, 1)
#endif
};


//inline


#endif
