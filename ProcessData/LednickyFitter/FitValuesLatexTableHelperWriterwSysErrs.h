/* FitValuesLatexTableHelperWriterwSysErrs.h */


#ifndef FITVALUESLATEXTABLEHELPERWRITERWSYSERRS_H
#define FITVALUESLATEXTABLEHELPERWRITERWSYSERRS_H

#include "FitValuesWriterwSysErrs.h"
#include "FitValuesLatexTableHelperWriter.h"

class FitValuesLatexTableHelperWriterwSysErrs : public FitValuesWriterwSysErrs {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitValuesLatexTableHelperWriterwSysErrs();
  virtual ~FitValuesLatexTableHelperWriterwSysErrs();

  static void WriteLatexTableHelperEntry(ostream &aOut, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType,   ResPrimMaxDecayType tResPrimMaxDecayType);
  static void WriteLatexTableHelperSection(ostream &aOut, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType, ResPrimMaxDecayType tResPrimMaxDecayType);
  static void WriteLatexTableHelperHeader(ostream &aOut);
  static void WriteLatexTableHelper(TString aHelperBaseLocation, TString aMasterFileLocatio, TString aSystematicsFileLocation, AnalysisType aAnType, IncludeResidualsType aResType, ResPrimMaxDecayType tResPrimMaxDecayType);

  static void WriteLatexTableHelperEntryForSingle(ostream &aOut, TString aMasterFileLocation, TString aSystematicsFileLocation, AnalysisType aAnType, TString aFitInfoTString);
  static void WriteSingleLatexTableHelper(TString aResultsDate, AnalysisType aAnType, TString aFitInfoTString);

  //inline (i.e. simple) functions


private:
  IncludeResidualsType fResType;

#ifdef __ROOT__
  ClassDef(FitValuesLatexTableHelperWriterwSysErrs, 1)
#endif
};


//inline


#endif
