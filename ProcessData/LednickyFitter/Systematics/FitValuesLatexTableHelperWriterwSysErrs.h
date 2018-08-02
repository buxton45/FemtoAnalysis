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

  static void WriteLatexTableHelperEntry(ostream &aOut, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType);
  static void WriteLatexTableHelperSection(ostream &aOut, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType);
  static void WriteLatexTableHelperHeader(ostream &aOut);
  static void WriteLatexTableHelper(TString aHelperBaseLocation, TString aMasterFileLocatio, TString aSystematicsFileLocationn, AnalysisType aAnType, IncludeResidualsType aResType);

  //inline (i.e. simple) functions


private:
  IncludeResidualsType fResType;

#ifdef __ROOT__
  ClassDef(FitValuesLatexTableHelperWriterwSysErrs, 1)
#endif
};


//inline


#endif
