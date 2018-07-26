/* FitValuesLatexTableHelperWriter.h */


#ifndef FITVALUESLATEXTABLEHELPERWRITER_H
#define FITVALUESLATEXTABLEHELPERWRITER_H

#include "FitValuesWriter.h"

class FitValuesLatexTableHelperWriter : public FitValuesWriter {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitValuesLatexTableHelperWriter(TString aMasterFileLocation, TString aResultsDate, AnalysisType aAnType);
  virtual ~FitValuesLatexTableHelperWriter();




  static TString GetTwoLetterID(TString aFitInfoTString, IncludeResidualsType aResType);
  static TString GetFitInfoTStringFromTwoLetterID(TString aTwoLetterID, IncludeResidualsType aResType);
  static TString GetLatexTableOverallLabel(TString aFitInfoTString);
  static vector<TString> GetFitInfoTStringAndLatexTableOverallLabel(TString aTwoLetterID, IncludeResidualsType aResType);
  static void WriteLatexTableHelperEntry(ostream &aOut, TString aMasterFileLocation, TString aFitInfoTString, AnalysisType aAnType);
  static void WriteLatexTableHelperSection(ostream &aOut, TString aMasterFileLocation, TString aFitInfoTString);
  static void WriteLatexTableHelperHeader(ostream &aOut);
  static void WriteLatexTableHelper(TString aHelperLocation, TString aMasterFileLocation, IncludeResidualsType aResType);

  //inline (i.e. simple) functions


private:
  IncludeResidualsType fResType;

#ifdef __ROOT__
  ClassDef(FitValuesLatexTableHelperWriter, 1)
#endif
};


//inline


#endif
