/* FitValuesLatexTableHelperWriter.h */


#ifndef FITVALUESLATEXTABLEHELPERWRITER_H
#define FITVALUESLATEXTABLEHELPERWRITER_H

#include "FitValuesWriter.h"

class FitValuesLatexTableHelperWriter : public FitValuesWriter {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitValuesLatexTableHelperWriter();
  virtual ~FitValuesLatexTableHelperWriter();


  static TString GetFitInfoTStringFromTwoLetterID(TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType);
  static TString GetFitInfoTStringFromTwoLetterID_LamKch(TString aTwoLetterID, IncludeResidualsType aResType);
  static TString GetFitInfoTStringFromTwoLetterID_LamK0(TString aTwoLetterID, IncludeResidualsType aResType);

  static TString GetLatexTableOverallLabel(TString aTwoLetterID);
  static void WriteLatexTableHelperEntry(ostream &aOut, TString aMasterFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType);
  static void WriteLatexTableHelperSection(ostream &aOut, TString aMasterFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType);
  static void WriteLatexTableHelperHeader(ostream &aOut);
  static void WriteLatexTableHelper(TString aHelperBaseLocation, TString aMasterFileLocation, AnalysisType aAnType, IncludeResidualsType aResType);

  //inline (i.e. simple) functions


private:
  IncludeResidualsType fResType;

#ifdef __ROOT__
  ClassDef(FitValuesLatexTableHelperWriter, 1)
#endif
};


//inline


#endif
