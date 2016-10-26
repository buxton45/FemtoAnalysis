///////////////////////////////////////////////////////////////////////////
// SystematicAnalysis:                                                   //
//                                                                       //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef SYSTEMATICANALYSIS_H
#define SYSTEMATICANALYSIS_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "Analysis.h"
class Analysis;



class SystematicAnalysis {

public:
  SystematicAnalysis(TString aFileLocationBase, AnalysisType aAnalysisType, CentralityType aCentralityType,
                   TString aDirNameModifierBase1, vector<double> &aModifierValues1,
                   TString aDirNameModifierBase2="", vector<double> &aModifierValues2 = vector<double>());

  virtual ~SystematicAnalysis();


protected:
  TString fFileLocationBase;
  AnalysisType fAnalysisType;
  CentralityType fCentralityType;

  TString fDirNameModifierBase1;
  TString fDirNameModifierBase2;

  vector<double> fModifierValues1;
  vector<double> fModifierValues2;

  vector<Analysis> fAnalyses;





#ifdef __ROOT__
  ClassDef(SystematicAnalysis, 1)
#endif
};



#endif
