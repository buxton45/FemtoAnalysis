///////////////////////////////////////////////////////////////////////////
// FitParameter.h:  a simple parameter class                             //
//  This will hold information for a single parameter                    //
//  This information includes:                                           //
//    Parameter number, name, start value, step size,                    //
//    lower and upper bounds, fixed boolean                              //
///////////////////////////////////////////////////////////////////////////

#ifndef FITPARAMETER_H
#define FITPARAMETER_H

//includes and any constant variable declarations
#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <algorithm>  //std::sort

#include "TString.h"

using std::cout;
using std::cin;
using std::endl;
using std::vector;

#include "Types.h"
#include "Types_LambdaValues.h"

class FitParameter {

public:

  //Constructor, destructor, copy constructor, assignment operator
  FitParameter(ParameterType aParamType, double aStartValue, bool aIsFixed = false, double aLowerParamBound = 0., double aUpperParamBound = 0., double aStepSize = 0.001);
  virtual ~FitParameter();


  void SetFixedToValue(double aValue);

  void SetSharedLocal(bool aIsShared, int aSharedAnalysis);
  void SetSharedLocal(bool aIsShared, vector<int> &aSharedAnalyses);
  vector<int> GetSharedWithLocal();

  void SetSharedGlobal(bool aIsShared, int aSharedAnalysis);
  void SetSharedGlobal(bool aIsShared, const vector<int> &aSharedAnalyses);
  vector<int> GetSharedWithGlobal();

  void SetAttributes(double aStartValue, bool aIsFixed, double aLowerParamBound, double aUpperParamBound, double aStepSize = -1.);
  void SetOwnerInfo(AnalysisType aAnType, CentralityType aCentType, BFieldType aBFieldType);
  TString GetOwnerName();

  //inline (i.e. simple) functions
  void SetType(ParameterType aParamType);
  void SetName(TString aName);
  void SetStartValue(double aStartValue);
  void SetFixed(bool aIsFixed);
  void SetLowerBound(double aLowerBound);
  void SetUpperBound(double aUpperBound);
  void SetStepSize(double aStepSize);

  ParameterType GetType();
  TString GetName();
  double GetStartValue();
  bool IsFixed();
  bool IsSharedLocal();
  bool IsSharedGlobal();
  double GetLowerBound();
  double GetUpperBound();
  double GetStepSize();


  void SetFitValue(double aFitValue);
  double GetFitValue();

  void SetFitValueError(double aFitValueError);
  double GetFitValueError();

  void SetFitValueSysError(double aFitValueSysError);
  double GetFitValueSysError();

  void SetMinuitParamNumber(int aParamNumber);
  int GetMinuitParamNumber();

  void SetFitInfo(TString aFitInfo);
  TString GetFitInfo();

  AnalysisType GetOwnerAnalysisType();
  CentralityType GetOwnerCentralityType();
  BFieldType GetOwnerBFieldType();

private:

  ParameterType fParamType;
  TString fParamName;
  double fStartValue;
  bool fIsFixed;
  bool fIsSharedLocal;
  bool fIsSharedGlobal;
  double fLowerParamBound;
  double fUpperParamBound;
  double fStepSize;

  double fFitValue;
  double fFitValueError;
  double fFitValueSysError;

  vector<int> fSharedWithLocal;
  vector<int> fSharedWithGlobal;

  int fMinuitParamNumber;
  
  ParamOwnerInfo fOwnerInfo;
  TString fFitInfo;

#ifdef __ROOT__
  ClassDef(FitParameter, 1)
#endif
};


//inline stuff
inline void FitParameter::SetType(ParameterType aParamType) {fParamType = aParamType;}
inline void FitParameter::SetName(TString aName) {fParamName = aName;}
inline void FitParameter::SetStartValue(double aStartValue) {fStartValue = aStartValue;}
inline void FitParameter::SetFixed(bool aIsFixed) {fIsFixed = aIsFixed;}
inline void FitParameter::SetLowerBound(double aLowerBound) {fLowerParamBound = aLowerBound;}
inline void FitParameter::SetUpperBound(double aUpperBound) {fUpperParamBound = aUpperBound;}
inline void FitParameter::SetStepSize(double aStepSize) {fStepSize = aStepSize;}

inline ParameterType FitParameter::GetType() {return fParamType;}
inline TString FitParameter::GetName() {return fParamName;}
inline double FitParameter::GetStartValue() {return fStartValue;}
inline bool FitParameter::IsFixed() {return fIsFixed;}
inline bool FitParameter::IsSharedLocal() {return fIsSharedLocal;}
inline bool FitParameter::IsSharedGlobal() {return fIsSharedGlobal;}
inline double FitParameter::GetLowerBound() {return fLowerParamBound;}
inline double FitParameter::GetUpperBound() {return fUpperParamBound;}
inline double FitParameter::GetStepSize() {return fStepSize;}

inline void FitParameter::SetFitValue(double aFitValue) {fFitValue = aFitValue;}
inline double FitParameter::GetFitValue() {return fFitValue;}

inline void FitParameter::SetFitValueError(double aFitValueError) {fFitValueError = aFitValueError;}
inline double FitParameter::GetFitValueError() {return fFitValueError;}

inline void FitParameter::SetFitValueSysError(double aFitValueSysError) {fFitValueSysError = aFitValueSysError;}
inline double FitParameter::GetFitValueSysError() {return fFitValueSysError;}

inline void FitParameter::SetMinuitParamNumber(int aParamNumber) {fMinuitParamNumber = aParamNumber;}
inline int FitParameter::GetMinuitParamNumber() {return fMinuitParamNumber;}

inline void FitParameter::SetFitInfo(TString aFitInfo) {fFitInfo=aFitInfo;}
inline TString FitParameter::GetFitInfo() {return fFitInfo;}


inline AnalysisType FitParameter::GetOwnerAnalysisType() {return fOwnerInfo.analysisType;}
inline CentralityType FitParameter::GetOwnerCentralityType() {return fOwnerInfo.centralityType;}
inline BFieldType FitParameter::GetOwnerBFieldType() {return fOwnerInfo.bFieldType;}

#endif


