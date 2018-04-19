/* ThermCf.h */
/* Basically turned old CompareBackgrounds.h methods into a class */

#ifndef THERMCF_H
#define THERMCF_H

//includes and any constant variable declarations
#include <iostream>
#include <iomanip>
#include <complex>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"
#include "TDirectoryFile.h"

#include "TSystem.h"
#include "TMinuit.h"
#include "TVirtualFitter.h"

using std::cout;
using std::endl;
using std::vector;

#include "CfLite.h"
class CfLite;

#include "CfHeavy.h"
class CfHeavy;

#include "Types.h"
#include "Types_LambdaValues.h"
#include "Types_ThermBgdParams.h"

#include "AnalysisInfo.h"
class AnalysisInfo;

class ThermCf {

public:

  //Constructor, destructor, copy constructor, assignment operator
  ThermCf(TString aFileName, TString aCfDescriptor, AnalysisType aAnalysisType, CentralityType aCentralityType, bool aCombineConj, bool aCombineLamKchPM, ThermEventsType aThermEventsType, int aRebin, double aMinNorm, double aMaxNorm);
  virtual ~ThermCf();

  static void SetStyleAndColor(TH1* aHist, int aMarkerStyle, int aColor, double aMarkerSize=0.75);
  static TH1* GetThermHist(TString aFileLocation, TString aHistName);
  static CfHeavy* CombineTwoCfHeavy(TString aName, CfHeavy* aCfHeavy1, CfHeavy* aCfHeavy2);

  static CfHeavy* GetThermHeavyCf(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, int aImpactParam=8, bool aCombineConj=true, bool aUseAdamEvents=false, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, bool aUseNumRotPar2InsteadOfDen=false);
  static TH1* GetThermCf(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, int aImpactParam=8, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1, bool aUseNumRotPar2InsteadOfDen=false);
  static TH1* GetThermCf(AnalysisType aAnType, int aImpactParam=8, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1, bool aUseNumRotPar2InsteadOfDen=false);

  static CfHeavy* GetCentralityCombinedThermCfsHeavy(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, CentralityType aCentType, bool aCombineConj=true, bool aUseAdamEvents=false, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, bool aUseNumRotPar2InsteadOfDen=false);
  static CfHeavy* GetCentralityCombinedThermCfsHeavy(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, CentralityType aCentType, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, bool aUseNumRotPar2InsteadOfDen=false);
  static TH1* GetCentralityCombinedThermCfs(TString aFileName, TString aCfDescriptor, AnalysisType aAnType, CentralityType aCentType, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1, bool aUseNumRotPar2InsteadOfDen=false);
  static TH1* GetCentralityCombinedThermCfs(AnalysisType aAnType, CentralityType aCentType, bool aCombineConj=true, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1, bool aUseNumRotPar2InsteadOfDen=false);

  static CfHeavy* GetLamKchPMCombinedThermCfsHeavy(TString aFileName, TString aCfDescriptor, CentralityType aCentType, bool aUseAdamEvents=false, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, bool aUseNumRotPar2InsteadOfDen=false);
  static CfHeavy* GetLamKchPMCombinedThermCfsHeavy(TString aFileName, TString aCfDescriptor, CentralityType aCentType, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, bool aUseNumRotPar2InsteadOfDen=false);
  static TH1* GetLamKchPMCombinedThermCfs(TString aFileName, TString aCfDescriptor, CentralityType aCentType, ThermEventsType aEventsType=kMe, int aRebin=2, double aMinNorm=0.32, double aMaxNorm=0.40, int aMarkerStyle=20, int aColor=1, bool aUseNumRotPar2InsteadOfDen=false);

  //------------------------------------------------------------------------

  void BuildThermCf();


  //inline (i.e. simple) functions
  AnalysisType GetAnalysisType();
  CentralityType GetCentralityType();

  CfHeavy* GetThermCfHeavy();
  TH1* GetThermCf();

  void SetFileName(TString aFileName);
  void SetCfDescriptor(TString aCfDescriptor);

  void SetAnalysisType(AnalysisType aAnType);
  void SetCentralityType(CentralityType aCentType);

  void SetCombineConjugates(bool aCombine);
  void SetCombineLamKchPM(bool aCombine);
  void SetThermEventsType(ThermEventsType aType);

  void SetRebin(int aRebin);
  void SetMinNorm(double aMin);
  void SetMaxNorm(double aMax);

  void SetUseNumRotPar2InsteadOfDen(bool aUse);

private:
  TString fFileName;
  TString fCfDescriptor;

  AnalysisType fAnalysisType;
  CentralityType fCentralityType;

  bool fCombineConjugates;
  bool fCombineLamKchPM;
  ThermEventsType fThermEventsType;

  int fRebin;
  double fMinNorm;
  double fMaxNorm;

  bool fUseNumRotPar2InsteadOfDen;

  CfHeavy *fThermCfHeavy;


#ifdef __ROOT__
  ClassDef(ThermCf, 1)
#endif
};


//inline
inline AnalysisType ThermCf::GetAnalysisType() {return fAnalysisType;}
inline CentralityType ThermCf::GetCentralityType() {return fCentralityType;}

inline CfHeavy* ThermCf::GetThermCfHeavy() {return fThermCfHeavy;}
inline TH1* ThermCf::GetThermCf() {return fThermCfHeavy->GetHeavyCf();}


inline void ThermCf::SetFileName(TString aFileName) {fFileName=aFileName; BuildThermCf();}
inline void ThermCf::SetCfDescriptor(TString aCfDescriptor) {fCfDescriptor=aCfDescriptor; BuildThermCf();}

inline void ThermCf::SetAnalysisType(AnalysisType aAnType) {fAnalysisType=aAnType; BuildThermCf();}
inline void ThermCf::SetCentralityType(CentralityType aCentType) {fCentralityType=aCentType; BuildThermCf();}

inline void ThermCf::SetCombineConjugates(bool aCombine) {fCombineConjugates=aCombine; BuildThermCf();}
inline void ThermCf::SetCombineLamKchPM(bool aCombine) {fCombineLamKchPM=aCombine; BuildThermCf();}
inline void ThermCf::SetThermEventsType(ThermEventsType aType) {fThermEventsType=aType; BuildThermCf();}

inline void ThermCf::SetRebin(int aRebin) {fRebin=aRebin; BuildThermCf();}
inline void ThermCf::SetMinNorm(double aMin) {fMinNorm=aMin; BuildThermCf();}
inline void ThermCf::SetMaxNorm(double aMax) {fMaxNorm=aMax; BuildThermCf();}

inline void ThermCf::SetUseNumRotPar2InsteadOfDen(bool aUse) {fUseNumRotPar2InsteadOfDen = aUse;}

#endif
