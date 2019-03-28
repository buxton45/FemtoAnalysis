///////////////////////////////////////////////////////////////////////////
// CfHeavy:                                                              //
//                                                                       //
//    A collection of CfLite* to be combined via a weighted              //
//    average routine, hence the name "Heavy".  This class will contain  //
//    a vector<CfLite*>                                                  //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef CFHEAVY_H
#define CFHEAVY_H

#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <assert.h>

#include "TF1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TMath.h"

using std::cout;
using std::endl;
using std::vector;

#include "CfLite.h"
class CfLite;

class CfHeavy {

public:


  CfHeavy(TString aHeavyCfName, TString aHeavyCfTitle, vector<CfLite*> &aCfLiteCollection, double aMinNorm, double aMaxNorm);
  virtual ~CfHeavy();

  CfHeavy(const CfHeavy &aHeavy);
  CfHeavy& operator=(const CfHeavy& aHeavy);

  void CombineCfs();
  void CombineCfs(TString aReturnName, TString aReturnTitle);
  void Rebin(int aRebinFactor);
  void Rebin(int aNGroups, vector<double> &aGroups);
  void Rebin(int aRebinFactor, double aMinNorm, double aMaxNorm);

  void AddCfLite(CfLite* aCfLite);

  TObjArray* GetNumCollection();
  TObjArray* GetDenCollection();
  TObjArray* GetCfCollection();

  void SaveAllCollectionsAndCf(TString aPreName, TString aPostName, TFile *aFile);

  TH1* GetSimplyAddedNumDen(TString aReturnName, bool aGetNum);

  void BuildHeavyCfwErrorsByHand();

  void DivideCfByThermBgd(TH1* aThermBgd);

  //inline-----------------------
  TH1* GetHeavyCf();
  TH1* GetHeavyCfClone();

  TH1* GetHeavyCfwErrorsByHand();

  double GetMinNorm();
  double GetMaxNorm();

  TString GetHeavyCfName();
  TString GetHeavyCfTitle();

  CfLite* GetCfLite(int aPartAnNumber);
  vector<CfLite*> GetCfLiteCollection();

private:

  TString fHeavyCfName;
  TString fHeavyCfTitle;

  vector<CfLite*> fCfLiteCollection;
  int fCollectionSize;

  TH1* fHeavyCf;
  TH1* fHeavyCfwErrorsByHand;
  double fMinNorm, fMaxNorm;





#ifdef __ROOT__
  ClassDef(CfHeavy, 1)
#endif
};


inline TH1* CfHeavy::GetHeavyCf() {return fHeavyCf;}
inline TH1* CfHeavy::GetHeavyCfClone() {return (TH1*)fHeavyCf->Clone();}
inline TH1* CfHeavy::GetHeavyCfwErrorsByHand() {return fHeavyCfwErrorsByHand;}

inline double CfHeavy::GetMinNorm() {return fMinNorm;}
inline double CfHeavy::GetMaxNorm() {return fMaxNorm;}

inline TString CfHeavy::GetHeavyCfName() {return fHeavyCfName;}
inline TString CfHeavy::GetHeavyCfTitle() {return fHeavyCfTitle;}

inline CfLite* CfHeavy::GetCfLite(int aPartAnNumber) {return fCfLiteCollection[aPartAnNumber];}
inline vector<CfLite*> CfHeavy::GetCfLiteCollection() {return fCfLiteCollection;}

#endif











