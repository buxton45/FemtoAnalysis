///////////////////////////////////////////////////////////////////////////
// CfLite:                                                               //
//                                                                       //
//  This class holds one single Cf and the methods to build it           //
//  To build, must supply: Name, Title, TH1* Num, TH1* Den               //
//               Optional: MinNorm, MaxNorm (default for KStarCf)        //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef CFLITE_H
#define CFLITE_H

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



class CfLite {

public:


  CfLite(TString aName, TString aTitle, TH1* aNum, TH1* aDen, double aMinNorm, double aMaxNorm);
  virtual ~CfLite();

  CfLite(const CfLite &aLite);
  CfLite& operator=(const CfLite& aLite);

  void BuildCf(double aMinNorm,double aMaxNorm);
  void BuildCf(TString aName, TString aTitle, double aMinNorm, double aMaxNorm);

  void Rebin(int aRebinFactor);
  void Rebin(int aRebinFactor, double aMinNorm, double aMaxNorm);


  //inline-----------------------
  TString CfName();
  TString CfTitle();

  TH1* Num();
  TH1* Den();
  TH1* Cf();

  double GetMinNorm();
  double GetMaxNorm();

  int GetNumScale();
  int GetDenScale();

private:

  TString fCfName;
  TString fCfTitle;
  TH1 *fNum, *fDen, *fCf;
  double fMinNorm, fMaxNorm;
  double fNumScale, fDenScale;





#ifdef __ROOT__
  ClassDef(CfLite, 1)
#endif
};

inline TString CfLite::CfName() {return fCfName;}
inline TString CfLite::CfTitle() {return fCfTitle;}

inline TH1* CfLite::Num() {return fNum;}
inline TH1* CfLite::Den() {return fDen;}
inline TH1* CfLite::Cf() {return fCf;}

inline double CfLite::GetMinNorm() {return fMinNorm;}
inline double CfLite::GetMaxNorm() {return fMaxNorm;}

inline int CfLite::GetNumScale() {return fNumScale;}
inline int CfLite::GetDenScale() {return fDenScale;};






#endif











