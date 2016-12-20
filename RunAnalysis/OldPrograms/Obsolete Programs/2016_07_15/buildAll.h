///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAll:                                                             //
//  This will be the parent of buildAllcLamK0 and buildAllcLamcKch       //
//  i.e., this will hold all methods/variables common to both of the     //
//  above.                                                               //
///////////////////////////////////////////////////////////////////////////

#ifndef BUILDALL_H
#define BUILDALL_H

#include "TH1F.h"
#include "TH2F.h"
#include "TH1D.h"
#include "TString.h"
#include "TList.h"
#include "TF1.h"
#include "TVectorD.h"
#include "TLine.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TObjArray.h"

#include "TStyle.h"
#include "TLegend.h"
#include "TVectorD.h"
#include "TVectorT.h"
#include "TPaveText.h"

#include "TFile.h"

#include <vector>
#include <cmath>
#include <assert.h>
using namespace std;


//______________________________________________________________________________________________________________


const double LambdaMass = 1.115683, KaonMass = 0.493677;

//______________________________________________________________________________________________________________
bool reject;
double ffBgFitLow[2];
double ffBgFitHigh[2];

double PurityBgFitFunction(double *x, double *par)
{
  if( reject && !(x[0]>ffBgFitLow[0] && x[0]<ffBgFitLow[1]) && !(x[0]>ffBgFitHigh[0] && x[0]<ffBgFitHigh[1]) )
    {
      TF1::RejectPoint();
      return 0;
    }
  
  return par[0] + par[1]*x[0] + par[2]*pow(x[0],2) + par[3]*pow(x[0],3) + par[4]*pow(x[0],4);
}


//______________________________________________________________________________________________________________
//*****_____*****_____*****_____*****_____*****_____*****_____*****_____*****_____*****_____*****_____*****_____
//______________________________________________________________________________________________________________

class buildAll {

public:
  enum AnalysisType {kLamK0=0, kALamK0=1, kLamKchP=2, kALamKchP=3, kLamKchM=4, kALamKchM=5, kLamPiP=6, kALamPiP=7, kLamPiM=8, kALamPiM=9};
  enum ParticleType {kLam=0, kALam=1, kK0=2, kKchP=3, kKchM=4};

  buildAll(vector<TString> &aVectorOfFileNames);
  virtual ~buildAll();

  TH1F* buildCF(TString name, TString title, TH1* Num, TH1* Denom, int aMinNormBin, int aMaxNormBin);
  TH1F* CombineCFs(TString aReturnName, TString aReturnTitle, TObjArray* aCfCollection, TObjArray* aNumCollection, int aMinNormBin, int aMaxNormBin);

  void SetVectorOfFileNames(vector<TString> &aVectorOfNames);
  TObjArray* ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName);

  TH1F* GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName);
  TObjArray* BuildCollectionOfCfs(TString aContainedHistosBaseName, TObjArray* aNumCollection, TObjArray* aDenCollection, int aMinNormBin, int aMaxNormBin);

  void SetPurityRegimes(TH1F* aLambdaPurity);
  void SetPurityRegimes(TH1F* aLambdaPurity, TH1F* aK0Short1Purity);
  TObjArray* CalculatePurity(const char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2]);
  TH1F* CombineCollectionOfHistograms(TString aReturnHistoName, TObjArray* aCollectionOfHistograms);  //this is used to straight forward add histograms, SHOULD NOT BE USED FOR CFs!!!!!!

  void DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg);

  //-----7 May 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK0)
  //Separation Histograms
  TH2F* Get2DHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName);
  TObjArray* BuildSepCfs(int aRebinFactor, TString aContainedHistosBaseName, int aNumerOfXbins, TObjArray* a2DNumCollection, TObjArray* a2DDenCollection, int aMinNormBin, int aMaxNormBin);

  //-----7 July 2015 
  //Average Separation Cowboys and Sailors Histograms
  TObjArray* BuildCowCfs(int aRebinFactor, TString aContainedHistosBaseName, TObjArray* a2DNumCollection, TObjArray* a2DDenCollection, int aMinNormBin, int aMaxNormBin);


  //Inline Functions------------------------
  void SetDebug(bool aDebug);
  void SetOutputPurityFitInfo(bool aOutputPurityFitInfo);

  //General stuff


 //KStar CF------------------
  void SetMinNormCF(double aMinNormCF);
  void SetMaxNormCF(double aMaxNormCF);
  double GetMinNormCF();
  double GetMaxNormCF();

  void SetMinNormBinCF(int aMinNormBinCF);
  void SetMaxNormBinCF(int aMaxNormBinCF);
  int GetMinNormBinCF();
  int GetMaxNormBinCF();

  //Average Separation CF------------------
  void SetMinNormAvgSepCF(double aMinNormAvgSepCF);
  void SetMaxNormAvgSepCF(double aMaxNormAvgSepCF);
  double GetMinNormAvgSepCF();
  double GetMaxNormAvgSepCF();

  void SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF);
  void SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF);
  int GetMinNormBinAvgSepCF();
  int GetMaxNormBinAvgSepCF();

  //Purity calculations------------------



protected:
  bool fDebug;
  bool fOutputPurityFitInfo;

  //General stuff needed to extract/store the proper files/histograms etc.
  vector<TString> fVectorOfFileNames;

  //KStar CFs-------------------------
  double fMinNormCF, fMaxNormCF;
  int fMinNormBinCF, fMaxNormBinCF;

  //Average Separation CF------------------
  double fMinNormAvgSepCF, fMaxNormAvgSepCF;
  int fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF;


  //Purity calculations------------------
  double fLamBgFitLow[2];
  double fLamBgFitHigh[2];
  double fLamROI[2];

  double fK0Short1BgFitLow[2];
  double fK0Short1BgFitHigh[2];
  double fK0Short1ROI[2];

  //Separation Histograms
  int fMinNormBinSepCF, fMaxNormBinSepCF;


#ifdef __ROOT__
  ClassDef(buildAll, 1)
#endif
};

inline void buildAll::SetDebug(bool aDebug) {fDebug = aDebug;}
inline void buildAll::SetOutputPurityFitInfo(bool aOutputPurityFitInfo) {fOutputPurityFitInfo = aOutputPurityFitInfo;}

//General stuff


//KStar CF------------------
inline void buildAll::SetMinNormCF(double aMinNormCF) {fMinNormCF = aMinNormCF;}
inline void buildAll::SetMaxNormCF(double aMaxNormCF) {fMaxNormCF = aMaxNormCF;}
inline double buildAll::GetMinNormCF() {return fMinNormCF;}
inline double buildAll::GetMaxNormCF() {return fMaxNormCF;}

inline void buildAll::SetMinNormBinCF(int aMinNormBinCF) {fMinNormBinCF = aMinNormBinCF;}
inline void buildAll::SetMaxNormBinCF(int aMaxNormBinCF) {fMaxNormBinCF = aMaxNormBinCF;}
inline int buildAll::GetMinNormBinCF() {return fMinNormBinCF;}
inline int buildAll::GetMaxNormBinCF() {return fMaxNormBinCF;}

//Average Separation CF------------------
inline void buildAll::SetMinNormAvgSepCF(double aMinNormAvgSepCF) {fMinNormAvgSepCF = aMinNormAvgSepCF;}
inline void buildAll::SetMaxNormAvgSepCF(double aMaxNormAvgSepCF) {fMaxNormAvgSepCF = aMaxNormAvgSepCF;}
inline double buildAll::GetMinNormAvgSepCF() {return fMinNormAvgSepCF;}
inline double buildAll::GetMaxNormAvgSepCF() {return fMaxNormAvgSepCF;}

inline void buildAll::SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF) {fMinNormBinAvgSepCF = aMinNormBinAvgSepCF;}
inline void buildAll::SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF) {fMaxNormBinAvgSepCF = aMaxNormBinAvgSepCF;}
inline int buildAll::GetMinNormBinAvgSepCF() {return fMinNormBinAvgSepCF;}
inline int buildAll::GetMaxNormBinAvgSepCF() {return fMaxNormBinAvgSepCF;}


//Purity calculations------------------


#endif

