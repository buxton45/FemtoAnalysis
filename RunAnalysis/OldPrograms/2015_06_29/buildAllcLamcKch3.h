///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamcKch:                                                     //
//                                                                       //
//  This will build everything I need in my analysis and correctly       //
//  combine multiple files when necessary                                //
//    Things this program will build                                     //
//      --KStar correlation functions                                    //
//      --Average separation correlation functions                       // 
//      --Purity results                                                 //  
//      --Event multiplicities/centralities                              // 
///////////////////////////////////////////////////////////////////////////

#ifndef BUILDALLCLAMCKCH3_H
#define BUILDALLCLAMCKCH3_H

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

class buildAllcLamcKch3 {

public:

  buildAllcLamcKch3(vector<TString> &aVectorOfFileNames, TString aDirNameLamKchP, TString aDirNameLamKchM, TString aDirNameALamKchP, TString aDirNameALamKchM);
  virtual ~buildAllcLamcKch3();

  TH1F* buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int aMinNormBin, int aMaxNormBin);
  TH1F* CombineCFs(TString aReturnName, TString aReturnTitle, TObjArray* aCfCollection, TObjArray* aNumCollection, int aMinNormBin, int aMaxNormBin);

  void SetVectorOfFileNames(vector<TString> &aVectorOfNames);
  TObjArray* ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName);
  void SetAnalysisDirectories();
  TObjArray* GetAnalysisDirectory(TString aFile, TString aDirectoryName);  //aFile needs only contain, for example Bp1, Bp2, ...
                                                                           //aDirectoryName needs equal EXACTLY LamKchP, LamKchM, ALamKchP or ALamKchM
  TH1F* GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName);
  TObjArray* LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName);
  TObjArray* BuildCollectionOfCfs(TString aContainedHistosBaseName, TObjArray* aNumCollection, TObjArray* aDenCollection, int aMinNormBin, int aMaxNormBin);

  void BuildCFCollections();
  void DrawFinalCFs(TCanvas *aCanvas);

    //--27 April 2015
  TObjArray* GetCfCollection(TString aType, TString aDirectoryName);  //aType should be either Num, Den, or Cf
								      //aDirectoryName needs equal EXACTLY LamKchP, LamKchM, ALamKchP or ALamKchM
    //--------------

  void BuildAvgSepCollections();
  void DrawFinalAvgSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM);

  void SetPurityRegimes(TH1F* aLambdaPurity);
  TObjArray* CalculatePurity(const char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2]);
  TH1F* CombineCollectionOfHistograms(TString aReturnHistoName, TObjArray* aCollectionOfHistograms);  //this is used to straight forward add histograms, SHOULD NOT BE USED FOR CFs!!!!!!
  void BuildPurityCollections();
  void DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg);
  void DrawFinalPurity(TCanvas *aCanvas);


    //----29 April 2015
  void SaveAll(TFile* aFile); //option should be, for example, RECREATE, NEW, CREATE, UPDATE, etc.


  //-----7 May 2015
  //Separation Histograms
  TH2F* Get2DHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName);
  TObjArray* LoadCollectionOf2DHistograms(TString aDirectoryName, TString aHistoName);
  TObjArray* BuildSepCfs(int aRebinFactor, TString aContainedHistosBaseName, int aNumerOfXbins, TObjArray* a2DNumCollection, TObjArray* a2DDenCollection, int aMinNormBin, int aMaxNormBin);
  void BuildSepCollections(int aRebinFactor);
  void DrawFinalSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM);
  void DrawAvgOfFinalSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM);


  //Inline Functions------------------------
  void SetDebug(bool aDebug);
  void SetOutputPurityFitInto(bool aOutputPurityFitInfo);

  //General stuff
  TString GetDirNameLamKchP();
  TString GetDirNameALamKchP();

  TObjArray* GetNumCollection_LamKchP();
  TObjArray* GetDenCollection_LamKchP();

 //KStar CF------------------
  void SetMinNormCF(double aMinNormCF);
  void SetMaxNormCF(double aMaxNormCF);
  double GetMinNormCF();
  double GetMaxNormCF();

  void SetMinNormBinCF(int aMinNormBinCF);
  void SetMaxNormBinCF(int aMaxNormBinCF);
  int GetMinNormBinCF();
  int GetMaxNormBinCF();

  TH1F* GetCf_LamKchP_Tot();
  TH1F* GetCf_ALamKchP_Tot();

    //1 April 2015
  TH1F* GetCf_LamKchM_Tot();
  TH1F* GetCf_ALamKchM_Tot();

  //Average Separation CF------------------
  void SetMinNormAvgSepCF(double aMinNormAvgSepCF);
  void SetMaxNormAvgSepCF(double aMaxNormAvgSepCF);
  double GetMinNormAvgSepCF();
  double GetMaxNormAvgSepCF();

  void SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF);
  void SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF);
  int GetMinNormBinAvgSepCF();
  int GetMaxNormBinAvgSepCF();

    //--31 March 2015
  TH1F* GetAvgSepCf_TrackPos_LamKchP_Tot();
  TH1F* GetAvgSepCf_TrackNeg_LamKchM_Tot();
  TH1F* GetAvgSepCf_TrackPos_ALamKchP_Tot();
  TH1F* GetAvgSepCf_TrackNeg_ALamKchM_Tot();

    //--1 April 2015
  TH1F* GetCf_AverageLamKchPM_Tot();
  TH1F* GetCf_AverageALamKchPM_Tot();

  //Purity calculations------------------





private:
  bool fDebug;
  bool fOutputPurityFitInfo;

  //General stuff needed to extract/store the proper files/histograms etc.
  vector<TString> fVectorOfFileNames;
  TString fDirNameLamKchP, fDirNameLamKchM, fDirNameALamKchP, fDirNameALamKchM;

  TObjArray *fDirLamKchPBp1, *fDirLamKchPBp2, *fDirLamKchPBm1, *fDirLamKchPBm2, *fDirLamKchPBm3;
  TObjArray *fDirLamKchMBp1, *fDirLamKchMBp2, *fDirLamKchMBm1, *fDirLamKchMBm2, *fDirLamKchMBm3;
  TObjArray *fDirALamKchPBp1, *fDirALamKchPBp2, *fDirALamKchPBm1, *fDirALamKchPBm2, *fDirALamKchPBm3;
  TObjArray *fDirALamKchMBp1, *fDirALamKchMBp2, *fDirALamKchMBm1, *fDirALamKchMBm2, *fDirALamKchMBm3;

  //KStar CFs-------------------------
  double fMinNormCF, fMaxNormCF;
  int fMinNormBinCF, fMaxNormBinCF;

  TObjArray *fNumCollection_LamKchP, *fDenCollection_LamKchP, *fCfCollection_LamKchP;
  TH1F *fCf_LamKchP_BpTot, *fCf_LamKchP_BmTot, *fCf_LamKchP_Tot;

  TObjArray *fNumCollection_LamKchM, *fDenCollection_LamKchM, *fCfCollection_LamKchM;
  TH1F *fCf_LamKchM_BpTot, *fCf_LamKchM_BmTot, *fCf_LamKchM_Tot;

  TObjArray *fNumCollection_ALamKchP, *fDenCollection_ALamKchP, *fCfCollection_ALamKchP;
  TH1F *fCf_ALamKchP_BpTot, *fCf_ALamKchP_BmTot, *fCf_ALamKchP_Tot;

  TObjArray *fNumCollection_ALamKchM, *fDenCollection_ALamKchM, *fCfCollection_ALamKchM;
  TH1F *fCf_ALamKchM_BpTot, *fCf_ALamKchM_BmTot, *fCf_ALamKchM_Tot;

    //---1 April 2015
  TH1F *fCf_AverageLamKchPM_Tot, *fCf_AverageALamKchPM_Tot;


  //Average Separation CF------------------
  double fMinNormAvgSepCF, fMaxNormAvgSepCF;
  int fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF;
  //_____ Track+ ___________________
  TObjArray *fAvgSepNumCollection_TrackPos_LamKchP, *fAvgSepDenCollection_TrackPos_LamKchP, *fAvgSepCfCollection_TrackPos_LamKchP;
  TH1F *fAvgSepCf_TrackPos_LamKchP_BpTot, *fAvgSepCf_TrackPos_LamKchP_BmTot, *fAvgSepCf_TrackPos_LamKchP_Tot;

  TObjArray *fAvgSepNumCollection_TrackPos_LamKchM, *fAvgSepDenCollection_TrackPos_LamKchM, *fAvgSepCfCollection_TrackPos_LamKchM;
  TH1F *fAvgSepCf_TrackPos_LamKchM_BpTot, *fAvgSepCf_TrackPos_LamKchM_BmTot, *fAvgSepCf_TrackPos_LamKchM_Tot;

  TObjArray *fAvgSepNumCollection_TrackPos_ALamKchP, *fAvgSepDenCollection_TrackPos_ALamKchP, *fAvgSepCfCollection_TrackPos_ALamKchP;
  TH1F *fAvgSepCf_TrackPos_ALamKchP_BpTot, *fAvgSepCf_TrackPos_ALamKchP_BmTot, *fAvgSepCf_TrackPos_ALamKchP_Tot;

  TObjArray *fAvgSepNumCollection_TrackPos_ALamKchM, *fAvgSepDenCollection_TrackPos_ALamKchM, *fAvgSepCfCollection_TrackPos_ALamKchM;
  TH1F *fAvgSepCf_TrackPos_ALamKchM_BpTot, *fAvgSepCf_TrackPos_ALamKchM_BmTot, *fAvgSepCf_TrackPos_ALamKchM_Tot;
  //_____ Track- ___________________
  TObjArray *fAvgSepNumCollection_TrackNeg_LamKchP, *fAvgSepDenCollection_TrackNeg_LamKchP, *fAvgSepCfCollection_TrackNeg_LamKchP;
  TH1F *fAvgSepCf_TrackNeg_LamKchP_BpTot, *fAvgSepCf_TrackNeg_LamKchP_BmTot, *fAvgSepCf_TrackNeg_LamKchP_Tot;

  TObjArray *fAvgSepNumCollection_TrackNeg_LamKchM, *fAvgSepDenCollection_TrackNeg_LamKchM, *fAvgSepCfCollection_TrackNeg_LamKchM;
  TH1F *fAvgSepCf_TrackNeg_LamKchM_BpTot, *fAvgSepCf_TrackNeg_LamKchM_BmTot, *fAvgSepCf_TrackNeg_LamKchM_Tot;

  TObjArray *fAvgSepNumCollection_TrackNeg_ALamKchP, *fAvgSepDenCollection_TrackNeg_ALamKchP, *fAvgSepCfCollection_TrackNeg_ALamKchP;
  TH1F *fAvgSepCf_TrackNeg_ALamKchP_BpTot, *fAvgSepCf_TrackNeg_ALamKchP_BmTot, *fAvgSepCf_TrackNeg_ALamKchP_Tot;

  TObjArray *fAvgSepNumCollection_TrackNeg_ALamKchM, *fAvgSepDenCollection_TrackNeg_ALamKchM, *fAvgSepCfCollection_TrackNeg_ALamKchM;
  TH1F *fAvgSepCf_TrackNeg_ALamKchM_BpTot, *fAvgSepCf_TrackNeg_ALamKchM_BmTot, *fAvgSepCf_TrackNeg_ALamKchM_Tot;


  //Purity calculations------------------
  double fLamBgFitLow[2];
  double fLamBgFitHigh[2];
  double fLamROI[2];


  TObjArray *fLambdaPurityHistogramCollection_LamKchP, *fLambdaPurityHistogramCollection_LamKchM, *fAntiLambdaPurityHistogramCollection_ALamKchP, *fAntiLambdaPurityHistogramCollection_ALamKchM;
  vector<TObjArray*> fLambdaPurityListCollection_LamKchP, fLambdaPurityListCollection_LamKchM, fAntiLambdaPurityListCollection_ALamKchP, fAntiLambdaPurityListCollection_ALamKchM;
    //The vector contains 5 elements, one for each file (Bp1,Bp2,Bm1,Bm2,Bm3)
    //Each element (TObjArray) of the vector contains 5 elements 
    //  which hold {fitBg, [Bgd, SigpBgd, Sig, and Pur], [ROImin, ROImax], [BgFitLowMin, BgFitLowMax], [BgFitHighMin, BgFitHighMax]}
  TH1F *fLambdaPurityTot_LamKchP, *fLambdaPurityTot_LamKchM, *fAntiLambdaPurityTot_ALamKchP, *fAntiLambdaPurityTot_ALamKchM;
  TObjArray *fLambdaPurityListTot_LamKchP, *fLambdaPurityListTot_LamKchM, *fAntiLambdaPurityListTot_ALamKchP, *fAntiLambdaPurityListTot_ALamKchM;


  //-----7 May 2015
  //Separation Histograms
  int fMinNormBinSepCF, fMaxNormBinSepCF;
  //_____ Track+ ___________________
  TObjArray *f2DSepNumCollection_TrackPos_LamKchP, *f2DSepDenCollection_TrackPos_LamKchP;
  TObjArray *f1DSepCfCollection_TrackPos_LamKchP;

  TObjArray *f2DSepNumCollection_TrackPos_LamKchM, *f2DSepDenCollection_TrackPos_LamKchM;
  TObjArray *f1DSepCfCollection_TrackPos_LamKchM;

  TObjArray *f2DSepNumCollection_TrackPos_ALamKchP, *f2DSepDenCollection_TrackPos_ALamKchP;
  TObjArray *f1DSepCfCollection_TrackPos_ALamKchP;

  TObjArray *f2DSepNumCollection_TrackPos_ALamKchM, *f2DSepDenCollection_TrackPos_ALamKchM;
  TObjArray *f1DSepCfCollection_TrackPos_ALamKchM;

  //_____ Track- ___________________
  TObjArray *f2DSepNumCollection_TrackNeg_LamKchP, *f2DSepDenCollection_TrackNeg_LamKchP;
  TObjArray *f1DSepCfCollection_TrackNeg_LamKchP;

  TObjArray *f2DSepNumCollection_TrackNeg_LamKchM, *f2DSepDenCollection_TrackNeg_LamKchM;
  TObjArray *f1DSepCfCollection_TrackNeg_LamKchM;

  TObjArray *f2DSepNumCollection_TrackNeg_ALamKchP, *f2DSepDenCollection_TrackNeg_ALamKchP;
  TObjArray *f1DSepCfCollection_TrackNeg_ALamKchP;

  TObjArray *f2DSepNumCollection_TrackNeg_ALamKchM, *f2DSepDenCollection_TrackNeg_ALamKchM;
  TObjArray *f1DSepCfCollection_TrackNeg_ALamKchM;



#ifdef __ROOT__
  ClassDef(buildAllcLamcKch3, 1)
#endif
};

inline void buildAllcLamcKch3::SetDebug(bool aDebug) {fDebug = aDebug;}
inline void buildAllcLamcKch3::SetOutputPurityFitInto(bool aOutputPurityFitInfo) {fOutputPurityFitInfo = aOutputPurityFitInfo;}

//General stuff
inline TString buildAllcLamcKch3::GetDirNameLamKchP() {return fDirNameLamKchP;}
inline TString buildAllcLamcKch3::GetDirNameALamKchP() {return fDirNameALamKchP;}

inline TObjArray* buildAllcLamcKch3::GetNumCollection_LamKchP() {return fNumCollection_LamKchP;}
inline TObjArray* buildAllcLamcKch3::GetDenCollection_LamKchP() {return fDenCollection_LamKchP;}

//KStar CF------------------
inline void buildAllcLamcKch3::SetMinNormCF(double aMinNormCF) {fMinNormCF = aMinNormCF;}
inline void buildAllcLamcKch3::SetMaxNormCF(double aMaxNormCF) {fMaxNormCF = aMaxNormCF;}
inline double buildAllcLamcKch3::GetMinNormCF() {return fMinNormCF;}
inline double buildAllcLamcKch3::GetMaxNormCF() {return fMaxNormCF;}

inline void buildAllcLamcKch3::SetMinNormBinCF(int aMinNormBinCF) {fMinNormBinCF = aMinNormBinCF;}
inline void buildAllcLamcKch3::SetMaxNormBinCF(int aMaxNormBinCF) {fMaxNormBinCF = aMaxNormBinCF;}
inline int buildAllcLamcKch3::GetMinNormBinCF() {return fMinNormBinCF;}
inline int buildAllcLamcKch3::GetMaxNormBinCF() {return fMaxNormBinCF;}

inline TH1F* buildAllcLamcKch3::GetCf_LamKchP_Tot() {return fCf_LamKchP_Tot;}
inline TH1F* buildAllcLamcKch3::GetCf_ALamKchP_Tot() {return fCf_ALamKchP_Tot;}

  //--1 April 2015
inline TH1F* buildAllcLamcKch3::GetCf_LamKchM_Tot() {return fCf_LamKchM_Tot;}
inline TH1F* buildAllcLamcKch3::GetCf_ALamKchM_Tot() {return fCf_ALamKchM_Tot;}

//Average Separation CF------------------
inline void buildAllcLamcKch3::SetMinNormAvgSepCF(double aMinNormAvgSepCF) {fMinNormAvgSepCF = aMinNormAvgSepCF;}
inline void buildAllcLamcKch3::SetMaxNormAvgSepCF(double aMaxNormAvgSepCF) {fMaxNormAvgSepCF = aMaxNormAvgSepCF;}
inline double buildAllcLamcKch3::GetMinNormAvgSepCF() {return fMinNormAvgSepCF;}
inline double buildAllcLamcKch3::GetMaxNormAvgSepCF() {return fMaxNormAvgSepCF;}

inline void buildAllcLamcKch3::SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF) {fMinNormBinAvgSepCF = aMinNormBinAvgSepCF;}
inline void buildAllcLamcKch3::SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF) {fMaxNormBinAvgSepCF = aMaxNormBinAvgSepCF;}
inline int buildAllcLamcKch3::GetMinNormBinAvgSepCF() {return fMinNormBinAvgSepCF;}
inline int buildAllcLamcKch3::GetMaxNormBinAvgSepCF() {return fMaxNormBinAvgSepCF;}

    //--31 March 2015
inline TH1F* buildAllcLamcKch3::GetAvgSepCf_TrackPos_LamKchP_Tot() {return fAvgSepCf_TrackPos_LamKchP_Tot;}
inline TH1F* buildAllcLamcKch3::GetAvgSepCf_TrackNeg_LamKchM_Tot() {return fAvgSepCf_TrackNeg_LamKchM_Tot;}
inline TH1F* buildAllcLamcKch3::GetAvgSepCf_TrackPos_ALamKchP_Tot() {return fAvgSepCf_TrackPos_ALamKchP_Tot;}
inline TH1F* buildAllcLamcKch3::GetAvgSepCf_TrackNeg_ALamKchM_Tot() {return fAvgSepCf_TrackNeg_ALamKchM_Tot;}

    //--1 April 2015
inline TH1F* buildAllcLamcKch3::GetCf_AverageLamKchPM_Tot() {return fCf_AverageLamKchPM_Tot;}
inline TH1F* buildAllcLamcKch3::GetCf_AverageALamKchPM_Tot() {return fCf_AverageALamKchPM_Tot;}

//Purity calculations------------------


#endif

