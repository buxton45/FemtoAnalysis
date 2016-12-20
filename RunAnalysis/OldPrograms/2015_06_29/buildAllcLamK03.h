///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamK0:                                                      //
//                                                                       //
//  This will build everything I need in my analysis and correctly       //
//  combine multiple files when necessary                                //
//    Things this program will build                                     //
//      --KStar correlation functions                                    //
//      --Average separation correlation functions                       // 
//      --Purity results                                                 //  
//      --Event multiplicities/centralities                              // 
///////////////////////////////////////////////////////////////////////////

#ifndef BUILDALLCLAMK03_H
#define BUILDALLCLAMK03_H

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

class buildAllcLamK03 {

public:

  buildAllcLamK03(vector<TString> &aVectorOfFileNames, TString aDirNameLamK0, TString aDirNameALamK0);
  virtual ~buildAllcLamK03();

  TH1F* buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int aMinNormBin, int aMaxNormBin);
  TH1F* CombineCFs(TString aReturnName, TString aReturnTitle, TObjArray* aCfCollection, TObjArray* aNumCollection, int aMinNormBin, int aMaxNormBin);

  void SetVectorOfFileNames(vector<TString> &aVectorOfNames);
  TObjArray* ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName);
  void SetAnalysisDirectories();
  TObjArray* GetAnalysisDirectory(TString aFile, TString aDirectoryName);  //aFile needs only contain, for example Bp1, Bp2, ...
                                                                           //aDirectoryName needs equal EXACTLY LamK0 or ALamK0
  TH1F* GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName);
  TObjArray* LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName);
  TObjArray* BuildCollectionOfCfs(TString aContainedHistosBaseName, TObjArray* aNumCollection, TObjArray* aDenCollection, int aMinNormBin, int aMaxNormBin);

  void BuildCFCollections();
  void DrawFinalCFs(TCanvas *aCanvas);

    //--27 April 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK03)
  TObjArray* GetCfCollection(TString aType, TString aDirectoryName);  //aType should be either Num, Den, or Cf
								      //aDirectoryName needs equal EXACTLY LamK0 or ALamK0
    //--------------

  void BuildAvgSepCollections();
  void DrawFinalAvgSepCFs(TCanvas *aCanvasLamK0, TCanvas *aCanvasALamK0);

  void SetPurityRegimes(TH1F* aLambdaPurity, TH1F* aK0Short1Purity);
  TObjArray* CalculatePurity(const char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2]);
  TH1F* CombineCollectionOfHistograms(TString aReturnHistoName, TObjArray* aCollectionOfHistograms);  //this is used to straight forward add histograms, SHOULD NOT BE USED FOR CFs!!!!!!
  void BuildPurityCollections();
  void DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg);
  void DrawFinalPurity(TCanvas *aCanvas);


    //----29 April 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK03)
  void SaveAll(TFile* aFile); //option should be, for example, RECREATE, NEW, CREATE, UPDATE, etc.


  //-----7 May 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK03)
  //Separation Histograms
  TH2F* Get2DHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName);
  TObjArray* LoadCollectionOf2DHistograms(TString aDirectoryName, TString aHistoName);
  TObjArray* BuildSepCfs(int aRebinFactor, TString aContainedHistosBaseName, int aNumerOfXbins, TObjArray* a2DNumCollection, TObjArray* a2DDenCollection, int aMinNormBin, int aMaxNormBin);
  void BuildSepCollections(int aRebinFactor);
  void DrawFinalSepCFs(TCanvas *aCanvasLamK0LikeSigns, TCanvas *aCanvasLamK0UnlikeSigns, TCanvas *aCanvasALamK0LikeSigns, TCanvas *aCanvasALamK0UnlikeSigns);
  void DrawAvgOfFinalSepCFs(TCanvas *aCanvasLamK0LikeSigns, TCanvas *aCanvasLamK0UnlikeSigns, TCanvas *aCanvasALamK0LikeSigns, TCanvas *aCanvasALamK0UnlikeSigns);


  //Inline Functions------------------------
  void SetDebug(bool aDebug);
  void SetOutputPurityFitInto(bool aOutputPurityFitInfo);

  //General stuff
  TString GetDirNameLamK0();
  TString GetDirNameALamK0();

  TObjArray* GetNumCollection_LamK0();
  TObjArray* GetDenCollection_LamK0();

 //KStar CF------------------
  void SetMinNormCF(double aMinNormCF);
  void SetMaxNormCF(double aMaxNormCF);
  double GetMinNormCF();
  double GetMaxNormCF();

  void SetMinNormBinCF(int aMinNormBinCF);
  void SetMaxNormBinCF(int aMaxNormBinCF);
  int GetMinNormBinCF();
  int GetMaxNormBinCF();

  TH1F* GetCf_LamK0_Tot();
  TH1F* GetCf_ALamK0_Tot();

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
  TH1F* GetAvgSepCf_PosPos_LamK0_Tot();
  TH1F* GetAvgSepCf_NegNeg_LamK0_Tot();
  TH1F* GetAvgSepCf_PosPos_ALamK0_Tot();
  TH1F* GetAvgSepCf_NegNeg_ALamK0_Tot();

  //Purity calculations------------------





private:
  bool fDebug;
  bool fOutputPurityFitInfo;

  //General stuff needed to extract/store the proper files/histograms etc.
  vector<TString> fVectorOfFileNames;
  TString fDirNameLamK0, fDirNameALamK0;

  TObjArray *fDirLamK0Bp1, *fDirLamK0Bp2, *fDirLamK0Bm1, *fDirLamK0Bm2, *fDirLamK0Bm3;
  TObjArray *fDirALamK0Bp1, *fDirALamK0Bp2, *fDirALamK0Bm1, *fDirALamK0Bm2, *fDirALamK0Bm3;

  //KStar CFs-------------------------
  double fMinNormCF, fMaxNormCF;
  int fMinNormBinCF, fMaxNormBinCF;

  TObjArray *fNumCollection_LamK0, *fDenCollection_LamK0, *fCfCollection_LamK0;
  TH1F *fCf_LamK0_BpTot, *fCf_LamK0_BmTot, *fCf_LamK0_Tot;

  TObjArray *fNumCollection_ALamK0, *fDenCollection_ALamK0, *fCfCollection_ALamK0;
  TH1F *fCf_ALamK0_BpTot, *fCf_ALamK0_BmTot, *fCf_ALamK0_Tot;


  //Average Separation CF------------------
  double fMinNormAvgSepCF, fMaxNormAvgSepCF;
  int fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF;
  //_____ ++ ___________________
  TObjArray *fAvgSepNumCollection_PosPos_LamK0, *fAvgSepDenCollection_PosPos_LamK0, *fAvgSepCfCollection_PosPos_LamK0;
  TH1F *fAvgSepCf_PosPos_LamK0_BpTot, *fAvgSepCf_PosPos_LamK0_BmTot, *fAvgSepCf_PosPos_LamK0_Tot;

  TObjArray *fAvgSepNumCollection_PosPos_ALamK0, *fAvgSepDenCollection_PosPos_ALamK0, *fAvgSepCfCollection_PosPos_ALamK0;
  TH1F *fAvgSepCf_PosPos_ALamK0_BpTot, *fAvgSepCf_PosPos_ALamK0_BmTot, *fAvgSepCf_PosPos_ALamK0_Tot;
  //_____ +- ___________________
  TObjArray *fAvgSepNumCollection_PosNeg_LamK0, *fAvgSepDenCollection_PosNeg_LamK0, *fAvgSepCfCollection_PosNeg_LamK0;
  TH1F *fAvgSepCf_PosNeg_LamK0_BpTot, *fAvgSepCf_PosNeg_LamK0_BmTot, *fAvgSepCf_PosNeg_LamK0_Tot;

  TObjArray *fAvgSepNumCollection_PosNeg_ALamK0, *fAvgSepDenCollection_PosNeg_ALamK0, *fAvgSepCfCollection_PosNeg_ALamK0;
  TH1F *fAvgSepCf_PosNeg_ALamK0_BpTot, *fAvgSepCf_PosNeg_ALamK0_BmTot, *fAvgSepCf_PosNeg_ALamK0_Tot;
  //_____ -+ ___________________
  TObjArray *fAvgSepNumCollection_NegPos_LamK0, *fAvgSepDenCollection_NegPos_LamK0, *fAvgSepCfCollection_NegPos_LamK0;
  TH1F *fAvgSepCf_NegPos_LamK0_BpTot, *fAvgSepCf_NegPos_LamK0_BmTot, *fAvgSepCf_NegPos_LamK0_Tot;

  TObjArray *fAvgSepNumCollection_NegPos_ALamK0, *fAvgSepDenCollection_NegPos_ALamK0, *fAvgSepCfCollection_NegPos_ALamK0;
  TH1F *fAvgSepCf_NegPos_ALamK0_BpTot, *fAvgSepCf_NegPos_ALamK0_BmTot, *fAvgSepCf_NegPos_ALamK0_Tot;
  //_____ -- ___________________
  TObjArray *fAvgSepNumCollection_NegNeg_LamK0, *fAvgSepDenCollection_NegNeg_LamK0, *fAvgSepCfCollection_NegNeg_LamK0;
  TH1F *fAvgSepCf_NegNeg_LamK0_BpTot, *fAvgSepCf_NegNeg_LamK0_BmTot, *fAvgSepCf_NegNeg_LamK0_Tot;

  TObjArray *fAvgSepNumCollection_NegNeg_ALamK0, *fAvgSepDenCollection_NegNeg_ALamK0, *fAvgSepCfCollection_NegNeg_ALamK0;
  TH1F *fAvgSepCf_NegNeg_ALamK0_BpTot, *fAvgSepCf_NegNeg_ALamK0_BmTot, *fAvgSepCf_NegNeg_ALamK0_Tot;


  //Purity calculations------------------
  double fLamBgFitLow[2];
  double fLamBgFitHigh[2];
  double fLamROI[2];

  double fK0Short1BgFitLow[2];
  double fK0Short1BgFitHigh[2];
  double fK0Short1ROI[2];

  TObjArray *fLambdaPurityHistogramCollection, *fK0Short1PurityHistogramCollection, *fAntiLambdaPurityHistogramCollection, *fK0Short2PurityHistogramCollection;
  vector<TObjArray*> fLambdaPurityListCollection, fK0Short1PurityListCollection, fAntiLambdaPurityListCollection, fK0Short2PurityListCollection;
    //The vector contains 5 elements, one for each file (Bp1,Bp2,Bm1,Bm2,Bm3)
    //Each element (TObjArray) of the vector contains 5 elements 
    //  which hold {fitBg, [Bgd, SigpBgd, Sig, and Pur], [ROImin, ROImax], [BgFitLowMin, BgFitLowMax], [BgFitHighMin, BgFitHighMax]}
  TH1F *fLambdaPurityTot, *fK0Short1PurityTot, *fAntiLambdaPurityTot, *fK0Short2PurityTot;
  TObjArray *fLambdaPurityListTot, *fK0Short1PurityListTot, *fAntiLambdaPurityListTot, *fK0Short2PurityListTot;


  //-----7 May 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK03)
  //Separation Histograms
  int fMinNormBinSepCF, fMaxNormBinSepCF;
  //_____ PosPos ___________________
  TObjArray *f2DSepNumCollection_PosPos_LamK0, *f2DSepDenCollection_PosPos_LamK0;
  TObjArray *f1DSepCfCollection_PosPos_LamK0;

  TObjArray *f2DSepNumCollection_PosPos_ALamK0, *f2DSepDenCollection_PosPos_ALamK0;
  TObjArray *f1DSepCfCollection_PosPos_ALamK0;

  //_____ PosNeg ___________________
  TObjArray *f2DSepNumCollection_PosNeg_LamK0, *f2DSepDenCollection_PosNeg_LamK0;
  TObjArray *f1DSepCfCollection_PosNeg_LamK0;

  TObjArray *f2DSepNumCollection_PosNeg_ALamK0, *f2DSepDenCollection_PosNeg_ALamK0;
  TObjArray *f1DSepCfCollection_PosNeg_ALamK0;

  //_____ NegPos ___________________
  TObjArray *f2DSepNumCollection_NegPos_LamK0, *f2DSepDenCollection_NegPos_LamK0;
  TObjArray *f1DSepCfCollection_NegPos_LamK0;

  TObjArray *f2DSepNumCollection_NegPos_ALamK0, *f2DSepDenCollection_NegPos_ALamK0;
  TObjArray *f1DSepCfCollection_NegPos_ALamK0;

  //_____ NegNeg ___________________
  TObjArray *f2DSepNumCollection_NegNeg_LamK0, *f2DSepDenCollection_NegNeg_LamK0;
  TObjArray *f1DSepCfCollection_NegNeg_LamK0;

  TObjArray *f2DSepNumCollection_NegNeg_ALamK0, *f2DSepDenCollection_NegNeg_ALamK0;
  TObjArray *f1DSepCfCollection_NegNeg_ALamK0;



#ifdef __ROOT__
  ClassDef(buildAllcLamK03, 1)
#endif
};

inline void buildAllcLamK03::SetDebug(bool aDebug) {fDebug = aDebug;}
inline void buildAllcLamK03::SetOutputPurityFitInto(bool aOutputPurityFitInfo) {fOutputPurityFitInfo = aOutputPurityFitInfo;}

//General stuff
inline TString buildAllcLamK03::GetDirNameLamK0() {return fDirNameLamK0;}
inline TString buildAllcLamK03::GetDirNameALamK0() {return fDirNameALamK0;}

inline TObjArray* buildAllcLamK03::GetNumCollection_LamK0() {return fNumCollection_LamK0;}
inline TObjArray* buildAllcLamK03::GetDenCollection_LamK0() {return fDenCollection_LamK0;}

//KStar CF------------------
inline void buildAllcLamK03::SetMinNormCF(double aMinNormCF) {fMinNormCF = aMinNormCF;}
inline void buildAllcLamK03::SetMaxNormCF(double aMaxNormCF) {fMaxNormCF = aMaxNormCF;}
inline double buildAllcLamK03::GetMinNormCF() {return fMinNormCF;}
inline double buildAllcLamK03::GetMaxNormCF() {return fMaxNormCF;}

inline void buildAllcLamK03::SetMinNormBinCF(int aMinNormBinCF) {fMinNormBinCF = aMinNormBinCF;}
inline void buildAllcLamK03::SetMaxNormBinCF(int aMaxNormBinCF) {fMaxNormBinCF = aMaxNormBinCF;}
inline int buildAllcLamK03::GetMinNormBinCF() {return fMinNormBinCF;}
inline int buildAllcLamK03::GetMaxNormBinCF() {return fMaxNormBinCF;}

inline TH1F* buildAllcLamK03::GetCf_LamK0_Tot() {return fCf_LamK0_Tot;}
inline TH1F* buildAllcLamK03::GetCf_ALamK0_Tot() {return fCf_ALamK0_Tot;}

//Average Separation CF------------------
inline void buildAllcLamK03::SetMinNormAvgSepCF(double aMinNormAvgSepCF) {fMinNormAvgSepCF = aMinNormAvgSepCF;}
inline void buildAllcLamK03::SetMaxNormAvgSepCF(double aMaxNormAvgSepCF) {fMaxNormAvgSepCF = aMaxNormAvgSepCF;}
inline double buildAllcLamK03::GetMinNormAvgSepCF() {return fMinNormAvgSepCF;}
inline double buildAllcLamK03::GetMaxNormAvgSepCF() {return fMaxNormAvgSepCF;}

inline void buildAllcLamK03::SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF) {fMinNormBinAvgSepCF = aMinNormBinAvgSepCF;}
inline void buildAllcLamK03::SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF) {fMaxNormBinAvgSepCF = aMaxNormBinAvgSepCF;}
inline int buildAllcLamK03::GetMinNormBinAvgSepCF() {return fMinNormBinAvgSepCF;}
inline int buildAllcLamK03::GetMaxNormBinAvgSepCF() {return fMaxNormBinAvgSepCF;}

    //--31 March 2015
inline TH1F* buildAllcLamK03::GetAvgSepCf_PosPos_LamK0_Tot() {return fAvgSepCf_PosPos_LamK0_Tot;}
inline TH1F* buildAllcLamK03::GetAvgSepCf_NegNeg_LamK0_Tot() {return fAvgSepCf_NegNeg_LamK0_Tot;}
inline TH1F* buildAllcLamK03::GetAvgSepCf_PosPos_ALamK0_Tot() {return fAvgSepCf_PosPos_ALamK0_Tot;}
inline TH1F* buildAllcLamK03::GetAvgSepCf_NegNeg_ALamK0_Tot() {return fAvgSepCf_NegNeg_ALamK0_Tot;}

//Purity calculations------------------


#endif

