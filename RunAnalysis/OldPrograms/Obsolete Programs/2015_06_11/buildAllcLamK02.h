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

#ifndef BUILDALLCLAMK02_H
#define BUILDALLCLAMK02_H

#include "TH1F.h"
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

#include <vector>
#include <cmath>
using namespace std;


//______________________________________________________________________________________________________________
typedef vector<TH1F*> VecOfHistos;

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

class buildAllcLamK02 {

public:

  buildAllcLamK02(vector<TString> &aVectorOfFileNames, TString aDirNameLamK0, TString aDirNameALamK0);
  virtual ~buildAllcLamK02();

  TH1F* buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int fMinNormBin, int fMaxNormBin);
  TH1F* CombineCFs(TString aReturnName, TString aReturnTitle, vector<TH1F*> &aCfCollection, vector<TH1F*> &aNumCollection, int aMinNormBin, int aMaxNormBin);

  void SetVectorOfFileNames(vector<TString> &aVectorOfNames);
  TObjArray* ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName);
  void SetAnalysisDirectories();
  TObjArray* GetAnalysisDirectory(TString aFile, TString aDirectoryName);  //aFile needs only contain, for example Bp1, Bp2, ...
                                                                           //aDirectoryName needs equal EXACTLY LamK0 or ALamK0
  TH1F* GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName);
  vector<TH1F*> LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName);
  vector<TH1F*> BuildCollectionOfCfs(vector<TH1F*> &aNumCollection, vector<TH1F*> &aDenCollection, int aMinNormBin, int aMaxNormBin);

  void BuildCFCollections();
  void DrawFinalCFs(TCanvas *aCanvas);

  void BuildAvgSepCollections();
  void DrawFinalAvgSepCFs(TCanvas *aCanvasLamK0, TCanvas *aCanvasALamK0);

  void SetPurityRegimes(TH1F* aLambdaPurity, TH1F* aK0Short1Purity);
  TObjArray* CalculatePurity(const char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2]);
  TH1F* CombineCollectionOfHistograms(TString aReturnHistoName, vector<TH1F*> &aCollectionOfHistograms);  //this is used to straight forward add histograms, SHOULD NOT BE USED FOR CFs!!!!!!
  void BuildPurityCollections();
  void DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg);
  void DrawFinalPurity(TCanvas *aCanvas);



  //Inline Functions------------------------
  //General stuff
  TString GetDirNameLamK0();
  TString GetDirNameALamK0();

  vector<TH1F*> GetNumCollection_LamK0();
  vector<TH1F*> GetDenCollection_LamK0();

 //KStar CF------------------
  void SetMinNormBinCF(int aMinNormBinCF);
  void SetMaxNormBinCF(int aMaxNormBinCF);
  int GetMinNormBinCF();
  int GetMaxNormBinCF();

  TH1F* GetCf_LamK0_Tot();
  TH1F* GetCf_ALamK0_Tot();

  //Average Separation CF------------------
  void SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF);
  void SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF);
  int GetMinNormBinAvgSepCF();
  int GetMaxNormBinAvgSepCF();

  //Purity calculations------------------





private:
  //General stuff needed to extract/store the proper files/histograms etc.
  vector<TString> fVectorOfFileNames;
  TString fDirNameLamK0, fDirNameALamK0;

  TObjArray *fDirLamK0Bp1, *fDirLamK0Bp2, *fDirLamK0Bm1, *fDirLamK0Bm2, *fDirLamK0Bm3;
  TObjArray *fDirALamK0Bp1, *fDirALamK0Bp2, *fDirALamK0Bm1, *fDirALamK0Bm2, *fDirALamK0Bm3;

  //KStar CFs-------------------------
  int fMinNormBinCF, fMaxNormBinCF;

  vector<TH1F*> fNumCollection_LamK0, fDenCollection_LamK0, fCfCollection_LamK0;
  TH1F *fCf_LamK0_BpTot, *fCf_LamK0_BmTot, *fCf_LamK0_Tot;

  vector<TH1F*> fNumCollection_ALamK0, fDenCollection_ALamK0, fCfCollection_ALamK0;
  TH1F *fCf_ALamK0_BpTot, *fCf_ALamK0_BmTot, *fCf_ALamK0_Tot;


  //Average Separation CF------------------
  int fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF;
  //_____ ++ ___________________
  vector<TH1F*> fAvgSepNumCollection_PosPos_LamK0, fAvgSepDenCollection_PosPos_LamK0, fAvgSepCfCollection_PosPos_LamK0;
  TH1F *fAvgSepCf_PosPos_LamK0_BpTot, *fAvgSepCf_PosPos_LamK0_BmTot, *fAvgSepCf_PosPos_LamK0_Tot;

  vector<TH1F*> fAvgSepNumCollection_PosPos_ALamK0, fAvgSepDenCollection_PosPos_ALamK0, fAvgSepCfCollection_PosPos_ALamK0;
  TH1F *fAvgSepCf_PosPos_ALamK0_BpTot, *fAvgSepCf_PosPos_ALamK0_BmTot, *fAvgSepCf_PosPos_ALamK0_Tot;
  //_____ +- ___________________
  vector<TH1F*> fAvgSepNumCollection_PosNeg_LamK0, fAvgSepDenCollection_PosNeg_LamK0, fAvgSepCfCollection_PosNeg_LamK0;
  TH1F *fAvgSepCf_PosNeg_LamK0_BpTot, *fAvgSepCf_PosNeg_LamK0_BmTot, *fAvgSepCf_PosNeg_LamK0_Tot;

  vector<TH1F*> fAvgSepNumCollection_PosNeg_ALamK0, fAvgSepDenCollection_PosNeg_ALamK0, fAvgSepCfCollection_PosNeg_ALamK0;
  TH1F *fAvgSepCf_PosNeg_ALamK0_BpTot, *fAvgSepCf_PosNeg_ALamK0_BmTot, *fAvgSepCf_PosNeg_ALamK0_Tot;
  //_____ -+ ___________________
  vector<TH1F*> fAvgSepNumCollection_NegPos_LamK0, fAvgSepDenCollection_NegPos_LamK0, fAvgSepCfCollection_NegPos_LamK0;
  TH1F *fAvgSepCf_NegPos_LamK0_BpTot, *fAvgSepCf_NegPos_LamK0_BmTot, *fAvgSepCf_NegPos_LamK0_Tot;

  vector<TH1F*> fAvgSepNumCollection_NegPos_ALamK0, fAvgSepDenCollection_NegPos_ALamK0, fAvgSepCfCollection_NegPos_ALamK0;
  TH1F *fAvgSepCf_NegPos_ALamK0_BpTot, *fAvgSepCf_NegPos_ALamK0_BmTot, *fAvgSepCf_NegPos_ALamK0_Tot;
  //_____ -- ___________________
  vector<TH1F*> fAvgSepNumCollection_NegNeg_LamK0, fAvgSepDenCollection_NegNeg_LamK0, fAvgSepCfCollection_NegNeg_LamK0;
  TH1F *fAvgSepCf_NegNeg_LamK0_BpTot, *fAvgSepCf_NegNeg_LamK0_BmTot, *fAvgSepCf_NegNeg_LamK0_Tot;

  vector<TH1F*> fAvgSepNumCollection_NegNeg_ALamK0, fAvgSepDenCollection_NegNeg_ALamK0, fAvgSepCfCollection_NegNeg_ALamK0;
  TH1F *fAvgSepCf_NegNeg_ALamK0_BpTot, *fAvgSepCf_NegNeg_ALamK0_BmTot, *fAvgSepCf_NegNeg_ALamK0_Tot;


  //Purity calculations------------------
  double fLamBgFitLow[2];
  double fLamBgFitHigh[2];
  double fLamROI[2];

  double fK0Short1BgFitLow[2];
  double fK0Short1BgFitHigh[2];
  double fK0Short1ROI[2];

  vector<TH1F*> fLambdaPurityHistogramCollection, fK0Short1PurityHistogramCollection, fAntiLambdaPurityHistogramCollection, fK0Short2PurityHistogramCollection;
  vector<TObjArray*> fLambdaPurityListCollection, fK0Short1PurityListCollection, fAntiLambdaPurityListCollection, fK0Short2PurityListCollection;
    //The vector contains 5 elements, one for each file (Bp1,Bp2,Bm1,Bm2,Bm3)
    //Each element (list) of the vector contains 5 elements 
    //  which hold {fitBg, [Bgd, SigpBgd, Sig, and Pur], [ROImin, ROImax], [BgFitLowMin, BgFitLowMax], [BgFitHighMin, BgFitHighMax]}
  TH1F *fLambdaPurityTot, *fK0Short1PurityTot, *fAntiLambdaPurityTot, *fK0Short2PurityTot;
  TObjArray *fLambdaPurityListTot, *fK0Short1PurityListTot, *fAntiLambdaPurityListTot, *fK0Short2PurityListTot;






#ifdef __ROOT__
  ClassDef(buildAllcLamK02, 1)
#endif
};

//General stuff
inline TString buildAllcLamK02::GetDirNameLamK0() {return fDirNameLamK0;}
inline TString buildAllcLamK02::GetDirNameALamK0() {return fDirNameALamK0;}

inline vector<TH1F*> buildAllcLamK02::GetNumCollection_LamK0() {return fNumCollection_LamK0;}
inline vector<TH1F*> buildAllcLamK02::GetDenCollection_LamK0() {return fDenCollection_LamK0;}

//KStar CF------------------
inline void buildAllcLamK02::SetMinNormBinCF(int aMinNormBinCF) {fMinNormBinCF = aMinNormBinCF;}
inline void buildAllcLamK02::SetMaxNormBinCF(int aMaxNormBinCF){fMaxNormBinCF = aMaxNormBinCF;}
inline int buildAllcLamK02::GetMinNormBinCF() {return fMinNormBinCF;}
inline int buildAllcLamK02::GetMaxNormBinCF() {return fMaxNormBinCF;}

inline TH1F* buildAllcLamK02::GetCf_LamK0_Tot() {return fCf_LamK0_Tot;}
inline TH1F* buildAllcLamK02::GetCf_ALamK0_Tot() {return fCf_ALamK0_Tot;}

//Average Separation CF------------------
inline void buildAllcLamK02::SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF) {fMinNormBinAvgSepCF = aMinNormBinAvgSepCF;}
inline void buildAllcLamK02::SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF){fMaxNormBinAvgSepCF = aMaxNormBinAvgSepCF;}
inline int buildAllcLamK02::GetMinNormBinAvgSepCF() {return fMinNormBinAvgSepCF;}
inline int buildAllcLamK02::GetMaxNormBinAvgSepCF() {return fMaxNormBinAvgSepCF;}

//Purity calculations------------------


#endif

