///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAllcLamcKch2:                                                      //
//                                                                       //
//  This will build everything I need in my analysis and correctly       //
//  combine multiple files when necessary                                //
//    Things this program will build                                     //
//      --KStar correlation functions                                    //
//      --Average separation correlation functions                       // 
//      --Purity results                                                 //  
//      --Event multiplicities/centralities                              // 
///////////////////////////////////////////////////////////////////////////

#ifndef BUILDALLCLAMCKCH2_H
#define BUILDALLCLAMCKCH2_H

#include "TH1F.h"
#include "TString.h"
#include "TList.h"
#include "TF1.h"
#include "TVectorD.h"
#include "TLine.h"
#include "TCanvas.h"

#include <vector>


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

class buildAllcLamcKch2 {

public:

  buildAllcLamcKch2(vector<TString> &aVectorOfFileNames, TString aDirNameLamKchP, TString aDirNameLamKchM, TString aDirNameALamKchP, TString aDirNameALamKchM);
  virtual ~buildAllcLamcKch2();

  TH1F* buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int fMinNormBin, int fMaxNormBin);
  TH1F* CombineCFs(TString aReturnName, TString aReturnTitle, vector<TH1F*> &aCfCollection, vector<TH1F*> &aNumCollection, int aMinNormBin, int aMaxNormBin);

  void SetVectorOfFileNames(vector<TString> &aVectorOfNames);
  TObjArray* ConnectAnalysisDirectory(TString aFileName, TString aDirectoryName);
  void SetAnalysisDirectories();
  TObjArray* GetAnalysisDirectory(TString aFile, TString aDirectoryName);  //aFile needs only contain, for example Bp1, Bp2, ...
                                                                           //aDirectoryName needs equal EXACTLY LamKchP, LamKchM, ALamKchP or ALamKchM
  TH1F* GetHistoClone(TObjArray* aAnalysisDirectory, TString aHistoName, TString aCloneHistoName);
  vector<TH1F*> LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName);
  vector<TH1F*> BuildCollectionOfCfs(vector<TH1F*> &aNumCollection, vector<TH1F*> &aDenCollection, int aMinNormBin, int aMaxNormBin);

  void BuildCFCollections();
  void DrawFinalCFs(TCanvas *aCanvas);

  void BuildAvgSepCollections();
  void DrawFinalAvgSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM);

  void SetPurityRegimes(TH1F* aLambdaPurity);
  TObjArray* CalculatePurity(char* aReturnFitName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2]);
  TH1F* CombineCollectionOfHistograms(TString aReturnHistoName, vector<TH1F*> &aCollectionOfHistograms);  //this is used to straight forward add histograms, SHOULD NOT BE USED FOR CFs!!!!!!
  void BuildPurityCollections();
  void DrawPurity(TH1F* aPurityHisto, TObjArray* aFitList, bool ZoomBg);
  void DrawFinalPurity(TCanvas *aCanvas);




  //Inline Functions------------------------
  //General stuff
  TObjArray* GetDirLamKchP();
  TString GetDirNameLamKchM();
  TString GetDirNameALamKchP();
  TString GetDirNameALamKchM();

  vector<TH1F*> GetNumCollection_LamKchP();
  vector<TH1F*> GetDenCollection_LamKchP();

  //KStar CF------------------
  void SetMinNormBinCF(int aMinNormBinCF);
  void SetMaxNormBinCF(int aMaxNormBinCF);
  int GetMinNormBinCF();
  int GetMaxNormBinCF();

  TH1F* GetCf_LamKchP_Tot();
  TH1F* GetCf_ALamKchP_Tot();

  //Average Separation CF------------------
  void SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF);
  void SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF);
  int GetMinNormBinAvgSepCF();
  int GetMaxNormBinAvgSepCF();

  //Purity calculations------------------







private:
  //General stuff needed to extract/store the proper files/histograms etc.
  vector<TString> fVectorOfFileNames;
  TString fDirNameLamKchP, fDirNameLamKchM, fDirNameALamKchP, fDirNameALamKchM;

  TObjArray *fDirLamKchPBp1, *fDirLamKchPBp2, *fDirLamKchPBm1, *fDirLamKchPBm2, *fDirLamKchPBm3;
  TObjArray *fDirLamKchMBp1, *fDirLamKchMBp2, *fDirLamKchMBm1, *fDirLamKchMBm2, *fDirLamKchMBm3;
  TObjArray *fDirALamKchPBp1, *fDirALamKchPBp2, *fDirALamKchPBm1, *fDirALamKchPBm2, *fDirALamKchPBm3;
  TObjArray *fDirALamKchMBp1, *fDirALamKchMBp2, *fDirALamKchMBm1, *fDirALamKchMBm2, *fDirALamKchMBm3;

  //KStar CFs-------------------------
  int fMinNormBinCF, fMaxNormBinCF;

  vector<TH1F*> fNumCollection_LamKchP, fDenCollection_LamKchP, fCfCollection_LamKchP;
  TH1F *fCf_LamKchP_BpTot, *fCf_LamKchP_BmTot, *fCf_LamKchP_Tot;

  vector<TH1F*> fNumCollection_LamKchM, fDenCollection_LamKchM, fCfCollection_LamKchM;
  TH1F *fCf_LamKchM_BpTot, *fCf_LamKchM_BmTot, *fCf_LamKchM_Tot;

  vector<TH1F*> fNumCollection_ALamKchP, fDenCollection_ALamKchP, fCfCollection_ALamKchP;
  TH1F *fCf_ALamKchP_BpTot, *fCf_ALamKchP_BmTot, *fCf_ALamKchP_Tot;

  vector<TH1F*> fNumCollection_ALamKchM, fDenCollection_ALamKchM, fCfCollection_ALamKchM;
  TH1F *fCf_ALamKchM_BpTot, *fCf_ALamKchM_BmTot, *fCf_ALamKchM_Tot;


  //Average Separation CF------------------
  int fMinNormBinAvgSepCF, fMaxNormBinAvgSepCF;
  //_____ Track+ ___________________
  vector<TH1F*> fAvgSepNumCollection_TrackPos_LamKchP, fAvgSepDenCollection_TrackPos_LamKchP, fAvgSepCfCollection_TrackPos_LamKchP;
  TH1F *fAvgSepCf_TrackPos_LamKchP_BpTot, *fAvgSepCf_TrackPos_LamKchP_BmTot, *fAvgSepCf_TrackPos_LamKchP_Tot;

  vector<TH1F*> fAvgSepNumCollection_TrackPos_LamKchM, fAvgSepDenCollection_TrackPos_LamKchM, fAvgSepCfCollection_TrackPos_LamKchM;
  TH1F *fAvgSepCf_TrackPos_LamKchM_BpTot, *fAvgSepCf_TrackPos_LamKchM_BmTot, *fAvgSepCf_TrackPos_LamKchM_Tot;

  vector<TH1F*> fAvgSepNumCollection_TrackPos_ALamKchP, fAvgSepDenCollection_TrackPos_ALamKchP, fAvgSepCfCollection_TrackPos_ALamKchP;
  TH1F *fAvgSepCf_TrackPos_ALamKchP_BpTot, *fAvgSepCf_TrackPos_ALamKchP_BmTot, *fAvgSepCf_TrackPos_ALamKchP_Tot;

  vector<TH1F*> fAvgSepNumCollection_TrackPos_ALamKchM, fAvgSepDenCollection_TrackPos_ALamKchM, fAvgSepCfCollection_TrackPos_ALamKchM;
  TH1F *fAvgSepCf_TrackPos_ALamKchM_BpTot, *fAvgSepCf_TrackPos_ALamKchM_BmTot, *fAvgSepCf_TrackPos_ALamKchM_Tot;
  //_____ Track- ___________________
  vector<TH1F*> fAvgSepNumCollection_TrackNeg_LamKchP, fAvgSepDenCollection_TrackNeg_LamKchP, fAvgSepCfCollection_TrackNeg_LamKchP;
  TH1F *fAvgSepCf_TrackNeg_LamKchP_BpTot, *fAvgSepCf_TrackNeg_LamKchP_BmTot, *fAvgSepCf_TrackNeg_LamKchP_Tot;

  vector<TH1F*> fAvgSepNumCollection_TrackNeg_LamKchM, fAvgSepDenCollection_TrackNeg_LamKchM, fAvgSepCfCollection_TrackNeg_LamKchM;
  TH1F *fAvgSepCf_TrackNeg_LamKchM_BpTot, *fAvgSepCf_TrackNeg_LamKchM_BmTot, *fAvgSepCf_TrackNeg_LamKchM_Tot;

  vector<TH1F*> fAvgSepNumCollection_TrackNeg_ALamKchP, fAvgSepDenCollection_TrackNeg_ALamKchP, fAvgSepCfCollection_TrackNeg_ALamKchP;
  TH1F *fAvgSepCf_TrackNeg_ALamKchP_BpTot, *fAvgSepCf_TrackNeg_ALamKchP_BmTot, *fAvgSepCf_TrackNeg_ALamKchP_Tot;

  vector<TH1F*> fAvgSepNumCollection_TrackNeg_ALamKchM, fAvgSepDenCollection_TrackNeg_ALamKchM, fAvgSepCfCollection_TrackNeg_ALamKchM;
  TH1F *fAvgSepCf_TrackNeg_ALamKchM_BpTot, *fAvgSepCf_TrackNeg_ALamKchM_BmTot, *fAvgSepCf_TrackNeg_ALamKchM_Tot;



  //Purity calculations------------------
  double fLamBgFitLow[2];
  double fLamBgFitHigh[2];
  double fLamROI[2];

  vector<TH1F*> fLambdaPurityHistogramCollection_LamKchP, fLambdaPurityHistogramCollection_LamKchM, fAntiLambdaPurityHistogramCollection_ALamKchP, fAntiLambdaPurityHistogramCollection_ALamKchM;
  vector<TObjArray*> fLambdaPurityListCollection_LamKchP, fLambdaPurityListCollection_LamKchM, fAntiLambdaPurityListCollection_ALamKchP, fAntiLambdaPurityListCollection_ALamKchM;
    //The vector contains 5 elements, one for each file (Bp1,Bp2,Bm1,Bm2,Bm3)
    //Each element (list) of the vector contains 5 elements 
    //  which hold {fitBg, [Bgd, SigpBgd, Sig, and Pur], [ROImin, ROImax], [BgFitLowMin, BgFitLowMax], [BgFitHighMin, BgFitHighMax]}
  TH1F *fLambdaPurityTot_LamKchP, *fLambdaPurityTot_LamKchM, *fAntiLambdaPurityTot_ALamKchP, *fAntiLambdaPurityTot_ALamKchM;
  TObjArray *fLambdaPurityListTot_LamKchP, *fLambdaPurityListTot_LamKchM, *fAntiLambdaPurityListTot_ALamKchP, *fAntiLambdaPurityListTot_ALamKchM;






#ifdef __ROOT__
  ClassDef(buildAllcLamcKch2, 1)
#endif
};

//General stuff
inline TObjArray* buildAllcLamcKch2::GetDirLamKchP() {return fDirLamKchPBp1;}
inline TString buildAllcLamcKch2::GetDirNameLamKchM() {return fDirNameLamKchM;}
inline TString buildAllcLamcKch2::GetDirNameALamKchP() {return fDirNameALamKchP;}
inline TString buildAllcLamcKch2::GetDirNameALamKchM() {return fDirNameALamKchM;}

inline vector<TH1F*> buildAllcLamcKch2::GetNumCollection_LamKchP() {return fNumCollection_LamKchP;}
inline vector<TH1F*> buildAllcLamcKch2::GetDenCollection_LamKchP() {return fDenCollection_LamKchP;}

//KStar CF------------------
inline void buildAllcLamcKch2::SetMinNormBinCF(int aMinNormBinCF) {fMinNormBinCF = aMinNormBinCF;}
inline void buildAllcLamcKch2::SetMaxNormBinCF(int aMaxNormBinCF){fMaxNormBinCF = aMaxNormBinCF;}
inline int buildAllcLamcKch2::GetMinNormBinCF() {return fMinNormBinCF;}
inline int buildAllcLamcKch2::GetMaxNormBinCF() {return fMaxNormBinCF;}

inline TH1F* buildAllcLamcKch2::GetCf_LamKchP_Tot() {return fCf_LamKchP_Tot;}
inline TH1F* buildAllcLamcKch2::GetCf_ALamKchP_Tot() {return fCf_ALamKchP_Tot;}

//Average Separation CF------------------
inline void buildAllcLamcKch2::SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF) {fMinNormBinAvgSepCF = aMinNormBinAvgSepCF;}
inline void buildAllcLamcKch2::SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF){fMaxNormBinAvgSepCF = aMaxNormBinAvgSepCF;}
inline int buildAllcLamcKch2::GetMinNormBinAvgSepCF() {return fMinNormBinAvgSepCF;}
inline int buildAllcLamcKch2::GetMaxNormBinAvgSepCF() {return fMaxNormBinAvgSepCF;}

//Purity calculations------------------


#endif

