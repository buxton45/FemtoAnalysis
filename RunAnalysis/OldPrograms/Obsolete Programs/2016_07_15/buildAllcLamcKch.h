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

#ifndef BUILDALLCLAMCKCH_H
#define BUILDALLCLAMCKCH_H

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

#include "buildAll.cxx"

//______________________________________________________________________________________________________________
//*****_____*****_____*****_____*****_____*****_____*****_____*****_____*****_____*****_____*****_____*****_____
//______________________________________________________________________________________________________________

class buildAllcLamcKch : public buildAll {

public:

  buildAllcLamcKch(vector<TString> &aVectorOfFileNames, TString aDirNameLamKchP, TString aDirNameLamKchM, TString aDirNameALamKchP, TString aDirNameALamKchM);
  virtual ~buildAllcLamcKch();


  void SetAnalysisDirectories();
  TObjArray* GetAnalysisDirectory(TString aFile, TString aDirectoryName);  //aFile needs only contain, for example Bp1, Bp2, ...
                                                                           //aDirectoryName needs equal EXACTLY LamKchP, LamKchM, ALamKchP or ALamKchM

  TObjArray* LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName);

  void BuildCFCollections();
  void DrawFinalCFs(TCanvas *aCanvas);

    //--27 April 2015
  TObjArray* GetCfCollection(TString aType, TString aDirectoryName);  //aType should be either Num, Den, or Cf
								      //aDirectoryName needs equal EXACTLY LamKchP, LamKchM, ALamKchP or ALamKchM
    //--------------

  void BuildAvgSepCollections();
  void DrawFinalAvgSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM);

  void BuildPurityCollections();
  void DrawFinalPurity(TCanvas *aCanvas);


    //----29 April 2015
  void SaveAll(TFile* aFile); //option should be, for example, RECREATE, NEW, CREATE, UPDATE, etc.


  //-----7 May 2015
  //Separation Histograms
  TObjArray* LoadCollectionOf2DHistograms(TString aDirectoryName, TString aHistoName);

  void BuildSepCollections(int aRebinFactor = 1);
  void DrawFinalSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM);
  void DrawAvgOfFinalSepCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM);

  //-----7 July 2015 
  //Average Separation Cowboys and Sailors Histograms
  void BuildCowCollections(int aRebinFactor = 1);
  void DrawFinalCowCFs(TCanvas *aCanvasLamKchP, TCanvas *aCanvasLamKchM, TCanvas *aCanvasALamKchP, TCanvas *aCanvasALamKchM);

  //KStar Cfs binned in posneg KStarOut
  TObjArray* DecomposeKStar2DCfs(TString aAnalysisTag, TObjArray *a2DNumCollection, TObjArray* a2DDenCollection);
  void BuildKStarCfsBinnedInKStarOut();
  void DrawKStarCfsBinnedInKStarOut(TCanvas *aCanvas);


  //Inline Functions------------------------

  //General stuff
  TString GetDirNameLamKchP();
  TString GetDirNameALamKchP();

  TObjArray* GetNumCollection_LamKchP();
  TObjArray* GetDenCollection_LamKchP();

 //KStar CF------------------
  TH1F* GetCf_LamKchP_Tot();
  TH1F* GetCf_ALamKchP_Tot();

    //1 April 2015
  TH1F* GetCf_LamKchM_Tot();
  TH1F* GetCf_ALamKchM_Tot();

  //Average Separation CF------------------
    //--31 March 2015
  TH1F* GetAvgSepCf_TrackPos_LamKchP_Tot();
  TH1F* GetAvgSepCf_TrackNeg_LamKchM_Tot();
  TH1F* GetAvgSepCf_TrackPos_ALamKchP_Tot();
  TH1F* GetAvgSepCf_TrackNeg_ALamKchM_Tot();

    //--1 April 2015
  TH1F* GetCf_AverageLamKchPM_Tot();
  TH1F* GetCf_AverageALamKchPM_Tot();

  //Purity calculations------------------


  //Other--------------------------------
  double GetMCKchPurity(AnalysisType aAnalysisType, bool aBeforePairCut);


private:

  //General stuff needed to extract/store the proper files/histograms etc.
  TString fDirNameLamKchP, fDirNameLamKchM, fDirNameALamKchP, fDirNameALamKchM;

  TObjArray *fDirLamKchPBp1, *fDirLamKchPBp2, *fDirLamKchPBm1, *fDirLamKchPBm2, *fDirLamKchPBm3;
  TObjArray *fDirLamKchMBp1, *fDirLamKchMBp2, *fDirLamKchMBm1, *fDirLamKchMBm2, *fDirLamKchMBm3;
  TObjArray *fDirALamKchPBp1, *fDirALamKchPBp2, *fDirALamKchPBm1, *fDirALamKchPBm2, *fDirALamKchPBm3;
  TObjArray *fDirALamKchMBp1, *fDirALamKchMBp2, *fDirALamKchMBm1, *fDirALamKchMBm2, *fDirALamKchMBm3;

  //KStar CFs-------------------------
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
  TObjArray *fLambdaPurityHistogramCollection_LamKchP, *fLambdaPurityHistogramCollection_LamKchM, *fAntiLambdaPurityHistogramCollection_ALamKchP, *fAntiLambdaPurityHistogramCollection_ALamKchM;
  vector<TObjArray*> fLambdaPurityListCollection_LamKchP, fLambdaPurityListCollection_LamKchM, fAntiLambdaPurityListCollection_ALamKchP, fAntiLambdaPurityListCollection_ALamKchM;
    //The vector contains 5 elements, one for each file (Bp1,Bp2,Bm1,Bm2,Bm3)
    //Each element (TObjArray) of the vector contains 5 elements 
    //  which hold {fitBg, [Bgd, SigpBgd, Sig, and Pur], [ROImin, ROImax], [BgFitLowMin, BgFitLowMax], [BgFitHighMin, BgFitHighMax]}
  TH1F *fLambdaPurityTot_LamKchP, *fLambdaPurityTot_LamKchM, *fAntiLambdaPurityTot_ALamKchP, *fAntiLambdaPurityTot_ALamKchM;
  TObjArray *fLambdaPurityListTot_LamKchP, *fLambdaPurityListTot_LamKchM, *fAntiLambdaPurityListTot_ALamKchP, *fAntiLambdaPurityListTot_ALamKchM;


  //-----7 May 2015
  //Separation Histograms
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


  //-----7 July 2015 
  //Average Separation Cowboys and Sailors Histograms
  //_____ Track+ ___________________
  TObjArray *f2DCowNumCollection_TrackPos_LamKchP, *f2DCowDenCollection_TrackPos_LamKchP;
  TObjArray *f1DCowCfCollection_TrackPos_LamKchP;

  TObjArray *f2DCowNumCollection_TrackPos_LamKchM, *f2DCowDenCollection_TrackPos_LamKchM;
  TObjArray *f1DCowCfCollection_TrackPos_LamKchM;

  TObjArray *f2DCowNumCollection_TrackPos_ALamKchP, *f2DCowDenCollection_TrackPos_ALamKchP;
  TObjArray *f1DCowCfCollection_TrackPos_ALamKchP;

  TObjArray *f2DCowNumCollection_TrackPos_ALamKchM, *f2DCowDenCollection_TrackPos_ALamKchM;
  TObjArray *f1DCowCfCollection_TrackPos_ALamKchM;

  //_____ Track- ___________________
  TObjArray *f2DCowNumCollection_TrackNeg_LamKchP, *f2DCowDenCollection_TrackNeg_LamKchP;
  TObjArray *f1DCowCfCollection_TrackNeg_LamKchP;

  TObjArray *f2DCowNumCollection_TrackNeg_LamKchM, *f2DCowDenCollection_TrackNeg_LamKchM;
  TObjArray *f1DCowCfCollection_TrackNeg_LamKchM;

  TObjArray *f2DCowNumCollection_TrackNeg_ALamKchP, *f2DCowDenCollection_TrackNeg_ALamKchP;
  TObjArray *f1DCowCfCollection_TrackNeg_ALamKchP;

  TObjArray *f2DCowNumCollection_TrackNeg_ALamKchM, *f2DCowDenCollection_TrackNeg_ALamKchM;
  TObjArray *f1DCowCfCollection_TrackNeg_ALamKchM;

  //KStar Cfs binned in posneg KStarOut
  TH1D *fKStarCfPosKStarOut_LamKchP, *fKStarCfNegKStarOut_LamKchP, *fKStarCfRatioPosNeg_LamKchP; 
  TH1D *fKStarCfPosKStarOut_LamKchM, *fKStarCfNegKStarOut_LamKchM, *fKStarCfRatioPosNeg_LamKchM; 
  TH1D *fKStarCfPosKStarOut_ALamKchP, *fKStarCfNegKStarOut_ALamKchP, *fKStarCfRatioPosNeg_ALamKchP; 
  TH1D *fKStarCfPosKStarOut_ALamKchM, *fKStarCfNegKStarOut_ALamKchM, *fKStarCfRatioPosNeg_ALamKchM; 


#ifdef __ROOT__
  ClassDef(buildAllcLamcKch, 1)
#endif
};

//General stuff
inline TString buildAllcLamcKch::GetDirNameLamKchP() {return fDirNameLamKchP;}
inline TString buildAllcLamcKch::GetDirNameALamKchP() {return fDirNameALamKchP;}

inline TObjArray* buildAllcLamcKch::GetNumCollection_LamKchP() {return fNumCollection_LamKchP;}
inline TObjArray* buildAllcLamcKch::GetDenCollection_LamKchP() {return fDenCollection_LamKchP;}

//KStar CF------------------
inline TH1F* buildAllcLamcKch::GetCf_LamKchP_Tot() {return fCf_LamKchP_Tot;}
inline TH1F* buildAllcLamcKch::GetCf_ALamKchP_Tot() {return fCf_ALamKchP_Tot;}

  //--1 April 2015
inline TH1F* buildAllcLamcKch::GetCf_LamKchM_Tot() {return fCf_LamKchM_Tot;}
inline TH1F* buildAllcLamcKch::GetCf_ALamKchM_Tot() {return fCf_ALamKchM_Tot;}

//Average Separation CF------------------
    //--31 March 2015
inline TH1F* buildAllcLamcKch::GetAvgSepCf_TrackPos_LamKchP_Tot() {return fAvgSepCf_TrackPos_LamKchP_Tot;}
inline TH1F* buildAllcLamcKch::GetAvgSepCf_TrackNeg_LamKchM_Tot() {return fAvgSepCf_TrackNeg_LamKchM_Tot;}
inline TH1F* buildAllcLamcKch::GetAvgSepCf_TrackPos_ALamKchP_Tot() {return fAvgSepCf_TrackPos_ALamKchP_Tot;}
inline TH1F* buildAllcLamcKch::GetAvgSepCf_TrackNeg_ALamKchM_Tot() {return fAvgSepCf_TrackNeg_ALamKchM_Tot;}

    //--1 April 2015
inline TH1F* buildAllcLamcKch::GetCf_AverageLamKchPM_Tot() {return fCf_AverageLamKchPM_Tot;}
inline TH1F* buildAllcLamcKch::GetCf_AverageALamKchPM_Tot() {return fCf_AverageALamKchPM_Tot;}

//Purity calculations------------------


#endif

