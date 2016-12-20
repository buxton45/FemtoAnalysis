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

#ifndef BUILDALLCLAMK0_H
#define BUILDALLCLAMK0_H

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

class buildAllcLamK0 : public buildAll {

public:

  buildAllcLamK0(vector<TString> &aVectorOfFileNames, TString aDirNameLamK0, TString aDirNameALamK0);
  virtual ~buildAllcLamK0();


  void SetAnalysisDirectories();
  TObjArray* GetAnalysisDirectory(TString aFile, TString aDirectoryName);  //aFile needs only contain, for example Bp1, Bp2, ...
                                                                           //aDirectoryName needs equal EXACTLY LamK0 or ALamK0

  TObjArray* LoadCollectionOfHistograms(TString aDirectoryName, TString aHistoName);


  void BuildCFCollections();
  void DrawFinalCFs(TCanvas *aCanvas);

    //--27 April 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK0)
  TObjArray* GetCfCollection(TString aType, TString aDirectoryName);  //aType should be either Num, Den, or Cf
								      //aDirectoryName needs equal EXACTLY LamK0 or ALamK0
    //--------------

  void BuildAvgSepCollections();
  void DrawFinalAvgSepCFs(TCanvas *aCanvasLamK0, TCanvas *aCanvasALamK0);

  void BuildPurityCollections();

  void DrawFinalPurity(TCanvas *aCanvas);


    //----29 April 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK0)
  void SaveAll(TFile* aFile); //option should be, for example, RECREATE, NEW, CREATE, UPDATE, etc.


  //-----7 May 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK0)
  //Separation Histograms
  TObjArray* LoadCollectionOf2DHistograms(TString aDirectoryName, TString aHistoName);

  void BuildSepCollections(int aRebinFactor = 1);
  void DrawFinalSepCFs(TCanvas *aCanvasLamK0LikeSigns, TCanvas *aCanvasLamK0UnlikeSigns, TCanvas *aCanvasALamK0LikeSigns, TCanvas *aCanvasALamK0UnlikeSigns);
  void DrawAvgOfFinalSepCFs(TCanvas *aCanvasLamK0LikeSigns, TCanvas *aCanvasLamK0UnlikeSigns, TCanvas *aCanvasALamK0LikeSigns, TCanvas *aCanvasALamK0UnlikeSigns);

  //-----7 July 2015 
  //Average Separation Cowboys and Sailors Histograms
  void BuildCowCollections(int aRebinFactor = 1);
  void DrawFinalCowCFs(TCanvas *aCanvasLamK0, TCanvas *aCanvasALamK0);


  //Inline Functions------------------------
  //General stuff
  TString GetDirNameLamK0();
  TString GetDirNameALamK0();

  TObjArray* GetNumCollection_LamK0();
  TObjArray* GetDenCollection_LamK0();

 //KStar CF------------------
  TH1F* GetCf_LamK0_Tot();
  TH1F* GetCf_ALamK0_Tot();

  //Average Separation CF------------------

    //--31 March 2015
  TH1F* GetAvgSepCf_PosPos_LamK0_Tot();
  TH1F* GetAvgSepCf_NegNeg_LamK0_Tot();
  TH1F* GetAvgSepCf_PosPos_ALamK0_Tot();
  TH1F* GetAvgSepCf_NegNeg_ALamK0_Tot();

  //Purity calculations------------------





private:

  //General stuff needed to extract/store the proper files/histograms etc.
  TString fDirNameLamK0, fDirNameALamK0;

  TObjArray *fDirLamK0Bp1, *fDirLamK0Bp2, *fDirLamK0Bm1, *fDirLamK0Bm2, *fDirLamK0Bm3;
  TObjArray *fDirALamK0Bp1, *fDirALamK0Bp2, *fDirALamK0Bm1, *fDirALamK0Bm2, *fDirALamK0Bm3;

  //KStar CFs-------------------------
  TObjArray *fNumCollection_LamK0, *fDenCollection_LamK0, *fCfCollection_LamK0;
  TH1F *fCf_LamK0_BpTot, *fCf_LamK0_BmTot, *fCf_LamK0_Tot;

  TObjArray *fNumCollection_ALamK0, *fDenCollection_ALamK0, *fCfCollection_ALamK0;
  TH1F *fCf_ALamK0_BpTot, *fCf_ALamK0_BmTot, *fCf_ALamK0_Tot;


  //Average Separation CF------------------
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
  TObjArray *fLambdaPurityHistogramCollection, *fK0Short1PurityHistogramCollection, *fAntiLambdaPurityHistogramCollection, *fK0Short2PurityHistogramCollection;
  vector<TObjArray*> fLambdaPurityListCollection, fK0Short1PurityListCollection, fAntiLambdaPurityListCollection, fK0Short2PurityListCollection;
    //The vector contains 5 elements, one for each file (Bp1,Bp2,Bm1,Bm2,Bm3)
    //Each element (TObjArray) of the vector contains 5 elements 
    //  which hold {fitBg, [Bgd, SigpBgd, Sig, and Pur], [ROImin, ROImax], [BgFitLowMin, BgFitLowMax], [BgFitHighMin, BgFitHighMax]}
  TH1F *fLambdaPurityTot, *fK0Short1PurityTot, *fAntiLambdaPurityTot, *fK0Short2PurityTot;
  TObjArray *fLambdaPurityListTot, *fK0Short1PurityListTot, *fAntiLambdaPurityListTot, *fK0Short2PurityListTot;


  //-----7 May 2015 in buildAllcLamcKch3, 23 June 2015 in this file (buildAllcLamK0)
  //Separation Histograms
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




  //-----7 July 2015 
  //Average Separation Cowboys and Sailors Histograms
  //_____ PosPos ___________________
  TObjArray *f2DCowNumCollection_PosPos_LamK0, *f2DCowDenCollection_PosPos_LamK0;
  TObjArray *f1DCowCfCollection_PosPos_LamK0;

  TObjArray *f2DCowNumCollection_PosPos_ALamK0, *f2DCowDenCollection_PosPos_ALamK0;
  TObjArray *f1DCowCfCollection_PosPos_ALamK0;

  //_____ PosNeg ___________________
  TObjArray *f2DCowNumCollection_PosNeg_LamK0, *f2DCowDenCollection_PosNeg_LamK0;
  TObjArray *f1DCowCfCollection_PosNeg_LamK0;

  TObjArray *f2DCowNumCollection_PosNeg_ALamK0, *f2DCowDenCollection_PosNeg_ALamK0;
  TObjArray *f1DCowCfCollection_PosNeg_ALamK0;

  //_____ NegPos ___________________
  TObjArray *f2DCowNumCollection_NegPos_LamK0, *f2DCowDenCollection_NegPos_LamK0;
  TObjArray *f1DCowCfCollection_NegPos_LamK0;

  TObjArray *f2DCowNumCollection_NegPos_ALamK0, *f2DCowDenCollection_NegPos_ALamK0;
  TObjArray *f1DCowCfCollection_NegPos_ALamK0;

  //_____ NegNeg ___________________
  TObjArray *f2DCowNumCollection_NegNeg_LamK0, *f2DCowDenCollection_NegNeg_LamK0;
  TObjArray *f1DCowCfCollection_NegNeg_LamK0;

  TObjArray *f2DCowNumCollection_NegNeg_ALamK0, *f2DCowDenCollection_NegNeg_ALamK0;
  TObjArray *f1DCowCfCollection_NegNeg_ALamK0;



#ifdef __ROOT__
  ClassDef(buildAllcLamK0, 1)
#endif
};


//General stuff
inline TString buildAllcLamK0::GetDirNameLamK0() {return fDirNameLamK0;}
inline TString buildAllcLamK0::GetDirNameALamK0() {return fDirNameALamK0;}

inline TObjArray* buildAllcLamK0::GetNumCollection_LamK0() {return fNumCollection_LamK0;}
inline TObjArray* buildAllcLamK0::GetDenCollection_LamK0() {return fDenCollection_LamK0;}

//KStar CF------------------
inline TH1F* buildAllcLamK0::GetCf_LamK0_Tot() {return fCf_LamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCf_ALamK0_Tot() {return fCf_ALamK0_Tot;}

//Average Separation CF------------------
    //--31 March 2015
inline TH1F* buildAllcLamK0::GetAvgSepCf_PosPos_LamK0_Tot() {return fAvgSepCf_PosPos_LamK0_Tot;}
inline TH1F* buildAllcLamK0::GetAvgSepCf_NegNeg_LamK0_Tot() {return fAvgSepCf_NegNeg_LamK0_Tot;}
inline TH1F* buildAllcLamK0::GetAvgSepCf_PosPos_ALamK0_Tot() {return fAvgSepCf_PosPos_ALamK0_Tot;}
inline TH1F* buildAllcLamK0::GetAvgSepCf_NegNeg_ALamK0_Tot() {return fAvgSepCf_NegNeg_ALamK0_Tot;}

//Purity calculations------------------


#endif

