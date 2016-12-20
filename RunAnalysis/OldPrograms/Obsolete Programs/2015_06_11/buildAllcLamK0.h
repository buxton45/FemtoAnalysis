///////////////////////////////////////////////////////////////////////////
//                                                                       //
// buildAll_cLamK0:                                                      //
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

//#include "UsefulMacros.C"
#include "TH1F.h"
#include "TString.h"
#include "TList.h"
#include "TF1.h"
#include "TVectorD.h"
#include "TLine.h"

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

class buildAllcLamK0 {

public:

  buildAllcLamK0();
  virtual ~buildAllcLamK0();

  TH1F* GetHistoClone(TString FileName, TString ArrayName, TString HistoName);
  TH1F* buildCF(TString name, TString title, TH1F* Num, TH1F* Denom, int fMinNormBin, int fMaxNormBin);
  TH1F* CombineCFs(TString name, TString title, TList* CfList, TList* NumList, int fMinNormBin, int fMaxNormBin);
  TList* Merge2Lists(TList* List1, TList* List2);

  void SetHistograms(TString aFileName);  //This needs to be called first in order to assign/grab all histograms etc.
  void buildCorrCombined(bool BuildBpTot, bool BuildBmTot, bool BuildTot);

  TH1F* GetCf(TString aFileName, TString aAnalysis, TString aHistogram);  //aFileName needs only contain Bp1, Bp2, Bm1, Bm2 or Bm3
                                                                           //aAnalysis should be exactly LamK0 or ALamK0
                                                                           //aHistogram should be exactly Num, Den or Cf

  TH1F* GetAvgSepCf(TString aFileName, TString aAnalysis, TString aDaughters, TString aHistogram); //aFileName, aAnalysis and aHistogram are as above
                                                                                                   //aDaughters should be exactly PosPos, PosNeg, NegPos or NegNeg


  TList* CalculatePurity(char* aName, TH1F* aPurityHisto, double aBgFitLow[2], double aBgFitHigh[2], double aROI[2]);
  void PurityDrawAll(TH1F* PurityHisto, TList* FitList, bool ZoomBg);







private:
  TList* fListOfFileNames;

  //KStar CF------------------
  int fMinNormBinCF;
  int fMaxNormBinCF;

  TH1F *fNumLamK0Bp1, *fNumLamK0Bp2, *fNumLamK0Bm1, *fNumLamK0Bm2, *fNumLamK0Bm3;
  TList *fNumList_LamK0_BpTot, *fNumList_LamK0_BmTot, *fNumList_LamK0_Tot;
  TH1F *fDenLamK0Bp1, *fDenLamK0Bp2, *fDenLamK0Bm1, *fDenLamK0Bm2, *fDenLamK0Bm3;
  TList *fDenList_LamK0_BpTot, *fDenList_LamK0_BmTot, *fDenList_LamK0_Tot;
  TH1F *fCfLamK0Bp1, *fCfLamK0Bp2, *fCfLamK0Bm1, *fCfLamK0Bm2, *fCfLamK0Bm3;
  TList *fCfList_LamK0_BpTot, *fCfList_LamK0_BmTot, *fCfList_LamK0_Tot;
  TH1F *fCfLamK0BpTot, *fCfLamK0BmTot, *fCfLamK0Tot;

  TH1F *fNumALamK0Bp1, *fNumALamK0Bp2, *fNumALamK0Bm1, *fNumALamK0Bm2, *fNumALamK0Bm3;
  TList *fNumList_ALamK0_BpTot, *fNumList_ALamK0_BmTot, *fNumList_ALamK0_Tot;
  TH1F *fDenALamK0Bp1, *fDenALamK0Bp2, *fDenALamK0Bm1, *fDenALamK0Bm2, *fDenALamK0Bm3;
  TList *fDenList_ALamK0_BpTot, *fDenList_ALamK0_BmTot, *fDenList_ALamK0_Tot;
  TH1F *fCfALamK0Bp1, *fCfALamK0Bp2, *fCfALamK0Bm1, *fCfALamK0Bm2, *fCfALamK0Bm3;
  TList *fCfList_ALamK0_BpTot, *fCfList_ALamK0_BmTot, *fCfList_ALamK0_Tot;
  TH1F *fCfALamK0BpTot, *fCfALamK0BmTot, *fCfALamK0Tot;

  //Average Separation CF------------------
  int fMinNormBinAvgSepCF;
  int fMaxNormBinAvgSepCF;
  //_____ ++ ___________________
  TH1F *fNumPosPosAvgSepCfLamK0Bp1, *fNumPosPosAvgSepCfLamK0Bp2, *fNumPosPosAvgSepCfLamK0Bm1, *fNumPosPosAvgSepCfLamK0Bm2, *fNumPosPosAvgSepCfLamK0Bm3;
  TList *fNumPosPosAvgSepCfList_LamK0_BpTot, *fNumPosPosAvgSepCfList_LamK0_BmTot, *fNumPosPosAvgSepCfList_LamK0_Tot;
  TH1F *fDenPosPosAvgSepCfLamK0Bp1, *fDenPosPosAvgSepCfLamK0Bp2, *fDenPosPosAvgSepCfLamK0Bm1, *fDenPosPosAvgSepCfLamK0Bm2, *fDenPosPosAvgSepCfLamK0Bm3;
  TList *fDenPosPosAvgSepCfList_LamK0_BpTot, *fDenPosPosAvgSepCfList_LamK0_BmTot, *fDenPosPosAvgSepCfList_LamK0_Tot;
  TH1F *fCfPosPosAvgSepCfLamK0Bp1, *fCfPosPosAvgSepCfLamK0Bp2, *fCfPosPosAvgSepCfLamK0Bm1, *fCfPosPosAvgSepCfLamK0Bm2, *fCfPosPosAvgSepCfLamK0Bm3;
  TList *fCfPosPosAvgSepCfList_LamK0_BpTot, *fCfPosPosAvgSepCfList_LamK0_BmTot, *fCfPosPosAvgSepCfList_LamK0_Tot;
  TH1F *fCfPosPosAvgSepCfLamK0BpTot, *fCfPosPosAvgSepCfLamK0BmTot, *fCfPosPosAvgSepCfLamK0Tot;

  TH1F *fNumPosPosAvgSepCfALamK0Bp1, *fNumPosPosAvgSepCfALamK0Bp2, *fNumPosPosAvgSepCfALamK0Bm1, *fNumPosPosAvgSepCfALamK0Bm2, *fNumPosPosAvgSepCfALamK0Bm3;
  TList *fNumPosPosAvgSepCfList_ALamK0_BpTot, *fNumPosPosAvgSepCfList_ALamK0_BmTot, *fNumPosPosAvgSepCfList_ALamK0_Tot;
  TH1F *fDenPosPosAvgSepCfALamK0Bp1, *fDenPosPosAvgSepCfALamK0Bp2, *fDenPosPosAvgSepCfALamK0Bm1, *fDenPosPosAvgSepCfALamK0Bm2, *fDenPosPosAvgSepCfALamK0Bm3;
  TList *fDenPosPosAvgSepCfList_ALamK0_BpTot, *fDenPosPosAvgSepCfList_ALamK0_BmTot, *fDenPosPosAvgSepCfList_ALamK0_Tot;
  TH1F *fCfPosPosAvgSepCfALamK0Bp1, *fCfPosPosAvgSepCfALamK0Bp2, *fCfPosPosAvgSepCfALamK0Bm1, *fCfPosPosAvgSepCfALamK0Bm2, *fCfPosPosAvgSepCfALamK0Bm3;
  TList *fCfPosPosAvgSepCfList_ALamK0_BpTot, *fCfPosPosAvgSepCfList_ALamK0_BmTot, *fCfPosPosAvgSepCfList_ALamK0_Tot;
  TH1F *fCfPosPosAvgSepCfALamK0BpTot, *fCfPosPosAvgSepCfALamK0BmTot, *fCfPosPosAvgSepCfALamK0Tot;

  //_____ +- ___________________
  TH1F *fNumPosNegAvgSepCfLamK0Bp1, *fNumPosNegAvgSepCfLamK0Bp2, *fNumPosNegAvgSepCfLamK0Bm1, *fNumPosNegAvgSepCfLamK0Bm2, *fNumPosNegAvgSepCfLamK0Bm3;
  TList *fNumPosNegAvgSepCfList_LamK0_BpTot, *fNumPosNegAvgSepCfList_LamK0_BmTot, *fNumPosNegAvgSepCfList_LamK0_Tot;
  TH1F *fDenPosNegAvgSepCfLamK0Bp1, *fDenPosNegAvgSepCfLamK0Bp2, *fDenPosNegAvgSepCfLamK0Bm1, *fDenPosNegAvgSepCfLamK0Bm2, *fDenPosNegAvgSepCfLamK0Bm3;
  TList *fDenPosNegAvgSepCfList_LamK0_BpTot, *fDenPosNegAvgSepCfList_LamK0_BmTot, *fDenPosNegAvgSepCfList_LamK0_Tot;
  TH1F *fCfPosNegAvgSepCfLamK0Bp1, *fCfPosNegAvgSepCfLamK0Bp2, *fCfPosNegAvgSepCfLamK0Bm1, *fCfPosNegAvgSepCfLamK0Bm2, *fCfPosNegAvgSepCfLamK0Bm3;
  TList *fCfPosNegAvgSepCfList_LamK0_BpTot, *fCfPosNegAvgSepCfList_LamK0_BmTot, *fCfPosNegAvgSepCfList_LamK0_Tot;
  TH1F *fCfPosNegAvgSepCfLamK0BpTot, *fCfPosNegAvgSepCfLamK0BmTot, *fCfPosNegAvgSepCfLamK0Tot;

  TH1F *fNumPosNegAvgSepCfALamK0Bp1, *fNumPosNegAvgSepCfALamK0Bp2, *fNumPosNegAvgSepCfALamK0Bm1, *fNumPosNegAvgSepCfALamK0Bm2, *fNumPosNegAvgSepCfALamK0Bm3;
  TList *fNumPosNegAvgSepCfList_ALamK0_BpTot, *fNumPosNegAvgSepCfList_ALamK0_BmTot, *fNumPosNegAvgSepCfList_ALamK0_Tot;
  TH1F *fDenPosNegAvgSepCfALamK0Bp1, *fDenPosNegAvgSepCfALamK0Bp2, *fDenPosNegAvgSepCfALamK0Bm1, *fDenPosNegAvgSepCfALamK0Bm2, *fDenPosNegAvgSepCfALamK0Bm3;
  TList *fDenPosNegAvgSepCfList_ALamK0_BpTot, *fDenPosNegAvgSepCfList_ALamK0_BmTot, *fDenPosNegAvgSepCfList_ALamK0_Tot;
  TH1F *fCfPosNegAvgSepCfALamK0Bp1, *fCfPosNegAvgSepCfALamK0Bp2, *fCfPosNegAvgSepCfALamK0Bm1, *fCfPosNegAvgSepCfALamK0Bm2, *fCfPosNegAvgSepCfALamK0Bm3;
  TList *fCfPosNegAvgSepCfList_ALamK0_BpTot, *fCfPosNegAvgSepCfList_ALamK0_BmTot, *fCfPosNegAvgSepCfList_ALamK0_Tot;
  TH1F *fCfPosNegAvgSepCfALamK0BpTot, *fCfPosNegAvgSepCfALamK0BmTot, *fCfPosNegAvgSepCfALamK0Tot;

  //_____ -+ ___________________
  TH1F *fNumNegPosAvgSepCfLamK0Bp1, *fNumNegPosAvgSepCfLamK0Bp2, *fNumNegPosAvgSepCfLamK0Bm1, *fNumNegPosAvgSepCfLamK0Bm2, *fNumNegPosAvgSepCfLamK0Bm3;
  TList *fNumNegPosAvgSepCfList_LamK0_BpTot, *fNumNegPosAvgSepCfList_LamK0_BmTot, *fNumNegPosAvgSepCfList_LamK0_Tot;
  TH1F *fDenNegPosAvgSepCfLamK0Bp1, *fDenNegPosAvgSepCfLamK0Bp2, *fDenNegPosAvgSepCfLamK0Bm1, *fDenNegPosAvgSepCfLamK0Bm2, *fDenNegPosAvgSepCfLamK0Bm3;
  TList *fDenNegPosAvgSepCfList_LamK0_BpTot, *fDenNegPosAvgSepCfList_LamK0_BmTot, *fDenNegPosAvgSepCfList_LamK0_Tot;
  TH1F *fCfNegPosAvgSepCfLamK0Bp1, *fCfNegPosAvgSepCfLamK0Bp2, *fCfNegPosAvgSepCfLamK0Bm1, *fCfNegPosAvgSepCfLamK0Bm2, *fCfNegPosAvgSepCfLamK0Bm3;
  TList *fCfNegPosAvgSepCfList_LamK0_BpTot, *fCfNegPosAvgSepCfList_LamK0_BmTot, *fCfNegPosAvgSepCfList_LamK0_Tot;
  TH1F *fCfNegPosAvgSepCfLamK0BpTot, *fCfNegPosAvgSepCfLamK0BmTot, *fCfNegPosAvgSepCfLamK0Tot;

  TH1F *fNumNegPosAvgSepCfALamK0Bp1, *fNumNegPosAvgSepCfALamK0Bp2, *fNumNegPosAvgSepCfALamK0Bm1, *fNumNegPosAvgSepCfALamK0Bm2, *fNumNegPosAvgSepCfALamK0Bm3;
  TList *fNumNegPosAvgSepCfList_ALamK0_BpTot, *fNumNegPosAvgSepCfList_ALamK0_BmTot, *fNumNegPosAvgSepCfList_ALamK0_Tot;
  TH1F *fDenNegPosAvgSepCfALamK0Bp1, *fDenNegPosAvgSepCfALamK0Bp2, *fDenNegPosAvgSepCfALamK0Bm1, *fDenNegPosAvgSepCfALamK0Bm2, *fDenNegPosAvgSepCfALamK0Bm3;
  TList *fDenNegPosAvgSepCfList_ALamK0_BpTot, *fDenNegPosAvgSepCfList_ALamK0_BmTot, *fDenNegPosAvgSepCfList_ALamK0_Tot;
  TH1F *fCfNegPosAvgSepCfALamK0Bp1, *fCfNegPosAvgSepCfALamK0Bp2, *fCfNegPosAvgSepCfALamK0Bm1, *fCfNegPosAvgSepCfALamK0Bm2, *fCfNegPosAvgSepCfALamK0Bm3;
  TList *fCfNegPosAvgSepCfList_ALamK0_BpTot, *fCfNegPosAvgSepCfList_ALamK0_BmTot, *fCfNegPosAvgSepCfList_ALamK0_Tot;
  TH1F *fCfNegPosAvgSepCfALamK0BpTot, *fCfNegPosAvgSepCfALamK0BmTot, *fCfNegPosAvgSepCfALamK0Tot;

  //_____ -- ___________________
  TH1F *fNumNegNegAvgSepCfLamK0Bp1, *fNumNegNegAvgSepCfLamK0Bp2, *fNumNegNegAvgSepCfLamK0Bm1, *fNumNegNegAvgSepCfLamK0Bm2, *fNumNegNegAvgSepCfLamK0Bm3;
  TList *fNumNegNegAvgSepCfList_LamK0_BpTot, *fNumNegNegAvgSepCfList_LamK0_BmTot, *fNumNegNegAvgSepCfList_LamK0_Tot;
  TH1F *fDenNegNegAvgSepCfLamK0Bp1, *fDenNegNegAvgSepCfLamK0Bp2, *fDenNegNegAvgSepCfLamK0Bm1, *fDenNegNegAvgSepCfLamK0Bm2, *fDenNegNegAvgSepCfLamK0Bm3;
  TList *fDenNegNegAvgSepCfList_LamK0_BpTot, *fDenNegNegAvgSepCfList_LamK0_BmTot, *fDenNegNegAvgSepCfList_LamK0_Tot;
  TH1F *fCfNegNegAvgSepCfLamK0Bp1, *fCfNegNegAvgSepCfLamK0Bp2, *fCfNegNegAvgSepCfLamK0Bm1, *fCfNegNegAvgSepCfLamK0Bm2, *fCfNegNegAvgSepCfLamK0Bm3;
  TList *fCfNegNegAvgSepCfList_LamK0_BpTot, *fCfNegNegAvgSepCfList_LamK0_BmTot, *fCfNegNegAvgSepCfList_LamK0_Tot;
  TH1F *fCfNegNegAvgSepCfLamK0BpTot, *fCfNegNegAvgSepCfLamK0BmTot, *fCfNegNegAvgSepCfLamK0Tot;

  TH1F *fNumNegNegAvgSepCfALamK0Bp1, *fNumNegNegAvgSepCfALamK0Bp2, *fNumNegNegAvgSepCfALamK0Bm1, *fNumNegNegAvgSepCfALamK0Bm2, *fNumNegNegAvgSepCfALamK0Bm3;
  TList *fNumNegNegAvgSepCfList_ALamK0_BpTot, *fNumNegNegAvgSepCfList_ALamK0_BmTot, *fNumNegNegAvgSepCfList_ALamK0_Tot;
  TH1F *fDenNegNegAvgSepCfALamK0Bp1, *fDenNegNegAvgSepCfALamK0Bp2, *fDenNegNegAvgSepCfALamK0Bm1, *fDenNegNegAvgSepCfALamK0Bm2, *fDenNegNegAvgSepCfALamK0Bm3;
  TList *fDenNegNegAvgSepCfList_ALamK0_BpTot, *fDenNegNegAvgSepCfList_ALamK0_BmTot, *fDenNegNegAvgSepCfList_ALamK0_Tot;
  TH1F *fCfNegNegAvgSepCfALamK0Bp1, *fCfNegNegAvgSepCfALamK0Bp2, *fCfNegNegAvgSepCfALamK0Bm1, *fCfNegNegAvgSepCfALamK0Bm2, *fCfNegNegAvgSepCfALamK0Bm3;
  TList *fCfNegNegAvgSepCfList_ALamK0_BpTot, *fCfNegNegAvgSepCfList_ALamK0_BmTot, *fCfNegNegAvgSepCfList_ALamK0_Tot;
  TH1F *fCfNegNegAvgSepCfALamK0BpTot, *fCfNegNegAvgSepCfALamK0BmTot, *fCfNegNegAvgSepCfALamK0Tot;

  //Purity calculations------------------
  double fLamBgFitLow[2];
  double fLamBgFitHigh[2];
  double fLamROI[2];

  double fK0Short1BgFitLow[2];
  double fK0Short1BgFitHigh[2];
  double fK0Short1ROI[2];

  TH1F* fLambdaPurityBp1;
    TList* fLamListBp1;     //will hold fitBg2, [Bgd, SigpBgd, Sig, and Pur], lROImin, lROImax, lBgFitLowMin, lBigFitLowMax, lBgFitHighMin, and lBgFitHighMax

  TH1F* fLambdaPurityBp2;
    TList* fLamListBp2;

  TH1F* fLambdaPurityBm1;
    TList* fLamListBm1;

  TH1F* fLambdaPurityBm2;
    TList* fLamListBm2;

  TH1F* fLambdaPurityBm3;
    TList* fLamListBm3;


  TH1F* fK0ShortPurity1Bp1;
    TList* fK0Short1ListBp1;

  TH1F* fK0ShortPurity1Bp2;
    TList* fK0Short1ListBp2;

  TH1F* fK0ShortPurity1Bm1;
    TList* fK0Short1ListBm1;

  TH1F* fK0ShortPurity1Bm2;
    TList* fK0Short1ListBm2;

  TH1F* fK0ShortPurity1Bm3;
    TList* fK0Short1ListBm3;


  TH1F* fAntiLambdaPurityBp1;
    TList* fALamListBp1;

  TH1F* fAntiLambdaPurityBp2;
    TList* fALamListBp2;

  TH1F* fAntiLambdaPurityBm1;
    TList* fALamListBm1;

  TH1F* fAntiLambdaPurityBm2;
    TList* fALamListBm2;

  TH1F* fAntiLambdaPurityBm3;
    TList* fALamListBm3;


  TH1F* fK0ShortPurity2Bp1;
    TList* fK0Short2ListBp1;

  TH1F* fK0ShortPurity2Bp2;
    TList* fK0Short2ListBp2;

  TH1F* fK0ShortPurity2Bm1;
    TList* fK0Short2ListBm1;

  TH1F* fK0ShortPurity2Bm2;
    TList* fK0Short2ListBm2;

  TH1F* fK0ShortPurity2Bm3;
    TList* fK0Short2ListBm3;





#ifdef __ROOT__
  ClassDef(buildAllcLamK0, 1)
#endif
};

//KStar CF------------------
inline void buildAllcLamK0::SetMinNormBinCF(int aMinNormBinCF) {fMinNormBinCF = aMinNormBinCF;}
inline void buildAllcLamK0::SetMaxNormBinCF(int aMaxNormBinCF){fMaxNormBinCF = aMaxNormBinCF;}
inline int buildAllcLamK0::GetMinNormBinCF() {return fMinNormBinCF;}
inline int buildAllcLamK0::GetMaxNormBinCF() {return fMaxNormBinCF;}

inline TList* buildAllcLamK0::GetNumList_LamK0_BpTot(){return fNumList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumList_LamK0_BmTot(){return fNumList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumList_LamK0_Tot(){return fNumList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetDenList_LamK0_BpTot(){return fDenList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenList_LamK0_BmTot(){return fDenList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenList_LamK0_Tot(){return fDenList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetCfList_LamK0_BpTot(){return fCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfList_LamK0_BmTot(){return fCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfList_LamK0_Tot(){return fCfList_LamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfLamK0BpTot(){return fCfLamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfLamK0BmTot(){return fCfLamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfLamK0Tot(){return fCfLamK0Tot;}

inline TList* buildAllcLamK0::GetNumList_ALamK0_BpTot(){return fNumList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumList_ALamK0_BmTot(){return fNumList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumList_ALamK0_Tot(){return fNumList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetDenList_ALamK0_BpTot(){return fDenList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenList_ALamK0_BmTot(){return fDenList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenList_ALamK0_Tot(){return fDenList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetCfList_ALamK0_BpTot(){return fCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfList_ALamK0_BmTot(){return fCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfList_ALamK0_Tot(){return fCfList_ALamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfALamK0BpTot(){return fCfALamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfALamK0BmTot(){return fCfALamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfALamK0Tot(){return fCfALamK0Tot;}

//Average Separation CF------------------
inline void buildAllcLamK0::SetMinNormBinAvgSepCF(int aMinNormBinAvgSepCF) {fMinNormBinAvgSepCF = aMinNormBinAvgSepCF;}
inline void buildAllcLamK0::SetMaxNormBinAvgSepCF(int aMaxNormBinAvgSepCF){fMaxNormBinAvgSepCF = aMaxNormBinAvgSepCF;}
inline int buildAllcLamK0::GetMinNormBinAvgSepCF() {return fMinNormBinAvgSepCF;}
inline int buildAllcLamK0::GetMaxNormBinAvgSepCF() {return fMaxNormBinAvgSepCF;}
//_____ ++ ___________________
inline TList* buildAllcLamK0::GetNumPosPosAvgSepCfList_LamK0_BpTot(){return fNumPosPosAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumPosPosAvgSepCfList_LamK0_BmTot(){return fNumPosPosAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumPosPosAvgSepCfList_LamK0_Tot(){return fNumPosPosAvgSepCfList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetDenPosPosAvgSepCfList_LamK0_BpTot(){return fDenPosPosAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenPosPosAvgSepCfList_LamK0_BmTot(){return fDenPosPosAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenPosPosAvgSepCfList_LamK0_Tot(){return fDenPosPosAvgSepCfList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetCfPosPosAvgSepCfList_LamK0_BpTot(){return fCfPosPosAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfPosPosAvgSepCfList_LamK0_BmTot(){return fCfPosPosAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfPosPosAvgSepCfList_LamK0_Tot(){return fCfPosPosAvgSepCfList_LamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfPosPosAvgSepCfLamK0BpTot(){return fCfPosPosAvgSepCfLamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfPosPosAvgSepCfLamK0BmTot(){return fCfPosPosAvgSepCfLamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfPosPosAvgSepCfLamK0Tot(){return fCfPosPosAvgSepCfLamK0Tot;}

inline TList* buildAllcLamK0::GetNumPosPosAvgSepCfList_ALamK0_BpTot(){return fNumPosPosAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumPosPosAvgSepCfList_ALamK0_BmTot(){return fNumPosPosAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumPosPosAvgSepCfList_ALamK0_Tot(){return fNumPosPosAvgSepCfList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetDenPosPosAvgSepCfList_ALamK0_BpTot(){return fDenPosPosAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenPosPosAvgSepCfList_ALamK0_BmTot(){return fDenPosPosAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenPosPosAvgSepCfList_ALamK0_Tot(){return fDenPosPosAvgSepCfList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetCfPosPosAvgSepCfList_ALamK0_BpTot(){return fCfPosPosAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfPosPosAvgSepCfList_ALamK0_BmTot(){return fCfPosPosAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfPosPosAvgSepCfList_ALamK0_Tot(){return fCfPosPosAvgSepCfList_ALamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfPosPosAvgSepCfALamK0BpTot(){return fCfPosPosAvgSepCfALamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfPosPosAvgSepCfALamK0BmTot(){return fCfPosPosAvgSepCfALamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfPosPosAvgSepCfALamK0Tot(){return fCfPosPosAvgSepCfALamK0Tot;}

//_____ +- ___________________
inline TList* buildAllcLamK0::GetNumPosNegAvgSepCfList_LamK0_BpTot(){return fNumPosNegAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumPosNegAvgSepCfList_LamK0_BmTot(){return fNumPosNegAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumPosNegAvgSepCfList_LamK0_Tot(){return fNumPosNegAvgSepCfList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetDenPosNegAvgSepCfList_LamK0_BpTot(){return fDenPosNegAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenPosNegAvgSepCfList_LamK0_BmTot(){return fDenPosNegAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenPosNegAvgSepCfList_LamK0_Tot(){return fDenPosNegAvgSepCfList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetCfPosNegAvgSepCfList_LamK0_BpTot(){return fCfPosNegAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfPosNegAvgSepCfList_LamK0_BmTot(){return fCfPosNegAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfPosNegAvgSepCfList_LamK0_Tot(){return fCfPosNegAvgSepCfList_LamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfPosNegAvgSepCfLamK0BpTot(){return fCfPosNegAvgSepCfLamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfPosNegAvgSepCfLamK0BmTot(){return fCfPosNegAvgSepCfLamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfPosNegAvgSepCfLamK0Tot(){return fCfPosNegAvgSepCfLamK0Tot;}

inline TList* buildAllcLamK0::GetNumPosNegAvgSepCfList_ALamK0_BpTot(){return fNumPosNegAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumPosNegAvgSepCfList_ALamK0_BmTot(){return fNumPosNegAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumPosNegAvgSepCfList_ALamK0_Tot(){return fNumPosNegAvgSepCfList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetDenPosNegAvgSepCfList_ALamK0_BpTot(){return fDenPosNegAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenPosNegAvgSepCfList_ALamK0_BmTot(){return fDenPosNegAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenPosNegAvgSepCfList_ALamK0_Tot(){return fDenPosNegAvgSepCfList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetCfPosNegAvgSepCfList_ALamK0_BpTot(){return fCfPosNegAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfPosNegAvgSepCfList_ALamK0_BmTot(){return fCfPosNegAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfPosNegAvgSepCfList_ALamK0_Tot(){return fCfPosNegAvgSepCfList_ALamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfPosNegAvgSepCfALamK0BpTot(){return fCfPosNegAvgSepCfALamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfPosNegAvgSepCfALamK0BmTot(){return fCfPosNegAvgSepCfALamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfPosNegAvgSepCfALamK0Tot(){return fCfPosNegAvgSepCfALamK0Tot;}

//_____ -+ ___________________
inline TList* buildAllcLamK0::GetNumNegPosAvgSepCfList_LamK0_BpTot(){return fNumNegPosAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumNegPosAvgSepCfList_LamK0_BmTot(){return fNumNegPosAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumNegPosAvgSepCfList_LamK0_Tot(){return fNumNegPosAvgSepCfList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetDenNegPosAvgSepCfList_LamK0_BpTot(){return fDenNegPosAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenNegPosAvgSepCfList_LamK0_BmTot(){return fDenNegPosAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenNegPosAvgSepCfList_LamK0_Tot(){return fDenNegPosAvgSepCfList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetCfNegPosAvgSepCfList_LamK0_BpTot(){return fCfNegPosAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfNegPosAvgSepCfList_LamK0_BmTot(){return fCfNegPosAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfNegPosAvgSepCfList_LamK0_Tot(){return fCfNegPosAvgSepCfList_LamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfNegPosAvgSepCfLamK0BpTot(){return fCfNegPosAvgSepCfLamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfNegPosAvgSepCfLamK0BmTot(){return fCfNegPosAvgSepCfLamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfNegPosAvgSepCfLamK0Tot(){return fCfNegPosAvgSepCfLamK0Tot;}

inline TList* buildAllcLamK0::GetNumNegPosAvgSepCfList_ALamK0_BpTot(){return fNumNegPosAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumNegPosAvgSepCfList_ALamK0_BmTot(){return fNumNegPosAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumNegPosAvgSepCfList_ALamK0_Tot(){return fNumNegPosAvgSepCfList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetDenNegPosAvgSepCfList_ALamK0_BpTot(){return fDenNegPosAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenNegPosAvgSepCfList_ALamK0_BmTot(){return fDenNegPosAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenNegPosAvgSepCfList_ALamK0_Tot(){return fDenNegPosAvgSepCfList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetCfNegPosAvgSepCfList_ALamK0_BpTot(){return fCfNegPosAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfNegPosAvgSepCfList_ALamK0_BmTot(){return fCfNegPosAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfNegPosAvgSepCfList_ALamK0_Tot(){return fCfNegPosAvgSepCfList_ALamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfNegPosAvgSepCfALamK0BpTot(){return fCfNegPosAvgSepCfALamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfNegPosAvgSepCfALamK0BmTot(){return fCfNegPosAvgSepCfALamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfNegPosAvgSepCfALamK0Tot(){return fCfNegPosAvgSepCfALamK0Tot;}

//_____ -- ___________________
inline TList* buildAllcLamK0::GetNumNegNegAvgSepCfList_LamK0_BpTot(){return fNumNegNegAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumNegNegAvgSepCfList_LamK0_BmTot(){return fNumNegNegAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumNegNegAvgSepCfList_LamK0_Tot(){return fNumNegNegAvgSepCfList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetDenNegNegAvgSepCfList_LamK0_BpTot(){return fDenNegNegAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenNegNegAvgSepCfList_LamK0_BmTot(){return fDenNegNegAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenNegNegAvgSepCfList_LamK0_Tot(){return fDenNegNegAvgSepCfList_LamK0_Tot;}
inline TList* buildAllcLamK0::GetCfNegNegAvgSepCfList_LamK0_BpTot(){return fCfNegNegAvgSepCfList_LamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfNegNegAvgSepCfList_LamK0_BmTot(){return fCfNegNegAvgSepCfList_LamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfNegNegAvgSepCfList_LamK0_Tot(){return fCfNegNegAvgSepCfList_LamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfNegNegAvgSepCfLamK0BpTot(){return fCfNegNegAvgSepCfLamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfNegNegAvgSepCfLamK0BmTot(){return fCfNegNegAvgSepCfLamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfNegNegAvgSepCfLamK0Tot(){return fCfNegNegAvgSepCfLamK0Tot;}

inline TList* buildAllcLamK0::GetNumNegNegAvgSepCfList_ALamK0_BpTot(){return fNumNegNegAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetNumNegNegAvgSepCfList_ALamK0_BmTot(){return fNumNegNegAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetNumNegNegAvgSepCfList_ALamK0_Tot(){return fNumNegNegAvgSepCfList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetDenNegNegAvgSepCfList_ALamK0_BpTot(){return fDenNegNegAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetDenNegNegAvgSepCfList_ALamK0_BmTot(){return fDenNegNegAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetDenNegNegAvgSepCfList_ALamK0_Tot(){return fDenNegNegAvgSepCfList_ALamK0_Tot;}
inline TList* buildAllcLamK0::GetCfNegNegAvgSepCfList_ALamK0_BpTot(){return fCfNegNegAvgSepCfList_ALamK0_BpTot;}
inline TList* buildAllcLamK0::GetCfNegNegAvgSepCfList_ALamK0_BmTot(){return fCfNegNegAvgSepCfList_ALamK0_BmTot;}
inline TList* buildAllcLamK0::GetCfNegNegAvgSepCfList_ALamK0_Tot(){return fCfNegNegAvgSepCfList_ALamK0_Tot;}
inline TH1F* buildAllcLamK0::GetCfNegNegAvgSepCfALamK0BpTot(){return fCfNegNegAvgSepCfALamK0BpTot;}
inline TH1F* buildAllcLamK0::GetCfNegNegAvgSepCfALamK0BmTot(){return fCfNegNegAvgSepCfALamK0BmTot;}
inline TH1F* buildAllcLamK0::GetCfNegNegAvgSepCfALamK0Tot(){return fCfNegNegAvgSepCfALamK0Tot;}

//Purity calculations------------------
inline TH1F* buildAllcLamK0::GetLambdaPurityBp1(){return fLambdaPurityBp1;}
inline TList* buildAllcLamK0::GetLamListBp1(){return fLamListBp1;}
inline TH1F* buildAllcLamK0::GetLambdaPurityBp2(){return fLambdaPurityBp2;}
inline TList* buildAllcLamK0::GetLamListBp2(){return fLamListBp2;}
inline TH1F* buildAllcLamK0::GetLambdaPurityBm1(){return fLambdaPurityBm1;}
inline TList* buildAllcLamK0::GetLamListBm1(){return fLamListBm1;}
inline TH1F* buildAllcLamK0::GetLambdaPurityBm2(){return fLambdaPurityBm2;}
inline TList* buildAllcLamK0::GetLamListBm2(){return fLamListBm2;}
inline TH1F* buildAllcLamK0::GetLambdaPurityBm3(){return fLambdaPurityBm3;}
inline TList* buildAllcLamK0::GetLamListBm3(){return fLamListBm3;}

inline TH1F* buildAllcLamK0::GetK0Short1PurityBp1(){return fK0Short1PurityBp1;}
inline TList* buildAllcLamK0::GetK0Short1ListBp1(){return fK0Short1ListBp1;}
inline TH1F* buildAllcLamK0::GetK0Short1PurityBp2(){return fK0Short1PurityBp2;}
inline TList* buildAllcLamK0::GetK0Short1ListBp2(){return fK0Short1ListBp2;}
inline TH1F* buildAllcLamK0::GetK0Short1PurityBm1(){return fK0Short1PurityBm1;}
inline TList* buildAllcLamK0::GetK0Short1ListBm1(){return fK0Short1ListBm1;}
inline TH1F* buildAllcLamK0::GetK0Short1PurityBm2(){return fK0Short1PurityBm2;}
inline TList* buildAllcLamK0::GetK0Short1ListBm2(){return fK0Short1ListBm2;}
inline TH1F* buildAllcLamK0::GetK0Short1PurityBm3(){return fK0Short1PurityBm3;}
inline TList* buildAllcLamK0::GetK0Short1ListBm3(){return fK0Short1ListBm3;}

inline TH1F* buildAllcLamK0::GetAntiLambdaPurityBp1(){return fAntiLambdaPurityBp1;}
inline TList* buildAllcLamK0::GetALamListBp1(){return fALamListBp1;}
inline TH1F* buildAllcLamK0::GetAntiLambdaPurityBp2(){return fAntiLambdaPurityBp2;}
inline TList* buildAllcLamK0::GetALamListBp2(){return fALamListBp2;}
inline TH1F* buildAllcLamK0::GetAntiLambdaPurityBm1(){return fAntiLambdaPurityBm1;}
inline TList* buildAllcLamK0::GetALamListBm1(){return fALamListBm1;}
inline TH1F* buildAllcLamK0::GetAntiLambdaPurityBm2(){return fAntiLambdaPurityBm2;}
inline TList* buildAllcLamK0::GetALamListBm2(){return fALamListBm2;}
inline TH1F* buildAllcLamK0::GetAntiLambdaPurityBm3(){return fAntiLambdaPurityBm3;}
inline TList* buildAllcLamK0::GetALamListBm3(){return fALamListBm3;}

inline TH1F* buildAllcLamK0::GetK0Short2PurityBp1(){return fK0Short2PurityBp1;}
inline TList* buildAllcLamK0::GetK0Short2ListBp1(){return fK0Short2ListBp1;}
inline TH1F* buildAllcLamK0::GetK0Short2PurityBp2(){return fK0Short2PurityBp2;}
inline TList* buildAllcLamK0::GetK0Short2ListBp2(){return fK0Short2ListBp2;}
inline TH1F* buildAllcLamK0::GetK0Short2PurityBm1(){return fK0Short2PurityBm1;}
inline TList* buildAllcLamK0::GetK0Short2ListBm1(){return fK0Short2ListBm1;}
inline TH1F* buildAllcLamK0::GetK0Short2PurityBm2(){return fK0Short2PurityBm2;}
inline TList* buildAllcLamK0::GetK0Short2ListBm2(){return fK0Short2ListBm2;}
inline TH1F* buildAllcLamK0::GetK0Short2PurityBm3(){return fK0Short2PurityBm3;}
inline TList* buildAllcLamK0::GetK0Short2ListBm3(){return fK0Short2ListBm3;}

#endif
