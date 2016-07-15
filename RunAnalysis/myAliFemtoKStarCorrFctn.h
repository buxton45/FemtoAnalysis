/*
 *  myAliFemtoKStarCorrFctn.h
 *
 *  Based off AliFemtoQinvCorrFctn and Kubera's KStarCF
 */

#ifndef MYALIFEMTOKSTARCORRFCTN_H
#define MYALIFEMTOKSTARCORRFCTN_H

#include "TH1D.h"
#include "TH2D.h"
#include "TNtuple.h"

#include "AliFemtoCorrFctn.h"

#include "AliAODInputHandler.h"
#include "AliAnalysisManager.h"

class myAliFemtoKStarCorrFctn : public AliFemtoCorrFctn {
public:
  myAliFemtoKStarCorrFctn(const char* title, const int& nbins, const float& KStarLo, const float& KStarHi);
  myAliFemtoKStarCorrFctn(const myAliFemtoKStarCorrFctn& aCorrFctn);
  virtual ~myAliFemtoKStarCorrFctn();

  myAliFemtoKStarCorrFctn& operator=(const myAliFemtoKStarCorrFctn& aCorrFctn);

  virtual AliFemtoString Report();
  virtual void AddRealPair(AliFemtoPair* aPair);
  virtual void AddMixedPair(AliFemtoPair* aPair);

  virtual void Finish();

  void CalculateDetaDphis(Bool_t, Double_t);
  void CalculatePairKinematics(Bool_t);

  TH1D* Numerator();
  TH1D* Denominator();
  TH1D* Ratio();

  virtual TList* GetOutputList();
  void Write();

private:
  TH1D* fNumerator;          // numerator - real pairs
  TH1D* fDenominator;        // denominator - mixed pairs
  TH1D* fRatio;              // ratio - correlation function
  TH1D* fkTMonitor;          // Monitor the kT of pairs in the function

  Bool_t fDetaDphiscal;
  Bool_t fPairKinematics;

  Double_t fRaddedps;
  TH2D* fNumDEtaDPhiS;
  TH2D* fDenDEtaDPhiS;

  TNtuple* fPairKStar; //PairReader for CorrFit

#ifdef __ROOT__
  ClassDef(myAliFemtoKStarCorrFctn, 1)
#endif
};

inline  TH1D* myAliFemtoKStarCorrFctn::Numerator(){return fNumerator;}
inline  TH1D* myAliFemtoKStarCorrFctn::Denominator(){return fDenominator;}
inline  TH1D* myAliFemtoKStarCorrFctn::Ratio(){return fRatio;}


#endif
