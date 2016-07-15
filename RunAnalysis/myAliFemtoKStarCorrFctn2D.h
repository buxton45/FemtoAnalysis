/*
 *  myAliFemtoKStarCorrFctn2D.h
 *
 *  For now, the second dimension will simply bin in positive and negative k_out*
 *  May be used later when I want to bin in kT
 */

#ifndef MYALIFEMTOKSTARCORRFCTN2D_H
#define MYALIFEMTOKSTARCORRFCTN2D_H

#include "TH1D.h"
#include "TH2D.h"
#include "TNtuple.h"

#include "AliFemtoCorrFctn.h"

#include "AliAODInputHandler.h"
#include "AliAnalysisManager.h"

class myAliFemtoKStarCorrFctn2D : public AliFemtoCorrFctn {
public:
  myAliFemtoKStarCorrFctn2D(const char* title, const int& nbinsKStar, const float& KStarLo, const float& KStarHi, const int& nbinsY, const float& YLo, const float& YHi);
  myAliFemtoKStarCorrFctn2D(const myAliFemtoKStarCorrFctn2D& aCorrFctn);
  virtual ~myAliFemtoKStarCorrFctn2D();

  myAliFemtoKStarCorrFctn2D& operator=(const myAliFemtoKStarCorrFctn2D& aCorrFctn);

  virtual AliFemtoString Report();
  virtual void AddRealPair(AliFemtoPair* aPair);
  virtual void AddMixedPair(AliFemtoPair* aPair);

  virtual void Finish();

  TH2D* NumeratorKStarOut();
  TH2D* NumeratorKStarSide();
  TH2D* NumeratorKStarLong();

  TH2D* DenominatorKStarOut();
  TH2D* DenominatorKStarSide();
  TH2D* DenominatorKStarLong();

  virtual TList* GetOutputList();
  void Write();

private:
  TH2D* fNumeratorKStarOut;          // numerator - real pairs, will be binned in +- values of k*out
  TH2D* fNumeratorKStarSide;          // numerator - real pairs, will be binned in +- values of k*side
  TH2D* fNumeratorKStarLong;          // numerator - real pairs, will be binned in +- values of k*long


  TH2D* fDenominatorKStarOut;        // denominator - mixed pairs, will be binned in +- values of k*out
  TH2D* fDenominatorKStarSide;        // denominator - mixed pairs, will be binned in +- values of k*side
  TH2D* fDenominatorKStarLong;        // denominator - mixed pairs, will be binned in +- values of k*long

#ifdef __ROOT__
  ClassDef(myAliFemtoKStarCorrFctn2D, 1)
#endif
};

inline  TH2D* myAliFemtoKStarCorrFctn2D::NumeratorKStarOut(){return fNumeratorKStarOut;}
inline  TH2D* myAliFemtoKStarCorrFctn2D::NumeratorKStarSide(){return fNumeratorKStarSide;}
inline  TH2D* myAliFemtoKStarCorrFctn2D::NumeratorKStarLong(){return fNumeratorKStarLong;}


inline  TH2D* myAliFemtoKStarCorrFctn2D::DenominatorKStarOut(){return fDenominatorKStarOut;}
inline  TH2D* myAliFemtoKStarCorrFctn2D::DenominatorKStarSide(){return fDenominatorKStarSide;}
inline  TH2D* myAliFemtoKStarCorrFctn2D::DenominatorKStarLong(){return fDenominatorKStarLong;}


#endif
