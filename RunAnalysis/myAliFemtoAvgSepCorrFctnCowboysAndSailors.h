/*
 *  myAliFemtoAvgSepCorrFctnCowboysAndSailors.h
 *
 */

#ifndef MYALIFEMTOAVGSEPCORRFCTNCOWBOYSANDSAILORS_H
#define MYALIFEMTOAVGSEPCORRFCTNCOWBOYSANDSAILORS_H

#include "TH1D.h"
#include "TH2D.h"
#include "TNtuple.h"

#include "AliFemtoCorrFctn.h"

#include "AliAODInputHandler.h"
#include "AliAnalysisManager.h"

class myAliFemtoAvgSepCorrFctnCowboysAndSailors : public AliFemtoCorrFctn {
public:
  enum PairType {kPosPos = 0, kPosNeg = 1, kNegPos = 2, kNegNeg = 3, kTrackPos = 4, kTrackNeg = 5, kTrackTrack = 6};
  enum V0Type {kLambda = 0, kAntiLambda = 1, kK0Short = 2};
  myAliFemtoAvgSepCorrFctnCowboysAndSailors(const char* title, const int& nbinsX, const float& XLo, const float& XHi, const int& nbinsY, const float& AvgSepLo, const float& AvgSepHi);
  myAliFemtoAvgSepCorrFctnCowboysAndSailors(const myAliFemtoAvgSepCorrFctnCowboysAndSailors& aCorrFctn);
  virtual ~myAliFemtoAvgSepCorrFctnCowboysAndSailors();

  myAliFemtoAvgSepCorrFctnCowboysAndSailors& operator=(const myAliFemtoAvgSepCorrFctnCowboysAndSailors& aCorrFctn);

  virtual AliFemtoString Report();
  virtual void AddRealPair(AliFemtoPair* aPair);
  virtual void AddMixedPair(AliFemtoPair* aPair);

  virtual void Finish();

  //--used when both are V0s
  TH2D* NumeratorPosPos();
  TH2D* NumeratorPosNeg();
  TH2D* NumeratorNegPos();
  TH2D* NumeratorNegNeg();
  TH2D* DenominatorPosPos();
  TH2D* DenominatorPosNeg();
  TH2D* DenominatorNegPos();
  TH2D* DenominatorNegNeg();


  //--used when one V0 and one track
  TH2D* NumeratorTrackPos();
  TH2D* NumeratorTrackNeg();
  TH2D* DenominatorTrackPos();
  TH2D* DenominatorTrackNeg();

 //--used when both are tracks
  TH2D* NumeratorTrackTrack();
  TH2D* DenominatorTrackTrack();

  virtual TList* GetOutputList();
  void Write();

  void FillAvgSepHisto(AliFemtoPair* pair, PairType fType, TH2D* aHisto, double aWeight);

private:
  //--used when both are V0s
  TH2D* fNumeratorPosPos;          // numerator ++
  TH2D* fNumeratorPosNeg;          // numerator +-
  TH2D* fNumeratorNegPos;          // numerator -+
  TH2D* fNumeratorNegNeg;          // numerator --
  TH2D* fDenominatorPosPos;          // denominator ++
  TH2D* fDenominatorPosNeg;          // denominator +-
  TH2D* fDenominatorNegPos;          // denominator -+
  TH2D* fDenominatorNegNeg;          // denominator --

  //--used when one V0 and one track
  TH2D* fNumeratorTrackPos;          // numerator Track +
  TH2D* fNumeratorTrackNeg;          // numerator Track -
  TH2D* fDenominatorTrackPos;          // denominator Track +
  TH2D* fDenominatorTrackNeg;          // denominator Track -

 //--used when both are tracks
  TH2D* fNumeratorTrackTrack;          // numerator Track Track
  TH2D* fDenominatorTrackTrack;          // denominator Track Track


#ifdef __ROOT__
  ClassDef(myAliFemtoAvgSepCorrFctnCowboysAndSailors, 1)
#endif
};

inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::NumeratorPosPos(){return fNumeratorPosPos;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::NumeratorPosNeg(){return fNumeratorPosNeg;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::NumeratorNegPos(){return fNumeratorNegPos;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::NumeratorNegNeg(){return fNumeratorNegNeg;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::DenominatorPosPos(){return fDenominatorPosPos;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::DenominatorPosNeg(){return fDenominatorPosNeg;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::DenominatorNegPos(){return fDenominatorNegPos;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::DenominatorNegNeg(){return fDenominatorNegNeg;}

inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::NumeratorTrackPos(){return fNumeratorTrackPos;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::NumeratorTrackNeg(){return fNumeratorTrackNeg;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::DenominatorTrackPos(){return fDenominatorTrackPos;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::DenominatorTrackNeg(){return fDenominatorTrackNeg;}

inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::NumeratorTrackTrack(){return fNumeratorTrackTrack;}
inline  TH2D* myAliFemtoAvgSepCorrFctnCowboysAndSailors::DenominatorTrackTrack(){return fDenominatorTrackTrack;}

#endif
