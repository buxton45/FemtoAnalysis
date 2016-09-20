////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  myAliFemtoModelCorrFctnKStar                                              //
//                                                                            //
//  Author: Jesse Buxton jesse.thomas.buxton@cern.ch                          //
////////////////////////////////////////////////////////////////////////////////

#ifndef MYALIFEMTOMODELCORRFCTNKSTAR_H
#define MYALIFEMTOMODELCORRFCTNKSTAR_H

#include "AliFemtoCorrFctn.h"
#include "AliFemtoModelCorrFctn.h"
#include "AliFemtoModelHiddenInfo.h"
#include "AliFemtoPair.h"
#include "AliFemtoModelManager.h"

#include <limits>


class myAliFemtoModelCorrFctnKStar : public AliFemtoModelCorrFctn 
{
public:

  enum AnalysisType {kLamK0=0, kALamK0=1, kLamKchP=2, kALamKchP=3, kLamKchM=4, kALamKchM=5, kXiKchP=6, kAXiKchP=7, kXiKchM=8, kAXiKchM=9, kLamLam=10, kALamALam=11, kLamALam=12, kLamPiP=13, kALamPiP=14, kLamPiM=15, kALamPiM=16};

  myAliFemtoModelCorrFctnKStar();
  myAliFemtoModelCorrFctnKStar(const char *title, int aNbins, double aKStarLo, double aKStarHi);
  virtual ~myAliFemtoModelCorrFctnKStar();

  myAliFemtoModelCorrFctnKStar(const myAliFemtoModelCorrFctnKStar& aCorrFctn);
  myAliFemtoModelCorrFctnKStar& operator=(const myAliFemtoModelCorrFctnKStar& aCorrFctn);

  void SetPIDs(AnalysisType aAnalysisType);
  void SetAnalysisType(int aType);
  bool TestPIDs(int aPID1, int aPID2);

  double GetKStarTrue(AliFemtoPair* aPair);

  virtual void AddRealPair(AliFemtoPair* aPair);
  virtual void AddMixedPair(AliFemtoPair* aPair);

  virtual TList* GetOutputList();
  virtual void Write();

  //inline
  void SetRemoveMisidentified(bool aSet);


protected:

  bool fRemoveMisidentified;
  AnalysisType fAnalysisType;

  int fPart1ID, fPart2ID;

  //----- Defined in AliFemtoModelCorrFctn.h -----//
  /*
   *  TH1D *fNumeratorTrue;      // Numerator made with RECONSTRUCTED k* of pairs from SAME event
                                 //   Weight from fManager, if exists (otherwise weight = 1)
   *  TH1D *fNumeratorFake;      // Numerator made with RECONSTRUCTED k* of pairs from MIXED events
                                 //   Weight from fManager, if exists (otherwise weight = 1)
   *  TH1D *fDenominator;        // Denominator made with RECONSTRUCTED k* of pairs from MIXED events
                                 //   Weight = 1 always 

   *  TH1D *fNumeratorTrueIdeal; // Numerator made with TRUE k* of pairs from SAME event
                                 //   Weight from fManager, if exists (otherwise weight = 1)
   *  TH1D *fNumeratorFakeIdeal; // Numerator made with TRUE k* of pairs from MIXED events
                                 //   Weight from fManager, if exists (otherwise weight = 1)
   *  TH1D *fDenominatorIdeal;   // Denominator made with TRUE k* of pairs from MIXED events
                                 //   Weight = 1 always
  */
  //----- END: Defined in AliFemtoModelCorrFctn.h -----//


  TH1D *fNumTrueUnitWeights;      // Numerator made with RECONSTRUCTED k* of pairs from SAME event with unit weights
  TH1D *fNumTrueIdealUnitWeights; // Numerator made with TRUE k* of pairs from SAME event with unit weights

  TH2D *fKTrueKRecSame;           // 2D histogram of k*_{True} vs k*_{Reconstructed} of pairs from SAME event
  TH2D *fKTrueKRecMixed;          // 2D histogram of k*_{True} vs k*_{Reconstructed} of pairs from MIXED events

  TH2D *fKTrueKRecRotSame;        // fKTrueVsKRecSame rotated by 45 degrees
  TH2D *fKTrueKRecRotMixed;       // fKTrueVsKRecMixed rotated by 45 degrees


#ifdef __ROOT__
  /// \cond CLASSIMP
  ClassDef(myAliFemtoModelCorrFctnKStar, 1);
  /// \endcond
#endif

};

inline void myAliFemtoModelCorrFctnKStar::SetRemoveMisidentified(bool aSet) {fRemoveMisidentified=aSet;}


#endif
