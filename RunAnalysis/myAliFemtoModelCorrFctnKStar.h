////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  myAliFemtoModelCorrFctnKStar                                              //
//                                                                            //
//  Author: Jesse Buxton jesse.thomas.buxton@cern.ch                          //
////////////////////////////////////////////////////////////////////////////////

#ifndef ALIFEMTOMODELCORRFCTNKSTAR_H
#define ALIFEMTOMODELCORRFCTNKSTAR_H

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

  TH1D *fNumeratorUnitWeightTrue;
  TH1D *fNumeratorUnitWeightTrueIdeal;

  TH2D *fKTrueKRecSame;
  TH2D *fKTrueKRecMixed;

  TH2D *fKTrueKRecRotSame;
  TH2D *fKTrueKRecRotMixed;




#ifdef __ROOT__
  /// \cond CLASSIMP
  ClassDef(myAliFemtoModelCorrFctnKStar, 1);
  /// \endcond
#endif

};

inline void myAliFemtoModelCorrFctnKStar::SetRemoveMisidentified(bool aSet) {fRemoveMisidentified=aSet;}


#endif
