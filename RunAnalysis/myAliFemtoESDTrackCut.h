////////////////////////////////////////////////////////////////////////////
//                                                                        //
// myAliFemtoESDTrackCut:                                                 //
//  This will give me more control over the Kch track cuts                //
//  For now, the only this I want to change is the method IsKaonNSigma    //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef MYALIFEMTOESDTRACKCUT_H
#define MYALIFEMTOESDTRACKCUT_H

#include "AliFemtoESDTrackCut.h"

class myAliFemtoESDTrackCut : public AliFemtoESDTrackCut {

public:

  myAliFemtoESDTrackCut();
  myAliFemtoESDTrackCut(const myAliFemtoESDTrackCut& aCut);
  myAliFemtoESDTrackCut& operator=(const myAliFemtoESDTrackCut& aCut);
  virtual ~myAliFemtoESDTrackCut();

  virtual bool Pass(const AliFemtoTrack* aTrack);

  void SetPionRejection(bool aReject);

private:
  bool fPionRejection;  // 26/02/2016

  bool IsKaonNSigma(float mom, float nsigmaTPC, float nsigmaTOF);

  bool IsElectron(float nsigmaTPCE);  // 26/02/2016


#ifdef __ROOT__
  ClassDef(myAliFemtoESDTrackCut, 1)
#endif

};

inline void myAliFemtoESDTrackCut::SetPionRejection(bool aReject) {fPionRejection = aReject;}

#endif
