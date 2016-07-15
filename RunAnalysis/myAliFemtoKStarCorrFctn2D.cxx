///////////////////////////////////////////////////////////////////////////
//                                                                       //
// myAliFemtoKStarCorrFctn2D:                                            //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#include "myAliFemtoKStarCorrFctn2D.h"
//#include "AliFemtoHisto.h"
#include <cstdio>

#ifdef __ROOT__
ClassImp(myAliFemtoKStarCorrFctn2D)
#endif

//____________________________
myAliFemtoKStarCorrFctn2D::myAliFemtoKStarCorrFctn2D(const char* title, const int& nbinsKStar, const float& KStarLo, const float& KStarHi, const int& nbinsY, const float& YLo, const float& YHi):
  fNumeratorKStarOut(0),
  fNumeratorKStarSide(0),
  fNumeratorKStarLong(0),

  fDenominatorKStarOut(0),
  fDenominatorKStarSide(0),
  fDenominatorKStarLong(0)

{
  char tAppendKStarOut[11] = "KStarOut";
  char tAppendKStarSide[11] = "KStarSide";
  char tAppendKStarLong[11] = "KStarLong";

  // set up numerators
  char tTitNumKStarOut[101] = "Num";
  strncat(tTitNumKStarOut,title, 100);
  strncat(tTitNumKStarOut,tAppendKStarOut,10);
  fNumeratorKStarOut = new TH2D(tTitNumKStarOut,title,nbinsKStar,KStarLo,KStarHi,nbinsY,YLo,YHi);

  char tTitNumKStarSide[101] = "Num";
  strncat(tTitNumKStarSide,title, 100);
  strncat(tTitNumKStarSide,tAppendKStarSide,10);
  fNumeratorKStarSide = new TH2D(tTitNumKStarSide,title,nbinsKStar,KStarLo,KStarHi,nbinsY,YLo,YHi);

  char tTitNumKStarLong[101] = "Num";
  strncat(tTitNumKStarLong,title, 100);
  strncat(tTitNumKStarLong,tAppendKStarLong,10);
  fNumeratorKStarLong = new TH2D(tTitNumKStarLong,title,nbinsKStar,KStarLo,KStarHi,nbinsY,YLo,YHi);


  // set up denominators
  char tTitDenKStarOut[101] = "Den";
  strncat(tTitDenKStarOut,title, 100);
  strncat(tTitDenKStarOut,tAppendKStarOut,10);
  fDenominatorKStarOut = new TH2D(tTitDenKStarOut,title,nbinsKStar,KStarLo,KStarHi,nbinsY,YLo,YHi);

  char tTitDenKStarSide[101] = "Den";
  strncat(tTitDenKStarSide,title, 100);
  strncat(tTitDenKStarSide,tAppendKStarSide,10);
  fDenominatorKStarSide = new TH2D(tTitDenKStarSide,title,nbinsKStar,KStarLo,KStarHi,nbinsY,YLo,YHi);

  char tTitDenKStarLong[101] = "Den";
  strncat(tTitDenKStarLong,title, 100);
  strncat(tTitDenKStarLong,tAppendKStarLong,10);
  fDenominatorKStarLong = new TH2D(tTitDenKStarLong,title,nbinsKStar,KStarLo,KStarHi,nbinsY,YLo,YHi);



  //-----call Sumw2()

  fNumeratorKStarOut->Sumw2();
  fNumeratorKStarSide->Sumw2();
  fNumeratorKStarLong->Sumw2();

  fDenominatorKStarOut->Sumw2();
  fDenominatorKStarSide->Sumw2();
  fDenominatorKStarLong->Sumw2();

}

//____________________________
myAliFemtoKStarCorrFctn2D::myAliFemtoKStarCorrFctn2D(const myAliFemtoKStarCorrFctn2D& aCorrFctn) :
  AliFemtoCorrFctn(),
  fNumeratorKStarOut(0),
  fNumeratorKStarSide(0),
  fNumeratorKStarLong(0),

  fDenominatorKStarOut(0),
  fDenominatorKStarSide(0),
  fDenominatorKStarLong(0)

{
  // copy constructor
  fNumeratorKStarOut = new TH2D(*aCorrFctn.fNumeratorKStarOut);
  fNumeratorKStarSide = new TH2D(*aCorrFctn.fNumeratorKStarSide);
  fNumeratorKStarLong = new TH2D(*aCorrFctn.fNumeratorKStarLong);


  fDenominatorKStarOut = new TH2D(*aCorrFctn.fDenominatorKStarOut);
  fDenominatorKStarSide = new TH2D(*aCorrFctn.fDenominatorKStarSide);
  fDenominatorKStarLong = new TH2D(*aCorrFctn.fDenominatorKStarLong);

}
//____________________________
myAliFemtoKStarCorrFctn2D::~myAliFemtoKStarCorrFctn2D(){
  // destructor
  delete fNumeratorKStarOut;
  delete fNumeratorKStarSide;
  delete fNumeratorKStarLong;

  delete fDenominatorKStarOut;
  delete fDenominatorKStarSide;
  delete fDenominatorKStarLong;

}
//_________________________
myAliFemtoKStarCorrFctn2D& myAliFemtoKStarCorrFctn2D::operator=(const myAliFemtoKStarCorrFctn2D& aCorrFctn)
{
  // assignment operator
  if (this == &aCorrFctn)
    return *this;

  if (fNumeratorKStarOut) delete fNumeratorKStarOut;
  fNumeratorKStarOut = new TH2D(*aCorrFctn.fNumeratorKStarOut);

  if (fNumeratorKStarSide) delete fNumeratorKStarSide;
  fNumeratorKStarSide = new TH2D(*aCorrFctn.fNumeratorKStarSide);

  if (fNumeratorKStarLong) delete fNumeratorKStarLong;
  fNumeratorKStarLong = new TH2D(*aCorrFctn.fNumeratorKStarLong);


  if (fDenominatorKStarOut) delete fDenominatorKStarOut;
  fDenominatorKStarOut = new TH2D(*aCorrFctn.fDenominatorKStarOut);

  if (fDenominatorKStarSide) delete fDenominatorKStarSide;
  fDenominatorKStarSide = new TH2D(*aCorrFctn.fDenominatorKStarSide);

  if (fDenominatorKStarLong) delete fDenominatorKStarLong;
  fDenominatorKStarLong = new TH2D(*aCorrFctn.fDenominatorKStarLong);

  return *this;
}

//_________________________
void myAliFemtoKStarCorrFctn2D::Finish(){
  // here is where we should normalize, fit, etc...
  // we should NOT Draw() the histos (as I had done it below),
  // since we want to insulate ourselves from root at this level
  // of the code.  Do it instead at root command line with browser.
  //  fNumerator->Draw();
  //fDenominator->Draw();

}

//____________________________
AliFemtoString myAliFemtoKStarCorrFctn2D::Report(){
  // construct report
  string stemp = "KStar Correlation Function Report:\n";
  char ctemp[100];
  snprintf(ctemp , 100, "Number of entries in NumeratorKStarOut:\t%E\n",fNumeratorKStarOut->GetEntries());
  stemp += ctemp;
  snprintf(ctemp , 100, "Number of entries in NumeratorKStarSide:\t%E\n",fNumeratorKStarSide->GetEntries());
  stemp += ctemp;
  snprintf(ctemp , 100, "Number of entries in NumeratorKStarLong:\t%E\n",fNumeratorKStarLong->GetEntries());
  stemp += ctemp;


  snprintf(ctemp , 100, "Number of entries in DenominatorKStarOut:\t%E\n",fDenominatorKStarOut->GetEntries());
  stemp += ctemp;
  snprintf(ctemp , 100, "Number of entries in DenominatorKStarSide:\t%E\n",fDenominatorKStarSide->GetEntries());
  stemp += ctemp;
  snprintf(ctemp , 100, "Number of entries in DenominatorKStarLong:\t%E\n",fDenominatorKStarLong->GetEntries());
  stemp += ctemp;

  //  stemp += mCoulombWeight->Report();
  AliFemtoString returnThis = stemp;
  return returnThis;
}
//____________________________
void myAliFemtoKStarCorrFctn2D::AddRealPair(AliFemtoPair* pair){
  // add true pair
  if (fPairCut)
    if (!fPairCut->Pass(pair)) return;

  double tKStar = fabs(pair->KStar());   // note - qInv() will be negative for identical pairs...

  double tSign;

  if(pair->KStarOut() > 0.) {tSign = 1.;}
  else{tSign = -1.;}
  fNumeratorKStarOut->Fill(tKStar,tSign);

  if(pair->KStarSide() > 0.) {tSign = 1.;}
  else{tSign = -1.;}
  fNumeratorKStarSide->Fill(tKStar,tSign);

  if(pair->KStarLong() > 0.) {tSign = 1.;}
  else{tSign = -1.;}
  fNumeratorKStarLong->Fill(tKStar,tSign);

}

//____________________________
void myAliFemtoKStarCorrFctn2D::AddMixedPair(AliFemtoPair* pair){
  // add mixed (background) pair
  if (fPairCut)
    if (!fPairCut->Pass(pair)) return;

  double weight = 1.0;
  double tKStar = fabs(pair->KStar());   // note - qInv() will be negative for identical pairs...

  double tSign;

  if(pair->KStarOut() > 0.) {tSign = 1.;}
  else{tSign = -1.;}
  fDenominatorKStarOut->Fill(tKStar,tSign,weight);

  if(pair->KStarSide() > 0.) {tSign = 1.;}
  else{tSign = -1.;}
  fDenominatorKStarSide->Fill(tKStar,tSign,weight);

  if(pair->KStarLong() > 0.) {tSign = 1.;}
  else{tSign = -1.;}
  fDenominatorKStarLong->Fill(tKStar,tSign,weight);

}
//____________________________
void myAliFemtoKStarCorrFctn2D::Write(){
  // Write out neccessary objects
  fNumeratorKStarOut->Write();
  fNumeratorKStarSide->Write();
  fNumeratorKStarLong->Write();

  fDenominatorKStarOut->Write();
  fDenominatorKStarSide->Write();
  fDenominatorKStarLong->Write();

}
//______________________________
TList* myAliFemtoKStarCorrFctn2D::GetOutputList()
{
  // Prepare the list of objects to be written to the output
  TList *tOutputList = new TList();

  tOutputList->Add(fNumeratorKStarOut);
  tOutputList->Add(fNumeratorKStarSide);
  tOutputList->Add(fNumeratorKStarLong);


  tOutputList->Add(fDenominatorKStarOut);
  tOutputList->Add(fDenominatorKStarSide);
  tOutputList->Add(fDenominatorKStarLong);


  return tOutputList;
}
