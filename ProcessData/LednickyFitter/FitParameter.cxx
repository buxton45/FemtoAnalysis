///////////////////////////////////////////////////////////////////////////
// FitParameter:                                                         //
///////////////////////////////////////////////////////////////////////////


#include "FitParameter.h"

#ifdef __ROOT__
ClassImp(FitParameter)
#endif




//________________________________________________________________________________________________________________
//****************************************************************************************************************
//________________________________________________________________________________________________________________


//________________________________________________________________________________________________________________
FitParameter::FitParameter(ParameterType aParamType, double aStartValue, bool aIsFixed, double aLowerParamBound, double aUpperParamBound, double aStepSize):

  fParamType(aParamType),
  fParamName(cParameterNames[fParamType]),
  fStartValue(aStartValue),
  fIsFixed(aIsFixed),
  fIsSharedLocal(false),
  fIsSharedGlobal(false),
  fLowerParamBound(aLowerParamBound),
  fUpperParamBound(aUpperParamBound),
  fStepSize(aStepSize),
  fFitValue(0),
  fFitValueError(0),
  fSharedWithLocal(0),
  fSharedWithGlobal(0),
  fMinuitParamNumber(-1),
  fOwnerInfo(),
  fFitInfo("")


{


}



//________________________________________________________________________________________________________________
FitParameter::~FitParameter()
{
  cout << "FitParameter object is being deleted!!!" << endl;
}


//________________________________________________________________________________________________________________
void FitParameter::SetFixedToValue(double aValue)
{
  fStartValue = aValue;
  fIsFixed = true;
}


//________________________________________________________________________________________________________________
void FitParameter::SetSharedLocal(bool aIsShared, int aSharedAnalysis)
{
  std::sort(fSharedWithLocal.begin(), fSharedWithLocal.end());

  if(aIsShared == false)
  {
    cout << "WARNING!!!!!  fIsSharedLocal is being set to false" << endl;
    if(fSharedWithLocal.size() != 0)
    {
      cout << "FURTHER WARNING!!! fSharedWithLocal.size() != 0, and will be deleted!!!!!" << endl;
      cout << "Do you really wish to continue?  Enter 0 (no) or 1(yes):  " << endl;
      int tResponse;
      cin >> tResponse;
      if(!tResponse)
      {
        cout << "fIsSharedLocal will continue to be true, and fSharedWithLocal will not be deleted" << endl;
        fIsSharedLocal = true;
      }
      else
      {
        cout << "fIsSharedLocal will be set to FALSE, and fSharedWithLocal WILL BE DELETED!" << endl;
        fIsSharedLocal = false;
        fSharedWithLocal.clear();
      }
    }
    else{fIsSharedLocal = false;}
  }

  else
  {
    fIsSharedLocal = true;

    bool tAlreadyShared = false;
    for(unsigned int i=0; i<fSharedWithLocal.size(); i++)
    {
      if(fSharedWithLocal[i] == aSharedAnalysis) {tAlreadyShared = true;}
    }
    if(!tAlreadyShared) {fSharedWithLocal.push_back(aSharedAnalysis);}

  }

  std::sort(fSharedWithLocal.begin(), fSharedWithLocal.end());
}


//________________________________________________________________________________________________________________
void FitParameter::SetSharedLocal(bool aIsShared, vector<int> &aSharedAnalyses)
{
  std::sort(fSharedWithLocal.begin(), fSharedWithLocal.end());

  if(aIsShared == false)
  {
    cout << "WARNING!!!!!  fIsSharedLocal is being set to false" << endl;
    if(fSharedWithLocal.size() != 0)
    {
      cout << "FURTHER WARNING!!! fSharedWithLocal.size() != 0, and will be deleted!!!!!" << endl;
      cout << "Do you really wish to continue?  Enter 0 (no) or 1(yes):  " << endl;
      int tResponse;
      cin >> tResponse;
      if(!tResponse)
      {
        cout << "fIsSharedLocal will continue to be true, and fSharedWithLocal will not be deleted" << endl;
        fIsSharedLocal = true;
      }
      else
      {
        cout << "fIsSharedLocal will be set to FALSE, and fSharedWithLocal WILL BE DELETED!" << endl;
        fIsSharedLocal = false;
        fSharedWithLocal.clear();
      }
    }
    else{fIsSharedLocal = false;}
  }

  else
  {
    fIsSharedLocal = true;

    for(unsigned int i=0; i<aSharedAnalyses.size(); i++)
    {
      bool tAlreadyShared = false;
      for(unsigned int j=0; j<fSharedWithLocal.size(); j++)
      {
        if(aSharedAnalyses[i] == fSharedWithLocal[j]) {tAlreadyShared = true;}
      }
      if(!tAlreadyShared) {fSharedWithLocal.push_back(aSharedAnalyses[i]);}
    }
  }

  std::sort(fSharedWithLocal.begin(), fSharedWithLocal.end());

}



//________________________________________________________________________________________________________________
vector<int> FitParameter::GetSharedWithLocal() 
{
  std::sort(fSharedWithLocal.begin(), fSharedWithLocal.end());
  return fSharedWithLocal;
}

//________________________________________________________________________________________________________________
void FitParameter::SetSharedGlobal(bool aIsShared, int aSharedAnalysis)
{
  std::sort(fSharedWithGlobal.begin(), fSharedWithGlobal.end());

  if(aIsShared == false)
  {
    cout << "WARNING!!!!!  fIsSharedGlobal is being set to false" << endl;
    if(fSharedWithGlobal.size() != 0)
    {
      cout << "FURTHER WARNING!!! fSharedWithGlobal.size() != 0, and will be deleted!!!!!" << endl;
      cout << "Do you really wish to continue?  Enter 0 (no) or 1(yes):  " << endl;
      int tResponse;
      cin >> tResponse;
      if(!tResponse)
      {
        cout << "fIsSharedGlobal will continue to be true, and fSharedWithGlobal will not be deleted" << endl;
        fIsSharedGlobal = true;
      }
      else
      {
        cout << "fIsSharedGlobal will be set to FALSE, and fSharedWithGlobal WILL BE DELETED!" << endl;
        fIsSharedGlobal = false;
        fSharedWithGlobal.clear();
      }
    }
    else{fIsSharedGlobal = false;}
  }

  else
  {
    fIsSharedGlobal = true;

    bool tAlreadyShared = false;
    for(unsigned int i=0; i<fSharedWithGlobal.size(); i++)
    {
      if(fSharedWithGlobal[i] == aSharedAnalysis) {tAlreadyShared = true;}
    }
    if(!tAlreadyShared) {fSharedWithGlobal.push_back(aSharedAnalysis);}

  }

  std::sort(fSharedWithGlobal.begin(), fSharedWithGlobal.end());
}


//________________________________________________________________________________________________________________
void FitParameter::SetSharedGlobal(bool aIsShared, const vector<int> &aSharedAnalyses)
{
  std::sort(fSharedWithGlobal.begin(), fSharedWithGlobal.end());

  if(aIsShared == false)
  {
    cout << "WARNING!!!!!  fIsSharedGlobal is being set to false" << endl;
    if(fSharedWithGlobal.size() != 0)
    {
      cout << "FURTHER WARNING!!! fSharedWithGlobal.size() != 0, and will be deleted!!!!!" << endl;
      cout << "Do you really wish to continue?  Enter 0 (no) or 1(yes):  " << endl;
      int tResponse;
      cin >> tResponse;
      if(!tResponse)
      {
        cout << "fIsSharedGlobal will continue to be true, and fSharedWithGlobal will not be deleted" << endl;
        fIsSharedGlobal = true;
      }
      else
      {
        cout << "fIsSharedGlobal will be set to FALSE, and fSharedWithGlobal WILL BE DELETED!" << endl;
        fIsSharedGlobal = false;
        fSharedWithGlobal.clear();
      }
    }
    else{fIsSharedGlobal = false;}
  }

  else
  {
    fIsSharedGlobal = true;

    for(unsigned int i=0; i<aSharedAnalyses.size(); i++)
    {
      bool tAlreadyShared = false;
      for(unsigned int j=0; j<fSharedWithGlobal.size(); j++)
      {
        if(aSharedAnalyses[i] == fSharedWithGlobal[j]) {tAlreadyShared = true;}
      }
      if(!tAlreadyShared) {fSharedWithGlobal.push_back(aSharedAnalyses[i]);}
    }
  }

  std::sort(fSharedWithGlobal.begin(), fSharedWithGlobal.end());

}



//________________________________________________________________________________________________________________
vector<int> FitParameter::GetSharedWithGlobal() 
{
  std::sort(fSharedWithGlobal.begin(), fSharedWithGlobal.end());
  return fSharedWithGlobal;
}


//________________________________________________________________________________________________________________
void FitParameter::SetAttributes(double aStartValue, bool aIsFixed, double aLowerParamBound, double aUpperParamBound, double aStepSize)
{
  fStartValue = aStartValue;
  fIsFixed = aIsFixed;
  fLowerParamBound = aLowerParamBound;
  fUpperParamBound = aUpperParamBound;
  if(aStepSize > 0.) fStepSize = aStepSize;
}

//________________________________________________________________________________________________________________
void FitParameter::SetOwnerInfo(AnalysisType aAnType, CentralityType aCentType, BFieldType aBFieldType)
{
  fOwnerInfo = ParamOwnerInfo(aAnType, aCentType, aBFieldType);
}

//________________________________________________________________________________________________________________
TString FitParameter::GetOwnerName()
{
  return TString::Format("%s%s%s", cAnalysisBaseTags[fOwnerInfo.analysisType], 
                                   cCentralityTags[fOwnerInfo.centralityType], 
                                   cBFieldTags[fOwnerInfo.bFieldType]);
  //ex. LamK0_0010_FemtoPlus
}


