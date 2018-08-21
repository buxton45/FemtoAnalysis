/* FitValuesLatexTableHelperWriterwSysErrs.cxx */

#include "FitValuesLatexTableHelperWriterwSysErrs.h"

#ifdef __ROOT__
ClassImp(FitValuesLatexTableHelperWriterwSysErrs)
#endif




//________________________________________________________________________________________________________________
FitValuesLatexTableHelperWriterwSysErrs::FitValuesLatexTableHelperWriterwSysErrs() : 
  FitValuesWriterwSysErrs(),
  fResType(kInclude3Residuals)
{

}





//________________________________________________________________________________________________________________
FitValuesLatexTableHelperWriterwSysErrs::~FitValuesLatexTableHelperWriterwSysErrs()
{
  //no-op
}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriterwSysErrs::WriteLatexTableHelperEntry(ostream &aOut, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType)
{
  TString tFitInfoTString = FitValuesLatexTableHelperWriter::GetFitInfoTStringFromTwoLetterID(aTwoLetterID, aAnType, aResType);

  FitParameter *tPar_lam0010, *tPar_lam1030, *tPar_lam3050;
  FitParameter *tPar_R0010, *tPar_R1030, *tPar_R3050;
  FitParameter *tPar_Ref0, *tPar_Imf0, *tPar_d0;

  tPar_lam0010 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, tFitInfoTString, aAnType, k0010, kLambda);
  tPar_lam1030 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, tFitInfoTString, aAnType, k1030, kLambda);
  tPar_lam3050 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, tFitInfoTString, aAnType, k3050, kLambda);

  tPar_R0010 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, tFitInfoTString, aAnType, k0010, kRadius);
  tPar_R1030 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, tFitInfoTString, aAnType, k1030, kRadius);
  tPar_R3050 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, tFitInfoTString, aAnType, k3050, kRadius);

  tPar_Ref0 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, tFitInfoTString, aAnType, k0010, kRef0);
  tPar_Imf0 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, tFitInfoTString, aAnType, k0010, kImf0);
  tPar_d0 = GetFitParameterSys(aMasterFileLocation, aSystematicsFileLocation, tFitInfoTString, aAnType, k0010, kd0);


  //------------------------------------------------------------------
  TString tAnalysisTag;
  if(aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP) tAnalysisTag = cAnalysisBaseTags[aAnType];
  else if(aAnType==kLamK0) tAnalysisTag = TString("LamKs");
  else if(aAnType==kALamK0) tAnalysisTag = TString("ALamKs");
  else assert(0);

  aOut << TString::Format("\\newarray\\%s%s", aTwoLetterID.Data(), tAnalysisTag.Data()) << endl;
  aOut << TString::Format("\\readarray{%s%s}{", aTwoLetterID.Data(), tAnalysisTag.Data()) << endl;

  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f & ", tPar_lam0010->GetFitValue(), tPar_lam0010->GetFitValueError(), tPar_lam0010->GetFitValueSysError()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f & ", tPar_R0010->GetFitValue(), tPar_R0010->GetFitValueError(), tPar_R0010->GetFitValueSysError()) << endl;

  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f & ", tPar_lam1030->GetFitValue(), tPar_lam1030->GetFitValueError(), tPar_lam1030->GetFitValueSysError()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f & ", tPar_R1030->GetFitValue(), tPar_R1030->GetFitValueError(), tPar_R1030->GetFitValueSysError()) << endl;

  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f & ", tPar_lam3050->GetFitValue(), tPar_lam3050->GetFitValueError(), tPar_lam3050->GetFitValueSysError()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f & ", tPar_R3050->GetFitValue(), tPar_R3050->GetFitValueError(), tPar_R3050->GetFitValueSysError()) << endl;

  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f & ", tPar_Ref0->GetFitValue(), tPar_Ref0->GetFitValueError(), tPar_Ref0->GetFitValueSysError()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f & ", tPar_Imf0->GetFitValue(), tPar_Imf0->GetFitValueError(), tPar_Imf0->GetFitValueSysError()) << endl;
  aOut << TString::Format("                       %0.2f & %0.2f & %0.2f}", tPar_d0->GetFitValue(), tPar_d0->GetFitValueError(), tPar_d0->GetFitValueSysError()) << endl << endl;


}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriterwSysErrs::WriteLatexTableHelperSection(ostream &aOut, TString aMasterFileLocation, TString aSystematicsFileLocation, TString aTwoLetterID, AnalysisType aAnType, IncludeResidualsType aResType)
{
  if(aAnType==kLamKchP || aAnType==kALamKchM || aAnType==kLamKchM || aAnType==kALamKchP)
  {
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aSystematicsFileLocation, aTwoLetterID, kLamKchP,  aResType);
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aSystematicsFileLocation, aTwoLetterID, kALamKchM, aResType);
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aSystematicsFileLocation, aTwoLetterID, kLamKchM,  aResType);
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aSystematicsFileLocation, aTwoLetterID, kALamKchP, aResType);
  }
  else if(aAnType==kLamK0 || aAnType==kALamK0)
  {
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aSystematicsFileLocation, aTwoLetterID, kLamK0,   aResType);
    WriteLatexTableHelperEntry(aOut, aMasterFileLocation, aSystematicsFileLocation, aTwoLetterID, kALamK0,  aResType);
  }
  else assert(0);
}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriterwSysErrs::WriteLatexTableHelperHeader(ostream &aOut)
{
  aOut << "\%NOTE: Arrays seem to only be able to be named with alphabetical characters" << endl;
  aOut << "\%      No '_' or event numbers" << endl;
  aOut << endl;
  aOut << "\% ----------Shorthand definitions----------" << endl;
  aOut << "\% A = FitGen_NoLamShare" << endl;
  aOut << "\% B = FitGen_ShareLamConj" << endl;
  aOut << endl;
  aOut << "\% C = Dualie_NoLamShare" << endl;
  aOut << "\% D = Dualie_ShareLamConj" << endl;
  aOut << "\% E = Dualie_ShareSingleLam" << endl;
  aOut << endl;
  aOut << "\%   ------------------" << endl;
  aOut << "\% a = PolyNFB" << endl;
  aOut << "\% b = LinrNFB" << endl;
  aOut << "\% c = LinrNFB_StavCf" << endl;
  aOut << "\% d = NoNFB_StavCf" << endl;
  aOut << endl;
  aOut << "\% ----------How arrays are stored----------" << endl;
  aOut << "\% aArray = {lam0010(1), lam0010StatErr(2), lam0010SysErr(3)" << endl;
  aOut << "\%           R0010(4),   R0010StatErr(5),   R0010SysErr(6)" << endl;

  aOut << "\%           lam1030(7), lam1030StatErr(8), lam1030SysErr(9)" << endl;
  aOut << "\%           R1030(10),  R1030StatErr(11),  R1030SysErr(12)" << endl;

  aOut << "\%           lam3050(13), lam3050StatErr(14), lam3050SysErr(15)" << endl;
  aOut << "\%           R3050(16),   R3050StatErr(17),   R3050SysErr(18)" << endl;

  aOut << "\%           ReF0(19),   ReF0StatErr(20),   ReF0SysErr(21)" << endl;
  aOut << "\%           ImF0(22),   ImF0StatErr(23),   ImF0SysErr(24)" << endl;
  aOut << "\%           d0(25),     d0StatErr(26),     d0SysErr(27)}" << endl;

  aOut << endl;
  aOut << endl;
  aOut << "\%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << endl;
}


//________________________________________________________________________________________________________________
void FitValuesLatexTableHelperWriterwSysErrs::WriteLatexTableHelper(TString aHelperBaseLocation, TString aMasterFileLocation, TString aSystematicsFileLocation, AnalysisType aAnType, IncludeResidualsType aResType)
{
  assert(aAnType==kLamKchP || aAnType==kLamK0);
  TString tPairDesc = "";
  if     (aAnType==kLamKchP) tPairDesc = TString("_cLamcKch");
  else if(aAnType==kLamK0) tPairDesc = TString("_cLamK0");

  TString tHelperLocation = TString::Format("%s%s%s.tex", aHelperBaseLocation.Data(), tPairDesc.Data(), cIncludeResidualsTypeTags[aResType]);

  std::ofstream tOut;
  tOut.open(tHelperLocation);
  WriteLatexTableHelperHeader(tOut);

  vector<vector<TString> > tAllTwoLetterID = {{"Aa", "Ba", "Ca", "Da", "Ea"},
                                              {"Ab", "Bb", "Cb", "Db", "Eb"},
                                              {"Ac", "Bc", "Cc", "Dc", "Ec"}};
  if(aAnType==kLamK0 || aAnType==kALamK0) tAllTwoLetterID = vector<vector<TString> > {{"Aa"}, {"Ab"}, {"Ac"}};
  //-------------------------------------------
  TString tTwoLetterID = "";
  for(unsigned int i=0; i<tAllTwoLetterID.size(); i++)
  {
    for(unsigned int j=0; j<tAllTwoLetterID[i].size(); j++)
    {
      tTwoLetterID = tAllTwoLetterID[i][j];
      tOut << TString::Format("%% --------------- %s = %s%s ---------------", tTwoLetterID.Data(), FitValuesLatexTableHelperWriter::GetLatexTableOverallLabel(tTwoLetterID).Data(), cIncludeResidualsTypeTags[aResType]) << endl;
      WriteLatexTableHelperSection(tOut, aMasterFileLocation, aSystematicsFileLocation, tTwoLetterID, aAnType, aResType);
    }
    tOut << "\%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << endl;
  }

  tOut.close();
}

