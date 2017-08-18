/* PIDMapping */

#ifndef PIDMAPPING_H
#define PIDMAPPING_H

#include "TString.h"

#include <vector>
#include <complex>
#include <iostream>
using namespace std;

#include "Types.h"

extern TString GetParticleName(int aPID);
extern bool IncludeAsPrimary(int aPID1, int aPID2);
extern bool PairAccountedForInResiduals(int aPID1, int aPID2);
extern bool IncludeInOthers(int aPID1, int aPID2);
//------------------------------------------

extern vector<int> cUniqueFathersPIDsFull;
extern vector<TString> cUniqueFathersNamesFull;

extern vector<int> cUniqueFathersPIDsIncludeAsPrimary;
extern vector<int> cUniqueFathersPIDsExcludeAsPrimary;

extern vector<int> cUniqueLamFathersPIDs;
extern vector<int> cUniqueALamFathersPIDs;
extern vector<int> cAllLambdaFathers;

extern vector<int> cUniqueK0ShortFathersPIDs;
extern vector<int> cAllK0ShortFathers;

extern vector<int> cUniqueKchPFathersPIDs;
extern vector<int> cUniqueKchMFathersPIDs;
extern vector<int> cAllKchFathers;

extern vector<int> cUniqueProtFathersPIDs;
extern vector<int> cUniqueAProtFathersPIDs;
extern vector<int> cAllProtonFathers;


//------------------------------------------


#endif
