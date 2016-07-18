///////////////////////////////////////////////////////////////////////////
// Square:                                                               //
///////////////////////////////////////////////////////////////////////////

#ifndef SQUARE_H
#define SQUARE_H

#include <stdio.h>
#include <iostream>
#include "timer.h"
#include "utils.h"


class Square {

public:

  Square();
  virtual ~Square();

  void RunSquare(float * h_out, float * h_in, const int ARRAY_SIZE);

private:


};

#endif
