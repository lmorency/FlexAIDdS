#include "fileio.h"
#include "flexaid_exception.h"
#include <string>

void Terminate(int status){
  throw FlexAIDException("FlexAID terminated with exit code " + std::to_string(status), status);
}

int OpenFile_B(char* filename, const char* mode, FILE **f){
  *f = fopen(filename,mode);
  if (*f == NULL){
    fprintf(stderr, "ERROR: Could not open file %s.\n", filename);
    return 0;
  }
  return 1;
}

void CloseFile_B(FILE **f, const char* mode){
  (void)mode;
  fclose(*f);
}
