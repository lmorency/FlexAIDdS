#include "fileio.h"

void Terminate(int status){
  printf("Exit code: %d\n",status);
  exit(status);
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
