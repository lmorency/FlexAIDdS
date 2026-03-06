#ifndef FILEIO_H
#define FILEIO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PATH__ 255
#define MAX_NUM_FILES 15

// File handling functions
int    OpenFile_B(char* filename, const char* mode, FILE **f);
void   CloseFile_B(FILE **f, const char* mode);

// Program termination
void   Terminate(int status);

#endif // FILEIO_H
