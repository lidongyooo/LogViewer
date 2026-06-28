#ifndef AKLV_PLATFORM_H
#define AKLV_PLATFORM_H

#include <stddef.h>

char **aklv_platform_open_files(size_t *count_out);
void aklv_platform_free_open_files(char **paths, size_t count);

#endif
