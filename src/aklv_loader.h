#ifndef AKLV_LOADER_H
#define AKLV_LOADER_H

#include "aklv_index.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct AklvLoader AklvLoader;

typedef struct {
    AklvFile *file;
    char *path;
    char error[512];
    int status;
} AklvLoaderEvent;

AklvLoader *aklv_loader_create(uint32_t worker_count);
void aklv_loader_destroy(AklvLoader *loader);
bool aklv_loader_enqueue(AklvLoader *loader, const char *path);
bool aklv_loader_poll(AklvLoader *loader, AklvLoaderEvent *event_out);
void aklv_loader_event_destroy(AklvLoaderEvent *event);

#endif
