#ifndef AKLV_SEARCH_H
#define AKLV_SEARCH_H

#include "aklv_index.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    AklvFile **items;
    uint32_t count;
    uint32_t active_index;
    uint64_t active_selected_line;
} AklvFileSnapshot;

typedef struct {
    uint32_t file_id;
    uint64_t line_no;
    uint64_t byte_offset;
} AklvSearchResult;

typedef struct {
    uint64_t generation;
    bool running;
    bool complete;
    double elapsed_sec;
    char query[256];
    AklvSearchResult *items;
    uint64_t count;
    uint64_t capacity;
} AklvSearchResults;

typedef struct AklvSearchService AklvSearchService;

void aklv_file_snapshot_destroy(AklvFileSnapshot *snapshot);

AklvSearchService *aklv_search_service_create(void);
void aklv_search_service_destroy(AklvSearchService *service);

void aklv_search_service_submit(AklvSearchService *service,
                                const char *query,
                                const AklvFileSnapshot *snapshot);
void aklv_search_service_cancel(AklvSearchService *service);

void aklv_search_results_init(AklvSearchResults *results);
void aklv_search_results_destroy(AklvSearchResults *results);
void aklv_search_results_swap_from_service(AklvSearchService *service, AklvSearchResults *target);

int aklv_search_selftest(void);

#endif
