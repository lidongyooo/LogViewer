#include "aklv_search.h"

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef AKLV_SEARCH_CHUNK_LINES
#define AKLV_SEARCH_CHUNK_LINES UINT64_C(2000000)
#endif
#ifndef AKLV_SEARCH_CANCEL_CHECK_LINES
#define AKLV_SEARCH_CANCEL_CHECK_LINES UINT64_C(200000)
#endif
#define AKLV_SIMD_SHORT_QUERY_MAX 16
#define AKLV_QUERY_INLINE_PATTERN_BYTES 256
#define AKLV_QUERY_INLINE_GRAMS 256

typedef enum {
    AKLV_QUERY_KERNEL_BMH = 0,
    AKLV_QUERY_KERNEL_SIMD_SHORT
} AklvQueryKernel;

typedef struct {
    const unsigned char *lower;
    unsigned char *pattern;
    unsigned char inline_pattern[AKLV_QUERY_INLINE_PATTERN_BYTES];
    size_t pattern_len;
    size_t skip[256];
    AklvQueryKernel kernel;
    size_t anchor_pos;
    unsigned char anchor;
    unsigned char anchor_pair;
    bool anchor_has_pair;
    uint32_t *grams;
    uint32_t inline_grams[AKLV_QUERY_INLINE_GRAMS];
    size_t gram_count;
    size_t gram_capacity;
    bool pattern_heap;
    bool grams_heap;
} AklvCompiledQuery;

typedef struct {
    AklvSearchResult *items;
    uint64_t count;
    uint64_t capacity;
} AklvResultVec;

typedef struct {
    uint64_t seq;
    uint32_t file_index;
    uint64_t first_line;
    uint64_t last_line;
} AklvSearchTask;

typedef struct {
    AklvSearchTask *items;
    uint64_t count;
    uint64_t capacity;
} AklvSearchTaskVec;

typedef struct {
    const AklvRoaring **items;
    size_t count;
    bool owns_items;
} AklvPostingPlan;

typedef struct {
    const AklvGramIndex *gram_index;
    bool has_candidates;
    AklvPostingPlan plan;
    const AklvRoaring *inline_items[AKLV_QUERY_INLINE_GRAMS];
} AklvSearchFilePlan;

typedef struct {
    pthread_t thread;
    struct AklvSearchService *service;
    AklvResultVec local_results;
    const AklvRoaringContainer **container_scratch;
    uint64_t *container_index_scratch;
    size_t container_scratch_capacity;
    const AklvRoaringContainer *container_stack[AKLV_QUERY_INLINE_GRAMS];
    uint64_t container_index_stack[AKLV_QUERY_INLINE_GRAMS];
} AklvSearchWorker;

typedef struct {
    uint64_t generation;
    char *query;
    AklvCompiledQuery compiled;
    AklvFileSnapshot snapshot;
    AklvSearchFilePlan *file_plans;
    uint32_t file_plan_count;
    uint64_t start_ns;
} AklvSearchJob;

struct AklvSearchService {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_cond_t done_cond;
    bool stop;
    bool queued_ready;
    bool job_running;
    atomic_bool cancel_stop;
    atomic_uint_fast64_t cancel_generation;

    uint64_t request_generation;
    uint64_t work_epoch;
    uint32_t thread_count;
    uint32_t active_workers;

    const AklvSearchTask *tasks;
    AklvResultVec *task_results;
    uint64_t task_count;
    atomic_uint_fast64_t next_task;

    AklvSearchJob queued_job;
    AklvSearchJob active_job;
    AklvSearchWorker *workers;

    AklvSearchResults pending;
};

static bool aklv_generation_cancelled(const AklvSearchService *service, uint64_t generation);
static void aklv_search_job_file_plans_destroy(AklvSearchJob *job);
static pthread_once_t g_aklv_ascii_lower_once = PTHREAD_ONCE_INIT;
static unsigned char g_aklv_ascii_lower[256];

static uint64_t aklv_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static void aklv_init_ascii_lower_once(void) {
    for (size_t i = 0; i < 256; i++) {
        g_aklv_ascii_lower[i] = (unsigned char)i;
    }
    for (unsigned char c = 'A'; c <= 'Z'; c++) {
        g_aklv_ascii_lower[c] = (unsigned char)(c + ('a' - 'A'));
    }
}

static const unsigned char *aklv_ascii_lower_table(void) {
    pthread_once(&g_aklv_ascii_lower_once, aklv_init_ascii_lower_once);
    return g_aklv_ascii_lower;
}

static bool aklv_bmh_equal_at(const AklvCompiledQuery *compiled,
                              const unsigned char *haystack,
                              size_t needle_len) {
    for (size_t i = 0; i < needle_len; i++) {
        if (compiled->lower[haystack[i]] != compiled->pattern[i]) {
            return false;
        }
    }
    return true;
}

static const unsigned char *aklv_bmh_find(const AklvCompiledQuery *compiled,
                                          const unsigned char *haystack,
                                          size_t haystack_len) {
    size_t needle_len = compiled->pattern_len;
    if (needle_len == 0 || haystack_len < needle_len) {
        return NULL;
    }
    if (needle_len == 1) {
        unsigned char c = compiled->pattern[0];
        if (c >= 'a' && c <= 'z') {
            return aklv_find_either_byte_simd(haystack,
                                              haystack_len,
                                              c,
                                              (unsigned char)(c - ('a' - 'A')));
        }
        return aklv_find_byte_simd(haystack, haystack_len, c);
    }

    size_t pos = 0;
    while (pos <= haystack_len - needle_len) {
        unsigned char last = compiled->lower[haystack[pos + needle_len - 1]];
        if (last == compiled->pattern[needle_len - 1] &&
            aklv_bmh_equal_at(compiled, haystack + pos, needle_len)) {
            return haystack + pos;
        }
        pos += compiled->skip[last];
    }
    return NULL;
}

static void aklv_compiled_query_destroy(AklvCompiledQuery *compiled) {
    if (compiled == NULL) {
        return;
    }
    if (compiled->pattern_heap) {
        free(compiled->pattern);
    }
    if (compiled->grams_heap) {
        free(compiled->grams);
    }
    memset(compiled, 0, sizeof(*compiled));
}

static void aklv_compiled_query_rebind_inline(AklvCompiledQuery *compiled) {
    if (compiled == NULL) {
        return;
    }
    if (!compiled->pattern_heap) {
        compiled->pattern = compiled->inline_pattern;
    }
    if (!compiled->grams_heap) {
        compiled->grams = compiled->inline_grams;
    }
}

static int aklv_compare_u32(const void *a, const void *b) {
    uint32_t va = *(const uint32_t *)a;
    uint32_t vb = *(const uint32_t *)b;
    return (va > vb) - (va < vb);
}

static bool aklv_compiled_query_init_grams(AklvCompiledQuery *compiled) {
    size_t len = compiled->pattern_len;
    compiled->grams = compiled->inline_grams;
    compiled->gram_capacity = AKLV_QUERY_INLINE_GRAMS;
    if (len < AKLV_ASCII_GRAM_MAX_SIZE) {
        return true;
    }
    size_t max_count = len - AKLV_ASCII_GRAM_MAX_SIZE + 1;
    if (max_count > AKLV_QUERY_INLINE_GRAMS) {
        compiled->grams = malloc(max_count * sizeof(*compiled->grams));
        if (compiled->grams == NULL) {
            return false;
        }
        compiled->gram_capacity = max_count;
        compiled->grams_heap = true;
    }

    size_t i = 0;
    while (i < len) {
        while (i < len && !aklv_is_index_byte(compiled->pattern[i])) {
            i++;
        }
        size_t run_start = i;
        while (i < len && aklv_is_index_byte(compiled->pattern[i])) {
            i++;
        }
        size_t run_len = i - run_start;
        if (run_len < AKLV_ASCII_GRAM_MAX_SIZE) {
            continue;
        }
        uint32_t first = compiled->pattern[run_start];
        uint32_t second = compiled->pattern[run_start + 1];
        for (size_t pos = run_start + AKLV_ASCII_GRAM_MAX_SIZE - 1; pos < i; pos++) {
            uint32_t third = compiled->pattern[pos];
            uint32_t gram = (AKLV_ASCII_GRAM_MAX_SIZE << 24) |
                            (first << 16) |
                            (second << 8) |
                            third;
            if (compiled->gram_count == compiled->gram_capacity) {
                return false;
            }
            compiled->grams[compiled->gram_count++] = gram;
            first = second;
            second = third;
        }
    }
    if (compiled->gram_count > 1) {
        qsort(compiled->grams, compiled->gram_count, sizeof(*compiled->grams), aklv_compare_u32);
        size_t out = 1;
        for (size_t read = 1; read < compiled->gram_count; read++) {
            if (compiled->grams[read] != compiled->grams[out - 1]) {
                compiled->grams[out++] = compiled->grams[read];
            }
        }
        compiled->gram_count = out;
    }
    return true;
}

static bool aklv_compiled_query_init(AklvCompiledQuery *compiled,
                                     const char *query,
                                     const AklvFileSnapshot *snapshot) {
    (void)snapshot;
    memset(compiled, 0, sizeof(*compiled));
    compiled->lower = aklv_ascii_lower_table();
    size_t len = strlen(query);
    compiled->pattern_len = len;
    compiled->pattern = compiled->inline_pattern;
    if (len > AKLV_QUERY_INLINE_PATTERN_BYTES) {
        compiled->pattern = malloc(len);
        if (compiled->pattern == NULL) {
            return false;
        }
        compiled->pattern_heap = true;
    }
    for (size_t i = 0; i < len; i++) {
        compiled->pattern[i] = compiled->lower[(unsigned char)query[i]];
    }
    for (size_t i = 0; i < 256; i++) {
        compiled->skip[i] = len == 0 ? 1 : len;
    }
    if (len > 1) {
        for (size_t i = 0; i + 1 < len; i++) {
            compiled->skip[compiled->pattern[i]] = len - 1 - i;
        }
    }

    if (!aklv_compiled_query_init_grams(compiled)) {
        aklv_compiled_query_destroy(compiled);
        return false;
    }

    compiled->kernel = len > 0 && len <= AKLV_SIMD_SHORT_QUERY_MAX
        ? AKLV_QUERY_KERNEL_SIMD_SHORT
        : AKLV_QUERY_KERNEL_BMH;
    compiled->anchor_pos = len > 0 ? len - 1 : 0;
    if (len > 0) {
        compiled->anchor_pos = len - 1;
        compiled->anchor = compiled->pattern[compiled->anchor_pos];
        if (compiled->anchor >= 'a' && compiled->anchor <= 'z') {
            compiled->anchor_pair = (unsigned char)(compiled->anchor - ('a' - 'A'));
            compiled->anchor_has_pair = true;
        }
    }
    return true;
}

static bool aklv_compiled_equal_at(const AklvCompiledQuery *compiled, const unsigned char *haystack) {
    for (size_t i = 0; i < compiled->pattern_len; i++) {
        if (compiled->lower[haystack[i]] != compiled->pattern[i]) {
            return false;
        }
    }
    return true;
}

static const unsigned char *aklv_find_next_anchor(const AklvCompiledQuery *compiled,
                                                  const unsigned char *haystack,
                                                  size_t haystack_len) {
    if (compiled->anchor_has_pair) {
        return aklv_find_either_byte_simd(haystack,
                                          haystack_len,
                                          compiled->anchor,
                                          compiled->anchor_pair);
    }
    return aklv_find_byte_simd(haystack, haystack_len, compiled->anchor);
}

static const unsigned char *aklv_simd_short_find(const AklvCompiledQuery *compiled,
                                                 const unsigned char *haystack,
                                                 size_t haystack_len) {
    size_t len = compiled->pattern_len;
    if (len == 0 || haystack_len < len) {
        return NULL;
    }
    const unsigned char *scan = haystack + compiled->anchor_pos;
    size_t scan_len = haystack_len - compiled->anchor_pos;
    while (scan_len > 0) {
        const unsigned char *anchor = aklv_find_next_anchor(compiled, scan, scan_len);
        if (anchor == NULL) {
            return NULL;
        }
        const unsigned char *candidate = anchor - compiled->anchor_pos;
        if ((size_t)(haystack + haystack_len - candidate) >= len &&
            aklv_compiled_equal_at(compiled, candidate)) {
            return candidate;
        }
        size_t consumed = (size_t)(anchor - scan) + 1;
        scan += consumed;
        scan_len -= consumed;
    }
    return NULL;
}

static const unsigned char *aklv_compiled_find(const AklvCompiledQuery *compiled,
                                               const unsigned char *haystack,
                                               size_t haystack_len) {
    if (compiled->kernel == AKLV_QUERY_KERNEL_SIMD_SHORT) {
        return aklv_simd_short_find(compiled, haystack, haystack_len);
    }
    return aklv_bmh_find(compiled, haystack, haystack_len);
}

static uint32_t aklv_default_search_thread_count(void) {
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1) {
        cores = 1;
    }
    long threads = cores - 1;
    if (threads < 1) {
        threads = 1;
    }
    if (threads > 16) {
        threads = 16;
    }
    return (uint32_t)threads;
}

static void aklv_result_vec_clear(AklvResultVec *vec) {
    vec->count = 0;
}

static void aklv_result_vec_destroy(AklvResultVec *vec) {
    free(vec->items);
    memset(vec, 0, sizeof(*vec));
}

static bool aklv_result_vec_reserve(AklvResultVec *vec, uint64_t capacity) {
    if (capacity <= vec->capacity) {
        return true;
    }
    if (capacity > (uint64_t)(SIZE_MAX / sizeof(*vec->items))) {
        return false;
    }
    AklvSearchResult *new_items = realloc(vec->items, (size_t)capacity * sizeof(*vec->items));
    if (new_items == NULL) {
        return false;
    }
    vec->items = new_items;
    vec->capacity = capacity;
    return true;
}

static bool aklv_result_vec_append(AklvResultVec *vec,
                                      uint32_t file_id,
                                      uint64_t line_no,
                                      uint64_t byte_offset) {
    if (vec->count == vec->capacity) {
        uint64_t new_capacity = vec->capacity == 0 ? 4096 : vec->capacity * 2;
        if (new_capacity < vec->capacity ||
            new_capacity > (uint64_t)(SIZE_MAX / sizeof(*vec->items))) {
            return false;
        }
        if (!aklv_result_vec_reserve(vec, new_capacity)) {
            return false;
        }
    }
    vec->items[vec->count].file_id = file_id;
    vec->items[vec->count].line_no = line_no;
    vec->items[vec->count].byte_offset = byte_offset;
    vec->count++;
    return true;
}

static void aklv_task_vec_destroy(AklvSearchTaskVec *vec) {
    free(vec->items);
    memset(vec, 0, sizeof(*vec));
}

static bool aklv_task_vec_reserve(AklvSearchTaskVec *vec, uint64_t capacity) {
    if (capacity <= vec->capacity) {
        return true;
    }
    if (capacity > (uint64_t)(SIZE_MAX / sizeof(*vec->items))) {
        return false;
    }
    AklvSearchTask *new_items = realloc(vec->items, (size_t)capacity * sizeof(*vec->items));
    if (new_items == NULL) {
        return false;
    }
    vec->items = new_items;
    vec->capacity = capacity;
    return true;
}

static bool aklv_task_vec_append(AklvSearchTaskVec *vec,
                                 uint32_t file_index,
                                 uint64_t first_line,
                                 uint64_t last_line) {
    if (first_line == 0 || last_line < first_line) {
        return true;
    }
    if (vec->count == vec->capacity) {
        uint64_t new_capacity = vec->capacity == 0 ? 64 : vec->capacity * 2;
        if (new_capacity < vec->capacity ||
            new_capacity > (uint64_t)(SIZE_MAX / sizeof(*vec->items))) {
            return false;
        }
        if (!aklv_task_vec_reserve(vec, new_capacity)) {
            return false;
        }
    }
    uint64_t seq = vec->count;
    vec->items[seq].seq = seq;
    vec->items[seq].file_index = file_index;
    vec->items[seq].first_line = first_line;
    vec->items[seq].last_line = last_line;
    vec->count++;
    return true;
}

static bool aklv_task_vec_append_chunks(AklvSearchTaskVec *vec,
                                        uint32_t file_index,
                                        uint64_t first_line,
                                        uint64_t last_line) {
    if (first_line == 0 || last_line < first_line) {
        return true;
    }
    uint64_t line = first_line;
    while (line <= last_line) {
        uint64_t chunk_last = line + AKLV_SEARCH_CHUNK_LINES - 1;
        if (chunk_last < line || chunk_last > last_line) {
            chunk_last = last_line;
        }
        if (!aklv_task_vec_append(vec, file_index, line, chunk_last)) {
            return false;
        }
        if (chunk_last == UINT64_MAX) {
            break;
        }
        line = chunk_last + 1;
    }
    return true;
}

static void aklv_search_job_destroy(AklvSearchJob *job) {
    free(job->query);
    aklv_search_job_file_plans_destroy(job);
    aklv_compiled_query_destroy(&job->compiled);
    aklv_file_snapshot_destroy(&job->snapshot);
    memset(job, 0, sizeof(*job));
}

void aklv_file_snapshot_destroy(AklvFileSnapshot *snapshot) {
    if (snapshot == NULL) {
        return;
    }
    if (snapshot->items != NULL) {
        for (uint32_t i = 0; i < snapshot->count; i++) {
            aklv_file_release(snapshot->items[i]);
        }
        free(snapshot->items);
    }
    memset(snapshot, 0, sizeof(*snapshot));
}

static bool aklv_file_snapshot_clone(AklvFileSnapshot *dst, const AklvFileSnapshot *src) {
    memset(dst, 0, sizeof(*dst));
    dst->count = src->count;
    dst->active_index = src->active_index;
    dst->active_selected_line = src->active_selected_line;
    if (src->count == 0) {
        return true;
    }
    dst->items = calloc(src->count, sizeof(*dst->items));
    if (dst->items == NULL) {
        memset(dst, 0, sizeof(*dst));
        return false;
    }
    for (uint32_t i = 0; i < src->count; i++) {
        dst->items[i] = src->items[i];
        aklv_file_retain(dst->items[i]);
    }
    return true;
}

static bool aklv_search_job_clone(AklvSearchJob *dst,
                                  const char *query,
                                  const AklvFileSnapshot *snapshot) {
    memset(dst, 0, sizeof(*dst));
    dst->query = aklv_strdup(query);
    if (dst->query == NULL || !aklv_file_snapshot_clone(&dst->snapshot, snapshot) ||
        !aklv_compiled_query_init(&dst->compiled, query, &dst->snapshot)) {
        aklv_search_job_destroy(dst);
        return false;
    }
    return true;
}

void aklv_search_results_init(AklvSearchResults *results) {
    memset(results, 0, sizeof(*results));
}

void aklv_search_results_destroy(AklvSearchResults *results) {
    if (results == NULL) {
        return;
    }
    free(results->items);
    memset(results, 0, sizeof(*results));
}

static void aklv_search_results_clear(AklvSearchResults *results) {
    results->count = 0;
    results->complete = false;
    results->running = false;
    results->elapsed_sec = 0.0;
    results->query[0] = '\0';
}

static bool aklv_search_results_reserve(AklvSearchResults *results, uint64_t capacity) {
    if (capacity <= results->capacity) {
        return true;
    }
    if (capacity > (uint64_t)(SIZE_MAX / sizeof(*results->items))) {
        return false;
    }
    AklvSearchResult *new_items = realloc(results->items, (size_t)capacity * sizeof(*results->items));
    if (new_items == NULL) {
        return false;
    }
    results->items = new_items;
    results->capacity = capacity;
    return true;
}

static void aklv_task_results_destroy(AklvResultVec *results, uint64_t count) {
    if (results == NULL) {
        return;
    }
    for (uint64_t i = 0; i < count; i++) {
        aklv_result_vec_destroy(&results[i]);
    }
    free(results);
}

static bool aklv_publish_task_results(AklvSearchService *service,
                                      uint64_t generation,
                                      const char *query,
                                      double elapsed_sec,
                                      AklvResultVec *task_results,
                                      uint64_t task_count,
                                      bool complete) {
    if (task_count != 0 && task_results == NULL) {
        return false;
    }
    uint64_t total = 0;
    uint64_t nonempty_count = 0;
    uint64_t nonempty_index = 0;
    for (uint64_t i = 0; i < task_count; i++) {
        if (task_results[i].count > UINT64_MAX - total) {
            return false;
        }
        if (task_results[i].count != 0) {
            nonempty_index = i;
            nonempty_count++;
        }
        total += task_results[i].count;
    }
    if (total > (uint64_t)(SIZE_MAX / sizeof(AklvSearchResult))) {
        return false;
    }
    AklvSearchResult *merged = NULL;
    if (total > 0) {
        if (nonempty_count == 1) {
            AklvResultVec *src = &task_results[nonempty_index];
            if (src->items == NULL) {
                return false;
            }
            merged = src->items;
            src->items = NULL;
            src->count = 0;
            src->capacity = 0;
        } else {
            merged = malloc((size_t)total * sizeof(*merged));
            if (merged == NULL) {
                return false;
            }
            uint64_t pos = 0;
            for (uint64_t i = 0; i < task_count; i++) {
                if (task_results[i].count == 0) {
                    continue;
                }
                memcpy(merged + pos,
                       task_results[i].items,
                       (size_t)task_results[i].count * sizeof(*merged));
                pos += task_results[i].count;
            }
        }
    }

    pthread_mutex_lock(&service->mutex);
    if (service->stop || service->request_generation != generation) {
        pthread_mutex_unlock(&service->mutex);
        free(merged);
        return false;
    }
    free(service->pending.items);
    service->pending.items = merged;
    service->pending.count = total;
    service->pending.capacity = total;
    service->pending.generation = generation;
    service->pending.running = !complete;
    service->pending.complete = complete;
    service->pending.elapsed_sec = elapsed_sec;
    snprintf(service->pending.query, sizeof(service->pending.query), "%s", query);
    pthread_mutex_unlock(&service->mutex);
    return true;
}

static void aklv_mark_search_complete(AklvSearchService *service,
                                      uint64_t generation,
                                      const char *query,
                                      double elapsed_sec) {
    pthread_mutex_lock(&service->mutex);
    if (!service->stop && service->request_generation == generation) {
        service->pending.generation = generation;
        service->pending.running = false;
        service->pending.complete = true;
        service->pending.elapsed_sec = elapsed_sec;
        snprintf(service->pending.query, sizeof(service->pending.query), "%s", query);
    }
    pthread_mutex_unlock(&service->mutex);
}

static bool aklv_line_matches_at(const AklvFile *file,
                                 uint64_t line_no,
                                 const AklvCompiledQuery *compiled,
                                 AklvResultVec *out) {
    const unsigned char *base = file->mapped.data;
    const size_t file_size = file->mapped.size;
    const uint64_t pos = line_no - 1;
    const uint64_t block = pos >> AKLV_LINE_INDEX_BLOCK_SHIFT;
    const uint64_t in_block = pos & AKLV_LINE_INDEX_BLOCK_MASK;
    size_t packed = file->index.offset_blocks[block][in_block];
    size_t offset = aklv_line_index_unpack_offset(packed);
    size_t next_offset = file_size;
    if (line_no < file->index.count) {
        uint64_t next_pos = pos + 1;
        next_offset = aklv_line_index_unpack_offset(
            file->index.offset_blocks[next_pos >> AKLV_LINE_INDEX_BLOCK_SHIFT]
                                     [next_pos & AKLV_LINE_INDEX_BLOCK_MASK]);
    }
    AklvLineView line;
    line.start = base + offset;
    line.len = next_offset > offset ? next_offset - offset : 0;
    while (line.len > 0 && (line.start[line.len - 1] == '\n' || line.start[line.len - 1] == '\r')) {
        line.len--;
    }

    uint32_t prefix_skip = aklv_line_index_unpack_prefix_skip(packed);
    AklvLineView effective = line;
    if (prefix_skip == AKLV_LINE_PREFIX_OVERFLOW) {
        effective = aklv_effective_search_line(line);
    } else if (prefix_skip > 0 && prefix_skip < effective.len) {
        effective.start += prefix_skip;
        effective.len -= prefix_skip;
    } else if (prefix_skip >= effective.len) {
        effective.start += effective.len;
        effective.len = 0;
    }
    if (aklv_compiled_find(compiled, effective.start, effective.len) != NULL) {
        return aklv_result_vec_append(out, file->id, line_no, (uint64_t)offset);
    }
    return true;
}

static void aklv_posting_plan_destroy(AklvPostingPlan *plan) {
    if (plan->owns_items) {
        free(plan->items);
    }
    memset(plan, 0, sizeof(*plan));
}

static void aklv_search_job_file_plans_destroy(AklvSearchJob *job) {
    if (job == NULL || job->file_plans == NULL) {
        return;
    }
    for (uint32_t i = 0; i < job->file_plan_count; i++) {
        aklv_posting_plan_destroy(&job->file_plans[i].plan);
    }
    free(job->file_plans);
    job->file_plans = NULL;
    job->file_plan_count = 0;
}

static bool aklv_build_posting_plan(const AklvGramIndex *gram_index,
                                    const AklvCompiledQuery *compiled,
                                    AklvPostingPlan *plan,
                                    const AklvRoaring **scratch,
                                    size_t scratch_count) {
    memset(plan, 0, sizeof(*plan));
    if (compiled->gram_count == 0) {
        return false;
    }
    if (compiled->gram_count <= scratch_count) {
        plan->items = scratch;
    } else {
        plan->items = malloc(compiled->gram_count * sizeof(*plan->items));
        if (plan->items == NULL) {
            return false;
        }
        plan->owns_items = true;
    }
    for (size_t i = 0; i < compiled->gram_count; i++) {
        const AklvRoaring *posting = aklv_gram_index_get(gram_index, compiled->grams[i]);
        if (posting == NULL || aklv_roaring_cardinality(posting) == 0) {
            aklv_posting_plan_destroy(plan);
            return false;
        }
        uint64_t cardinality = aklv_roaring_cardinality(posting);
        size_t pos = plan->count;
        while (pos > 0 && aklv_roaring_cardinality(plan->items[pos - 1]) > cardinality) {
            plan->items[pos] = plan->items[pos - 1];
            pos--;
        }
        plan->items[pos] = posting;
        plan->count++;
    }
    return true;
}

static void aklv_worker_destroy_scratch(AklvSearchWorker *worker) {
    if (worker == NULL) {
        return;
    }
    free(worker->container_scratch);
    free(worker->container_index_scratch);
    worker->container_scratch = NULL;
    worker->container_index_scratch = NULL;
    worker->container_scratch_capacity = 0;
}

static bool aklv_worker_reserve_container_scratch(AklvSearchWorker *worker,
                                                  size_t count,
                                                  const AklvRoaringContainer ***containers_out,
                                                  uint64_t **indices_out) {
    if (count <= AKLV_QUERY_INLINE_GRAMS) {
        *containers_out = worker->container_stack;
        *indices_out = worker->container_index_stack;
        return true;
    }
    if (count <= worker->container_scratch_capacity) {
        *containers_out = worker->container_scratch;
        *indices_out = worker->container_index_scratch;
        return true;
    }
    if (count > SIZE_MAX / sizeof(*worker->container_scratch) ||
        count > SIZE_MAX / sizeof(*worker->container_index_scratch)) {
        return false;
    }
    const AklvRoaringContainer **new_containers =
        realloc(worker->container_scratch, count * sizeof(*new_containers));
    if (new_containers == NULL) {
        return false;
    }
    worker->container_scratch = new_containers;
    uint64_t *new_indices =
        realloc(worker->container_index_scratch, count * sizeof(*new_indices));
    if (new_indices == NULL) {
        return false;
    }
    worker->container_index_scratch = new_indices;
    worker->container_scratch_capacity = count;
    *containers_out = worker->container_scratch;
    *indices_out = worker->container_index_scratch;
    return true;
}

static bool aklv_candidate_has_all_container_grams(const AklvPostingPlan *plan,
                                                   const AklvRoaringContainer **containers,
                                                   uint16_t low) {
    for (size_t i = 1; i < plan->count; i++) {
        if (!aklv_roaring_container_contains_low(containers[i], low)) {
            return false;
        }
    }
    return true;
}

static bool aklv_container_is_bitmap_like(const AklvRoaringContainer *container) {
    return container != NULL &&
           (container->type == AKLV_ROARING_BITMAP || container->type == AKLV_ROARING_FULL);
}

static uint64_t aklv_container_word_bits(const AklvRoaringContainer *container, uint32_t word_index) {
    if (container->type == AKLV_ROARING_FULL) {
        return UINT64_MAX;
    }
    return container->bitmap[word_index];
}

static bool aklv_result_for_candidate_line(const AklvFile *file,
                                           uint64_t line_no,
                                           const AklvCompiledQuery *compiled,
                                           bool exact_validation_needed,
                                           AklvResultVec *out) {
    if (exact_validation_needed) {
        return aklv_line_matches_at(file, line_no, compiled, out);
    }
    return aklv_result_vec_append(out,
                                  file->id,
                                  line_no,
                                  (uint64_t)aklv_line_index_offset(&file->index, line_no));
}

static bool aklv_collect_bitmap_intersection_matches(const AklvFile *file,
                                                     const AklvCompiledQuery *compiled,
                                                     const AklvPostingPlan *plan,
                                                     const AklvRoaringContainer **containers,
                                                     uint16_t min_low,
                                                     uint16_t max_low,
                                                     bool exact_validation_needed,
                                                     uint64_t generation,
                                                     const AklvSearchService *service,
                                                     uint64_t *next_cancel_check,
                                                     AklvResultVec *out) {
    uint32_t first_word = min_low >> 6;
    uint32_t last_word = max_low >> 6;
    for (uint32_t word_index = first_word; word_index <= last_word; word_index++) {
        uint64_t bits = aklv_container_word_bits(containers[0], word_index);
        for (size_t i = 1; bits != 0 && i < plan->count; i++) {
            bits &= aklv_container_word_bits(containers[i], word_index);
        }
        if (word_index == first_word) {
            uint32_t skip = min_low & 63U;
            if (skip != 0) {
                bits &= UINT64_MAX << skip;
            }
        }
        if (word_index == last_word) {
            uint32_t keep = (max_low & 63U) + 1U;
            if (keep < 64U) {
                bits &= (UINT64_C(1) << keep) - 1U;
            }
        }
        while (bits != 0) {
            uint32_t bit = (uint32_t)__builtin_ctzll(bits);
            bits &= bits - 1;
            uint64_t line_no = (containers[0]->key << 16) | (uint64_t)((word_index << 6) | bit);
            if (line_no >= *next_cancel_check) {
                if (aklv_generation_cancelled(service, generation)) {
                    return false;
                }
                uint64_t next = line_no + AKLV_SEARCH_CANCEL_CHECK_LINES;
                *next_cancel_check = next > line_no ? next : UINT64_MAX;
            }
            if (!aklv_result_for_candidate_line(file,
                                                line_no,
                                                compiled,
                                                exact_validation_needed,
                                                out)) {
                return false;
            }
        }
        if (word_index == UINT32_MAX) {
            break;
        }
    }
    return true;
}

static bool aklv_collect_driver_container_matches(const AklvFile *file,
                                                  const AklvCompiledQuery *compiled,
                                                  const AklvPostingPlan *plan,
                                                  const AklvRoaringContainer **containers,
                                                  uint64_t *container_indices,
                                                  const AklvRoaringContainer *driver_container,
                                                  uint16_t min_low,
                                                  uint16_t max_low,
                                                  uint64_t generation,
                                                  const AklvSearchService *service,
                                                  uint64_t *next_cancel_check,
                                                  AklvResultVec *out) {
    containers[0] = driver_container;
    uint64_t driver_key = driver_container->key;
    for (size_t i = 1; i < plan->count; i++) {
        const AklvRoaring *posting = plan->items[i];
        uint64_t pos = container_indices[i];
        while (pos < posting->count && posting->containers[pos].key < driver_key) {
            pos++;
        }
        container_indices[i] = pos;
        if (pos >= posting->count || posting->containers[pos].key != driver_key) {
            return true;
        }
        containers[i] = &posting->containers[pos];
    }

    bool exact_validation_needed =
        !(compiled->pattern_len == AKLV_ASCII_GRAM_MAX_SIZE && compiled->gram_count == 1);
    bool bitmap_fast_path = true;
    for (size_t i = 0; i < plan->count; i++) {
        if (!aklv_container_is_bitmap_like(containers[i])) {
            bitmap_fast_path = false;
            break;
        }
    }
    if (bitmap_fast_path) {
        bool ok = aklv_collect_bitmap_intersection_matches(file,
                                                           compiled,
                                                           plan,
                                                           containers,
                                                           min_low,
                                                           max_low,
                                                           exact_validation_needed,
                                                           generation,
                                                           service,
                                                           next_cancel_check,
                                                           out);
        return ok;
    }

    bool ok = true;
    AklvRoaringIter iter;
    aklv_roaring_iter_init(&iter, plan->items[0], (driver_container->key << 16) | (uint64_t)min_low);
    uint64_t line_no = 0;
    while (aklv_roaring_iter_next(&iter, &line_no)) {
        uint64_t key = line_no >> 16;
        uint16_t low = (uint16_t)(line_no & 0xffffU);
        if (key != driver_container->key || low > max_low) {
            break;
        }
        if (line_no >= *next_cancel_check) {
            if (aklv_generation_cancelled(service, generation)) {
                ok = false;
                break;
            }
            uint64_t next = line_no + AKLV_SEARCH_CANCEL_CHECK_LINES;
            *next_cancel_check = next > line_no ? next : UINT64_MAX;
        }
        if (!aklv_candidate_has_all_container_grams(plan, containers, low)) {
            continue;
        }
        if (!aklv_result_for_candidate_line(file,
                                            line_no,
                                            compiled,
                                            exact_validation_needed,
                                            out)) {
            ok = false;
            break;
        }
    }
    return ok;
}

static bool aklv_collect_file_range_matches_with_plan(const AklvFile *file,
                                                      const AklvPostingPlan *plan,
                                                      AklvSearchWorker *worker,
                                                      uint64_t first_line,
                                                      uint64_t last_line,
                                                      const AklvCompiledQuery *compiled,
                                                      uint64_t generation,
                                                      const AklvSearchService *service,
                                                      AklvResultVec *out) {
    if (file->mapped.size == 0 || file->index.count == 0 || first_line == 0 ||
        compiled->pattern_len == 0 || plan == NULL || plan->count == 0) {
        return true;
    }
    if (last_line > file->index.count) {
        last_line = file->index.count;
    }
    if (first_line > last_line) {
        return true;
    }

    const AklvRoaringContainer **containers = NULL;
    uint64_t *container_indices = NULL;
    if (!aklv_worker_reserve_container_scratch(worker,
                                               plan->count,
                                               &containers,
                                               &container_indices)) {
        return false;
    }
    uint64_t next_cancel_check = first_line;
    bool ok = true;
    uint64_t first_key = first_line >> 16;
    uint64_t last_key = last_line >> 16;
    uint64_t first_container =
        aklv_roaring_lower_bound_container_index(plan->items[0], first_key);
    for (size_t i = 1; i < plan->count; i++) {
        container_indices[i] = aklv_roaring_lower_bound_container_index(plan->items[i], first_key);
    }
    for (uint64_t ci = first_container; ci < plan->items[0]->count; ci++) {
        const AklvRoaringContainer *container = &plan->items[0]->containers[ci];
        if (container->key > last_key) {
            break;
        }
        uint16_t min_low = container->key == first_key ? (uint16_t)(first_line & 0xffffU) : 0;
        uint16_t max_low = container->key == last_key ? (uint16_t)(last_line & 0xffffU) : UINT16_MAX;
        if (!aklv_collect_driver_container_matches(file,
                                                   compiled,
                                                   plan,
                                                   containers,
                                                   container_indices,
                                                   container,
                                                   min_low,
                                                   max_low,
                                                   generation,
                                                   service,
                                                   &next_cancel_check,
                                                   out)) {
            ok = false;
            break;
        }
    }
    return ok;
}

static bool aklv_collect_file_range_matches_worker_plan(AklvSearchWorker *worker,
                                                        const AklvFile *file,
                                                        const AklvPostingPlan *plan,
                                                        uint64_t first_line,
                                                        uint64_t last_line,
                                                        const AklvCompiledQuery *compiled,
                                                        uint64_t generation,
                                                        const AklvSearchService *service,
                                                        AklvResultVec *out) {
    if (plan == NULL || plan->count == 0) {
        return true;
    }
    return aklv_collect_file_range_matches_with_plan(file,
                                                     plan,
                                                     worker,
                                                     first_line,
                                                     last_line,
                                                     compiled,
                                                     generation,
                                                     service,
                                                     out);
}

static bool aklv_collect_cached_file_range_matches(AklvSearchWorker *worker,
                                                   const AklvFile *file,
                                                   const AklvSearchFilePlan *file_plan,
                                                   uint64_t first_line,
                                                   uint64_t last_line,
                                                   const AklvCompiledQuery *compiled,
                                                   uint64_t generation,
                                                   const AklvSearchService *service,
                                                   AklvResultVec *out) {
    if (file_plan == NULL || !file_plan->has_candidates) {
        return true;
    }
    return aklv_collect_file_range_matches_worker_plan(worker,
                                                       file,
                                                       &file_plan->plan,
                                                       first_line,
                                                       last_line,
                                                       compiled,
                                                       generation,
                                                       service,
                                                       out);
}

static bool aklv_search_job_prepare_file_plans(AklvSearchJob *job,
                                               uint64_t generation,
                                               const AklvSearchService *service) {
    if (job == NULL || job->compiled.gram_count == 0) {
        return true;
    }
    aklv_search_job_file_plans_destroy(job);
    if (job->snapshot.count == 0) {
        return true;
    }
    job->file_plans = calloc(job->snapshot.count, sizeof(*job->file_plans));
    if (job->file_plans == NULL) {
        return false;
    }
    job->file_plan_count = job->snapshot.count;
    for (uint32_t i = 0; i < job->snapshot.count; i++) {
        if (aklv_generation_cancelled(service, generation)) {
            return false;
        }
        AklvFile *file = job->snapshot.items[i];
        const AklvGramIndex *gram_index = NULL;
        AklvIndexCache *cache = &file->index.cache;
        if (cache->available) {
            char error[256] = {0};
            if (aklv_index_cache_get_index(cache,
                                           &file->mapped,
                                           &gram_index,
                                           error,
                                           sizeof(error)) != 0 ||
                gram_index == NULL) {
                return false;
            }
        } else {
            gram_index = &file->index.gram_index;
        }
        AklvSearchFilePlan *file_plan = &job->file_plans[i];
        file_plan->gram_index = gram_index;
        file_plan->has_candidates =
            aklv_build_posting_plan(gram_index,
                                    &job->compiled,
                                    &file_plan->plan,
                                    file_plan->inline_items,
                                    sizeof(file_plan->inline_items) /
                                        sizeof(file_plan->inline_items[0]));
    }
    return true;
}

static bool aklv_generation_cancelled(const AklvSearchService *service, uint64_t generation) {
    return atomic_load_explicit(&service->cancel_stop, memory_order_relaxed) ||
           atomic_load_explicit(&service->cancel_generation, memory_order_acquire) != generation;
}

static bool aklv_worker_pop_task(AklvSearchService *service, AklvSearchTask *task_out) {
    uint64_t index = atomic_fetch_add_explicit(&service->next_task, 1, memory_order_relaxed);
    if (index >= service->task_count) {
        return false;
    }
    *task_out = service->tasks[index];
    return true;
}

static void *aklv_search_worker_main(void *arg) {
    AklvSearchWorker *worker = (AklvSearchWorker *)arg;
    AklvSearchService *service = worker->service;
    uint64_t seen_epoch = 0;

    pthread_mutex_lock(&service->mutex);
    while (true) {
        while (!service->stop && service->work_epoch == seen_epoch) {
            pthread_cond_wait(&service->cond, &service->mutex);
        }
        if (service->stop) {
            if (service->work_epoch != seen_epoch && service->active_workers > 0) {
                service->active_workers--;
                if (service->active_workers == 0) {
                    pthread_cond_signal(&service->done_cond);
                }
            }
            pthread_mutex_unlock(&service->mutex);
            return NULL;
        }

        uint64_t epoch = service->work_epoch;
        uint64_t generation = service->active_job.generation;
        AklvSearchJob *job = &service->active_job;
        seen_epoch = epoch;
        pthread_mutex_unlock(&service->mutex);

        const AklvCompiledQuery *compiled = &job->compiled;
        AklvSearchTask task;

        if (compiled->pattern_len > 0) {
            while (!aklv_generation_cancelled(service, generation) &&
                   aklv_worker_pop_task(service, &task)) {
                if (task.file_index >= job->snapshot.count) {
                    continue;
                }
                aklv_result_vec_clear(&worker->local_results);
                AklvFile *file = job->snapshot.items[task.file_index];
                const AklvSearchFilePlan *file_plan = task.file_index < job->file_plan_count
                    ? &job->file_plans[task.file_index]
                    : NULL;
                aklv_collect_cached_file_range_matches(worker,
                                                       file,
                                                       file_plan,
                                                       task.first_line,
                                                       task.last_line,
                                                       compiled,
                                                       generation,
                                                       service,
                                                       &worker->local_results);
                if (task.seq < service->task_count && !aklv_generation_cancelled(service, generation)) {
                    if (worker->local_results.count != 0) {
                        aklv_result_vec_destroy(&service->task_results[task.seq]);
                        service->task_results[task.seq] = worker->local_results;
                        memset(&worker->local_results, 0, sizeof(worker->local_results));
                    } else {
                        aklv_result_vec_clear(&worker->local_results);
                    }
                }
            }
        }

        pthread_mutex_lock(&service->mutex);
        if (service->active_workers > 0) {
            service->active_workers--;
        }
        if (service->active_workers == 0) {
            pthread_cond_signal(&service->done_cond);
        }
    }
}

static bool aklv_run_phase(AklvSearchService *service,
                           uint64_t generation,
                           const AklvSearchTask *tasks,
                           uint64_t task_count,
                           AklvResultVec **results_out) {
    if (results_out != NULL) {
        *results_out = NULL;
    }
    if (task_count == 0) {
        return !aklv_generation_cancelled(service, generation);
    }
    AklvResultVec *task_results = calloc((size_t)task_count, sizeof(*task_results));
    if (task_results == NULL) {
        return false;
    }
    if (task_count == 1) {
        const AklvSearchJob *job = &service->active_job;
        const AklvCompiledQuery *compiled = &job->compiled;
        const AklvSearchTask *task = &tasks[0];
        bool ok = !aklv_generation_cancelled(service, generation) &&
                  task->file_index < job->snapshot.count;
        if (ok) {
            AklvSearchWorker *worker = &service->workers[0];
            aklv_result_vec_clear(&worker->local_results);
            AklvFile *file = job->snapshot.items[task->file_index];
            const AklvSearchFilePlan *file_plan = task->file_index < job->file_plan_count
                ? &job->file_plans[task->file_index]
                : NULL;
            ok = aklv_collect_cached_file_range_matches(worker,
                                                        file,
                                                        file_plan,
                                                        task->first_line,
                                                        task->last_line,
                                                        compiled,
                                                        generation,
                                                        service,
                                                        &worker->local_results);
            if (ok && !aklv_generation_cancelled(service, generation) &&
                worker->local_results.count != 0) {
                task_results[0] = worker->local_results;
                memset(&worker->local_results, 0, sizeof(worker->local_results));
            } else {
                aklv_result_vec_clear(&worker->local_results);
            }
        }
        if (!ok) {
            aklv_task_results_destroy(task_results, task_count);
            return false;
        }
        if (results_out != NULL) {
            *results_out = task_results;
        } else {
            aklv_task_results_destroy(task_results, task_count);
        }
        return true;
    }

    pthread_mutex_lock(&service->mutex);
    if (service->stop || service->request_generation != generation) {
        pthread_mutex_unlock(&service->mutex);
        aklv_task_results_destroy(task_results, task_count);
        return false;
    }
    service->tasks = tasks;
    service->task_results = task_results;
    service->task_count = task_count;
    atomic_store_explicit(&service->next_task, 0, memory_order_relaxed);
    service->active_workers = service->thread_count;
    service->work_epoch++;
    pthread_cond_broadcast(&service->cond);
    while (service->active_workers > 0) {
        pthread_cond_wait(&service->done_cond, &service->mutex);
    }
    service->tasks = NULL;
    service->task_results = NULL;
    service->task_count = 0;
    atomic_store_explicit(&service->next_task, 0, memory_order_relaxed);
    bool ok = !service->stop && service->request_generation == generation;
    pthread_mutex_unlock(&service->mutex);
    if (!ok) {
        aklv_task_results_destroy(task_results, task_count);
    } else if (results_out != NULL) {
        *results_out = task_results;
    } else {
        aklv_task_results_destroy(task_results, task_count);
    }
    return ok;
}

static uint64_t aklv_search_count_chunks(uint64_t first_line, uint64_t last_line) {
    if (first_line == 0 || last_line < first_line) {
        return 0;
    }
    uint64_t line_count = last_line - first_line + 1;
    return (line_count + AKLV_SEARCH_CHUNK_LINES - 1) / AKLV_SEARCH_CHUNK_LINES;
}

static bool aklv_estimate_all_task_count(const AklvSearchJob *job, uint64_t *count_out) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < job->snapshot.count; i++) {
        if (i >= job->file_plan_count || !job->file_plans[i].has_candidates) {
            continue;
        }
        AklvFile *file = job->snapshot.items[i];
        uint64_t add = aklv_search_count_chunks(1, file->index.count);
        if (add > UINT64_MAX - total) {
            return false;
        }
        total += add;
    }
    *count_out = total;
    return true;
}

static bool aklv_build_all_tasks(const AklvSearchJob *job, AklvSearchTaskVec *tasks) {
    uint64_t task_count = 0;
    if (!aklv_estimate_all_task_count(job, &task_count) ||
        !aklv_task_vec_reserve(tasks, task_count)) {
        return false;
    }
    for (uint32_t i = 0; i < job->snapshot.count; i++) {
        if (i >= job->file_plan_count || !job->file_plans[i].has_candidates) {
            continue;
        }
        AklvFile *file = job->snapshot.items[i];
        if (!aklv_task_vec_append_chunks(tasks, i, 1, file->index.count)) {
            return false;
        }
    }
    return true;
}

static double aklv_job_elapsed_sec(const AklvSearchJob *job) {
    uint64_t now = aklv_now_ns();
    if (now < job->start_ns || job->start_ns == 0) {
        return 0.0;
    }
    return (double)(now - job->start_ns) / 1000000000.0;
}

static void *aklv_search_coordinator_main(void *arg) {
    AklvSearchService *service = (AklvSearchService *)arg;

    pthread_mutex_lock(&service->mutex);
    while (!service->stop) {
        while (!service->stop && !service->queued_ready) {
            pthread_cond_wait(&service->cond, &service->mutex);
        }
        if (service->stop) {
            break;
        }

        service->active_job = service->queued_job;
        aklv_compiled_query_rebind_inline(&service->active_job.compiled);
        memset(&service->queued_job, 0, sizeof(service->queued_job));
        service->queued_ready = false;
        service->job_running = true;
        uint64_t generation = service->active_job.generation;
        pthread_mutex_unlock(&service->mutex);

        AklvSearchTaskVec tasks = {0};
        AklvResultVec *phase_results = NULL;
        bool ok = true;
        if (service->active_job.compiled.gram_count == 0) {
            ok = aklv_publish_task_results(service,
                                           generation,
                                           service->active_job.query,
                                           aklv_job_elapsed_sec(&service->active_job),
                                           NULL,
                                           0,
                                           true);
        } else {
            ok = aklv_search_job_prepare_file_plans(&service->active_job,
                                                    generation,
                                                    service) &&
                 aklv_build_all_tasks(&service->active_job, &tasks);
        }
        if (ok && tasks.count != 0) {
            ok = aklv_run_phase(service,
                                generation,
                                tasks.items,
                                tasks.count,
                                &phase_results);
        }
        if (ok && phase_results != NULL) {
            ok = aklv_publish_task_results(service,
                                           generation,
                                           service->active_job.query,
                                           aklv_job_elapsed_sec(&service->active_job),
                                           phase_results,
                                           tasks.count,
                                           false);
            aklv_task_results_destroy(phase_results, tasks.count);
            phase_results = NULL;
        }
        if (ok) {
            aklv_mark_search_complete(service,
                                      generation,
                                      service->active_job.query,
                                      aklv_job_elapsed_sec(&service->active_job));
        }
        aklv_task_vec_destroy(&tasks);

        pthread_mutex_lock(&service->mutex);
        service->job_running = false;
        aklv_search_job_destroy(&service->active_job);
    }
    pthread_mutex_unlock(&service->mutex);
    return NULL;
}

AklvSearchService *aklv_search_service_create(void) {
    AklvSearchService *service = calloc(1, sizeof(*service));
    if (service == NULL) {
        return NULL;
    }
    if (pthread_mutex_init(&service->mutex, NULL) != 0 ||
        pthread_cond_init(&service->cond, NULL) != 0 ||
        pthread_cond_init(&service->done_cond, NULL) != 0) {
        aklv_search_service_destroy(service);
        return NULL;
    }
    atomic_init(&service->cancel_stop, false);
    atomic_init(&service->cancel_generation, 0);
    aklv_search_results_init(&service->pending);
    service->thread_count = aklv_default_search_thread_count();
    service->workers = calloc(service->thread_count + 1, sizeof(*service->workers));
    if (service->workers == NULL) {
        aklv_search_service_destroy(service);
        return NULL;
    }

    service->workers[0].service = service;
    int err = pthread_create(&service->workers[0].thread, NULL, aklv_search_coordinator_main, service);
    if (err != 0) {
        aklv_search_service_destroy(service);
        return NULL;
    }

    for (uint32_t i = 0; i < service->thread_count; i++) {
        AklvSearchWorker *worker = &service->workers[i + 1];
        worker->service = service;
        err = pthread_create(&worker->thread, NULL, aklv_search_worker_main, worker);
        if (err != 0) {
            service->thread_count = i;
            aklv_search_service_destroy(service);
            return NULL;
        }
    }
    return service;
}

void aklv_search_service_destroy(AklvSearchService *service) {
    if (service == NULL) {
        return;
    }
    pthread_mutex_lock(&service->mutex);
    service->stop = true;
    service->request_generation++;
    atomic_store_explicit(&service->cancel_stop, true, memory_order_release);
    atomic_store_explicit(&service->cancel_generation, service->request_generation, memory_order_release);
    pthread_cond_broadcast(&service->cond);
    pthread_cond_broadcast(&service->done_cond);
    pthread_mutex_unlock(&service->mutex);

    if (service->workers != NULL) {
        if (service->workers[0].thread != 0) {
            pthread_join(service->workers[0].thread, NULL);
        }
        aklv_result_vec_destroy(&service->workers[0].local_results);
        aklv_worker_destroy_scratch(&service->workers[0]);
        for (uint32_t i = 0; i < service->thread_count; i++) {
            AklvSearchWorker *worker = &service->workers[i + 1];
            if (worker->thread != 0) {
                pthread_join(worker->thread, NULL);
            }
            aklv_result_vec_destroy(&worker->local_results);
            aklv_worker_destroy_scratch(worker);
        }
        free(service->workers);
    }
    aklv_search_job_destroy(&service->queued_job);
    aklv_search_job_destroy(&service->active_job);
    aklv_search_results_destroy(&service->pending);
    pthread_cond_destroy(&service->done_cond);
    pthread_cond_destroy(&service->cond);
    pthread_mutex_destroy(&service->mutex);
    free(service);
}

void aklv_search_service_submit(AklvSearchService *service,
                                const char *query,
                                const AklvFileSnapshot *snapshot) {
    if (service == NULL || query == NULL || query[0] == '\0' || snapshot == NULL || snapshot->count == 0) {
        return;
    }

    AklvSearchJob new_job;
    if (!aklv_search_job_clone(&new_job, query, snapshot)) {
        return;
    }

    pthread_mutex_lock(&service->mutex);
    service->request_generation++;
    new_job.generation = service->request_generation;
    atomic_store_explicit(&service->cancel_stop, false, memory_order_release);
    atomic_store_explicit(&service->cancel_generation, service->request_generation, memory_order_release);
    aklv_search_job_destroy(&service->queued_job);
    service->queued_job = new_job;
    aklv_compiled_query_rebind_inline(&service->queued_job.compiled);
    memset(&new_job, 0, sizeof(new_job));
    service->queued_ready = true;
    aklv_search_results_clear(&service->pending);
    service->pending.generation = service->request_generation;
    service->pending.running = true;
    service->pending.complete = false;
    service->pending.elapsed_sec = 0.0;
    snprintf(service->pending.query, sizeof(service->pending.query), "%s", query);
    service->queued_job.start_ns = aklv_now_ns();
    pthread_cond_broadcast(&service->cond);
    pthread_mutex_unlock(&service->mutex);
    aklv_search_job_destroy(&new_job);
}

void aklv_search_service_cancel(AklvSearchService *service) {
    if (service == NULL) {
        return;
    }
    pthread_mutex_lock(&service->mutex);
    service->request_generation++;
    atomic_store_explicit(&service->cancel_generation, service->request_generation, memory_order_release);
    service->queued_ready = false;
    aklv_search_job_destroy(&service->queued_job);
    aklv_search_results_clear(&service->pending);
    service->pending.generation = service->request_generation;
    service->pending.running = false;
    service->pending.complete = false;
    service->pending.elapsed_sec = 0.0;
    pthread_cond_broadcast(&service->cond);
    pthread_mutex_unlock(&service->mutex);
}

void aklv_search_results_swap_from_service(AklvSearchService *service, AklvSearchResults *target) {
    if (service == NULL || target == NULL) {
        return;
    }
    pthread_mutex_lock(&service->mutex);
    uint64_t pending_generation = service->pending.generation;
    uint64_t pending_count = service->pending.count;
    pthread_mutex_unlock(&service->mutex);

    if (target->generation != pending_generation || target->count > pending_count) {
        target->count = 0;
        target->generation = pending_generation;
    }
    if (pending_count > target->count &&
        !aklv_search_results_reserve(target, pending_count)) {
        return;
    }

    pthread_mutex_lock(&service->mutex);
    if (target->generation != service->pending.generation || target->count > service->pending.count) {
        target->count = 0;
        target->generation = service->pending.generation;
    }
    if (service->pending.count > target->count) {
        uint64_t old_count = target->count;
        if (service->pending.count <= target->capacity) {
            memcpy(target->items + old_count,
                   service->pending.items + old_count,
                   (size_t)(service->pending.count - old_count) * sizeof(*target->items));
            target->count = service->pending.count;
        }
    }
    target->running = service->pending.running;
    target->complete = service->pending.complete;
    target->elapsed_sec = service->pending.elapsed_sec;
    snprintf(target->query, sizeof(target->query), "%s", service->pending.query);
    pthread_mutex_unlock(&service->mutex);
}

int aklv_search_selftest(void) {
    AklvCompiledQuery compiled;
    AklvFileSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    if (!aklv_compiled_query_init(&compiled, "Target", &snapshot)) {
        return 1;
    }
    const unsigned char text[] = "prefix target suffix";
    int rc = aklv_compiled_find(&compiled, text, sizeof(text) - 1) != NULL ? 0 : 1;
    aklv_compiled_query_destroy(&compiled);
    return rc;
}
