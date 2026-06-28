#include "aklv_search.h"

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define AKLV_SEARCH_CHUNK_LINES UINT64_C(2000000)
#define AKLV_SEARCH_CANCEL_CHECK_LINES UINT64_C(200000)
#define AKLV_SIMD_SHORT_QUERY_MAX 16

typedef enum {
    AKLV_QUERY_KERNEL_BMH = 0,
    AKLV_QUERY_KERNEL_SIMD_SHORT
} AklvQueryKernel;

typedef struct {
    unsigned char lower[256];
    unsigned char *pattern;
    size_t pattern_len;
    size_t skip[256];
    AklvQueryKernel kernel;
    size_t anchor_pos;
    unsigned char anchor;
    unsigned char anchor_pair;
    bool anchor_has_pair;
    uint32_t *grams;
    size_t gram_count;
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
    pthread_t thread;
    struct AklvSearchService *service;
    AklvResultVec local_results;
} AklvSearchWorker;

typedef struct {
    uint64_t generation;
    char *query;
    AklvCompiledQuery compiled;
    AklvFileSnapshot snapshot;
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
    uint64_t next_task;

    AklvSearchJob queued_job;
    AklvSearchJob active_job;
    AklvSearchWorker *workers;

    AklvSearchResults pending;
};

static bool aklv_generation_cancelled(const AklvSearchService *service, uint64_t generation);

static uint64_t aklv_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static void aklv_init_ascii_lower(unsigned char lower[256]) {
    for (size_t i = 0; i < 256; i++) {
        lower[i] = (unsigned char)i;
    }
    for (unsigned char c = 'A'; c <= 'Z'; c++) {
        lower[c] = (unsigned char)(c + ('a' - 'A'));
    }
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
        const unsigned char *hit = aklv_find_byte_simd(haystack, haystack_len, c);
        if (c >= 'a' && c <= 'z') {
            const unsigned char *upper_hit =
                aklv_find_byte_simd(haystack, haystack_len, (unsigned char)(c - ('a' - 'A')));
            if (hit == NULL || (upper_hit != NULL && upper_hit < hit)) {
                hit = upper_hit;
            }
        }
        return hit;
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
    free(compiled->pattern);
    free(compiled->grams);
    memset(compiled, 0, sizeof(*compiled));
}

static bool aklv_compiled_query_init_grams(AklvCompiledQuery *compiled) {
    size_t len = compiled->pattern_len;
    if (len == 0) {
        return true;
    }
    size_t max_count = len;
    compiled->grams = malloc(max_count * sizeof(*compiled->grams));
    if (compiled->grams == NULL) {
        return false;
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
        for (size_t pos = run_start; pos + AKLV_ASCII_GRAM_MAX_SIZE <= i; pos++) {
            uint32_t gram = aklv_ngram_key(compiled->pattern + pos, AKLV_ASCII_GRAM_MAX_SIZE);
            bool seen = false;
            for (size_t j = 0; j < compiled->gram_count; j++) {
                if (compiled->grams[j] == gram) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                compiled->grams[compiled->gram_count++] = gram;
            }
        }
    }
    return true;
}

static bool aklv_compiled_query_init(AklvCompiledQuery *compiled,
                                     const char *query,
                                     const AklvFileSnapshot *snapshot) {
    (void)snapshot;
    memset(compiled, 0, sizeof(*compiled));
    aklv_init_ascii_lower(compiled->lower);
    size_t len = strlen(query);
    compiled->pattern_len = len;
    compiled->pattern = malloc(len == 0 ? 1 : len);
    if (compiled->pattern == NULL) {
        return false;
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
    const unsigned char *hit = aklv_find_byte_simd(haystack, haystack_len, compiled->anchor);
    if (compiled->anchor_has_pair) {
        const unsigned char *pair_hit = aklv_find_byte_simd(haystack, haystack_len, compiled->anchor_pair);
        if (hit == NULL || (pair_hit != NULL && pair_hit < hit)) {
            hit = pair_hit;
        }
    }
    return hit;
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
        AklvSearchTask *new_items = realloc(vec->items, (size_t)new_capacity * sizeof(*vec->items));
        if (new_items == NULL) {
            return false;
        }
        vec->items = new_items;
        vec->capacity = new_capacity;
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

static bool aklv_search_results_append_batch(AklvSearchResults *results,
                                             const AklvSearchResult *items,
                                             uint64_t count) {
    if (count == 0) {
        return true;
    }
    if (count > UINT64_MAX - results->count) {
        return false;
    }
    uint64_t needed = results->count + count;
    if (!aklv_search_results_reserve(results, needed)) {
        return false;
    }
    memcpy(results->items + results->count, items, (size_t)count * sizeof(*items));
    results->count = needed;
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
                                      const AklvResultVec *task_results,
                                      uint64_t task_count,
                                      bool complete) {
    pthread_mutex_lock(&service->mutex);
    if (service->stop || service->request_generation != generation) {
        pthread_mutex_unlock(&service->mutex);
        return false;
    }
    service->pending.generation = generation;
    service->pending.running = !complete;
    service->pending.complete = complete;
    service->pending.elapsed_sec = elapsed_sec;
    snprintf(service->pending.query, sizeof(service->pending.query), "%s", query);
    for (uint64_t i = 0; i < task_count; i++) {
        if (!aklv_search_results_append_batch(&service->pending,
                                              task_results[i].items,
                                              task_results[i].count)) {
            pthread_mutex_unlock(&service->mutex);
            return false;
        }
    }
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

static const AklvRoaring *aklv_select_driver_posting(const AklvFile *file,
                                                     const AklvCompiledQuery *compiled) {
    const AklvRoaring *best = NULL;
    uint64_t best_count = UINT64_MAX;
    for (size_t i = 0; i < compiled->gram_count; i++) {
        const AklvRoaring *posting = aklv_gram_index_get(&file->index.gram_index, compiled->grams[i]);
        if (posting == NULL || aklv_roaring_cardinality(posting) == 0) {
            return NULL;
        }
        uint64_t cardinality = aklv_roaring_cardinality(posting);
        if (cardinality < best_count) {
            best = posting;
            best_count = cardinality;
        }
    }
    return best;
}

static bool aklv_candidate_has_all_grams(const AklvFile *file,
                                         const AklvCompiledQuery *compiled,
                                         const AklvRoaring *driver,
                                         uint64_t line_no) {
    for (size_t i = 0; i < compiled->gram_count; i++) {
        const AklvRoaring *posting = aklv_gram_index_get(&file->index.gram_index, compiled->grams[i]);
        if (posting == NULL) {
            return false;
        }
        if (posting != driver && !aklv_roaring_contains(posting, line_no)) {
            return false;
        }
    }
    return true;
}

static bool aklv_collect_file_range_matches(const AklvFile *file,
                                            uint64_t first_line,
                                            uint64_t last_line,
                                            const AklvCompiledQuery *compiled,
                                            uint64_t generation,
                                            const AklvSearchService *service,
                                            AklvResultVec *out) {
    if (file->mapped.size == 0 || file->index.count == 0 || first_line == 0 ||
        compiled->pattern_len == 0) {
        return true;
    }
    if (last_line > file->index.count) {
        last_line = file->index.count;
    }
    if (first_line > last_line) {
        return true;
    }

    const AklvRoaring *driver = aklv_select_driver_posting(file, compiled);
    if (driver == NULL) {
        return true;
    }
    uint64_t next_cancel_check = first_line;
    AklvRoaringIter iter;
    aklv_roaring_iter_init(&iter, driver, first_line);
    uint64_t line_no = 0;
    while (aklv_roaring_iter_next(&iter, &line_no)) {
        if (line_no > last_line) {
            break;
        }
        if (line_no >= next_cancel_check) {
            if (aklv_generation_cancelled(service, generation)) {
                return false;
            }
            uint64_t next = line_no + AKLV_SEARCH_CANCEL_CHECK_LINES;
            next_cancel_check = next > line_no ? next : UINT64_MAX;
        }
        if (!aklv_candidate_has_all_grams(file, compiled, driver, line_no)) {
            continue;
        }
        if (!aklv_line_matches_at(file, line_no, compiled, out)) {
            return false;
        }
    }
    return true;
}

static bool aklv_generation_cancelled(const AklvSearchService *service, uint64_t generation) {
    return atomic_load_explicit(&service->cancel_stop, memory_order_relaxed) ||
           atomic_load_explicit(&service->cancel_generation, memory_order_acquire) != generation;
}

static bool aklv_worker_pop_task(AklvSearchService *service, AklvSearchTask *task_out) {
    bool ok = false;
    pthread_mutex_lock(&service->mutex);
    if (service->next_task < service->task_count) {
        *task_out = service->tasks[service->next_task++];
        ok = true;
    }
    pthread_mutex_unlock(&service->mutex);
    return ok;
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
                aklv_collect_file_range_matches(file,
                                                task.first_line,
                                                task.last_line,
                                                compiled,
                                                generation,
                                                service,
                                                &worker->local_results);
                if (task.seq < service->task_count && !aklv_generation_cancelled(service, generation)) {
                    aklv_result_vec_destroy(&service->task_results[task.seq]);
                    service->task_results[task.seq] = worker->local_results;
                    memset(&worker->local_results, 0, sizeof(worker->local_results));
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

    pthread_mutex_lock(&service->mutex);
    if (service->stop || service->request_generation != generation) {
        pthread_mutex_unlock(&service->mutex);
        aklv_task_results_destroy(task_results, task_count);
        return false;
    }
    service->tasks = tasks;
    service->task_results = task_results;
    service->task_count = task_count;
    service->next_task = 0;
    service->active_workers = service->thread_count;
    service->work_epoch++;
    pthread_cond_broadcast(&service->cond);
    while (service->active_workers > 0) {
        pthread_cond_wait(&service->done_cond, &service->mutex);
    }
    service->tasks = NULL;
    service->task_results = NULL;
    service->task_count = 0;
    service->next_task = 0;
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

static bool aklv_build_all_tasks(const AklvSearchJob *job, AklvSearchTaskVec *tasks) {
    for (uint32_t i = 0; i < job->snapshot.count; i++) {
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
        memset(&service->queued_job, 0, sizeof(service->queued_job));
        service->queued_ready = false;
        service->job_running = true;
        uint64_t generation = service->active_job.generation;
        pthread_mutex_unlock(&service->mutex);

        AklvSearchTaskVec tasks = {0};
        AklvResultVec *phase_results = NULL;
        bool ok = aklv_build_all_tasks(&service->active_job, &tasks);
        if (ok) {
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
        for (uint32_t i = 0; i < service->thread_count; i++) {
            AklvSearchWorker *worker = &service->workers[i + 1];
            if (worker->thread != 0) {
                pthread_join(worker->thread, NULL);
            }
            aklv_result_vec_destroy(&worker->local_results);
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
    pthread_cond_broadcast(&service->done_cond);
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
    pthread_cond_broadcast(&service->done_cond);
    pthread_mutex_unlock(&service->mutex);
}

void aklv_search_results_swap_from_service(AklvSearchService *service, AklvSearchResults *target) {
    if (service == NULL || target == NULL) {
        return;
    }
    pthread_mutex_lock(&service->mutex);
    if (target->generation != service->pending.generation || target->count > service->pending.count) {
        target->count = 0;
        target->generation = service->pending.generation;
    }
    if (service->pending.count > target->count) {
        uint64_t old_count = target->count;
        if (aklv_search_results_reserve(target, service->pending.count)) {
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
