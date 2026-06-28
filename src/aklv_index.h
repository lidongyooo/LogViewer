#ifndef AKLV_INDEX_H
#define AKLV_INDEX_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <pthread.h>
#include <stdint.h>

#define AKLV_LINE_INDEX_BLOCK_SHIFT 16
#define AKLV_LINE_INDEX_BLOCK_LINES (UINT64_C(1) << AKLV_LINE_INDEX_BLOCK_SHIFT)
#define AKLV_LINE_INDEX_BLOCK_MASK (AKLV_LINE_INDEX_BLOCK_LINES - 1)
#define AKLV_LINE_OFFSET_PACK_MASK ((size_t)UINT64_C(0x0000FFFFFFFFFFFF))
#define AKLV_LINE_PREFIX_SHIFT 48
#define AKLV_LINE_PREFIX_OVERFLOW UINT32_C(0xffff)
#define AKLV_SEARCH_SNIPPET_LIMIT 300
#define AKLV_ASCII_GRAM_MAX_SIZE 3
#define AKLV_INDEX_CACHE_VERSION UINT32_C(3)
#ifndef AKLV_INDEX_SHARD_MAX_BYTES
#define AKLV_INDEX_SHARD_MAX_BYTES UINT64_C(1073741824)
#endif

typedef enum {
    AKLV_ROARING_ARRAY = 0,
    AKLV_ROARING_BITMAP = 1,
    AKLV_ROARING_RUN = 2,
    AKLV_ROARING_FULL = 3
} AklvRoaringContainerType;

typedef struct {
    uint16_t start;
    uint16_t length;
} AklvRoaringRun;

typedef struct {
    int fd;
    const unsigned char *data;
    size_t size;
} AklvMappedFile;

typedef struct {
    const unsigned char *start;
    size_t len;
} AklvLineView;

typedef struct {
    uint64_t key;
    uint32_t cardinality;
    uint32_t capacity;
    uint16_t *array;
    uint64_t *bitmap;
    AklvRoaringRun *runs;
    uint32_t run_count;
    AklvRoaringContainerType type;
} AklvRoaringContainer;

typedef struct {
    AklvRoaringContainer *containers;
    uint64_t count;
    uint64_t capacity;
    uint64_t cardinality;
    bool borrowed_containers;
    bool borrowed_payloads;
} AklvRoaring;

typedef struct {
    const AklvRoaring *bitmap;
    uint64_t container_index;
    uint32_t array_index;
    uint32_t bitmap_word_index;
    uint64_t bitmap_word_bits;
    uint16_t min_low;
} AklvRoaringIter;

typedef struct {
    uint32_t gram;
    bool used;
    AklvRoaring lines;
} AklvGramPosting;

typedef struct {
    AklvGramPosting *items;
    uint64_t *used_slots;
    AklvRoaringContainer *container_arena;
    unsigned char *payload_arena;
    uint64_t count;
    uint64_t capacity;
    uint64_t used_capacity;
} AklvGramIndex;

typedef struct {
    uint32_t shard_index;
    uint64_t first_line;
    uint64_t last_line;
    char *path;
    bool gram_index_loaded;
    bool gram_index_loading;
    AklvGramIndex gram_index;
} AklvIndexCacheShard;

typedef struct {
    bool available;
    bool mutex_initialized;
    bool cond_initialized;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    char hash[33];
    char *dir;
    AklvIndexCacheShard *shards;
    uint32_t shard_count;
    uint32_t shard_capacity;
} AklvIndexCache;

typedef struct {
    size_t **offset_blocks;
    size_t *block_max_grams;
    AklvGramIndex gram_index;
    AklvIndexCache cache;
    uint64_t count;
    uint64_t block_count;
    uint64_t block_capacity;
} AklvLineIndex;

typedef struct AklvFile {
    atomic_uint refs;
    uint32_t id;
    char *path;
    char *name;
    AklvMappedFile mapped;
    AklvLineIndex index;
} AklvFile;

unsigned char aklv_fold_ascii_byte(unsigned char c);
const unsigned char *aklv_find_byte_simd(const unsigned char *text, size_t len, unsigned char needle);
bool aklv_is_index_byte(unsigned char c);

static inline size_t aklv_line_index_pack_offset(size_t offset, uint32_t prefix_skip) {
    return (offset & AKLV_LINE_OFFSET_PACK_MASK) | ((size_t)prefix_skip << AKLV_LINE_PREFIX_SHIFT);
}

static inline size_t aklv_line_index_unpack_offset(size_t packed) {
    return packed & AKLV_LINE_OFFSET_PACK_MASK;
}

static inline uint32_t aklv_line_index_unpack_prefix_skip(size_t packed) {
    return (uint32_t)(packed >> AKLV_LINE_PREFIX_SHIFT);
}

char *aklv_strdup(const char *text);
char *aklv_path_basename_dup(const char *path);

int aklv_map_file(const char *path, AklvMappedFile *mapped, char *error, size_t error_cap);
void aklv_unmap_file(AklvMappedFile *mapped);

void aklv_roaring_destroy(AklvRoaring *bitmap);
uint64_t aklv_roaring_cardinality(const AklvRoaring *bitmap);
uint64_t aklv_roaring_lower_bound_container_index(const AklvRoaring *bitmap, uint64_t key);
bool aklv_roaring_container_contains_low(const AklvRoaringContainer *container, uint16_t low);
void aklv_roaring_iter_init(AklvRoaringIter *iter, const AklvRoaring *bitmap, uint64_t min_value);
bool aklv_roaring_iter_next(AklvRoaringIter *iter, uint64_t *value_out);

void aklv_gram_index_destroy(AklvGramIndex *index);
const AklvRoaring *aklv_gram_index_get(const AklvGramIndex *index, uint32_t gram);

void aklv_line_index_destroy(AklvLineIndex *index);
int aklv_build_line_index(const AklvMappedFile *mapped,
                          AklvLineIndex *index,
                          const atomic_bool *cancel,
                          char *error,
                          size_t error_cap);
int aklv_index_cache_try_load(const char *path,
                              const AklvMappedFile *mapped,
                              AklvLineIndex *index,
                              char *error,
                              size_t error_cap);
int aklv_index_cache_store(const char *path,
                           const AklvMappedFile *mapped,
                           AklvLineIndex *index,
                           char *error,
                           size_t error_cap);
int aklv_index_cache_load_shard(const AklvIndexCacheShard *shard,
                                const AklvMappedFile *mapped,
                                AklvGramIndex *gram_index,
                                char *error,
                                size_t error_cap);
int aklv_index_cache_get_shard_index(AklvIndexCache *cache,
                                     AklvIndexCacheShard *shard,
                                     const AklvMappedFile *mapped,
                                     const AklvGramIndex **gram_index,
                                     char *error,
                                     size_t error_cap);
size_t aklv_line_index_offset(const AklvLineIndex *index, uint64_t line_no);
uint32_t aklv_line_index_prefix_skip(const AklvLineIndex *index, uint64_t line_no);

AklvLineView aklv_line_at_offset(const unsigned char *data, size_t size, size_t offset);
AklvLineView aklv_file_line_fast(const AklvFile *file, uint64_t line_no);
AklvLineView aklv_effective_search_line(AklvLineView line);

int aklv_file_open_path(const char *path,
                        uint32_t id,
                        const atomic_bool *cancel,
                        AklvFile **out,
                        char *error,
                        size_t error_cap);
void aklv_file_retain(AklvFile *file);
void aklv_file_release(AklvFile *file);

size_t aklv_copy_line_snippet(AklvLineView line, char *out, size_t out_cap, bool *truncated_out);

#endif
