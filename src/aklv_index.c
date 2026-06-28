#include "aklv_index.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <CommonCrypto/CommonDigest.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#define AKLV_GRAM_SCRATCH_INITIAL_CAPACITY 128
#define AKLV_GRAM_SCRATCH_INITIAL_USED_CAPACITY 512
#define AKLV_INDEX_CACHE_DIR "indexs"
#define AKLV_INDEX_CACHE_MAGIC "AKLVIDX1"
#define AKLV_INDEX_CACHE_HEADER_SIZE 4096
#define AKLV_CACHE_STREAM_BUFFER_BYTES (50 * 1024 * 1024)
#define AKLV_CACHE_U16_WRITE_BATCH (8192 * 50)
#define AKLV_CACHE_RUN_WRITE_BATCH (2048 * 50)
#define AKLV_CACHE_OFFSET_CONVERT_BATCH (8192 * 50)
#define AKLV_ASCII_GRAM_KEY_SPACE UINT64_C(2097152)
#define AKLV_ASCII_GRAM_KEY_SPACE_WORDS (AKLV_ASCII_GRAM_KEY_SPACE / 64)
#ifndef AKLV_WORKER_GRAM_RESERVE_BYTES_PER_GRAM
#define AKLV_WORKER_GRAM_RESERVE_BYTES_PER_GRAM UINT64_C(512)
#endif
#ifndef AKLV_WORKER_GRAM_RESERVE_MAX
#define AKLV_WORKER_GRAM_RESERVE_MAX UINT64_C(131072)
#endif
#ifndef AKLV_SYNC_CACHE_STORE_MAX_FILE_BYTES
#define AKLV_SYNC_CACHE_STORE_MAX_FILE_BYTES UINT64_C(134217728)
#endif

typedef struct {
    uint32_t *items;
    uint64_t count;
    uint64_t capacity;
    uint64_t *seen_words;
    uint32_t *used_slots;
    uint64_t used_count;
    uint64_t used_capacity;
} AklvGramScratch;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t shard_index;
    uint64_t file_size;
    uint64_t file_mtime_sec;
    uint64_t file_mtime_nsec;
    uint64_t line_count;
    uint64_t first_line;
    uint64_t last_line;
    uint64_t offset_count;
    uint64_t gram_count;
    uint64_t payload_bytes;
    uint32_t gram_size;
    uint32_t total_container_count;
} AklvIndexCacheFileHeader;

typedef struct {
    uint32_t gram;
    uint32_t container_count;
    uint64_t cardinality;
} AklvIndexCachePostingHeader;

typedef struct {
    uint64_t key;
    uint32_t cardinality;
    uint32_t type;
    uint32_t payload_count;
    uint32_t reserved;
} AklvIndexCacheContainerHeader;

typedef struct {
    uint64_t file_size;
    uint64_t mtime_sec;
    uint64_t mtime_nsec;
} AklvCacheFileIdentity;

typedef struct {
    uint32_t cardinality;
    uint32_t run_count;
    AklvRoaringContainerType type;
    uint32_t payload_count;
    size_t payload_bytes;
    uint16_t single_run_start;
    uint16_t single_run_end;
    bool single_run_direct;
    bool full_container_direct;
} AklvContainerRangePlan;

typedef struct {
    uint32_t container_count;
    uint32_t reserved;
    uint64_t cardinality;
    uint64_t serialized_bytes;
    uint64_t payload_bytes;
} AklvCachePostingRangePlan;

typedef struct {
    const AklvRoaringContainer *container;
    AklvContainerRangePlan plan;
    uint16_t min_low;
    uint16_t max_low;
} AklvCacheContainerWritePlan;

typedef struct {
    uint64_t *key_bytes;
    uint64_t key_count;
    uint64_t *bucket_offsets;
    uint32_t *posting_indices;
    uint64_t posting_index_count;
} AklvCacheKeyPlan;

typedef struct {
    unsigned char *cursor;
    unsigned char *end;
} AklvCachePayloadArena;

typedef struct {
    uint16_t *u16_values;
    AklvRoaringRun *runs;
    uint64_t *offset_values;
} AklvCacheWriteScratch;

static uint64_t aklv_cache_saturating_add_u64(uint64_t value, uint64_t add);
static uint32_t aklv_gram_dense_index(uint32_t gram);
static uint32_t aklv_lower_bound_u16(const uint16_t *items, uint32_t count, uint16_t value);

static void aklv_set_error(char *error, size_t error_cap, const char *fmt, ...) {
    if (error == NULL || error_cap == 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(error, error_cap, fmt, args);
    va_end(args);
}

static bool aklv_write_all(FILE *stream, const void *data, size_t size) {
    return size == 0 || fwrite(data, 1, size, stream) == size;
}

static bool aklv_read_all(FILE *stream, void *data, size_t size) {
    return size == 0 || fread(data, 1, size, stream) == size;
}

static bool aklv_file_stat_identity(const char *path,
                                    uint64_t *size_out,
                                    uint64_t *mtime_sec_out,
                                    uint64_t *mtime_nsec_out) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }
    if (size_out != NULL) {
        *size_out = st.st_size < 0 ? 0 : (uint64_t)st.st_size;
    }
    if (mtime_sec_out != NULL) {
#if defined(__APPLE__)
        *mtime_sec_out = (uint64_t)st.st_mtimespec.tv_sec;
#else
        *mtime_sec_out = (uint64_t)st.st_mtim.tv_sec;
#endif
    }
    if (mtime_nsec_out != NULL) {
#if defined(__APPLE__)
        *mtime_nsec_out = (uint64_t)st.st_mtimespec.tv_nsec;
#else
        *mtime_nsec_out = (uint64_t)st.st_mtim.tv_nsec;
#endif
    }
    return true;
}

static bool aklv_cache_dir_ensure(char *error, size_t error_cap) {
    struct stat st;
    if (stat(AKLV_INDEX_CACHE_DIR, &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            return true;
        }
        aklv_set_error(error, error_cap, "%s exists but is not a directory", AKLV_INDEX_CACHE_DIR);
        return false;
    }
    if (errno != ENOENT) {
        aklv_set_error(error, error_cap, "stat %s failed: %s", AKLV_INDEX_CACHE_DIR, strerror(errno));
        return false;
    }
    if (mkdir(AKLV_INDEX_CACHE_DIR, 0755) != 0 && errno != EEXIST) {
        aklv_set_error(error, error_cap, "mkdir %s failed: %s", AKLV_INDEX_CACHE_DIR, strerror(errno));
        return false;
    }
    return true;
}

static bool aklv_index_cache_hash_path(const char *path, char out[33]) {
    char resolved[PATH_MAX];
    char hash_input[PATH_MAX * 2];
    const char *full_path = path;
    if (realpath(path, resolved) != NULL) {
        full_path = resolved;
    }
    const char *slash = strrchr(full_path, '/');
    const char *name = slash == NULL ? full_path : slash + 1;
    snprintf(hash_input, sizeof(hash_input), "%s%s", full_path, name);
    unsigned char digest[CC_MD5_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CC_MD5(hash_input, (CC_LONG)strlen(hash_input), digest);
#pragma clang diagnostic pop
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(digest); i++) {
        out[i * 2] = hex[digest[i] >> 4];
        out[i * 2 + 1] = hex[digest[i] & 0xf];
    }
    out[32] = '\0';
    return true;
}

static char *aklv_index_cache_path_for(const char hash[33], uint32_t shard_index) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s_%u.index", AKLV_INDEX_CACHE_DIR, hash, shard_index);
    return aklv_strdup(path);
}

static atomic_uint_fast64_t g_aklv_cache_tmp_counter = 1;

static void aklv_index_cache_destroy(AklvIndexCache *cache) {
    if (cache == NULL) {
        return;
    }
    if (cache->shards != NULL) {
        for (uint32_t i = 0; i < cache->shard_count; i++) {
            aklv_gram_index_destroy(&cache->shards[i].gram_index);
            free(cache->shards[i].path);
        }
        free(cache->shards);
    }
    if (cache->cond_initialized) {
        pthread_cond_destroy(&cache->cond);
    }
    if (cache->mutex_initialized) {
        pthread_mutex_destroy(&cache->mutex);
    }
    free(cache->dir);
    memset(cache, 0, sizeof(*cache));
}

static bool aklv_index_cache_add_shard(AklvIndexCache *cache,
                                       uint32_t shard_index,
                                       uint64_t first_line,
                                       uint64_t last_line,
                                       const char *path) {
    if (cache->shard_count == cache->shard_capacity) {
        uint32_t new_capacity = cache->shard_capacity == 0 ? 8 : cache->shard_capacity * 2;
        if (new_capacity < cache->shard_capacity) {
            return false;
        }
        AklvIndexCacheShard *new_items =
            realloc(cache->shards, (size_t)new_capacity * sizeof(*new_items));
        if (new_items == NULL) {
            return false;
        }
        cache->shards = new_items;
        cache->shard_capacity = new_capacity;
    }
    AklvIndexCacheShard *shard = &cache->shards[cache->shard_count++];
    memset(shard, 0, sizeof(*shard));
    shard->shard_index = shard_index;
    shard->first_line = first_line;
    shard->last_line = last_line;
    shard->path = aklv_strdup(path);
    if (shard->path == NULL) {
        cache->shard_count--;
        return false;
    }
    return true;
}

unsigned char aklv_fold_ascii_byte(unsigned char c) {
    if (c >= 'A' && c <= 'Z') {
        return (unsigned char)(c + ('a' - 'A'));
    }
    return c;
}

bool aklv_is_index_byte(unsigned char c) {
    return c < 128;
}

const unsigned char *aklv_find_byte_simd(const unsigned char *text, size_t len, unsigned char needle) {
    if (text == NULL || len == 0) {
        return NULL;
    }
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const uint8x16_t target = vdupq_n_u8(needle);
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        uint8x16_t bytes = vld1q_u8(text + i);
        uint8x16_t eq = vceqq_u8(bytes, target);
        uint64x2_t lanes = vreinterpretq_u64_u8(eq);
        if ((vgetq_lane_u64(lanes, 0) | vgetq_lane_u64(lanes, 1)) != 0) {
            for (size_t j = 0; j < 16; j++) {
                if (text[i + j] == needle) {
                    return text + i + j;
                }
            }
        }
    }
    for (; i < len; i++) {
        if (text[i] == needle) {
            return text + i;
        }
    }
    return NULL;
#elif defined(__SSE2__)
    const __m128i target = _mm_set1_epi8((char)needle);
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        __m128i bytes = _mm_loadu_si128((const __m128i *)(const void *)(text + i));
        __m128i eq = _mm_cmpeq_epi8(bytes, target);
        int mask = _mm_movemask_epi8(eq);
        if (mask != 0) {
            return text + i + (size_t)__builtin_ctz((unsigned)mask);
        }
    }
    for (; i < len; i++) {
        if (text[i] == needle) {
            return text + i;
        }
    }
    return NULL;
#else
    return memchr(text, needle, len);
#endif
}

static void aklv_roaring_container_destroy(AklvRoaringContainer *container, bool borrowed_payloads) {
    if (container == NULL) {
        return;
    }
    if (!borrowed_payloads) {
        free(container->array);
        free(container->bitmap);
        free(container->runs);
    }
    memset(container, 0, sizeof(*container));
}

static bool aklv_roaring_container_promote(AklvRoaringContainer *container) {
    uint64_t *bitmap = calloc(1024, sizeof(*bitmap));
    if (bitmap == NULL) {
        return false;
    }
    for (uint32_t i = 0; i < container->cardinality; i++) {
        uint16_t low = container->array[i];
        bitmap[low >> 6] |= UINT64_C(1) << (low & 63U);
    }
    free(container->array);
    container->array = NULL;
    container->capacity = 0;
    container->bitmap = bitmap;
    container->type = AKLV_ROARING_BITMAP;
    return true;
}

static bool aklv_roaring_container_add(AklvRoaringContainer *container, uint16_t low) {
    if (container->type == AKLV_ROARING_FULL) {
        return true;
    }
    if (container->type == AKLV_ROARING_RUN) {
        uint32_t lo = 0;
        uint32_t hi = container->run_count;
        uint32_t value = low;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t start = container->runs[mid].start;
            uint32_t end = start + container->runs[mid].length;
            if (value < start) {
                hi = mid;
            } else if (value >= end) {
                lo = mid + 1;
            } else {
                return true;
            }
        }
        return false;
    }
    if (container->bitmap != NULL) {
        uint64_t mask = UINT64_C(1) << (low & 63U);
        uint64_t *word = &container->bitmap[low >> 6];
        if ((*word & mask) != 0) {
            return true;
        }
        *word |= mask;
        container->cardinality++;
        return true;
    }

    if (container->cardinality > 0 && container->array[container->cardinality - 1] == low) {
        return true;
    }
    if (container->cardinality >= 4096) {
        if (!aklv_roaring_container_promote(container)) {
            return false;
        }
        return aklv_roaring_container_add(container, low);
    }
    if (container->cardinality == container->capacity) {
        uint32_t new_capacity = container->capacity == 0 ? 4 : container->capacity * 2;
        if (new_capacity < container->capacity) {
            return false;
        }
        uint16_t *new_array = realloc(container->array, (size_t)new_capacity * sizeof(*new_array));
        if (new_array == NULL) {
            return false;
        }
        container->array = new_array;
        container->capacity = new_capacity;
    }
    container->array[container->cardinality++] = low;
    container->type = AKLV_ROARING_ARRAY;
    return true;
}

static AklvRoaringContainer *aklv_roaring_append_container(AklvRoaring *bitmap, uint64_t key) {
    if (bitmap->count == bitmap->capacity) {
        uint64_t new_capacity = bitmap->capacity == 0 ? 1 : bitmap->capacity * 2;
        if (new_capacity < bitmap->capacity ||
            new_capacity > (uint64_t)(SIZE_MAX / sizeof(*bitmap->containers))) {
            return NULL;
        }
        AklvRoaringContainer *new_items =
            realloc(bitmap->containers, (size_t)new_capacity * sizeof(*new_items));
        if (new_items == NULL) {
            return NULL;
        }
        memset(new_items + bitmap->capacity,
               0,
               (size_t)(new_capacity - bitmap->capacity) * sizeof(*new_items));
        bitmap->containers = new_items;
        bitmap->capacity = new_capacity;
    }
    AklvRoaringContainer *container = &bitmap->containers[bitmap->count++];
    memset(container, 0, sizeof(*container));
    container->key = key;
    return container;
}

static bool aklv_roaring_add(AklvRoaring *bitmap, uint64_t value) {
    uint64_t key = value >> 16;
    uint16_t low = (uint16_t)(value & 0xffffU);
    AklvRoaringContainer *container = NULL;
    if (bitmap->count > 0 && bitmap->containers[bitmap->count - 1].key == key) {
        container = &bitmap->containers[bitmap->count - 1];
    } else {
        container = aklv_roaring_append_container(bitmap, key);
        if (container == NULL) {
            return false;
        }
    }
    uint32_t before = container->cardinality;
    if (!aklv_roaring_container_add(container, low)) {
        return false;
    }
    bitmap->cardinality += container->cardinality - before;
    return true;
}

static uint64_t aklv_bit_range_mask(uint32_t low_bit, uint32_t high_bit) {
    uint64_t mask = UINT64_MAX << low_bit;
    if (high_bit < 63U) {
        mask &= (UINT64_C(1) << (high_bit + 1U)) - 1U;
    }
    return mask;
}

static bool aklv_roaring_container_add_all(AklvRoaringContainer *dst,
                                           const AklvRoaringContainer *src) {
    uint32_t before = dst->cardinality;
    if (src->type == AKLV_ROARING_FULL) {
        if (dst->type != AKLV_ROARING_FULL) {
            aklv_roaring_container_destroy(dst, false);
            dst->cardinality = 65536;
            dst->type = AKLV_ROARING_FULL;
        }
    } else if (src->type == AKLV_ROARING_RUN) {
        for (uint32_t i = 0; i < src->run_count; i++) {
            uint32_t start = src->runs[i].start;
            uint32_t end = start + src->runs[i].length;
            if (dst->bitmap != NULL) {
                uint32_t first_word = start >> 6;
                uint32_t last_word = (end - 1U) >> 6;
                for (uint32_t word_i = first_word; word_i <= last_word; word_i++) {
                    uint32_t word_start = word_i << 6;
                    uint32_t word_end = word_start + 63U;
                    uint32_t lo = start > word_start ? start : word_start;
                    uint32_t hi = (end - 1U) < word_end ? (end - 1U) : word_end;
                    uint64_t mask = aklv_bit_range_mask(lo & 63U, hi & 63U);
                    uint64_t before_word = dst->bitmap[word_i];
                    uint64_t add = mask & ~before_word;
                    dst->bitmap[word_i] = before_word | mask;
                    dst->cardinality += (uint32_t)__builtin_popcountll(add);
                }
                continue;
            }
            for (uint32_t low = start; low < end; low++) {
                if (!aklv_roaring_container_add(dst, (uint16_t)low)) {
                    return false;
                }
            }
        }
    } else if (src->bitmap != NULL) {
        if (dst->bitmap != NULL) {
            for (uint32_t word_i = 0; word_i < 1024; word_i++) {
                uint64_t before_word = dst->bitmap[word_i];
                uint64_t add = src->bitmap[word_i] & ~before_word;
                dst->bitmap[word_i] = before_word | src->bitmap[word_i];
                dst->cardinality += (uint32_t)__builtin_popcountll(add);
            }
        } else {
            for (uint32_t word_i = 0; word_i < 1024; word_i++) {
                uint64_t bits = src->bitmap[word_i];
                while (bits != 0) {
                    uint32_t bit = (uint32_t)__builtin_ctzll(bits);
                    bits &= bits - 1;
                    if (!aklv_roaring_container_add(dst, (uint16_t)((word_i << 6) | bit))) {
                        return false;
                    }
                }
            }
        }
    } else {
        for (uint32_t i = 0; i < src->cardinality; i++) {
            if (!aklv_roaring_container_add(dst, src->array[i])) {
                return false;
            }
        }
    }
    if (dst->cardinality > 65536) {
        dst->cardinality = 65536;
    }
    (void)before;
    return true;
}

static bool aklv_roaring_move_append(AklvRoaring *dst, AklvRoaring *src) {
    if (src->count == 0) {
        return true;
    }
    if (dst->count == 0) {
        *dst = *src;
        memset(src, 0, sizeof(*src));
        return true;
    }

    uint64_t move_start = 0;
    AklvRoaringContainer *dst_last = &dst->containers[dst->count - 1];
    if (dst_last->key == src->containers[0].key) {
        uint32_t before = dst_last->cardinality;
        if (!aklv_roaring_container_add_all(dst_last, &src->containers[0])) {
            return false;
        }
        dst->cardinality += dst_last->cardinality - before;
        aklv_roaring_container_destroy(&src->containers[0], src->borrowed_payloads);
        move_start = 1;
    }

    uint64_t move_count = src->count - move_start;
    if (move_count == 0) {
        free(src->containers);
        memset(src, 0, sizeof(*src));
        return true;
    }
    if (dst->count + move_count < dst->count ||
        dst->count + move_count > (uint64_t)(SIZE_MAX / sizeof(*dst->containers))) {
        return false;
    }
    if (dst->count + move_count > dst->capacity) {
        uint64_t new_capacity = dst->capacity == 0 ? 1 : dst->capacity;
        while (new_capacity < dst->count + move_count) {
            if (new_capacity > UINT64_MAX / 2) {
                return false;
            }
            new_capacity *= 2;
        }
        AklvRoaringContainer *new_items =
            realloc(dst->containers, (size_t)new_capacity * sizeof(*new_items));
        if (new_items == NULL) {
            return false;
        }
        dst->containers = new_items;
        dst->capacity = new_capacity;
    }
    memcpy(dst->containers + dst->count,
           src->containers + move_start,
           (size_t)move_count * sizeof(*dst->containers));
    for (uint64_t i = move_start; i < src->count; i++) {
        dst->cardinality += src->containers[i].cardinality;
    }
    dst->count += move_count;
    free(src->containers);
    memset(src, 0, sizeof(*src));
    return true;
}

void aklv_roaring_destroy(AklvRoaring *bitmap) {
    if (bitmap == NULL) {
        return;
    }
    for (uint64_t i = 0; i < bitmap->count; i++) {
        aklv_roaring_container_destroy(&bitmap->containers[i], bitmap->borrowed_payloads);
    }
    if (!bitmap->borrowed_containers) {
        free(bitmap->containers);
    }
    memset(bitmap, 0, sizeof(*bitmap));
}

uint64_t aklv_roaring_cardinality(const AklvRoaring *bitmap) {
    return bitmap == NULL ? 0 : bitmap->cardinality;
}

static const AklvRoaringContainer *aklv_roaring_find_container(const AklvRoaring *bitmap,
                                                              uint64_t key,
                                                              uint64_t *index_out) {
    if (bitmap == NULL || bitmap->count == 0) {
        return NULL;
    }
    uint64_t lo = 0;
    uint64_t hi = bitmap->count;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        uint64_t mid_key = bitmap->containers[mid].key;
        if (mid_key < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < bitmap->count && bitmap->containers[lo].key == key) {
        if (index_out != NULL) {
            *index_out = lo;
        }
        return &bitmap->containers[lo];
    }
    if (index_out != NULL) {
        *index_out = lo;
    }
    return NULL;
}

uint64_t aklv_roaring_lower_bound_container_index(const AklvRoaring *bitmap, uint64_t key) {
    uint64_t index = 0;
    (void)aklv_roaring_find_container(bitmap, key, &index);
    return index;
}

bool aklv_roaring_container_contains_low(const AklvRoaringContainer *container, uint16_t low) {
    if (container == NULL) {
        return false;
    }
    if (container->type == AKLV_ROARING_FULL) {
        return true;
    }
    if (container->type == AKLV_ROARING_RUN) {
        uint32_t lo = 0;
        uint32_t hi = container->run_count;
        uint32_t value = low;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t start = container->runs[mid].start;
            uint32_t end = start + container->runs[mid].length;
            if (value < start) {
                hi = mid;
            } else if (value >= end) {
                lo = mid + 1;
            } else {
                return true;
            }
        }
        return false;
    }
    if (container->bitmap != NULL) {
        return (container->bitmap[low >> 6] & (UINT64_C(1) << (low & 63U))) != 0;
    }
    uint32_t lo = 0;
    uint32_t hi = container->cardinality;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (container->array[mid] < low) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < container->cardinality && container->array[lo] == low;
}

void aklv_roaring_iter_init(AklvRoaringIter *iter, const AklvRoaring *bitmap, uint64_t min_value) {
    memset(iter, 0, sizeof(*iter));
    iter->bitmap = bitmap;
    if (bitmap == NULL || bitmap->count == 0) {
        return;
    }
    uint64_t key = min_value >> 16;
    uint16_t low = (uint16_t)(min_value & 0xffffU);
    uint64_t index = 0;
    const AklvRoaringContainer *container = aklv_roaring_find_container(bitmap, key, &index);
    iter->container_index = index;
    if (container == NULL) {
        return;
    }
    switch (container->type) {
        case AKLV_ROARING_FULL:
            iter->array_index = low;
            break;
        case AKLV_ROARING_RUN: {
            uint32_t lo = 0;
            uint32_t hi = container->run_count;
            while (lo < hi) {
                uint32_t mid = lo + (hi - lo) / 2;
                uint32_t start = container->runs[mid].start;
                uint32_t end = start + container->runs[mid].length;
                if (end <= low) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            iter->bitmap_word_index = lo;
            iter->min_low = low;
            break;
        }
        case AKLV_ROARING_BITMAP:
            iter->bitmap_word_index = low >> 6;
            iter->min_low = low;
            break;
        case AKLV_ROARING_ARRAY:
        default:
            iter->array_index = aklv_lower_bound_u16(container->array, container->cardinality, low);
            break;
    }
}

bool aklv_roaring_iter_next(AklvRoaringIter *iter, uint64_t *value_out) {
    if (iter == NULL || iter->bitmap == NULL || value_out == NULL) {
        return false;
    }
    while (iter->container_index < iter->bitmap->count) {
        const AklvRoaringContainer *container = &iter->bitmap->containers[iter->container_index];
        if (container->type == AKLV_ROARING_FULL) {
            uint32_t low = iter->array_index;
            if (iter->min_low != 0 && low < iter->min_low) {
                low = iter->min_low;
            }
            iter->min_low = 0;
            if (low <= UINT16_MAX) {
                iter->array_index = low + 1;
                *value_out = (container->key << 16) | (uint64_t)(uint16_t)low;
                return true;
            }
        } else if (container->type == AKLV_ROARING_RUN) {
            while (iter->bitmap_word_index < container->run_count) {
                AklvRoaringRun run = container->runs[iter->bitmap_word_index];
                uint32_t run_start = run.start;
                uint32_t run_end = run_start + run.length;
                uint32_t low = iter->array_index == 0 ? run_start : iter->array_index;
                if (iter->min_low != 0 && low < iter->min_low) {
                    low = iter->min_low;
                }
                if (low < run_end) {
                    iter->min_low = 0;
                    iter->array_index = low + 1;
                    *value_out = (container->key << 16) | (uint64_t)(uint16_t)low;
                    return true;
                }
                iter->bitmap_word_index++;
                iter->array_index = 0;
            }
        } else if (container->bitmap == NULL) {
            while (iter->array_index < container->cardinality &&
                   container->array[iter->array_index] < iter->min_low) {
                iter->array_index++;
            }
            iter->min_low = 0;
            if (iter->array_index < container->cardinality) {
                uint16_t low = container->array[iter->array_index++];
                *value_out = (container->key << 16) | (uint64_t)low;
                return true;
            }
        } else {
            if (iter->bitmap_word_bits == 0) {
                uint32_t start_word = iter->bitmap_word_index;
                if (iter->min_low != 0) {
                    start_word = iter->min_low >> 6;
                }
                for (uint32_t word_index = start_word; word_index < 1024; word_index++) {
                    uint64_t bits = container->bitmap[word_index];
                    if (iter->min_low != 0 && word_index == (uint32_t)(iter->min_low >> 6)) {
                        uint32_t skip = iter->min_low & 63U;
                        if (skip != 0) {
                            bits &= UINT64_MAX << skip;
                        }
                    }
                    if (bits != 0) {
                        iter->bitmap_word_index = word_index;
                        iter->bitmap_word_bits = bits;
                        break;
                    }
                }
                iter->min_low = 0;
            }
            if (iter->bitmap_word_bits != 0) {
                uint32_t word_index = iter->bitmap_word_index;
                uint32_t bit = (uint32_t)__builtin_ctzll(iter->bitmap_word_bits);
                iter->bitmap_word_bits &= iter->bitmap_word_bits - 1;
                if (iter->bitmap_word_bits == 0) {
                    iter->bitmap_word_index++;
                }
                uint16_t low = (uint16_t)((word_index << 6) | bit);
                *value_out = (container->key << 16) | (uint64_t)low;
                return true;
            }
        }
        iter->container_index++;
        iter->array_index = 0;
        iter->bitmap_word_index = 0;
        iter->bitmap_word_bits = 0;
        iter->min_low = 0;
    }
    return false;
}

static uint64_t aklv_gram_hash(uint32_t gram) {
    uint64_t x = (uint64_t)gram + UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static void aklv_gram_scratch_destroy(AklvGramScratch *scratch) {
    if (scratch == NULL) {
        return;
    }
    free(scratch->items);
    free(scratch->seen_words);
    free(scratch->used_slots);
    memset(scratch, 0, sizeof(*scratch));
}

static void aklv_gram_scratch_reset(AklvGramScratch *scratch) {
    for (uint64_t i = 0; i < scratch->used_count; i++) {
        scratch->seen_words[scratch->used_slots[i]] = 0;
    }
    scratch->count = 0;
    scratch->used_count = 0;
}

static bool aklv_gram_scratch_reserve_items(AklvGramScratch *scratch, uint64_t needed) {
    if (needed <= scratch->capacity) {
        return true;
    }
    uint64_t new_capacity = scratch->capacity == 0 ? AKLV_GRAM_SCRATCH_INITIAL_CAPACITY : scratch->capacity;
    while (new_capacity < needed) {
        if (new_capacity > UINT64_MAX / 2) {
            return false;
        }
        new_capacity *= 2;
    }
    if (new_capacity > (uint64_t)(SIZE_MAX / sizeof(*scratch->items))) {
        return false;
    }
    uint32_t *new_items = realloc(scratch->items, (size_t)new_capacity * sizeof(*new_items));
    if (new_items == NULL) {
        return false;
    }
    scratch->items = new_items;
    scratch->capacity = new_capacity;
    return true;
}

static bool aklv_gram_scratch_reserve_used_slots(AklvGramScratch *scratch, uint64_t needed) {
    if (needed <= scratch->used_capacity) {
        return true;
    }
    uint64_t new_capacity = scratch->used_capacity == 0 ? AKLV_GRAM_SCRATCH_INITIAL_USED_CAPACITY : scratch->used_capacity;
    while (new_capacity < needed) {
        if (new_capacity > UINT64_MAX / 2) {
            return false;
        }
        new_capacity *= 2;
    }
    if (new_capacity > (uint64_t)(SIZE_MAX / sizeof(*scratch->used_slots))) {
        return false;
    }
    uint32_t *new_slots = realloc(scratch->used_slots, (size_t)new_capacity * sizeof(*new_slots));
    if (new_slots == NULL) {
        return false;
    }
    scratch->used_slots = new_slots;
    scratch->used_capacity = new_capacity;
    return true;
}

static bool aklv_gram_scratch_ensure_seen_words(AklvGramScratch *scratch) {
    if (scratch->seen_words != NULL) {
        return true;
    }
    scratch->seen_words = calloc((size_t)AKLV_ASCII_GRAM_KEY_SPACE_WORDS,
                                 sizeof(*scratch->seen_words));
    return scratch->seen_words != NULL;
}

static uint32_t aklv_gram_dense_index(uint32_t gram) {
    return ((gram >> 16) & 0xffU) << 14 |
           ((gram >> 8) & 0x7fU) << 7 |
           (gram & 0x7fU);
}

static int aklv_gram_scratch_prepare_line(AklvGramScratch *scratch,
                                          size_t max_grams,
                                          char *error,
                                          size_t error_cap) {
    uint64_t max_used_words = max_grams > AKLV_ASCII_GRAM_KEY_SPACE_WORDS
        ? AKLV_ASCII_GRAM_KEY_SPACE_WORDS
        : (uint64_t)max_grams;
    if (!aklv_gram_scratch_ensure_seen_words(scratch) ||
        !aklv_gram_scratch_reserve_items(scratch, max_grams) ||
        !aklv_gram_scratch_reserve_used_slots(scratch, max_used_words)) {
        aklv_set_error(error, error_cap, "failed to grow line gram scratch");
        return -1;
    }
    return 0;
}

static void aklv_gram_scratch_add_prepared(AklvGramScratch *scratch, uint32_t gram) {
    uint32_t dense = aklv_gram_dense_index(gram);
    uint32_t word_index = dense >> 6;
    uint64_t mask = UINT64_C(1) << (dense & 63U);
    uint64_t before = scratch->seen_words[word_index];
    if ((before & mask) != 0) {
        return;
    }
    if (before == 0) {
        scratch->used_slots[scratch->used_count++] = word_index;
    }
    scratch->seen_words[word_index] = before | mask;
    scratch->items[scratch->count++] = gram;
}

static bool aklv_gram_index_reserve_used_slots(AklvGramIndex *index, uint64_t needed) {
    if (needed <= index->used_capacity) {
        return true;
    }
    uint64_t new_capacity = index->used_capacity == 0 ? 1024 : index->used_capacity;
    while (new_capacity < needed) {
        if (new_capacity > UINT64_MAX / 2) {
            return false;
        }
        new_capacity *= 2;
    }
    if (new_capacity > (uint64_t)(SIZE_MAX / sizeof(*index->used_slots))) {
        return false;
    }
    uint64_t *new_slots = realloc(index->used_slots, (size_t)new_capacity * sizeof(*new_slots));
    if (new_slots == NULL) {
        return false;
    }
    index->used_slots = new_slots;
    index->used_capacity = new_capacity;
    return true;
}

static bool aklv_gram_index_rehash(AklvGramIndex *index, uint64_t new_capacity) {
    if (new_capacity > (uint64_t)(SIZE_MAX / sizeof(*index->items))) {
        return false;
    }
    AklvGramPosting *old_items = index->items;
    uint64_t *old_used_slots = index->used_slots;
    uint64_t old_count = index->count;
    uint64_t old_capacity = index->capacity;
    AklvGramPosting *new_items = calloc((size_t)new_capacity, sizeof(*new_items));
    if (new_items == NULL) {
        return false;
    }

    uint64_t new_used_capacity = index->used_capacity;
    if (new_used_capacity < old_count) {
        new_used_capacity = old_count;
    }
    uint64_t *new_used_slots = NULL;
    if (new_used_capacity > 0) {
        if (new_used_capacity > (uint64_t)(SIZE_MAX / sizeof(*new_used_slots))) {
            free(new_items);
            return false;
        }
        new_used_slots = malloc((size_t)new_used_capacity * sizeof(*new_used_slots));
        if (new_used_slots == NULL) {
            free(new_items);
            return false;
        }
    }
    if (old_count > 0 && old_used_slots == NULL) {
        free(new_items);
        free(new_used_slots);
        return false;
    }

    index->items = new_items;
    index->used_slots = new_used_slots;
    index->capacity = new_capacity;
    index->used_capacity = new_used_capacity;
    index->count = 0;
    if (old_items != NULL && old_count > 0) {
        for (uint64_t used_i = 0; used_i < old_count; used_i++) {
            uint64_t old_slot = old_used_slots[used_i];
            if (old_slot >= old_capacity || !old_items[old_slot].used) {
                continue;
            }
            uint64_t slot = aklv_gram_hash(old_items[old_slot].gram) & (new_capacity - 1);
            while (new_items[slot].used) {
                slot = (slot + 1) & (new_capacity - 1);
            }
            new_items[slot] = old_items[old_slot];
            memset(&old_items[old_slot], 0, sizeof(old_items[old_slot]));
            new_used_slots[index->count++] = slot;
        }
    }
    free(old_items);
    free(old_used_slots);
    return true;
}

static AklvGramPosting *aklv_gram_index_slot(AklvGramIndex *index, uint32_t gram, bool create) {
    if (index->capacity == 0) {
        if (!create || !aklv_gram_index_rehash(index, 1024)) {
            return NULL;
        }
    }
    if (create && (index->count + 1) * 10 >= index->capacity * 7) {
        if (index->capacity > UINT64_MAX / 2 ||
            !aklv_gram_index_rehash(index, index->capacity * 2)) {
            return NULL;
        }
    }

    uint64_t slot = aklv_gram_hash(gram) & (index->capacity - 1);
    while (index->items[slot].used) {
        if (index->items[slot].gram == gram) {
            return &index->items[slot];
        }
        slot = (slot + 1) & (index->capacity - 1);
    }
    if (!create) {
        return NULL;
    }
    if (!aklv_gram_index_reserve_used_slots(index, index->count + 1)) {
        return NULL;
    }
    index->items[slot].used = true;
    index->items[slot].gram = gram;
    index->used_slots[index->count++] = slot;
    return &index->items[slot];
}

static bool aklv_gram_index_add(AklvGramIndex *index, uint32_t gram, uint64_t line_no) {
    AklvGramPosting *posting = aklv_gram_index_slot(index, gram, true);
    if (posting == NULL) {
        return false;
    }
    return aklv_roaring_add(&posting->lines, line_no);
}

void aklv_gram_index_destroy(AklvGramIndex *index) {
    if (index == NULL) {
        return;
    }
    if (index->items != NULL && index->used_slots != NULL) {
        for (uint64_t i = 0; i < index->count; i++) {
            uint64_t slot = index->used_slots[i];
            if (slot < index->capacity && index->items[slot].used) {
                aklv_roaring_destroy(&index->items[slot].lines);
            }
        }
    }
    if (index->items != NULL) {
        free(index->items);
    }
    free(index->used_slots);
    free(index->container_arena);
    free(index->payload_arena);
    memset(index, 0, sizeof(*index));
}

const AklvRoaring *aklv_gram_index_get(const AklvGramIndex *index, uint32_t gram) {
    if (index == NULL || index->capacity == 0) {
        return NULL;
    }
    uint64_t slot = aklv_gram_hash(gram) & (index->capacity - 1);
    while (index->items[slot].used) {
        if (index->items[slot].gram == gram) {
            return &index->items[slot].lines;
        }
        slot = (slot + 1) & (index->capacity - 1);
    }
    return NULL;
}

static bool aklv_gram_index_merge_move(AklvGramIndex *dst, AklvGramIndex *src) {
    if (src == NULL || src->count == 0) {
        return true;
    }
    if (src->used_slots == NULL) {
        return false;
    }
    for (uint64_t i = 0; i < src->count; i++) {
        uint64_t slot = src->used_slots[i];
        if (slot >= src->capacity || !src->items[slot].used) {
            continue;
        }
        AklvGramPosting *src_posting = &src->items[slot];
        AklvGramPosting *dst_posting = aklv_gram_index_slot(dst, src_posting->gram, true);
        if (dst_posting == NULL ||
            !aklv_roaring_move_append(&dst_posting->lines, &src_posting->lines)) {
            return false;
        }
    }
    aklv_gram_index_destroy(src);
    return true;
}

static bool aklv_gram_index_reserve_for_count(AklvGramIndex *index, uint64_t item_count) {
    if (item_count == 0) {
        return true;
    }
    uint64_t needed = item_count + item_count / 2 + 1;
    uint64_t capacity = index->capacity == 0 ? 1024 : index->capacity;
    while (capacity < needed) {
        if (capacity > UINT64_MAX / 2) {
            return false;
        }
        capacity *= 2;
    }
    if (capacity <= index->capacity) {
        return aklv_gram_index_reserve_used_slots(index, item_count);
    }
    return aklv_gram_index_rehash(index, capacity) &&
           aklv_gram_index_reserve_used_slots(index, item_count);
}

char *aklv_strdup(const char *text) {
    if (text == NULL) {
        text = "";
    }
    size_t len = strlen(text);
    char *out = malloc(len + 1);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, text, len + 1);
    return out;
}

char *aklv_path_basename_dup(const char *path) {
    if (path == NULL || path[0] == '\0') {
        return aklv_strdup("(unnamed)");
    }
    const char *slash = strrchr(path, '/');
    return aklv_strdup(slash == NULL ? path : slash + 1);
}

int aklv_map_file(const char *path, AklvMappedFile *mapped, char *error, size_t error_cap) {
    memset(mapped, 0, sizeof(*mapped));
    mapped->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        aklv_set_error(error, error_cap, "open failed: %s: %s", path, strerror(errno));
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        aklv_set_error(error, error_cap, "fstat failed: %s: %s", path, strerror(errno));
        close(fd);
        return -1;
    }
    if (!S_ISREG(st.st_mode)) {
        aklv_set_error(error, error_cap, "not a regular file: %s", path);
        close(fd);
        return -1;
    }
    if (st.st_size < 0) {
        aklv_set_error(error, error_cap, "invalid file size: %s", path);
        close(fd);
        return -1;
    }

    mapped->fd = fd;
    mapped->size = (size_t)st.st_size;
    if (mapped->size == 0) {
        mapped->data = NULL;
        return 0;
    }

    void *ptr = mmap(NULL, mapped->size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        aklv_set_error(error, error_cap, "mmap failed: %s: %s", path, strerror(errno));
        close(fd);
        mapped->fd = -1;
        mapped->size = 0;
        return -1;
    }

#if defined(MADV_SEQUENTIAL)
    madvise(ptr, mapped->size, MADV_SEQUENTIAL);
#endif
    mapped->data = (const unsigned char *)ptr;
    return 0;
}

void aklv_unmap_file(AklvMappedFile *mapped) {
    if (mapped->data != NULL && mapped->size > 0) {
        munmap((void *)mapped->data, mapped->size);
    }
    if (mapped->fd >= 0) {
        close(mapped->fd);
    }
    memset(mapped, 0, sizeof(*mapped));
    mapped->fd = -1;
}

void aklv_line_index_destroy(AklvLineIndex *index) {
    aklv_gram_index_destroy(&index->gram_index);
    aklv_index_cache_destroy(&index->cache);
    if (index->offset_blocks != NULL) {
        for (uint64_t i = 0; i < index->block_count; i++) {
            if (index->offset_blocks[i] != NULL) {
                munmap(index->offset_blocks[i],
                       (size_t)AKLV_LINE_INDEX_BLOCK_LINES * sizeof(**index->offset_blocks));
            }
        }
        free(index->offset_blocks);
    }
    free(index->block_max_grams);
    memset(index, 0, sizeof(*index));
}

typedef struct {
    const AklvMappedFile *mapped;
    const AklvLineIndex *line_index;
    uint64_t first_line;
    uint64_t last_line;
    const atomic_bool *cancel;
    atomic_bool *build_cancel;
    AklvGramIndex gram_index;
    int status;
    char error[256];
} AklvGramBuildTask;

static bool aklv_gram_build_union_count(const AklvGramBuildTask *tasks,
                                        uint32_t task_count,
                                        uint64_t *count_out) {
    *count_out = 0;
    uint64_t *seen_words = calloc((size_t)AKLV_ASCII_GRAM_KEY_SPACE_WORDS, sizeof(*seen_words));
    if (seen_words == NULL) {
        return false;
    }
    uint64_t count = 0;
    for (uint32_t task_i = 0; task_i < task_count; task_i++) {
        const AklvGramIndex *index = &tasks[task_i].gram_index;
        for (uint64_t i = 0; i < index->count; i++) {
            uint64_t slot = index->used_slots[i];
            uint32_t dense = aklv_gram_dense_index(index->items[slot].gram);
            uint32_t word_index = dense >> 6;
            uint64_t mask = UINT64_C(1) << (dense & 63U);
            if ((seen_words[word_index] & mask) == 0) {
                seen_words[word_index] |= mask;
                count++;
            }
        }
    }
    free(seen_words);
    *count_out = count;
    return true;
}

static uint32_t aklv_default_index_thread_count(void) {
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < 1) {
        cores = 1;
    }
    long threads = cores - 1;
    if (threads < 1) {
        threads = 1;
    }
    if (threads > 8) {
        threads = 8;
    }
    return (uint32_t)threads;
}

static int aklv_line_index_grow_block_arrays(AklvLineIndex *index, char *error, size_t error_cap) {
    if (index->block_count < index->block_capacity) {
        return 0;
    }
    uint64_t new_capacity = index->block_capacity == 0 ? 8 : index->block_capacity * 2;
    if (new_capacity < index->block_capacity ||
        new_capacity > (uint64_t)(SIZE_MAX / sizeof(*index->offset_blocks))) {
        aklv_set_error(error, error_cap, "line index block table too large");
        return -1;
    }

    size_t offset_bytes = (size_t)new_capacity * sizeof(*index->offset_blocks);
    size_t old_offset_bytes = (size_t)index->block_capacity * sizeof(*index->offset_blocks);
    size_t max_grams_bytes = (size_t)new_capacity * sizeof(*index->block_max_grams);
    size_t old_max_grams_bytes = (size_t)index->block_capacity * sizeof(*index->block_max_grams);

    size_t **new_offsets = realloc(index->offset_blocks, offset_bytes);
    if (new_offsets == NULL) {
        aklv_set_error(error, error_cap, "realloc failed while growing line offsets");
        return -1;
    }
    index->offset_blocks = new_offsets;
    size_t *new_max_grams = realloc(index->block_max_grams, max_grams_bytes);
    if (new_max_grams == NULL) {
        aklv_set_error(error, error_cap, "realloc failed while growing line gram bounds");
        return -1;
    }
    index->block_max_grams = new_max_grams;
    memset((unsigned char *)index->offset_blocks + old_offset_bytes, 0, offset_bytes - old_offset_bytes);
    memset((unsigned char *)index->block_max_grams + old_max_grams_bytes,
           0,
           max_grams_bytes - old_max_grams_bytes);

    index->block_capacity = new_capacity;
    return 0;
}

static int aklv_line_index_reserve_block_arrays(AklvLineIndex *index,
                                                uint64_t block_capacity,
                                                char *error,
                                                size_t error_cap) {
    if (block_capacity <= index->block_capacity) {
        return 0;
    }
    if (block_capacity > (uint64_t)(SIZE_MAX / sizeof(*index->offset_blocks))) {
        aklv_set_error(error, error_cap, "line index block table too large");
        return -1;
    }
    size_t offset_bytes = (size_t)block_capacity * sizeof(*index->offset_blocks);
    size_t old_offset_bytes = (size_t)index->block_capacity * sizeof(*index->offset_blocks);
    size_t max_grams_bytes = (size_t)block_capacity * sizeof(*index->block_max_grams);
    size_t old_max_grams_bytes = (size_t)index->block_capacity * sizeof(*index->block_max_grams);
    size_t **new_offsets = realloc(index->offset_blocks, offset_bytes);
    if (new_offsets == NULL) {
        aklv_set_error(error, error_cap, "realloc failed while reserving line offsets");
        return -1;
    }
    index->offset_blocks = new_offsets;
    size_t *new_max_grams = realloc(index->block_max_grams, max_grams_bytes);
    if (new_max_grams == NULL) {
        aklv_set_error(error, error_cap, "realloc failed while reserving line gram bounds");
        return -1;
    }
    index->block_max_grams = new_max_grams;
    memset((unsigned char *)index->offset_blocks + old_offset_bytes, 0, offset_bytes - old_offset_bytes);
    memset((unsigned char *)index->block_max_grams + old_max_grams_bytes,
           0,
           max_grams_bytes - old_max_grams_bytes);
    index->block_capacity = block_capacity;
    return 0;
}

static int aklv_line_index_add_block(AklvLineIndex *index, char *error, size_t error_cap) {
    if (aklv_line_index_grow_block_arrays(index, error, error_cap) != 0) {
        return -1;
    }

    size_t offset_bytes = (size_t)AKLV_LINE_INDEX_BLOCK_LINES * sizeof(**index->offset_blocks);
    void *offsets = mmap(NULL, offset_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (offsets == MAP_FAILED) {
        aklv_set_error(error, error_cap, "mmap failed while allocating line offset block: %s", strerror(errno));
        return -1;
    }

    index->offset_blocks[index->block_count] = (size_t *)offsets;
    index->block_count++;
    return 0;
}

static int aklv_line_index_append_packed(AklvLineIndex *index,
                                         size_t packed,
                                         size_t max_grams,
                                         char *error,
                                         size_t error_cap) {
    uint64_t in_block = index->count & AKLV_LINE_INDEX_BLOCK_MASK;
    if (in_block == 0 && aklv_line_index_add_block(index, error, error_cap) != 0) {
        return -1;
    }
    uint64_t block = index->count >> AKLV_LINE_INDEX_BLOCK_SHIFT;
    index->offset_blocks[block][in_block] = packed;
    if (max_grams > index->block_max_grams[block]) {
        index->block_max_grams[block] = max_grams;
    }
    index->count++;
    return 0;
}

static int aklv_line_index_append(AklvLineIndex *index,
                                  size_t offset,
                                  uint32_t prefix_skip,
                                  size_t max_grams,
                                  char *error,
                                  size_t error_cap) {
    return aklv_line_index_append_packed(index,
                                         aklv_line_index_pack_offset(offset, prefix_skip),
                                         max_grams,
                                         error,
                                         error_cap);
}

static void aklv_container_range_stats(const AklvRoaringContainer *container,
                                       uint16_t min_low,
                                       uint16_t max_low,
                                       uint32_t *cardinality_out,
                                       uint32_t *run_count_out) {
    uint32_t cardinality = 0;
    uint32_t run_count = 0;
    bool in_run = false;
    uint32_t prev = 0;
    if (container->type == AKLV_ROARING_FULL) {
        cardinality = (uint32_t)max_low - (uint32_t)min_low + 1U;
        run_count = cardinality == 0 ? 0 : 1;
    } else if (container->type == AKLV_ROARING_RUN) {
        for (uint32_t i = 0; i < container->run_count; i++) {
            uint32_t start = container->runs[i].start;
            uint32_t end = start + container->runs[i].length - 1U;
            if (end < min_low) {
                continue;
            }
            if (start > max_low) {
                break;
            }
            if (start < min_low) {
                start = min_low;
            }
            if (end > max_low) {
                end = max_low;
            }
            cardinality += end - start + 1U;
            run_count++;
        }
    } else if (container->bitmap != NULL) {
        uint32_t first_word = min_low >> 6;
        uint32_t last_word = max_low >> 6;
        for (uint32_t word_i = first_word; word_i <= last_word; word_i++) {
            uint64_t bits = container->bitmap[word_i];
            if (word_i == first_word) {
                uint32_t skip = min_low & 63U;
                if (skip != 0) {
                    bits &= UINT64_MAX << skip;
                }
            }
            if (word_i == last_word) {
                uint32_t keep = (max_low & 63U) + 1U;
                if (keep < 64U) {
                    bits &= (UINT64_C(1) << keep) - 1U;
                }
            }
            while (bits != 0) {
                uint32_t bit = (uint32_t)__builtin_ctzll(bits);
                bits &= bits - 1;
                uint32_t low = (word_i << 6) | bit;
                if (!in_run || low != prev + 1U) {
                    run_count++;
                    in_run = true;
                }
                prev = low;
                cardinality++;
            }
            if (word_i == UINT32_MAX) {
                break;
            }
        }
    } else {
        for (uint32_t i = 0; i < container->cardinality; i++) {
            uint16_t low = container->array[i];
            if (low < min_low) {
                continue;
            }
            if (low > max_low) {
                break;
            }
            if (!in_run || (uint32_t)low != prev + 1U) {
                run_count++;
                in_run = true;
            }
            prev = low;
            cardinality++;
        }
    }
    *cardinality_out = cardinality;
    *run_count_out = run_count;
}

static bool aklv_container_first_last_low(const AklvRoaringContainer *container,
                                          uint16_t *first_out,
                                          uint16_t *last_out) {
    if (container->cardinality == 0) {
        return false;
    }
    if (container->type == AKLV_ROARING_FULL) {
        *first_out = 0;
        *last_out = UINT16_MAX;
        return true;
    }
    if (container->type == AKLV_ROARING_RUN) {
        if (container->run_count == 0) {
            return false;
        }
        AklvRoaringRun first = container->runs[0];
        AklvRoaringRun last = container->runs[container->run_count - 1];
        *first_out = first.start;
        *last_out = (uint16_t)((uint32_t)last.start + last.length - 1U);
        return true;
    }
    if (container->bitmap != NULL) {
        uint16_t first = 0;
        uint16_t last = 0;
        bool found = false;
        for (uint32_t word_i = 0; word_i < 1024; word_i++) {
            uint64_t word = container->bitmap[word_i];
            if (word != 0) {
                first = (uint16_t)((word_i << 6) | (uint32_t)__builtin_ctzll(word));
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
        for (uint32_t word_i = 1024; word_i > 0; word_i--) {
            uint64_t word = container->bitmap[word_i - 1];
            if (word != 0) {
                last = (uint16_t)(((word_i - 1) << 6) | (63U - (uint32_t)__builtin_clzll(word)));
                break;
            }
        }
        *first_out = first;
        *last_out = last;
        return true;
    }
    *first_out = container->array[0];
    *last_out = container->array[container->cardinality - 1];
    return true;
}

static AklvContainerRangePlan aklv_container_full_range_fast_plan(const AklvRoaringContainer *container,
                                                                  uint16_t first_low,
                                                                  uint16_t last_low) {
    AklvContainerRangePlan plan;
    memset(&plan, 0, sizeof(plan));
    plan.cardinality = container->cardinality;
    if (container->cardinality == 65536) {
        plan.type = AKLV_ROARING_FULL;
        return plan;
    }
    uint32_t span = (uint32_t)last_low - (uint32_t)first_low + 1U;
    size_t run_bytes = sizeof(AklvRoaringRun);
    size_t current_payload_bytes = 0;
    switch (container->type) {
        case AKLV_ROARING_RUN:
            current_payload_bytes = (size_t)container->run_count * sizeof(*container->runs);
            break;
        case AKLV_ROARING_BITMAP:
            current_payload_bytes = 1024 * sizeof(*container->bitmap);
            break;
        case AKLV_ROARING_ARRAY:
        default:
            current_payload_bytes = (size_t)container->cardinality * sizeof(*container->array);
            break;
    }
    if (span == container->cardinality && run_bytes <= current_payload_bytes) {
        plan.type = AKLV_ROARING_RUN;
        plan.run_count = 1;
        plan.payload_count = 1;
        plan.payload_bytes = run_bytes;
        plan.single_run_start = first_low;
        plan.single_run_end = last_low;
        plan.single_run_direct = true;
        return plan;
    }
    plan.type = container->type;
    plan.full_container_direct = true;
    switch (container->type) {
        case AKLV_ROARING_FULL:
            break;
        case AKLV_ROARING_RUN:
            plan.run_count = container->run_count;
            plan.payload_count = container->run_count;
            plan.payload_bytes = (size_t)container->run_count * sizeof(*container->runs);
            break;
        case AKLV_ROARING_BITMAP:
            plan.payload_count = 1024;
            plan.payload_bytes = 1024 * sizeof(*container->bitmap);
            break;
        case AKLV_ROARING_ARRAY:
        default:
            plan.type = AKLV_ROARING_ARRAY;
            plan.payload_count = container->cardinality;
            plan.payload_bytes = (size_t)container->cardinality * sizeof(*container->array);
            break;
    }
    return plan;
}

static AklvContainerRangePlan aklv_container_complete_fast_plan(const AklvRoaringContainer *container) {
    AklvContainerRangePlan plan;
    memset(&plan, 0, sizeof(plan));
    plan.cardinality = container->cardinality;
    if (container->cardinality == 0) {
        plan.type = AKLV_ROARING_ARRAY;
        return plan;
    }
    if (container->cardinality == 65536 || container->type == AKLV_ROARING_FULL) {
        plan.type = AKLV_ROARING_FULL;
        return plan;
    }
    plan.type = container->type;
    plan.full_container_direct = true;
    switch (container->type) {
        case AKLV_ROARING_RUN:
            plan.run_count = container->run_count;
            plan.payload_count = container->run_count;
            plan.payload_bytes = (size_t)container->run_count * sizeof(*container->runs);
            break;
        case AKLV_ROARING_BITMAP:
            plan.payload_count = 1024;
            plan.payload_bytes = 1024 * sizeof(*container->bitmap);
            break;
        case AKLV_ROARING_ARRAY: {
            uint16_t first_low = container->array[0];
            uint16_t last_low = container->array[container->cardinality - 1];
            uint32_t span = (uint32_t)last_low - (uint32_t)first_low + 1U;
            size_t array_bytes = (size_t)container->cardinality * sizeof(*container->array);
            if (span == container->cardinality && sizeof(AklvRoaringRun) <= array_bytes) {
                plan.type = AKLV_ROARING_RUN;
                plan.run_count = 1;
                plan.payload_count = 1;
                plan.payload_bytes = sizeof(AklvRoaringRun);
                plan.single_run_start = first_low;
                plan.single_run_end = last_low;
                plan.single_run_direct = true;
                plan.full_container_direct = false;
            } else {
                plan.payload_count = container->cardinality;
                plan.payload_bytes = array_bytes;
            }
            break;
        }
        case AKLV_ROARING_FULL:
        default:
            plan.type = AKLV_ROARING_FULL;
            plan.full_container_direct = false;
            break;
    }
    return plan;
}

static AklvContainerRangePlan aklv_container_range_plan(const AklvRoaringContainer *container,
                                                        uint16_t min_low,
                                                        uint16_t max_low) {
    AklvContainerRangePlan plan;
    memset(&plan, 0, sizeof(plan));
    if (min_low == 0 && max_low == UINT16_MAX) {
        return aklv_container_complete_fast_plan(container);
    }
    uint16_t first_low = 0;
    uint16_t last_low = 0;
    if (aklv_container_first_last_low(container, &first_low, &last_low) &&
        min_low <= first_low &&
        max_low >= last_low) {
        return aklv_container_full_range_fast_plan(container, first_low, last_low);
    }
    uint32_t cardinality = 0;
    uint32_t run_count = 0;
    aklv_container_range_stats(container, min_low, max_low, &cardinality, &run_count);
    plan.cardinality = cardinality;
    plan.run_count = run_count;
    if (cardinality == 0) {
        plan.type = AKLV_ROARING_ARRAY;
        return plan;
    }
    if (cardinality == 65536) {
        plan.type = AKLV_ROARING_FULL;
        return plan;
    }

    size_t array_bytes = (size_t)cardinality * sizeof(uint16_t);
    size_t run_bytes = run_count == 0 ? SIZE_MAX : (size_t)run_count * sizeof(AklvRoaringRun);
    size_t bitmap_bytes = 1024 * sizeof(uint64_t);
    if (run_bytes <= array_bytes && run_bytes <= bitmap_bytes) {
        plan.type = AKLV_ROARING_RUN;
        plan.payload_count = run_count;
        plan.payload_bytes = run_bytes;
    } else if (bitmap_bytes < array_bytes) {
        plan.type = AKLV_ROARING_BITMAP;
        plan.payload_count = 1024;
        plan.payload_bytes = bitmap_bytes;
    } else {
        plan.type = AKLV_ROARING_ARRAY;
        plan.payload_count = cardinality;
        plan.payload_bytes = array_bytes;
    }
    return plan;
}

static AklvCachePostingRangePlan aklv_cache_plan_posting_range(const AklvGramPosting *posting,
                                                               uint64_t first_line,
                                                               uint64_t last_line,
                                                               AklvCacheContainerWritePlan *container_plans,
                                                               uint32_t container_plan_capacity) {
    AklvCachePostingRangePlan posting_plan;
    memset(&posting_plan, 0, sizeof(posting_plan));
    uint64_t first_key = first_line >> 16;
    uint64_t last_key = last_line >> 16;
    uint64_t first_container =
        aklv_roaring_lower_bound_container_index(&posting->lines, first_key);
    for (uint64_t i = first_container; i < posting->lines.count; i++) {
        const AklvRoaringContainer *container = &posting->lines.containers[i];
        if (container->key > last_key) {
            break;
        }
        uint16_t min_low = container->key == first_key ? (uint16_t)(first_line & 0xffffU) : 0;
        uint16_t max_low = container->key == last_key ? (uint16_t)(last_line & 0xffffU) : UINT16_MAX;
        AklvContainerRangePlan plan =
            aklv_container_range_plan(container, min_low, max_low);
        if (plan.cardinality == 0) {
            continue;
        }
        if (posting_plan.container_count == container_plan_capacity) {
            posting_plan.container_count = UINT32_MAX;
            posting_plan.serialized_bytes = UINT64_MAX;
            return posting_plan;
        }
        AklvCacheContainerWritePlan *write_plan = &container_plans[posting_plan.container_count++];
        write_plan->container = container;
        write_plan->plan = plan;
        write_plan->min_low = min_low;
        write_plan->max_low = max_low;
        posting_plan.cardinality += plan.cardinality;
        uint64_t bytes = sizeof(AklvIndexCacheContainerHeader) + (uint64_t)plan.payload_bytes;
        posting_plan.serialized_bytes = aklv_cache_saturating_add_u64(posting_plan.serialized_bytes, bytes);
        posting_plan.payload_bytes =
            aklv_cache_saturating_add_u64(posting_plan.payload_bytes, (uint64_t)plan.payload_bytes);
    }
    if (posting_plan.container_count != 0) {
        posting_plan.serialized_bytes =
            aklv_cache_saturating_add_u64(posting_plan.serialized_bytes,
                                          sizeof(AklvIndexCachePostingHeader));
    }
    return posting_plan;
}

static bool aklv_cache_write_padding(FILE *stream, size_t bytes) {
    static const unsigned char zeroes[256] = {0};
    while (bytes > 0) {
        size_t chunk = bytes > sizeof(zeroes) ? sizeof(zeroes) : bytes;
        if (!aklv_write_all(stream, zeroes, chunk)) {
            return false;
        }
        bytes -= chunk;
    }
    return true;
}

static bool aklv_cache_write_u16_sequence(FILE *stream,
                                          AklvCacheWriteScratch *scratch,
                                          uint32_t first,
                                          uint32_t last) {
    if (scratch == NULL || scratch->u16_values == NULL) {
        return false;
    }
    bool ok = true;
    while (ok && first <= last) {
        uint32_t remaining = last - first + 1U;
        size_t count = remaining > AKLV_CACHE_U16_WRITE_BATCH
            ? AKLV_CACHE_U16_WRITE_BATCH
            : (size_t)remaining;
        for (size_t i = 0; i < count; i++) {
            scratch->u16_values[i] = (uint16_t)(first + (uint32_t)i);
        }
        if (!aklv_write_all(stream, scratch->u16_values, count * sizeof(*scratch->u16_values))) {
            ok = false;
            break;
        }
        first += (uint32_t)count;
    }
    return ok;
}

static uint32_t aklv_lower_bound_u16(const uint16_t *items, uint32_t count, uint16_t value) {
    uint32_t lo = 0;
    uint32_t hi = count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (items[mid] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static bool aklv_cache_flush_u16_batch(FILE *stream, uint16_t *values, size_t *count) {
    if (*count == 0) {
        return true;
    }
    bool ok = aklv_write_all(stream, values, *count * sizeof(*values));
    *count = 0;
    return ok;
}

static bool aklv_cache_append_u16(FILE *stream, uint16_t *values, size_t *count, uint16_t value) {
    values[(*count)++] = value;
    if (*count == AKLV_CACHE_U16_WRITE_BATCH) {
        return aklv_cache_flush_u16_batch(stream, values, count);
    }
    return true;
}

static bool aklv_cache_flush_run_batch(FILE *stream, AklvRoaringRun *runs, size_t *count) {
    if (*count == 0) {
        return true;
    }
    bool ok = aklv_write_all(stream, runs, *count * sizeof(*runs));
    *count = 0;
    return ok;
}

static bool aklv_cache_append_run(FILE *stream,
                                  AklvRoaringRun *runs,
                                  size_t *count,
                                  uint32_t start,
                                  uint32_t end) {
    AklvRoaringRun run;
    run.start = (uint16_t)start;
    run.length = (uint16_t)(end - start + 1U);
    runs[(*count)++] = run;
    if (*count == AKLV_CACHE_RUN_WRITE_BATCH) {
        return aklv_cache_flush_run_batch(stream, runs, count);
    }
    return true;
}

static bool aklv_cache_write_single_run(FILE *stream, uint16_t start, uint16_t end) {
    AklvRoaringRun run;
    run.start = start;
    run.length = (uint16_t)((uint32_t)end - (uint32_t)start + 1U);
    return aklv_write_all(stream, &run, sizeof(run));
}

static bool aklv_cache_write_full_container_payload(FILE *stream,
                                                    const AklvRoaringContainer *container) {
    switch (container->type) {
        case AKLV_ROARING_FULL:
            return true;
        case AKLV_ROARING_RUN:
            return aklv_write_all(stream,
                                  container->runs,
                                  (size_t)container->run_count * sizeof(*container->runs));
        case AKLV_ROARING_BITMAP:
            return aklv_write_all(stream, container->bitmap, 1024 * sizeof(*container->bitmap));
        case AKLV_ROARING_ARRAY:
        default:
            return aklv_write_all(stream,
                                  container->array,
                                  (size_t)container->cardinality * sizeof(*container->array));
    }
}

static bool aklv_cache_write_container_array_range(FILE *stream,
                                                   AklvCacheWriteScratch *scratch,
                                                   const AklvRoaringContainer *container,
                                                   uint16_t min_low,
                                                   uint16_t max_low) {
    if (scratch == NULL || scratch->u16_values == NULL) {
        return false;
    }
    if (container->type == AKLV_ROARING_FULL) {
        return aklv_cache_write_u16_sequence(stream, scratch, min_low, max_low);
    }
    if (container->type == AKLV_ROARING_RUN) {
        for (uint32_t i = 0; i < container->run_count; i++) {
            uint32_t start = container->runs[i].start;
            uint32_t end = start + container->runs[i].length - 1U;
            if (end < min_low) {
                continue;
            }
            if (start > max_low) {
                break;
            }
            if (start < min_low) {
                start = min_low;
            }
            if (end > max_low) {
                end = max_low;
            }
            if (!aklv_cache_write_u16_sequence(stream, scratch, start, end)) {
                return false;
            }
        }
        return true;
    }
    if (container->bitmap != NULL) {
        size_t value_count = 0;
        uint32_t first_word = min_low >> 6;
        uint32_t last_word = max_low >> 6;
        for (uint32_t word_i = first_word; word_i <= last_word; word_i++) {
            uint64_t bits = container->bitmap[word_i];
            if (word_i == first_word) {
                uint32_t skip = min_low & 63U;
                if (skip != 0) {
                    bits &= UINT64_MAX << skip;
                }
            }
            if (word_i == last_word) {
                uint32_t keep = (max_low & 63U) + 1U;
                if (keep < 64U) {
                    bits &= (UINT64_C(1) << keep) - 1U;
                }
            }
            while (bits != 0) {
                uint32_t bit = (uint32_t)__builtin_ctzll(bits);
                bits &= bits - 1;
                uint16_t value = (uint16_t)((word_i << 6) | bit);
                if (!aklv_cache_append_u16(stream, scratch->u16_values, &value_count, value)) {
                    return false;
                }
            }
            if (word_i == UINT32_MAX) {
                break;
            }
        }
        return aklv_cache_flush_u16_batch(stream, scratch->u16_values, &value_count);
    }
    uint32_t begin = aklv_lower_bound_u16(container->array, container->cardinality, min_low);
    uint32_t end = max_low == UINT16_MAX
        ? container->cardinality
        : aklv_lower_bound_u16(container->array, container->cardinality, (uint16_t)(max_low + 1U));
    return begin >= end ||
           aklv_write_all(stream,
                          container->array + begin,
                          (size_t)(end - begin) * sizeof(*container->array));
}

static bool aklv_cache_write_container_runs_range(FILE *stream,
                                                  AklvCacheWriteScratch *scratch,
                                                  const AklvRoaringContainer *container,
                                                  uint16_t min_low,
                                                  uint16_t max_low) {
    if (scratch == NULL || scratch->runs == NULL) {
        return false;
    }
    if (container->type == AKLV_ROARING_FULL) {
        AklvRoaringRun current = {0};
        current.start = min_low;
        current.length = (uint16_t)((uint32_t)max_low - (uint32_t)min_low + 1U);
        return aklv_write_all(stream, &current, sizeof(current));
    }
    if (container->type == AKLV_ROARING_RUN) {
        size_t run_count = 0;
        for (uint32_t i = 0; i < container->run_count; i++) {
            uint32_t start = container->runs[i].start;
            uint32_t end = start + container->runs[i].length - 1U;
            if (end < min_low) {
                continue;
            }
            if (start > max_low) {
                break;
            }
            if (start < min_low) {
                start = min_low;
            }
            if (end > max_low) {
                end = max_low;
            }
            if (!aklv_cache_append_run(stream, scratch->runs, &run_count, start, end)) {
                return false;
            }
        }
        return aklv_cache_flush_run_batch(stream, scratch->runs, &run_count);
    }

    size_t run_batch_count = 0;
    AklvRoaringRun current = {0};
    bool in_run = false;
    uint32_t prev = 0;
    if (container->bitmap != NULL) {
        uint32_t first_word = min_low >> 6;
        uint32_t last_word = max_low >> 6;
        for (uint32_t word_i = first_word; word_i <= last_word; word_i++) {
            uint64_t bits = container->bitmap[word_i];
            if (word_i == first_word) {
                uint32_t skip = min_low & 63U;
                if (skip != 0) {
                    bits &= UINT64_MAX << skip;
                }
            }
            if (word_i == last_word) {
                uint32_t keep = (max_low & 63U) + 1U;
                if (keep < 64U) {
                    bits &= (UINT64_C(1) << keep) - 1U;
                }
            }
            while (bits != 0) {
                uint32_t bit = (uint32_t)__builtin_ctzll(bits);
                bits &= bits - 1;
                uint32_t low = (word_i << 6) | bit;
                if (!in_run) {
                    current.start = (uint16_t)low;
                    in_run = true;
                } else if (low != prev + 1U) {
                    if (!aklv_cache_append_run(stream,
                                               scratch->runs,
                                               &run_batch_count,
                                               current.start,
                                               prev)) {
                        return false;
                    }
                    current.start = (uint16_t)low;
                }
                prev = low;
            }
        }
    } else {
        uint32_t begin = aklv_lower_bound_u16(container->array, container->cardinality, min_low);
        uint32_t end = max_low == UINT16_MAX
            ? container->cardinality
            : aklv_lower_bound_u16(container->array,
                                   container->cardinality,
                                   (uint16_t)(max_low + 1U));
        for (uint32_t i = begin; i < end; i++) {
            uint32_t low = container->array[i];
            if (!in_run) {
                current.start = (uint16_t)low;
                in_run = true;
            } else if (low != prev + 1U) {
                if (!aklv_cache_append_run(stream,
                                           scratch->runs,
                                           &run_batch_count,
                                           current.start,
                                           prev)) {
                    return false;
                }
                current.start = (uint16_t)low;
            }
            prev = low;
        }
    }
    if (in_run) {
        if (!aklv_cache_append_run(stream, scratch->runs, &run_batch_count, current.start, prev)) {
            return false;
        }
    }
    return aklv_cache_flush_run_batch(stream, scratch->runs, &run_batch_count);
}

static bool aklv_cache_write_container_bitmap_range(FILE *stream,
                                                    const AklvRoaringContainer *container,
                                                    uint16_t min_low,
                                                    uint16_t max_low) {
    if (container->bitmap != NULL && min_low == 0 && max_low == UINT16_MAX) {
        return aklv_write_all(stream, container->bitmap, 1024 * sizeof(*container->bitmap));
    }
    uint64_t words[1024];
    memset(words, 0, sizeof(words));
    if (container->type == AKLV_ROARING_FULL) {
        for (uint32_t word_i = 0; word_i < 1024; word_i++) {
            words[word_i] = UINT64_MAX;
        }
    } else if (container->bitmap != NULL) {
        memcpy(words, container->bitmap, sizeof(words));
    } else if (container->type == AKLV_ROARING_RUN) {
        for (uint32_t i = 0; i < container->run_count; i++) {
            uint32_t start = container->runs[i].start;
            uint32_t end = start + container->runs[i].length - 1U;
            if (end < min_low) {
                continue;
            }
            if (start > max_low) {
                break;
            }
            if (start < min_low) {
                start = min_low;
            }
            if (end > max_low) {
                end = max_low;
            }
            uint32_t first_word = start >> 6;
            uint32_t last_word = end >> 6;
            for (uint32_t word_i = first_word; word_i <= last_word; word_i++) {
                uint32_t word_start = word_i << 6;
                uint32_t word_end = word_start + 63U;
                uint32_t lo = start > word_start ? start : word_start;
                uint32_t hi = end < word_end ? end : word_end;
                words[word_i] |= aklv_bit_range_mask(lo & 63U, hi & 63U);
            }
        }
    } else {
        for (uint32_t i = 0; i < container->cardinality; i++) {
            uint16_t low = container->array[i];
            words[low >> 6] |= UINT64_C(1) << (low & 63U);
        }
    }
    for (uint32_t word_i = 0; word_i < 1024; word_i++) {
        uint32_t word_start = word_i << 6;
        uint32_t word_end = word_start + 63U;
        if (word_end < min_low || word_start > max_low) {
            words[word_i] = 0;
            continue;
        }
        if (word_start < min_low) {
            uint32_t skip = min_low & 63U;
            if (skip != 0) {
                words[word_i] &= UINT64_MAX << skip;
            }
        }
        if (word_end > max_low) {
            uint32_t keep = (max_low & 63U) + 1U;
            if (keep < 64U) {
                words[word_i] &= (UINT64_C(1) << keep) - 1U;
            }
        }
    }
    return aklv_write_all(stream, words, sizeof(words));
}

static bool aklv_cache_write_container_range(FILE *stream,
                                             AklvCacheWriteScratch *scratch,
                                             const AklvRoaringContainer *container,
                                             uint16_t min_low,
                                             uint16_t max_low,
                                             AklvContainerRangePlan plan) {
    if (plan.cardinality == 0) {
        return true;
    }
    AklvIndexCacheContainerHeader header;
    memset(&header, 0, sizeof(header));
    header.key = container->key;
    header.cardinality = plan.cardinality;
    header.type = (uint32_t)plan.type;
    header.payload_count = plan.payload_count;
    if (!aklv_write_all(stream, &header, sizeof(header))) {
        return false;
    }
    if (plan.full_container_direct && plan.type == container->type) {
        return aklv_cache_write_full_container_payload(stream, container);
    }
    switch (plan.type) {
        case AKLV_ROARING_FULL:
            return true;
        case AKLV_ROARING_RUN:
            if (plan.single_run_direct) {
                return aklv_cache_write_single_run(stream,
                                                   plan.single_run_start,
                                                   plan.single_run_end);
            }
            return aklv_cache_write_container_runs_range(stream, scratch, container, min_low, max_low);
        case AKLV_ROARING_BITMAP:
            return aklv_cache_write_container_bitmap_range(stream, container, min_low, max_low);
        case AKLV_ROARING_ARRAY:
            return aklv_cache_write_container_array_range(stream, scratch, container, min_low, max_low);
        default:
            return false;
    }
}

static bool aklv_cache_write_posting_range(FILE *stream,
                                           AklvCacheWriteScratch *scratch,
                                           const AklvGramPosting *posting,
                                           const AklvCacheContainerWritePlan *container_plans,
                                           AklvCachePostingRangePlan plan) {
    if (plan.container_count == 0) {
        return true;
    }
    AklvIndexCachePostingHeader header;
    memset(&header, 0, sizeof(header));
    header.gram = posting->gram;
    header.container_count = plan.container_count;
    header.cardinality = plan.cardinality;
    if (!aklv_write_all(stream, &header, sizeof(header))) {
        return false;
    }
    for (uint32_t i = 0; i < plan.container_count; i++) {
        const AklvCacheContainerWritePlan *write_plan = &container_plans[i];
        if (!aklv_cache_write_container_range(stream,
                                              scratch,
                                              write_plan->container,
                                              write_plan->min_low,
                                              write_plan->max_low,
                                              write_plan->plan)) {
            return false;
        }
    }
    return true;
}

static bool aklv_cache_collect_all_postings(const AklvGramIndex *gram_index,
                                            AklvGramPosting ***out_postings,
                                            uint64_t *out_count) {
    *out_postings = NULL;
    *out_count = 0;
    if (gram_index->count == 0) {
        return true;
    }
    if (gram_index->count > (uint64_t)(SIZE_MAX / sizeof(AklvGramPosting *))) {
        return false;
    }
    AklvGramPosting **postings = malloc((size_t)gram_index->count * sizeof(*postings));
    if (postings == NULL) {
        return false;
    }
    uint64_t count = 0;
    for (uint64_t i = 0; i < gram_index->count; i++) {
        uint64_t slot = gram_index->used_slots[i];
        AklvGramPosting *posting = &gram_index->items[slot];
        if (!posting->used || posting->lines.count == 0) {
            continue;
        }
        postings[count++] = posting;
    }
    *out_postings = postings;
    *out_count = count;
    return true;
}

static uint64_t aklv_cache_saturating_add_u64(uint64_t value, uint64_t add) {
    if (UINT64_MAX - value < add) {
        return UINT64_MAX;
    }
    return value + add;
}

static uint64_t aklv_cache_key_first_line(uint64_t key) {
    return key == 0 ? 1 : (key << 16);
}

static uint64_t aklv_cache_key_last_line(uint64_t key, uint64_t line_count) {
    uint64_t last = UINT64_MAX;
    if (key < (UINT64_MAX >> 16)) {
        last = ((key + 1) << 16) - 1;
    }
    return last > line_count ? line_count : last;
}

static uint64_t aklv_cache_container_estimated_payload_bytes(const AklvRoaringContainer *container) {
    if (container->cardinality == 0) {
        return 0;
    }
    switch (container->type) {
        case AKLV_ROARING_FULL:
            return 0;
        case AKLV_ROARING_RUN:
            return (uint64_t)container->run_count * sizeof(*container->runs);
        case AKLV_ROARING_BITMAP:
            return 1024 * sizeof(*container->bitmap);
        case AKLV_ROARING_ARRAY:
        default:
            return (uint64_t)container->cardinality * sizeof(uint16_t);
    }
}

static void aklv_cache_key_plan_destroy(AklvCacheKeyPlan *plan) {
    if (plan == NULL) {
        return;
    }
    free(plan->key_bytes);
    free(plan->bucket_offsets);
    free(plan->posting_indices);
    memset(plan, 0, sizeof(*plan));
}

static void aklv_cache_write_scratch_destroy(AklvCacheWriteScratch *scratch) {
    if (scratch == NULL) {
        return;
    }
    free(scratch->u16_values);
    free(scratch->runs);
    free(scratch->offset_values);
    memset(scratch, 0, sizeof(*scratch));
}

static bool aklv_cache_write_scratch_init(AklvCacheWriteScratch *scratch) {
    if (scratch == NULL) {
        return false;
    }
    memset(scratch, 0, sizeof(*scratch));
    scratch->u16_values = malloc((size_t)AKLV_CACHE_U16_WRITE_BATCH * sizeof(*scratch->u16_values));
    scratch->runs = malloc((size_t)AKLV_CACHE_RUN_WRITE_BATCH * sizeof(*scratch->runs));
    scratch->offset_values =
        malloc((size_t)AKLV_CACHE_OFFSET_CONVERT_BATCH * sizeof(*scratch->offset_values));
    if (scratch->u16_values == NULL || scratch->runs == NULL || scratch->offset_values == NULL) {
        aklv_cache_write_scratch_destroy(scratch);
        return false;
    }
    return true;
}

static bool aklv_cache_build_key_plan(const AklvLineIndex *index,
                                      AklvGramPosting *const *postings,
                                      uint64_t posting_count,
                                      AklvCacheKeyPlan *plan) {
    memset(plan, 0, sizeof(*plan));
    if (index->count == 0) {
        return true;
    }
    uint64_t key_count = (index->count >> 16) + 1;
    if (key_count > (uint64_t)(SIZE_MAX / sizeof(uint64_t)) ||
        key_count + 1 < key_count ||
        key_count + 1 > (uint64_t)(SIZE_MAX / sizeof(uint64_t))) {
        return false;
    }
    uint64_t *key_bytes = calloc((size_t)key_count, sizeof(*key_bytes));
    if (key_bytes == NULL) {
        return false;
    }

    for (uint64_t key = 0; key < key_count; key++) {
        uint64_t first_line = aklv_cache_key_first_line(key);
        uint64_t last_line = aklv_cache_key_last_line(key, index->count);
        if (first_line > last_line) {
            continue;
        }
        uint64_t line_count = last_line - first_line + 1;
        key_bytes[key] = aklv_cache_saturating_add_u64(key_bytes[key],
                                                       line_count * sizeof(uint64_t));
    }

    for (uint64_t i = 0; i < posting_count; i++) {
        const AklvGramPosting *posting = postings[i];
        for (uint64_t c = 0; c < posting->lines.count; c++) {
            const AklvRoaringContainer *container = &posting->lines.containers[c];
            if (container->key >= key_count) {
                continue;
            }
            if (container->cardinality == 0) {
                continue;
            }
            uint64_t payload_bytes = aklv_cache_container_estimated_payload_bytes(container);
            uint64_t bytes = sizeof(AklvIndexCachePostingHeader) +
                             sizeof(AklvIndexCacheContainerHeader) +
                             payload_bytes;
            key_bytes[container->key] =
                aklv_cache_saturating_add_u64(key_bytes[container->key], bytes);
        }
    }

    plan->key_bytes = key_bytes;
    plan->key_count = key_count;
    return true;
}

static bool aklv_cache_build_posting_buckets(AklvCacheKeyPlan *plan,
                                             AklvGramPosting *const *postings,
                                             uint64_t posting_count) {
    if (plan->bucket_offsets != NULL || plan->key_count == 0) {
        return true;
    }
    if (posting_count > UINT32_MAX ||
        plan->key_count + 1 < plan->key_count ||
        plan->key_count + 1 > (uint64_t)(SIZE_MAX / sizeof(uint64_t))) {
        return false;
    }
    uint64_t *bucket_counts = calloc((size_t)(plan->key_count + 1), sizeof(*bucket_counts));
    if (bucket_counts == NULL) {
        return false;
    }
    for (uint64_t i = 0; i < posting_count; i++) {
        const AklvGramPosting *posting = postings[i];
        for (uint64_t c = 0; c < posting->lines.count; c++) {
            const AklvRoaringContainer *container = &posting->lines.containers[c];
            if (container->key < plan->key_count && container->cardinality != 0) {
                bucket_counts[container->key + 1]++;
            }
        }
    }

    uint64_t posting_index_count = 0;
    for (uint64_t key = 0; key < plan->key_count; key++) {
        uint64_t count = bucket_counts[key + 1];
        if (count > UINT64_MAX - posting_index_count) {
            free(bucket_counts);
            return false;
        }
        bucket_counts[key] = posting_index_count;
        posting_index_count += count;
    }
    bucket_counts[plan->key_count] = posting_index_count;
    if (posting_index_count > (uint64_t)(SIZE_MAX / sizeof(uint32_t)) ||
        posting_count > UINT32_MAX) {
        free(bucket_counts);
        return false;
    }
    uint32_t *posting_indices = NULL;
    if (posting_index_count != 0) {
        posting_indices = malloc((size_t)posting_index_count * sizeof(*posting_indices));
        if (posting_indices == NULL) {
            free(bucket_counts);
            return false;
        }
    }
    uint64_t *write_offsets = calloc((size_t)plan->key_count, sizeof(*write_offsets));
    if (write_offsets == NULL) {
        free(posting_indices);
        free(bucket_counts);
        return false;
    }
    memcpy(write_offsets, bucket_counts, (size_t)plan->key_count * sizeof(*write_offsets));
    for (uint64_t i = 0; i < posting_count; i++) {
        const AklvGramPosting *posting = postings[i];
        for (uint64_t c = 0; c < posting->lines.count; c++) {
            const AklvRoaringContainer *container = &posting->lines.containers[c];
            if (container->key >= plan->key_count || container->cardinality == 0) {
                continue;
            }
            posting_indices[write_offsets[container->key]++] = (uint32_t)i;
        }
    }
    free(write_offsets);

    plan->bucket_offsets = bucket_counts;
    plan->posting_indices = posting_indices;
    plan->posting_index_count = posting_index_count;
    return true;
}

static bool aklv_cache_write_offset_range(FILE *stream,
                                          AklvCacheWriteScratch *scratch,
                                          const AklvLineIndex *index,
                                          uint64_t first_line,
                                          uint64_t last_line) {
    if (scratch == NULL || scratch->offset_values == NULL) {
        return false;
    }
    bool done = false;
    bool ok = true;
    while (ok && !done && first_line <= last_line) {
        size_t count = 0;
        while (count < AKLV_CACHE_OFFSET_CONVERT_BATCH && first_line <= last_line) {
            uint64_t pos = first_line - 1;
            size_t packed = index->offset_blocks[pos >> AKLV_LINE_INDEX_BLOCK_SHIFT]
                                                [pos & AKLV_LINE_INDEX_BLOCK_MASK];
            scratch->offset_values[count++] = (uint64_t)packed;
            if (first_line == last_line || first_line == UINT64_MAX) {
                done = true;
                break;
            }
            first_line++;
        }
        if (!aklv_write_all(stream, scratch->offset_values, count * sizeof(*scratch->offset_values))) {
            ok = false;
            break;
        }
    }
    return ok;
}

static bool aklv_cache_write_shard(const AklvCacheFileIdentity *identity,
                                   AklvLineIndex *index,
                                   AklvIndexCache *cache,
                                   const char hash[33],
                                   AklvGramPosting *const *postings,
                                   uint64_t all_posting_count,
                                   const AklvCacheKeyPlan *key_plan,
                                   AklvCacheContainerWritePlan *container_plans,
                                   AklvCacheWriteScratch *scratch,
                                   uint32_t container_plan_capacity,
                                   uint32_t *posting_marks,
                                   uint32_t *shard_posting_indices,
                                   uint32_t posting_mark,
                                   unsigned char *stream_buffer,
                                   uint32_t shard_index,
                                   uint64_t first_line,
                                   uint64_t last_line,
                                   char *error,
                                   size_t error_cap) {
    char *cache_path = aklv_index_cache_path_for(hash, shard_index);
    if (cache_path == NULL) {
        aklv_set_error(error, error_cap, "failed to allocate cache path");
        return false;
    }
    char tmp_path[PATH_MAX];
    uint64_t tmp_id = atomic_fetch_add_explicit(&g_aklv_cache_tmp_counter, 1, memory_order_relaxed);
    int tmp_len = snprintf(tmp_path,
                           sizeof(tmp_path),
                           "%s.%ld.%" PRIu64 ".tmp",
                           cache_path,
                           (long)getpid(),
                           tmp_id);
    if (tmp_len < 0 || (size_t)tmp_len >= sizeof(tmp_path)) {
        aklv_set_error(error, error_cap, "cache temp path too long: %s", cache_path);
        free(cache_path);
        return false;
    }
    FILE *stream = fopen(tmp_path, "wb");
    if (stream == NULL) {
        aklv_set_error(error, error_cap, "open cache file failed: %s: %s", tmp_path, strerror(errno));
        free(cache_path);
        return false;
    }
    if (stream_buffer != NULL) {
        setvbuf(stream, (char *)stream_buffer, _IOFBF, AKLV_CACHE_STREAM_BUFFER_BYTES);
    }

    AklvIndexCacheFileHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, AKLV_INDEX_CACHE_MAGIC, sizeof(header.magic));
    header.version = AKLV_INDEX_CACHE_VERSION;
    header.shard_index = shard_index;
    header.file_size = identity->file_size;
    header.file_mtime_sec = identity->mtime_sec;
    header.file_mtime_nsec = identity->mtime_nsec;
    header.line_count = index->count;
    header.first_line = first_line;
    header.last_line = last_line;
    header.offset_count = last_line >= first_line ? last_line - first_line + 1 : 0;
    header.gram_size = AKLV_ASCII_GRAM_MAX_SIZE;

    bool ok = aklv_write_all(stream, &header, sizeof(header)) &&
              aklv_cache_write_padding(stream, AKLV_INDEX_CACHE_HEADER_SIZE - sizeof(header)) &&
              aklv_cache_write_offset_range(stream, scratch, index, first_line, last_line);
    uint64_t shard_posting_count = 0;
    bool whole_index_shard = first_line == 1 && last_line >= index->count;
    if (ok && !whole_index_shard &&
        key_plan->key_count != 0 && posting_marks != NULL && shard_posting_indices != NULL) {
        uint64_t first_key = first_line >> 16;
        uint64_t last_key = last_line >> 16;
        if (last_key >= key_plan->key_count) {
            last_key = key_plan->key_count - 1;
        }
        for (uint64_t key = first_key; key <= last_key; key++) {
            for (uint64_t i = key_plan->bucket_offsets[key]; i < key_plan->bucket_offsets[key + 1]; i++) {
                uint32_t posting_index = key_plan->posting_indices[i];
                if (posting_marks[posting_index] == posting_mark) {
                    continue;
                }
                posting_marks[posting_index] = posting_mark;
                shard_posting_indices[shard_posting_count++] = posting_index;
            }
            if (key == UINT64_MAX) {
                break;
            }
        }
    }
    uint64_t write_posting_count = whole_index_shard ? all_posting_count : shard_posting_count;
    for (uint64_t i = 0; ok && i < write_posting_count; i++) {
        AklvGramPosting *posting = whole_index_shard ? postings[i] : postings[shard_posting_indices[i]];
        AklvCachePostingRangePlan plan =
            aklv_cache_plan_posting_range(posting,
                                          first_line,
                                          last_line,
                                          container_plans,
                                          container_plan_capacity);
        if (plan.container_count == UINT32_MAX ||
            plan.serialized_bytes == UINT64_MAX ||
            plan.payload_bytes == UINT64_MAX) {
            ok = false;
            break;
        }
        if (plan.container_count != 0 && plan.serialized_bytes != 0) {
            if (plan.container_count > UINT32_MAX - header.total_container_count) {
                ok = false;
                break;
            }
            header.gram_count++;
            header.total_container_count += plan.container_count;
            header.payload_bytes = aklv_cache_saturating_add_u64(header.payload_bytes,
                                                                 plan.payload_bytes);
            ok = aklv_cache_write_posting_range(stream,
                                                scratch,
                                                posting,
                                                container_plans,
                                                plan);
        }
    }
    if (ok) {
        ok = fflush(stream) == 0 &&
             fseek(stream, 0, SEEK_SET) == 0 &&
             aklv_write_all(stream, &header, sizeof(header));
    }
    if (fclose(stream) != 0) {
        ok = false;
    }
    if (!ok || rename(tmp_path, cache_path) != 0) {
        aklv_set_error(error, error_cap, "failed to write cache shard: %s", cache_path);
        unlink(tmp_path);
        free(cache_path);
        return false;
    }
    bool added = true;
    if (cache != NULL) {
        added = aklv_index_cache_add_shard(cache, shard_index, first_line, last_line, cache_path);
    }
    free(cache_path);
    if (!added) {
        aklv_set_error(error, error_cap, "failed to remember cache shard");
        return false;
    }
    return true;
}

static int aklv_index_cache_write_files(const char *path,
                                        const AklvMappedFile *mapped,
                                        AklvLineIndex *index,
                                        AklvIndexCache *cache,
                                        const char hash[33],
                                        char *error,
                                        size_t error_cap) {
    (void)mapped;
    AklvGramPosting **postings = NULL;
    uint64_t posting_count = 0;
    AklvCacheKeyPlan key_plan;
    AklvCacheContainerWritePlan *container_plans = NULL;
    uint32_t *posting_marks = NULL;
    uint32_t *shard_posting_indices = NULL;
    unsigned char *stream_buffer = NULL;
    AklvCacheWriteScratch scratch;
    AklvCacheFileIdentity identity;
    memset(&scratch, 0, sizeof(scratch));
    memset(&key_plan, 0, sizeof(key_plan));
    memset(&identity, 0, sizeof(identity));
    if (!aklv_file_stat_identity(path, &identity.file_size, &identity.mtime_sec, &identity.mtime_nsec)) {
        aklv_set_error(error, error_cap, "stat file failed before cache store: %s: %s", path, strerror(errno));
        return -1;
    }
    if (!aklv_cache_collect_all_postings(&index->gram_index, &postings, &posting_count) ||
        !aklv_cache_build_key_plan(index, postings, posting_count, &key_plan)) {
        free(postings);
        aklv_cache_key_plan_destroy(&key_plan);
        aklv_set_error(error, error_cap, "failed to plan cache shards");
        return -1;
    }
    bool needs_posting_buckets = false;
    if (key_plan.key_count > 0) {
        uint64_t total = AKLV_INDEX_CACHE_HEADER_SIZE;
        uint64_t last_line = 1;
        for (uint64_t key = 0; key < key_plan.key_count; key++) {
            uint64_t key_last = aklv_cache_key_last_line(key, index->count);
            uint64_t add = key_plan.key_bytes[key];
            if (last_line > 1 && (add > UINT64_MAX - total || total + add > AKLV_INDEX_SHARD_MAX_BYTES)) {
                needs_posting_buckets = true;
                break;
            }
            total = aklv_cache_saturating_add_u64(total, add);
            last_line = key_last;
        }
        if (last_line < index->count) {
            needs_posting_buckets = true;
        }
    }
    if (needs_posting_buckets && !aklv_cache_build_posting_buckets(&key_plan, postings, posting_count)) {
        free(postings);
        aklv_cache_key_plan_destroy(&key_plan);
        aklv_set_error(error, error_cap, "failed to build cache shard posting buckets");
        return -1;
    }
    if (needs_posting_buckets && posting_count > 0) {
        if (posting_count > (uint64_t)(SIZE_MAX / sizeof(*posting_marks))) {
            free(postings);
            aklv_cache_key_plan_destroy(&key_plan);
            aklv_set_error(error, error_cap, "too many postings to plan cache shard");
            return -1;
        }
        posting_marks = calloc((size_t)posting_count, sizeof(*posting_marks));
        shard_posting_indices = malloc((size_t)posting_count * sizeof(*shard_posting_indices));
        if (posting_marks == NULL || shard_posting_indices == NULL) {
            free(shard_posting_indices);
            free(posting_marks);
            free(postings);
            aklv_cache_key_plan_destroy(&key_plan);
            aklv_set_error(error, error_cap, "failed to allocate cache shard posting plan");
            return -1;
        }
    }

    uint64_t max_container_count = 0;
    for (uint64_t i = 0; i < posting_count; i++) {
        if (postings[i]->lines.count > max_container_count) {
            max_container_count = postings[i]->lines.count;
        }
    }
    if (max_container_count > (uint64_t)UINT32_MAX ||
        max_container_count > (uint64_t)(SIZE_MAX / sizeof(*container_plans))) {
        free(shard_posting_indices);
        free(posting_marks);
        free(postings);
        aklv_cache_key_plan_destroy(&key_plan);
        aklv_set_error(error, error_cap, "too many posting containers to plan cache shard");
        return -1;
    }
    if (max_container_count != 0) {
        container_plans = malloc((size_t)max_container_count * sizeof(*container_plans));
        if (container_plans == NULL) {
            free(shard_posting_indices);
            free(posting_marks);
            free(postings);
            aklv_cache_key_plan_destroy(&key_plan);
            aklv_set_error(error, error_cap, "failed to allocate cache container plan");
            return -1;
        }
    }
    stream_buffer = malloc(AKLV_CACHE_STREAM_BUFFER_BYTES);
    if (!aklv_cache_write_scratch_init(&scratch)) {
        free(stream_buffer);
        free(container_plans);
        free(shard_posting_indices);
        free(posting_marks);
        free(postings);
        aklv_cache_key_plan_destroy(&key_plan);
        aklv_set_error(error, error_cap, "failed to allocate cache write scratch");
        return -1;
    }

    uint32_t shard_index = 0;
    uint64_t first_line = 1;
    while (first_line <= index->count) {
        uint64_t last_good_line = index->count;
        if (key_plan.key_count > 0) {
            uint64_t first_key = first_line >> 16;
            uint64_t total = AKLV_INDEX_CACHE_HEADER_SIZE;
            uint64_t key = first_key;
            last_good_line = first_line;
            while (key < key_plan.key_count) {
                uint64_t key_last = aklv_cache_key_last_line(key, index->count);
                if (key_last < first_line) {
                    key++;
                    continue;
                }
                uint64_t add = key_plan.key_bytes[key];
                if ((add > UINT64_MAX - total || total + add > AKLV_INDEX_SHARD_MAX_BYTES) &&
                    key > first_key) {
                    break;
                }
                total = aklv_cache_saturating_add_u64(total, add);
                last_good_line = key_last;
                if (last_good_line >= index->count) {
                    break;
                }
                key++;
            }
            if (last_good_line < first_line) {
                last_good_line = first_line;
            }
        }
        if (!aklv_cache_write_shard(&identity,
                                    index,
                                    cache,
                                    hash,
                                    postings,
                                    posting_count,
                                    &key_plan,
                                    container_plans,
                                    &scratch,
                                    (uint32_t)max_container_count,
                                    posting_marks,
                                    shard_posting_indices,
                                    shard_index + 1,
                                    stream_buffer,
                                    shard_index,
                                    first_line,
                                    last_good_line,
                                    error,
                                    error_cap)) {
            aklv_cache_write_scratch_destroy(&scratch);
            free(stream_buffer);
            free(container_plans);
            free(shard_posting_indices);
            free(posting_marks);
            free(postings);
            aklv_cache_key_plan_destroy(&key_plan);
            return -1;
        }
        shard_index++;
        if (last_good_line == UINT64_MAX || last_good_line >= index->count) {
            break;
        }
        first_line = last_good_line + 1;
    }
    aklv_cache_write_scratch_destroy(&scratch);
    free(stream_buffer);
    free(container_plans);
    free(shard_posting_indices);
    free(posting_marks);
    free(postings);
    aklv_cache_key_plan_destroy(&key_plan);
    return 0;
}

int aklv_index_cache_store(const char *path,
                           const AklvMappedFile *mapped,
                           AklvLineIndex *index,
                           char *error,
                           size_t error_cap) {
    if (path == NULL || mapped == NULL || index == NULL || index->count == 0) {
        return 0;
    }
    if (!aklv_cache_dir_ensure(error, error_cap)) {
        return -1;
    }
    aklv_index_cache_destroy(&index->cache);
    memset(&index->cache, 0, sizeof(index->cache));
    if (pthread_mutex_init(&index->cache.mutex, NULL) != 0) {
        aklv_set_error(error, error_cap, "failed to initialize index cache lock");
        return -1;
    }
    index->cache.mutex_initialized = true;
    if (pthread_cond_init(&index->cache.cond, NULL) != 0) {
        aklv_set_error(error, error_cap, "failed to initialize index cache lock");
        aklv_index_cache_destroy(&index->cache);
        return -1;
    }
    index->cache.cond_initialized = true;
    index->cache.dir = aklv_strdup(AKLV_INDEX_CACHE_DIR);
    if (index->cache.dir == NULL || !aklv_index_cache_hash_path(path, index->cache.hash)) {
        aklv_set_error(error, error_cap, "failed to initialize index cache metadata");
        return -1;
    }

    if (aklv_index_cache_write_files(path,
                                     mapped,
                                     index,
                                     &index->cache,
                                     index->cache.hash,
                                     error,
                                     error_cap) != 0) {
        return -1;
    }
    index->cache.available = index->cache.shard_count > 0;
    aklv_gram_index_destroy(&index->gram_index);
    return 0;
}

static int aklv_index_cache_store_files_only(const char *path,
                                             const AklvMappedFile *mapped,
                                             AklvLineIndex *index,
                                             char *error,
                                             size_t error_cap) {
    if (path == NULL || mapped == NULL || index == NULL || index->count == 0) {
        return 0;
    }
    if (!aklv_cache_dir_ensure(error, error_cap)) {
        return -1;
    }
    char hash[33];
    if (!aklv_index_cache_hash_path(path, hash)) {
        aklv_set_error(error, error_cap, "failed to initialize index cache metadata");
        return -1;
    }
    return aklv_index_cache_write_files(path, mapped, index, NULL, hash, error, error_cap);
}

typedef struct {
    AklvFile *file;
} AklvIndexCacheStoreTask;

static void *aklv_index_cache_store_worker_main(void *arg) {
    AklvIndexCacheStoreTask *task = (AklvIndexCacheStoreTask *)arg;
    AklvFile *file = task->file;
    free(task);
    char error[256] = {0};
    (void)aklv_index_cache_store_files_only(file->path,
                                            &file->mapped,
                                            &file->index,
                                            error,
                                            sizeof(error));
    aklv_file_release(file);
    return NULL;
}

static void aklv_index_cache_store_async(AklvFile *file) {
    if (file == NULL) {
        return;
    }
    AklvIndexCacheStoreTask *task = calloc(1, sizeof(*task));
    if (task == NULL) {
        return;
    }
    aklv_file_retain(file);
    task->file = file;
    pthread_t thread;
    int rc = pthread_create(&thread, NULL, aklv_index_cache_store_worker_main, task);
    if (rc != 0) {
        aklv_file_release(file);
        free(task);
        return;
    }
    pthread_detach(thread);
}

static bool aklv_cache_read_header(FILE *stream,
                                   AklvIndexCacheFileHeader *header,
                                   char *error,
                                   size_t error_cap) {
    memset(header, 0, sizeof(*header));
    if (!aklv_read_all(stream, header, sizeof(*header))) {
        aklv_set_error(error, error_cap, "failed to read index cache header");
        return false;
    }
    if (memcmp(header->magic, AKLV_INDEX_CACHE_MAGIC, sizeof(header->magic)) != 0 ||
        header->version != AKLV_INDEX_CACHE_VERSION ||
        header->gram_size != AKLV_ASCII_GRAM_MAX_SIZE) {
        aklv_set_error(error, error_cap, "index cache header mismatch");
        return false;
    }
    if (fseek(stream, (long)AKLV_INDEX_CACHE_HEADER_SIZE, SEEK_SET) != 0) {
        aklv_set_error(error, error_cap, "failed to seek index cache payload");
        return false;
    }
    return true;
}

static bool aklv_cache_read_offsets(FILE *stream,
                                    AklvLineIndex *index,
                                    uint64_t count,
                                    char *error,
                                    size_t error_cap) {
    uint64_t read_count = 0;
    while (read_count < count) {
        uint64_t in_block = index->count & AKLV_LINE_INDEX_BLOCK_MASK;
        if (in_block == 0 && aklv_line_index_add_block(index, error, error_cap) != 0) {
            return false;
        }
        uint64_t block = index->count >> AKLV_LINE_INDEX_BLOCK_SHIFT;
        uint64_t writable = AKLV_LINE_INDEX_BLOCK_LINES - in_block;
        uint64_t remaining = count - read_count;
        if (writable > remaining) {
            writable = remaining;
        }
        if (writable > (uint64_t)(SIZE_MAX / sizeof(**index->offset_blocks)) ||
            !aklv_read_all(stream,
                           index->offset_blocks[block] + in_block,
                           (size_t)writable * sizeof(**index->offset_blocks))) {
            aklv_set_error(error, error_cap, "failed to read cached line offsets");
            return false;
        }
        index->count += writable;
        read_count += writable;
    }
    return true;
}

int aklv_index_cache_try_load(const char *path,
                              const AklvMappedFile *mapped,
                              AklvLineIndex *index,
                              char *error,
                              size_t error_cap) {
    memset(index, 0, sizeof(*index));
    if (path == NULL || mapped == NULL || mapped->size == 0) {
        return 1;
    }
    char hash[33];
    if (!aklv_index_cache_hash_path(path, hash)) {
        return 1;
    }
    uint64_t file_size = 0;
    uint64_t mtime_sec = 0;
    uint64_t mtime_nsec = 0;
    if (!aklv_file_stat_identity(path, &file_size, &mtime_sec, &mtime_nsec)) {
        return 1;
    }
    if (pthread_mutex_init(&index->cache.mutex, NULL) != 0) {
        aklv_set_error(error, error_cap, "failed to initialize index cache lock");
        return -1;
    }
    index->cache.mutex_initialized = true;
    if (pthread_cond_init(&index->cache.cond, NULL) != 0) {
        aklv_line_index_destroy(index);
        aklv_set_error(error, error_cap, "failed to initialize index cache lock");
        return -1;
    }
    index->cache.cond_initialized = true;
    index->cache.dir = aklv_strdup(AKLV_INDEX_CACHE_DIR);
    if (index->cache.dir == NULL) {
        aklv_line_index_destroy(index);
        aklv_set_error(error, error_cap, "failed to allocate index cache metadata");
        return -1;
    }
    snprintf(index->cache.hash, sizeof(index->cache.hash), "%s", hash);

    uint32_t shard_index = 0;
    uint64_t expected_first_line = 1;
    uint64_t expected_line_count = 0;
    while (true) {
        char *cache_path = aklv_index_cache_path_for(hash, shard_index);
        if (cache_path == NULL) {
            aklv_line_index_destroy(index);
            return -1;
        }
        FILE *stream = fopen(cache_path, "rb");
        if (stream == NULL) {
            free(cache_path);
            break;
        }
        AklvIndexCacheFileHeader header;
        bool ok = aklv_cache_read_header(stream, &header, error, error_cap);
        if (ok) {
            ok = header.shard_index == shard_index &&
                 header.file_size == file_size &&
                 header.file_mtime_sec == mtime_sec &&
                 header.file_mtime_nsec == mtime_nsec &&
                 header.first_line == expected_first_line &&
                 header.last_line >= header.first_line &&
                 header.offset_count == header.last_line - header.first_line + 1;
        }
        if (ok && shard_index == 0) {
            expected_line_count = header.line_count;
            uint64_t block_capacity =
                (expected_line_count + AKLV_LINE_INDEX_BLOCK_LINES - 1) >>
                AKLV_LINE_INDEX_BLOCK_SHIFT;
            ok = aklv_line_index_reserve_block_arrays(index,
                                                      block_capacity,
                                                      error,
                                                      error_cap) == 0;
        }
        if (ok) {
            ok = header.line_count == expected_line_count &&
                 aklv_cache_read_offsets(stream, index, header.offset_count, error, error_cap);
        }
        fclose(stream);
        if (!ok ||
            !aklv_index_cache_add_shard(&index->cache,
                                        shard_index,
                                        header.first_line,
                                        header.last_line,
                                        cache_path)) {
            free(cache_path);
            aklv_line_index_destroy(index);
            return 1;
        }
        expected_first_line = header.last_line + 1;
        free(cache_path);
        shard_index++;
        if (index->count == expected_line_count) {
            break;
        }
    }
    if (index->count == 0 || index->count != expected_line_count) {
        aklv_line_index_destroy(index);
        return 1;
    }
    index->cache.available = index->cache.shard_count > 0;
    return 0;
}

static void *aklv_cache_payload_arena_alloc(AklvCachePayloadArena *arena,
                                            size_t size,
                                            size_t alignment) {
    if (size == 0) {
        return NULL;
    }
    if (arena == NULL || arena->cursor == NULL || arena->end == NULL ||
        alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return NULL;
    }
    uintptr_t cursor = (uintptr_t)arena->cursor;
    uintptr_t aligned = (cursor + (uintptr_t)alignment - 1U) & ~((uintptr_t)alignment - 1U);
    if (aligned < cursor || aligned > (uintptr_t)arena->end) {
        return NULL;
    }
    unsigned char *ptr = (unsigned char *)aligned;
    if ((size_t)(arena->end - ptr) < size) {
        return NULL;
    }
    arena->cursor = ptr + size;
    return ptr;
}

static bool aklv_cache_read_container(FILE *stream,
                                      AklvRoaringContainer *container,
                                      AklvCachePayloadArena *payload_arena,
                                      char *error,
                                      size_t error_cap) {
    AklvIndexCacheContainerHeader header;
    if (!aklv_read_all(stream, &header, sizeof(header))) {
        aklv_set_error(error, error_cap, "failed to read cached roaring container");
        return false;
    }
    memset(container, 0, sizeof(*container));
    container->key = header.key;
    container->cardinality = header.cardinality;
    container->type = (AklvRoaringContainerType)header.type;
    switch (container->type) {
        case AKLV_ROARING_FULL:
            container->cardinality = 65536;
            return true;
        case AKLV_ROARING_RUN:
            container->run_count = header.payload_count;
            if (container->run_count > 0) {
                if ((uint64_t)container->run_count > (uint64_t)(SIZE_MAX / sizeof(*container->runs))) {
                    aklv_set_error(error, error_cap, "cached run container too large");
                    return false;
                }
                size_t bytes = (size_t)container->run_count * sizeof(*container->runs);
                container->runs =
                    aklv_cache_payload_arena_alloc(payload_arena, bytes, _Alignof(AklvRoaringRun));
                if (container->runs == NULL ||
                    !aklv_read_all(stream, container->runs, bytes)) {
                    aklv_set_error(error, error_cap, "failed to read cached run container");
                    aklv_roaring_container_destroy(container, payload_arena != NULL);
                    return false;
                }
            }
            return true;
        case AKLV_ROARING_BITMAP:
            if (header.payload_count != 1024) {
                aklv_set_error(error, error_cap, "invalid cached bitmap container");
                return false;
            }
            container->bitmap =
                aklv_cache_payload_arena_alloc(payload_arena,
                                               1024 * sizeof(*container->bitmap),
                                               _Alignof(uint64_t));
            if (container->bitmap == NULL ||
                !aklv_read_all(stream, container->bitmap, 1024 * sizeof(*container->bitmap))) {
                aklv_set_error(error, error_cap, "failed to read cached bitmap container");
                aklv_roaring_container_destroy(container, payload_arena != NULL);
                return false;
            }
            return true;
        case AKLV_ROARING_ARRAY:
            container->capacity = header.payload_count;
            container->cardinality = header.payload_count;
            if (container->capacity > 0) {
                if ((uint64_t)container->capacity > (uint64_t)(SIZE_MAX / sizeof(*container->array))) {
                    aklv_set_error(error, error_cap, "cached array container too large");
                    return false;
                }
                size_t bytes = (size_t)container->capacity * sizeof(*container->array);
                container->array =
                    aklv_cache_payload_arena_alloc(payload_arena, bytes, _Alignof(uint16_t));
                if (container->array == NULL ||
                    !aklv_read_all(stream, container->array, bytes)) {
                    aklv_set_error(error, error_cap, "failed to read cached array container");
                    aklv_roaring_container_destroy(container, payload_arena != NULL);
                    return false;
                }
            }
            return true;
        default:
            aklv_set_error(error, error_cap, "invalid cached container type");
            return false;
    }
}

int aklv_index_cache_load_shard(const AklvIndexCacheShard *shard,
                                const AklvMappedFile *mapped,
                                AklvGramIndex *gram_index,
                                char *error,
                                size_t error_cap) {
    (void)mapped;
    memset(gram_index, 0, sizeof(*gram_index));
    if (shard == NULL || shard->path == NULL) {
        return -1;
    }
    int rc = -1;
    unsigned char *stream_buffer = NULL;
    FILE *stream = fopen(shard->path, "rb");
    if (stream == NULL) {
        aklv_set_error(error, error_cap, "failed to open index shard: %s", shard->path);
        return -1;
    }
    stream_buffer = malloc(AKLV_CACHE_STREAM_BUFFER_BYTES);
    if (stream_buffer != NULL) {
        setvbuf(stream, (char *)stream_buffer, _IOFBF, AKLV_CACHE_STREAM_BUFFER_BYTES);
    }
    AklvIndexCacheFileHeader header;
    if (!aklv_cache_read_header(stream, &header, error, error_cap)) {
        goto done;
    }
    uint64_t offset_bytes = header.offset_count * sizeof(uint64_t);
    if (fseek(stream, (long)(AKLV_INDEX_CACHE_HEADER_SIZE + offset_bytes), SEEK_SET) != 0) {
        aklv_set_error(error, error_cap, "failed to seek cached gram postings");
        goto done;
    }
    if (!aklv_gram_index_reserve_for_count(gram_index, header.gram_count)) {
        aklv_set_error(error, error_cap, "failed to reserve cached gram index");
        goto done;
    }
    AklvRoaringContainer *container_arena = NULL;
    unsigned char *payload_arena = NULL;
    AklvCachePayloadArena payload_state = {0};
    uint32_t container_arena_used = 0;
    if (header.payload_bytes != 0) {
        uint64_t alignment_slop = (uint64_t)header.total_container_count * (_Alignof(uint64_t) - 1U);
        if (alignment_slop < header.total_container_count ||
            alignment_slop > UINT64_MAX - header.payload_bytes ||
            header.payload_bytes + alignment_slop > (uint64_t)SIZE_MAX) {
            aklv_set_error(error, error_cap, "cached payload arena too large");
            goto done;
        }
        size_t payload_arena_bytes = (size_t)(header.payload_bytes + alignment_slop);
        payload_arena = malloc(payload_arena_bytes);
        if (payload_arena == NULL) {
            aklv_set_error(error, error_cap, "failed to allocate cached payload arena");
            goto done;
        }
        payload_state.cursor = payload_arena;
        payload_state.end = payload_arena + payload_arena_bytes;
    }
    if (header.total_container_count != 0) {
        if ((uint64_t)header.total_container_count > (uint64_t)(SIZE_MAX / sizeof(*container_arena))) {
            aklv_set_error(error, error_cap, "cached container arena too large");
            goto done;
        }
        container_arena = calloc(header.total_container_count, sizeof(*container_arena));
        if (container_arena == NULL) {
            aklv_set_error(error, error_cap, "failed to allocate cached container arena");
            goto done;
        }
    }
    for (uint64_t i = 0; i < header.gram_count; i++) {
        AklvIndexCachePostingHeader posting_header;
        if (!aklv_read_all(stream, &posting_header, sizeof(posting_header))) {
            aklv_set_error(error, error_cap, "failed to read cached gram posting");
            goto done;
        }
        AklvGramPosting *posting = aklv_gram_index_slot(gram_index, posting_header.gram, true);
        if (posting == NULL) {
            aklv_set_error(error, error_cap, "failed to allocate cached gram posting");
            goto done;
        }
        if (posting_header.container_count > 0) {
            if (container_arena == NULL ||
                posting_header.container_count > header.total_container_count - container_arena_used) {
                aklv_set_error(error, error_cap, "cached container count mismatch");
                goto done;
            }
            posting->lines.containers = container_arena + container_arena_used;
            posting->lines.borrowed_containers = true;
            posting->lines.borrowed_payloads = payload_arena != NULL;
            container_arena_used += posting_header.container_count;
        }
        posting->lines.count = posting_header.container_count;
        posting->lines.capacity = posting_header.container_count;
        posting->lines.cardinality = posting_header.cardinality;
        for (uint32_t c = 0; c < posting_header.container_count; c++) {
            if (!aklv_cache_read_container(stream,
                                               &posting->lines.containers[c],
                                               &payload_state,
                                               error,
                                               error_cap)) {
                goto done;
            }
        }
    }
    gram_index->container_arena = container_arena;
    gram_index->payload_arena = payload_arena;
    container_arena = NULL;
    payload_arena = NULL;
    rc = 0;

done:
    fclose(stream);
    free(stream_buffer);
    if (rc != 0) {
        aklv_gram_index_destroy(gram_index);
        free(container_arena);
        free(payload_arena);
    }
    return rc;
}

int aklv_index_cache_get_shard_index(AklvIndexCache *cache,
                                     AklvIndexCacheShard *shard,
                                     const AklvMappedFile *mapped,
                                     const AklvGramIndex **gram_index,
                                     char *error,
                                     size_t error_cap) {
    if (gram_index != NULL) {
        *gram_index = NULL;
    }
    if (cache == NULL || shard == NULL || gram_index == NULL) {
        return -1;
    }
    pthread_mutex_lock(&cache->mutex);
    while (!shard->gram_index_loaded && shard->gram_index_loading) {
        pthread_cond_wait(&cache->cond, &cache->mutex);
    }
    if (shard->gram_index_loaded) {
        *gram_index = &shard->gram_index;
        pthread_mutex_unlock(&cache->mutex);
        return 0;
    }
    shard->gram_index_loading = true;
    pthread_mutex_unlock(&cache->mutex);

    AklvGramIndex loaded;
    int rc = aklv_index_cache_load_shard(shard, mapped, &loaded, error, error_cap);

    pthread_mutex_lock(&cache->mutex);
    if (rc == 0 && !shard->gram_index_loaded) {
        shard->gram_index = loaded;
        memset(&loaded, 0, sizeof(loaded));
        shard->gram_index_loaded = true;
    } else if (rc == 0) {
        aklv_gram_index_destroy(&loaded);
    }
    shard->gram_index_loading = false;
    pthread_cond_broadcast(&cache->cond);
    if (rc == 0 && shard->gram_index_loaded) {
        *gram_index = &shard->gram_index;
    } else {
        *gram_index = NULL;
    }
    pthread_mutex_unlock(&cache->mutex);
    return rc;
}

static int aklv_line_index_collect_ascii_grams_for_run(AklvGramScratch *scratch,
                                                        const unsigned char *run,
                                                        size_t run_len) {
    if (run_len < AKLV_ASCII_GRAM_MAX_SIZE) {
        return 0;
    }
    uint32_t first = aklv_fold_ascii_byte(run[0]);
    uint32_t second = aklv_fold_ascii_byte(run[1]);
    for (size_t i = AKLV_ASCII_GRAM_MAX_SIZE - 1; i < run_len; i++) {
        uint32_t third = aklv_fold_ascii_byte(run[i]);
        uint32_t gram = (AKLV_ASCII_GRAM_MAX_SIZE << 24) |
                        (first << 16) |
                        (second << 8) |
                        third;
        aklv_gram_scratch_add_prepared(scratch, gram);
        first = second;
        second = third;
    }
    return 0;
}

static int aklv_line_index_add_ngrams(AklvGramIndex *gram_index,
                                      AklvLineView effective,
                                      uint64_t line_no,
                                      AklvGramScratch *scratch,
                                      char *error,
                                      size_t error_cap) {
    aklv_gram_scratch_reset(scratch);
    size_t max_grams = effective.len >= AKLV_ASCII_GRAM_MAX_SIZE
        ? effective.len - AKLV_ASCII_GRAM_MAX_SIZE + 1
        : 0;
    if (max_grams > AKLV_ASCII_GRAM_KEY_SPACE) {
        max_grams = AKLV_ASCII_GRAM_KEY_SPACE;
    }
    if (max_grams == 0) {
        return 0;
    }
    if (aklv_gram_scratch_prepare_line(scratch, max_grams, error, error_cap) != 0) {
        return -1;
    }
    size_t i = 0;
    while (i < effective.len) {
        while (i < effective.len && !aklv_is_index_byte(effective.start[i])) {
            i++;
        }
        size_t run_start = i;
        while (i < effective.len && aklv_is_index_byte(effective.start[i])) {
            i++;
        }
        if (i > run_start &&
            aklv_line_index_collect_ascii_grams_for_run(scratch,
                                                        effective.start + run_start,
                                                        i - run_start) != 0) {
            return -1;
        }
    }
    for (uint64_t gram_i = 0; gram_i < scratch->count; gram_i++) {
        if (!aklv_gram_index_add(gram_index, scratch->items[gram_i], line_no)) {
            aklv_set_error(error, error_cap, "failed to append ngram posting");
            return -1;
        }
    }
    return 0;
}

static uint32_t aklv_effective_prefix_skip(AklvLineView line) {
    if (line.len > 0 && line.start[line.len - 1] == '\r') {
        line.len--;
    }
    if (line.len == 0 || line.start[0] != '[') {
        return 0;
    }
    const unsigned char *bang = aklv_find_byte_simd(line.start, line.len, '!');
    if (bang == NULL) {
        return 0;
    }
    size_t prefix_len = (size_t)((bang + 1) - line.start);
    if (prefix_len >= AKLV_LINE_PREFIX_OVERFLOW) {
        return AKLV_LINE_PREFIX_OVERFLOW;
    }
    return (uint32_t)prefix_len;
}

static AklvLineView aklv_line_index_effective_line_for_build(const AklvMappedFile *mapped,
                                                             const AklvLineIndex *index,
                                                             uint64_t line_no) {
    uint64_t pos = line_no - 1;
    size_t packed = index->offset_blocks[pos >> AKLV_LINE_INDEX_BLOCK_SHIFT]
                                        [pos & AKLV_LINE_INDEX_BLOCK_MASK];
    size_t offset = aklv_line_index_unpack_offset(packed);
    size_t next_offset = mapped->size;
    if (line_no < index->count) {
        uint64_t next_pos = pos + 1;
        next_offset = aklv_line_index_unpack_offset(
            index->offset_blocks[next_pos >> AKLV_LINE_INDEX_BLOCK_SHIFT]
                                [next_pos & AKLV_LINE_INDEX_BLOCK_MASK]);
    }
    AklvLineView line;
    line.start = mapped->data + offset;
    line.len = next_offset > offset ? next_offset - offset : 0;
    while (line.len > 0 && (line.start[line.len - 1] == '\n' || line.start[line.len - 1] == '\r')) {
        line.len--;
    }
    uint32_t prefix_skip = aklv_line_index_unpack_prefix_skip(packed);
    if (prefix_skip == AKLV_LINE_PREFIX_OVERFLOW) {
        return aklv_effective_search_line(line);
    }
    if (prefix_skip > 0 && prefix_skip < line.len) {
        line.start += prefix_skip;
        line.len -= prefix_skip;
    } else if (prefix_skip >= line.len) {
        line.start += line.len;
        line.len = 0;
    }
    return line;
}

static uint64_t aklv_gram_build_task_byte_span(const AklvGramBuildTask *task) {
    if (task->first_line == 0 || task->last_line < task->first_line || task->line_index->count == 0) {
        return 0;
    }
    size_t first_offset = aklv_line_index_offset(task->line_index, task->first_line);
    size_t end_offset = task->mapped->size;
    if (task->last_line < task->line_index->count) {
        end_offset = aklv_line_index_offset(task->line_index, task->last_line + 1);
    }
    return end_offset > first_offset ? (uint64_t)(end_offset - first_offset) : 0;
}

static size_t aklv_gram_build_task_max_line_grams(const AklvGramBuildTask *task) {
    if (task->first_line == 0 || task->last_line < task->first_line ||
        task->line_index->block_max_grams == NULL) {
        return 0;
    }
    uint64_t first_block = (task->first_line - 1) >> AKLV_LINE_INDEX_BLOCK_SHIFT;
    uint64_t last_block = (task->last_line - 1) >> AKLV_LINE_INDEX_BLOCK_SHIFT;
    if (last_block >= task->line_index->block_count) {
        last_block = task->line_index->block_count == 0 ? 0 : task->line_index->block_count - 1;
    }
    size_t max_grams = 0;
    for (uint64_t block = first_block; block <= last_block; block++) {
        if (task->line_index->block_max_grams[block] > max_grams) {
            max_grams = task->line_index->block_max_grams[block];
        }
        if (block == UINT64_MAX) {
            break;
        }
    }
    if (max_grams > AKLV_ASCII_GRAM_KEY_SPACE) {
        max_grams = AKLV_ASCII_GRAM_KEY_SPACE;
    }
    return max_grams;
}

static void *aklv_gram_build_worker_main(void *arg) {
    AklvGramBuildTask *task = (AklvGramBuildTask *)arg;
    AklvGramScratch scratch = {0};
    uint64_t reserve_count =
        aklv_gram_build_task_byte_span(task) / AKLV_WORKER_GRAM_RESERVE_BYTES_PER_GRAM + 1U;
    if (reserve_count > AKLV_WORKER_GRAM_RESERVE_MAX) {
        reserve_count = AKLV_WORKER_GRAM_RESERVE_MAX;
    }
    if (reserve_count > AKLV_ASCII_GRAM_KEY_SPACE) {
        reserve_count = AKLV_ASCII_GRAM_KEY_SPACE;
    }
    if (!aklv_gram_index_reserve_for_count(&task->gram_index, reserve_count)) {
        task->status = -1;
        snprintf(task->error, sizeof(task->error), "%s", "failed to reserve worker gram index");
        return NULL;
    }
    size_t max_line_grams = aklv_gram_build_task_max_line_grams(task);
    if (max_line_grams > 0 &&
        aklv_gram_scratch_prepare_line(&scratch,
                                       max_line_grams,
                                       task->error,
                                       sizeof(task->error)) != 0) {
        task->status = -1;
        return NULL;
    }
    for (uint64_t line_no = task->first_line; line_no <= task->last_line; line_no++) {
        if (task->build_cancel != NULL &&
            atomic_load_explicit(task->build_cancel, memory_order_relaxed)) {
            task->status = -2;
            snprintf(task->error, sizeof(task->error), "%s", "index build cancelled");
            break;
        }
        if (((line_no - task->first_line) & 0x3fffU) == 0 &&
            task->cancel != NULL &&
            atomic_load_explicit(task->cancel, memory_order_relaxed)) {
            task->status = -2;
            snprintf(task->error, sizeof(task->error), "%s", "index build cancelled");
            if (task->build_cancel != NULL) {
                atomic_store_explicit(task->build_cancel, true, memory_order_relaxed);
            }
            break;
        }
        AklvLineView effective =
            aklv_line_index_effective_line_for_build(task->mapped, task->line_index, line_no);
        if (aklv_line_index_add_ngrams(&task->gram_index,
                                       effective,
                                       line_no,
                                       &scratch,
                                       task->error,
                                       sizeof(task->error)) != 0) {
            task->status = -1;
            if (task->build_cancel != NULL) {
                atomic_store_explicit(task->build_cancel, true, memory_order_relaxed);
            }
            break;
        }
    }
    aklv_gram_scratch_destroy(&scratch);
    return NULL;
}

int aklv_build_line_index(const AklvMappedFile *mapped,
                          AklvLineIndex *index,
                          const atomic_bool *cancel,
                          char *error,
                          size_t error_cap) {
    memset(index, 0, sizeof(*index));
    if (mapped->size == 0) {
        return 0;
    }

    const unsigned char *end = mapped->data + mapped->size;
    const unsigned char *line_start = mapped->data;
    uint64_t max_line_count = (uint64_t)mapped->size;
    uint64_t max_block_count = (max_line_count + AKLV_LINE_INDEX_BLOCK_LINES - 1) >>
                               AKLV_LINE_INDEX_BLOCK_SHIFT;
    if (aklv_line_index_reserve_block_arrays(index, max_block_count, error, error_cap) != 0) {
        aklv_line_index_destroy(index);
        return -1;
    }
    uint64_t check_counter = 0;
    while (line_start < end) {
        if ((check_counter++ & 0x3fffU) == 0 && cancel != NULL && atomic_load_explicit(cancel, memory_order_relaxed)) {
            aklv_set_error(error, error_cap, "index build cancelled");
            aklv_line_index_destroy(index);
            return -2;
        }

        const unsigned char *newline = aklv_find_byte_simd(line_start, (size_t)(end - line_start), '\n');
        const unsigned char *line_end = newline == NULL ? end : newline;
        AklvLineView line;
        line.start = line_start;
        line.len = (size_t)(line_end - line_start);
        uint32_t prefix_skip = aklv_effective_prefix_skip(line);
        size_t effective_len = line.len;
        if (prefix_skip != AKLV_LINE_PREFIX_OVERFLOW) {
            effective_len = prefix_skip < line.len ? line.len - prefix_skip : 0;
        }
        size_t max_grams = effective_len >= AKLV_ASCII_GRAM_MAX_SIZE
            ? effective_len - AKLV_ASCII_GRAM_MAX_SIZE + 1
            : 0;
        size_t offset = (size_t)(line_start - mapped->data);

        if (aklv_line_index_append(index,
                                   offset,
                                   prefix_skip,
                                   max_grams,
                                   error,
                                   error_cap) != 0) {
            aklv_line_index_destroy(index);
            return -1;
        }

        if (newline == NULL || newline + 1 >= end) {
            break;
        }
        line_start = newline + 1;
    }

    if (cancel != NULL && atomic_load_explicit(cancel, memory_order_relaxed)) {
        aklv_set_error(error, error_cap, "index build cancelled");
        aklv_line_index_destroy(index);
        return -2;
    }
    if (index->count == 0) {
        return 0;
    }

    uint32_t thread_count = aklv_default_index_thread_count();
    if ((uint64_t)thread_count > index->count) {
        thread_count = (uint32_t)index->count;
    }
    if (thread_count == 0) {
        thread_count = 1;
    }

    AklvGramBuildTask *tasks = calloc(thread_count, sizeof(*tasks));
    if (tasks == NULL) {
        aklv_set_error(error, error_cap, "calloc failed while allocating index build tasks");
        aklv_line_index_destroy(index);
        return -1;
    }
    pthread_t *threads = NULL;
    if (thread_count > 1) {
        threads = calloc(thread_count, sizeof(*threads));
        if (threads == NULL) {
            aklv_set_error(error, error_cap, "calloc failed while allocating index build threads");
            free(tasks);
            aklv_line_index_destroy(index);
            return -1;
        }
    }

    atomic_bool build_cancel;
    atomic_init(&build_cancel, false);
    uint64_t base_lines = index->count / thread_count;
    uint64_t extra_lines = index->count % thread_count;
    uint64_t first_line = 1;
    for (uint32_t i = 0; i < thread_count; i++) {
        uint64_t span = base_lines + (i < extra_lines ? 1 : 0);
        tasks[i].mapped = mapped;
        tasks[i].line_index = index;
        tasks[i].first_line = first_line;
        tasks[i].last_line = first_line + span - 1;
        tasks[i].cancel = cancel;
        tasks[i].build_cancel = &build_cancel;
        first_line = tasks[i].last_line + 1;
    }

    uint32_t created_threads = 0;
    int rc = 0;
    if (thread_count == 1) {
        aklv_gram_build_worker_main(&tasks[0]);
    } else {
        for (uint32_t i = 0; i < thread_count; i++) {
            int pthread_rc = pthread_create(&threads[i], NULL, aklv_gram_build_worker_main, &tasks[i]);
            if (pthread_rc != 0) {
                aklv_set_error(error,
                               error_cap,
                               "pthread_create failed while building index: %s",
                               strerror(pthread_rc));
                atomic_store_explicit(&build_cancel, true, memory_order_relaxed);
                rc = -1;
                break;
            }
            created_threads++;
        }
        for (uint32_t i = 0; i < created_threads; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    if (rc == 0) {
        for (uint32_t i = 0; i < thread_count; i++) {
            if (tasks[i].status != 0) {
                rc = tasks[i].status;
                aklv_set_error(error,
                               error_cap,
                               "%s",
                               tasks[i].error[0] != '\0' ? tasks[i].error : "index build failed");
                break;
            }
        }
    }
    if (rc == 0) {
        uint64_t merged_count = 0;
        if (!aklv_gram_build_union_count(tasks, thread_count, &merged_count) ||
            !aklv_gram_index_reserve_for_count(&index->gram_index, merged_count)) {
            aklv_set_error(error, error_cap, "failed to reserve merged index table");
            rc = -1;
        }
    }
    if (rc == 0) {
        for (uint32_t i = 0; i < thread_count; i++) {
            if (cancel != NULL && atomic_load_explicit(cancel, memory_order_relaxed)) {
                aklv_set_error(error, error_cap, "index build cancelled");
                rc = -2;
                break;
            }
            if (!aklv_gram_index_merge_move(&index->gram_index, &tasks[i].gram_index)) {
                aklv_set_error(error, error_cap, "failed to merge index build shards");
                rc = -1;
                break;
            }
        }
    }

    for (uint32_t i = 0; i < thread_count; i++) {
        aklv_gram_index_destroy(&tasks[i].gram_index);
    }
    free(threads);
    free(tasks);
    if (rc != 0) {
        aklv_line_index_destroy(index);
        return rc;
    }
    return 0;
}

size_t aklv_line_index_offset(const AklvLineIndex *index, uint64_t line_no) {
    uint64_t pos = line_no - 1;
    return aklv_line_index_unpack_offset(index->offset_blocks[pos >> AKLV_LINE_INDEX_BLOCK_SHIFT]
                                                            [pos & AKLV_LINE_INDEX_BLOCK_MASK]);
}

uint32_t aklv_line_index_prefix_skip(const AklvLineIndex *index, uint64_t line_no) {
    uint64_t pos = line_no - 1;
    return aklv_line_index_unpack_prefix_skip(index->offset_blocks[pos >> AKLV_LINE_INDEX_BLOCK_SHIFT]
                                                                [pos & AKLV_LINE_INDEX_BLOCK_MASK]);
}

AklvLineView aklv_line_at_offset(const unsigned char *data, size_t size, size_t offset) {
    AklvLineView line;
    line.start = data + offset;
    line.len = 0;

    size_t end = offset;
    if (offset < size) {
        const unsigned char *newline = aklv_find_byte_simd(data + offset, size - offset, '\n');
        end = newline == NULL ? size : (size_t)(newline - data);
    }
    line.len = end - offset;
    if (line.len > 0 && line.start[line.len - 1] == '\r') {
        line.len--;
    }
    return line;
}

AklvLineView aklv_file_line_fast(const AklvFile *file, uint64_t line_no) {
    if (file == NULL || line_no == 0 || line_no > file->index.count || file->mapped.size == 0) {
        AklvLineView empty = {0};
        return empty;
    }
    size_t offset = aklv_line_index_offset(&file->index, line_no);
    size_t next = file->mapped.size;
    if (line_no < file->index.count) {
        next = aklv_line_index_offset(&file->index, line_no + 1);
    }
    AklvLineView line;
    line.start = file->mapped.data + offset;
    line.len = next > offset ? next - offset : 0;
    while (line.len > 0 && (line.start[line.len - 1] == '\n' || line.start[line.len - 1] == '\r')) {
        line.len--;
    }
    return line;
}

AklvLineView aklv_effective_search_line(AklvLineView line) {
    if (line.len > 0 && line.start[line.len - 1] == '\r') {
        line.len--;
    }
    if (line.len > 0 && line.start[0] == '[') {
        const unsigned char *bang = aklv_find_byte_simd(line.start, line.len, '!');
        if (bang != NULL) {
            size_t prefix_len = (size_t)((bang + 1) - line.start);
            line.start += prefix_len;
            line.len -= prefix_len;
        }
    }
    return line;
}

int aklv_file_open_path(const char *path,
                        uint32_t id,
                        const atomic_bool *cancel,
                        AklvFile **out,
                        char *error,
                        size_t error_cap) {
    *out = NULL;
    AklvFile *file = calloc(1, sizeof(*file));
    if (file == NULL) {
        aklv_set_error(error, error_cap, "calloc failed while opening file");
        return -1;
    }
    atomic_init(&file->refs, 1);
    file->id = id;
    file->mapped.fd = -1;
    file->path = aklv_strdup(path);
    file->name = aklv_path_basename_dup(path);
    if (file->path == NULL || file->name == NULL) {
        aklv_set_error(error, error_cap, "malloc failed while storing file path");
        aklv_file_release(file);
        return -1;
    }

    if (aklv_map_file(path, &file->mapped, error, error_cap) != 0) {
        aklv_file_release(file);
        return -1;
    }

    int rc = aklv_index_cache_try_load(path, &file->mapped, &file->index, error, error_cap);
    if (rc != 0) {
        rc = aklv_build_line_index(&file->mapped, &file->index, cancel, error, error_cap);
        if (rc != 0) {
            aklv_file_release(file);
            return rc;
        }
        if ((uint64_t)file->mapped.size > AKLV_SYNC_CACHE_STORE_MAX_FILE_BYTES) {
            aklv_index_cache_store_async(file);
        } else {
            char cache_error[256] = {0};
            if (aklv_index_cache_store(path, &file->mapped, &file->index, cache_error, sizeof(cache_error)) != 0) {
                (void)cache_error;
            }
        }
    }

    *out = file;
    return 0;
}

void aklv_file_retain(AklvFile *file) {
    if (file != NULL) {
        atomic_fetch_add_explicit(&file->refs, 1, memory_order_relaxed);
    }
}

void aklv_file_release(AklvFile *file) {
    if (file == NULL) {
        return;
    }
    if (atomic_fetch_sub_explicit(&file->refs, 1, memory_order_acq_rel) != 1) {
        return;
    }
    aklv_line_index_destroy(&file->index);
    aklv_unmap_file(&file->mapped);
    free(file->path);
    free(file->name);
    free(file);
}

size_t aklv_copy_line_snippet(AklvLineView line, char *out, size_t out_cap, bool *truncated_out) {
    if (out == NULL || out_cap == 0) {
        return 0;
    }
    size_t cap = out_cap - 1;
    size_t max_len = AKLV_SEARCH_SNIPPET_LIMIT;
    if (max_len > cap) {
        max_len = cap;
    }

    size_t copied = 0;
    bool truncated = false;
    for (size_t i = 0; i < line.len; i++) {
        if (copied >= max_len) {
            truncated = true;
            break;
        }
        unsigned char c = line.start[i];
        if (c == '\t') {
            c = ' ';
        }
        if (c < 0x20 || c == 0x7f) {
            c = '.';
        }
        out[copied++] = (char)c;
    }
    out[copied] = '\0';
    if (truncated_out != NULL) {
        *truncated_out = truncated || line.len > copied;
    }
    return copied;
}
