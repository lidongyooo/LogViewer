#include "aklv_index.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#define AKLV_GRAM_SCRATCH_INITIAL_CAPACITY 128
#define AKLV_GRAM_SCRATCH_INITIAL_TABLE_CAPACITY 512

typedef struct {
    uint32_t *items;
    uint64_t count;
    uint64_t capacity;
    uint32_t *table;
    uint32_t *used_slots;
    uint64_t table_capacity;
    uint64_t used_count;
    uint64_t used_capacity;
} AklvGramScratch;

static void aklv_set_error(char *error, size_t error_cap, const char *fmt, ...) {
    if (error == NULL || error_cap == 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(error, error_cap, fmt, args);
    va_end(args);
}

unsigned char aklv_fold_ascii_byte(unsigned char c) {
    if (c >= 'A' && c <= 'Z') {
        return (unsigned char)(c + ('a' - 'A'));
    }
    return c;
}

uint32_t aklv_ngram_key(const unsigned char *text, size_t len) {
    uint32_t key = (uint32_t)len << 24;
    for (size_t i = 0; i < len; i++) {
        key |= (uint32_t)aklv_fold_ascii_byte(text[i]) << (unsigned)(8 * (AKLV_ASCII_GRAM_MAX_SIZE - 1 - i));
    }
    return key;
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

static void aklv_roaring_container_destroy(AklvRoaringContainer *container) {
    if (container == NULL) {
        return;
    }
    free(container->array);
    free(container->bitmap);
    memset(container, 0, sizeof(*container));
}

static bool aklv_roaring_container_promote(AklvRoaringContainer *container);

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
    return true;
}

static bool aklv_roaring_container_add(AklvRoaringContainer *container, uint16_t low) {
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

static bool aklv_roaring_container_add_all(AklvRoaringContainer *dst,
                                           const AklvRoaringContainer *src) {
    uint32_t before = dst->cardinality;
    if (src->bitmap != NULL) {
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
    } else {
        for (uint32_t i = 0; i < src->cardinality; i++) {
            if (!aklv_roaring_container_add(dst, src->array[i])) {
                return false;
            }
        }
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
        aklv_roaring_container_destroy(&src->containers[0]);
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
        aklv_roaring_container_destroy(&bitmap->containers[i]);
    }
    free(bitmap->containers);
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

bool aklv_roaring_contains(const AklvRoaring *bitmap, uint64_t value) {
    uint64_t key = value >> 16;
    uint16_t low = (uint16_t)(value & 0xffffU);
    const AklvRoaringContainer *container = aklv_roaring_find_container(bitmap, key, NULL);
    if (container == NULL) {
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
    iter->min_low = container != NULL ? low : 0;
}

bool aklv_roaring_iter_next(AklvRoaringIter *iter, uint64_t *value_out) {
    if (iter == NULL || iter->bitmap == NULL || value_out == NULL) {
        return false;
    }
    while (iter->container_index < iter->bitmap->count) {
        const AklvRoaringContainer *container = &iter->bitmap->containers[iter->container_index];
        if (container->bitmap == NULL) {
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
    free(scratch->table);
    free(scratch->used_slots);
    memset(scratch, 0, sizeof(*scratch));
}

static void aklv_gram_scratch_reset(AklvGramScratch *scratch) {
    for (uint64_t i = 0; i < scratch->used_count; i++) {
        scratch->table[scratch->used_slots[i]] = 0;
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
    uint64_t new_capacity = scratch->used_capacity == 0 ? AKLV_GRAM_SCRATCH_INITIAL_TABLE_CAPACITY : scratch->used_capacity;
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

static bool aklv_gram_scratch_rehash(AklvGramScratch *scratch, uint64_t new_capacity) {
    uint32_t *new_table = calloc((size_t)new_capacity, sizeof(*new_table));
    if (new_table == NULL) {
        return false;
    }
    if (!aklv_gram_scratch_reserve_used_slots(scratch, scratch->count == 0 ? 1 : scratch->count)) {
        free(new_table);
        return false;
    }
    free(scratch->table);
    scratch->table = new_table;
    scratch->table_capacity = new_capacity;
    scratch->used_count = 0;
    for (uint64_t i = 0; i < scratch->count; i++) {
        uint32_t gram = scratch->items[i];
        uint64_t slot = aklv_gram_hash(gram) & (new_capacity - 1);
        while (scratch->table[slot] != 0) {
            slot = (slot + 1) & (new_capacity - 1);
        }
        scratch->table[slot] = gram;
        scratch->used_slots[scratch->used_count++] = (uint32_t)slot;
    }
    return true;
}

static bool aklv_gram_scratch_ensure_table(AklvGramScratch *scratch) {
    if (scratch->table_capacity == 0) {
        return aklv_gram_scratch_rehash(scratch, AKLV_GRAM_SCRATCH_INITIAL_TABLE_CAPACITY);
    }
    if ((scratch->count + 1) * 10 < scratch->table_capacity * 7) {
        return true;
    }
    if (scratch->table_capacity > UINT64_MAX / 2 ||
        scratch->table_capacity > (uint64_t)(UINT32_MAX / 2)) {
        return false;
    }
    return aklv_gram_scratch_rehash(scratch, scratch->table_capacity * 2);
}

static int aklv_gram_scratch_add(AklvGramScratch *scratch,
                                 uint32_t gram,
                                 char *error,
                                 size_t error_cap) {
    if (gram == 0) {
        aklv_set_error(error, error_cap, "invalid zero gram");
        return -1;
    }
    if (!aklv_gram_scratch_ensure_table(scratch) ||
        !aklv_gram_scratch_reserve_items(scratch, scratch->count + 1) ||
        !aklv_gram_scratch_reserve_used_slots(scratch, scratch->used_count + 1)) {
        aklv_set_error(error, error_cap, "failed to grow line gram scratch");
        return -1;
    }
    uint64_t slot = aklv_gram_hash(gram) & (scratch->table_capacity - 1);
    while (scratch->table[slot] != 0) {
        if (scratch->table[slot] == gram) {
            return 0;
        }
        slot = (slot + 1) & (scratch->table_capacity - 1);
    }
    scratch->table[slot] = gram;
    scratch->used_slots[scratch->used_count++] = (uint32_t)slot;
    scratch->items[scratch->count++] = gram;
    return 0;
}

static bool aklv_gram_index_rehash(AklvGramIndex *index, uint64_t new_capacity) {
    AklvGramPosting *old_items = index->items;
    uint64_t old_capacity = index->capacity;
    AklvGramPosting *new_items = calloc((size_t)new_capacity, sizeof(*new_items));
    if (new_items == NULL) {
        return false;
    }

    index->items = new_items;
    index->capacity = new_capacity;
    index->count = 0;
    for (uint64_t i = 0; i < old_capacity; i++) {
        if (!old_items[i].used) {
            continue;
        }
        uint64_t slot = aklv_gram_hash(old_items[i].gram) & (new_capacity - 1);
        while (new_items[slot].used) {
            slot = (slot + 1) & (new_capacity - 1);
        }
        new_items[slot] = old_items[i];
        memset(&old_items[i], 0, sizeof(old_items[i]));
        index->count++;
    }
    free(old_items);
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
    index->items[slot].used = true;
    index->items[slot].gram = gram;
    index->count++;
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
    if (index->items != NULL) {
        for (uint64_t i = 0; i < index->capacity; i++) {
            if (index->items[i].used) {
                aklv_roaring_destroy(&index->items[i].lines);
            }
        }
        free(index->items);
    }
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

static int aklv_compare_gram_posting_ptrs(const void *lhs, const void *rhs) {
    const AklvGramPosting *a = *(const AklvGramPosting * const *)lhs;
    const AklvGramPosting *b = *(const AklvGramPosting * const *)rhs;
    return (a->gram > b->gram) - (a->gram < b->gram);
}

static bool aklv_gram_index_merge_move(AklvGramIndex *dst, AklvGramIndex *src) {
    if (src == NULL || src->count == 0) {
        return true;
    }
    AklvGramPosting **postings = malloc((size_t)src->count * sizeof(*postings));
    if (postings == NULL) {
        return false;
    }
    uint64_t out = 0;
    for (uint64_t i = 0; i < src->capacity; i++) {
        if (src->items[i].used) {
            postings[out++] = &src->items[i];
        }
    }
    qsort(postings, (size_t)out, sizeof(*postings), aklv_compare_gram_posting_ptrs);
    for (uint64_t i = 0; i < out; i++) {
        AklvGramPosting *src_posting = postings[i];
        AklvGramPosting *dst_posting = aklv_gram_index_slot(dst, src_posting->gram, true);
        if (dst_posting == NULL ||
            !aklv_roaring_move_append(&dst_posting->lines, &src_posting->lines)) {
            free(postings);
            return false;
        }
    }
    free(postings);
    aklv_gram_index_destroy(src);
    return true;
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
    if (index->offset_blocks != NULL) {
        for (uint64_t i = 0; i < index->block_count; i++) {
            if (index->offset_blocks[i] != NULL) {
                munmap(index->offset_blocks[i],
                       (size_t)AKLV_LINE_INDEX_BLOCK_LINES * sizeof(**index->offset_blocks));
            }
        }
        free(index->offset_blocks);
    }
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

    size_t **new_offsets = realloc(index->offset_blocks, offset_bytes);
    if (new_offsets == NULL) {
        aklv_set_error(error, error_cap, "realloc failed while growing line offsets");
        return -1;
    }
    index->offset_blocks = new_offsets;
    memset((unsigned char *)index->offset_blocks + old_offset_bytes, 0, offset_bytes - old_offset_bytes);

    index->block_capacity = new_capacity;
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

static int aklv_line_index_append(AklvLineIndex *index,
                                  size_t offset,
                                  uint32_t prefix_skip,
                                  char *error,
                                  size_t error_cap) {
    uint64_t in_block = index->count & AKLV_LINE_INDEX_BLOCK_MASK;
    if (in_block == 0 && aklv_line_index_add_block(index, error, error_cap) != 0) {
        return -1;
    }
    uint64_t block = index->count >> AKLV_LINE_INDEX_BLOCK_SHIFT;
    index->offset_blocks[block][in_block] = aklv_line_index_pack_offset(offset, prefix_skip);
    index->count++;
    return 0;
}

static int aklv_line_index_collect_ascii_grams_for_run(AklvGramScratch *scratch,
                                                        const unsigned char *run,
                                                        size_t run_len,
                                                        char *error,
                                                        size_t error_cap) {
    if (run_len < AKLV_ASCII_GRAM_MAX_SIZE) {
        return 0;
    }
    for (size_t i = 0; i + AKLV_ASCII_GRAM_MAX_SIZE <= run_len; i++) {
        uint32_t gram = aklv_ngram_key(run + i, AKLV_ASCII_GRAM_MAX_SIZE);
        if (aklv_gram_scratch_add(scratch, gram, error, error_cap) != 0) {
            return -1;
        }
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
                                                        i - run_start,
                                                        error,
                                                        error_cap) != 0) {
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

static void *aklv_gram_build_worker_main(void *arg) {
    AklvGramBuildTask *task = (AklvGramBuildTask *)arg;
    AklvGramScratch scratch = {0};
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
        size_t offset = (size_t)(line_start - mapped->data);

        if (aklv_line_index_append(index,
                                   offset,
                                   prefix_skip,
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

AklvLineView aklv_file_line(const AklvFile *file, uint64_t line_no) {
    if (file == NULL || line_no == 0 || line_no > file->index.count || file->mapped.size == 0) {
        AklvLineView empty = {0};
        return empty;
    }
    size_t offset = aklv_line_index_offset(&file->index, line_no);
    return aklv_line_at_offset(file->mapped.data, file->mapped.size, offset);
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

AklvLineView aklv_file_effective_search_line(const AklvFile *file, uint64_t line_no) {
    return aklv_effective_search_line(aklv_file_line(file, line_no));
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

    int rc = aklv_build_line_index(&file->mapped, &file->index, cancel, error, error_cap);
    if (rc != 0) {
        aklv_file_release(file);
        return rc;
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
