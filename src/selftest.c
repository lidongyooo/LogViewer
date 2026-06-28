#include "aklv_index.h"
#include "aklv_search.h"

#include <inttypes.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int test_effective_line(void) {
    AklvLineView line;
    line.start = (const unsigned char *)"[lib] 0x100!0x200 target";
    line.len = strlen((const char *)line.start);
    AklvLineView effective = aklv_effective_search_line(line);
    if (effective.len != strlen("0x200 target") ||
        memcmp(effective.start, "0x200 target", effective.len) != 0) {
        fprintf(stderr, "effective search line failed\n");
        return 1;
    }
    return 0;
}

static int test_index(void) {
    const unsigned char data[] =
        "[skip] abc!Hello World\n"
        "plain target\n"
        "last line";
    AklvMappedFile mapped;
    memset(&mapped, 0, sizeof(mapped));
    mapped.fd = -1;
    mapped.data = data;
    mapped.size = sizeof(data) - 1;
    AklvLineIndex index;
    atomic_bool cancel;
    atomic_init(&cancel, false);
    char error[256] = {0};
    if (aklv_build_line_index(&mapped, &index, &cancel, error, sizeof(error)) != 0) {
        fprintf(stderr, "index build failed: %s\n", error);
        return 1;
    }
    int rc = 0;
    if (index.count != 3) {
        fprintf(stderr, "unexpected line count: %" PRIu64 "\n", index.count);
        rc = 1;
    }
    AklvLineView first = aklv_line_at_offset(data, mapped.size, aklv_line_index_offset(&index, 1));
    AklvLineView effective = aklv_effective_search_line(first);
    if (effective.len != strlen("Hello World") ||
        memcmp(effective.start, "Hello World", effective.len) != 0) {
        fprintf(stderr, "indexed effective line mismatch\n");
        rc = 1;
    }
    if (aklv_line_index_prefix_skip(&index, 1) != strlen("[skip] abc!")) {
        fprintf(stderr, "packed prefix skip mismatch: %u\n", aklv_line_index_prefix_skip(&index, 1));
        rc = 1;
    }
    aklv_line_index_destroy(&index);
    return rc;
}

static int test_search_service_stream(void) {
    char path[] = "/tmp/aklv_selftest_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    const char payload[] =
        "zero\n"
        "one target\n"
        "[skip target!visible only\n"
        "three target\n";
    if (write(fd, payload, sizeof(payload) - 1) != (ssize_t)(sizeof(payload) - 1)) {
        perror("write");
        close(fd);
        unlink(path);
        return 1;
    }
    close(fd);

    atomic_bool cancel;
    atomic_init(&cancel, false);
    AklvFile *file = NULL;
    char error[256] = {0};
    if (aklv_file_open_path(path, 1, &cancel, &file, error, sizeof(error)) != 0) {
        fprintf(stderr, "file open failed: %s\n", error);
        unlink(path);
        return 1;
    }

    AklvSearchService *service = aklv_search_service_create();
    if (service == NULL) {
        aklv_file_release(file);
        unlink(path);
        return 1;
    }
    AklvFile *items[1] = {file};
    AklvFileSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.items = items;
    snapshot.count = 1;
    snapshot.active_index = 0;
    snapshot.active_selected_line = 1;
    aklv_search_service_submit(service, "target", &snapshot);

    AklvSearchResults results;
    aklv_search_results_init(&results);
    int rc = 1;
    for (int i = 0; i < 200; i++) {
        aklv_search_results_swap_from_service(service, &results);
        if (results.complete && results.count == 2 &&
            results.items[0].line_no == 2 &&
            results.items[1].line_no == 4) {
            rc = 0;
            break;
        }
        usleep(10000);
    }

    if (rc != 0) {
        fprintf(stderr, "stream search selftest failed: complete=%d count=%" PRIu64 "\n",
                results.complete,
                results.count);
    }
    aklv_search_results_destroy(&results);
    aklv_search_service_destroy(service);
    aklv_file_release(file);
    unlink(path);
    return rc;
}

static int wait_for_search(AklvSearchService *service, AklvSearchResults *results) {
    for (int i = 0; i < 300; i++) {
        aklv_search_results_swap_from_service(service, results);
        if (results->complete) {
            return 0;
        }
        usleep(10000);
    }
    return 1;
}

static int test_search_service_roaring_index(void) {
    char path[] = "/tmp/aklv_roaring_selftest_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }

    FILE *stream = fdopen(fd, "w");
    if (stream == NULL) {
        perror("fdopen");
        close(fd);
        unlink(path);
        return 1;
    }
    fputs("[skip 0x99!visible alpha_token\n", stream);
    fputs("noise line\n", stream);
    fputs("rarity 0x99 target\n", stream);
    fputs("punctuation !!! marker\n", stream);
    for (int i = 0; i < 5000; i++) {
        fprintf(stream, "common token payload %d\n", i);
    }
    fputs("suffix ABCDEF\n", stream);
    if (fclose(stream) != 0) {
        perror("fclose");
        unlink(path);
        return 1;
    }

    atomic_bool cancel;
    atomic_init(&cancel, false);
    AklvFile *file = NULL;
    char error[256] = {0};
    if (aklv_file_open_path(path, 1, &cancel, &file, error, sizeof(error)) != 0) {
        fprintf(stderr, "roaring file open failed: %s\n", error);
        unlink(path);
        return 1;
    }

    AklvSearchService *service = aklv_search_service_create();
    if (service == NULL) {
        aklv_file_release(file);
        unlink(path);
        return 1;
    }
    AklvFile *items[1] = {file};
    AklvFileSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.items = items;
    snapshot.count = 1;

    AklvSearchResults results;
    aklv_search_results_init(&results);
    int rc = 0;

    aklv_search_service_submit(service, "0x99", &snapshot);
    if (wait_for_search(service, &results) != 0 || results.count != 1 || results.items[0].line_no != 3) {
        fprintf(stderr, "rare ascii search mismatch: complete=%d count=%" PRIu64 "\n",
                results.complete,
                results.count);
        rc = 1;
    }

    if (rc == 0) {
        aklv_search_service_submit(service, "tar", &snapshot);
        if (wait_for_search(service, &results) != 0 || results.count != 1 || results.items[0].line_no != 3) {
            fprintf(stderr, "ascii substring search mismatch: count=%" PRIu64 "\n", results.count);
            rc = 1;
        }
    }

    if (rc == 0) {
        aklv_search_service_submit(service, "0x99 target", &snapshot);
        if (wait_for_search(service, &results) != 0 || results.count != 1 || results.items[0].line_no != 3) {
            fprintf(stderr, "phrase verification search mismatch: count=%" PRIu64 "\n", results.count);
            rc = 1;
        }
    }

    if (rc == 0) {
        aklv_search_service_submit(service, "skip 0x99", &snapshot);
        if (wait_for_search(service, &results) != 0 || results.count != 0) {
            fprintf(stderr, "prefix crop search mismatch: count=%" PRIu64 "\n", results.count);
            rc = 1;
        }
    }

    if (rc == 0) {
        aklv_search_service_submit(service, "!!!", &snapshot);
        if (wait_for_search(service, &results) != 0 || results.count != 1 || results.items[0].line_no != 4) {
            fprintf(stderr, "punctuation ascii search mismatch: count=%" PRIu64 "\n", results.count);
            rc = 1;
        }
    }

    if (rc == 0) {
        aklv_search_service_submit(service, "0x99!", &snapshot);
        if (wait_for_search(service, &results) != 0 || results.count != 0) {
            fprintf(stderr, "cropped punctuation search mismatch: count=%" PRIu64 "\n", results.count);
            rc = 1;
        }
    }

    if (rc == 0) {
        aklv_search_service_submit(service, "0x99 target", &snapshot);
        if (wait_for_search(service, &results) != 0 || results.count != 1 || results.items[0].line_no != 3) {
            fprintf(stderr, "space-inclusive ascii search mismatch: count=%" PRIu64 "\n", results.count);
            rc = 1;
        }
    }

    if (rc == 0) {
        aklv_search_service_submit(service, "common", &snapshot);
        if (wait_for_search(service, &results) != 0 || results.count != 5000 ||
            results.items[0].line_no != 5 || results.items[4999].line_no != 5004) {
            fprintf(stderr, "high-cardinality roaring search mismatch: count=%" PRIu64 "\n", results.count);
            rc = 1;
        }
    }

    aklv_search_results_destroy(&results);
    aklv_search_service_destroy(service);
    aklv_file_release(file);
    unlink(path);
    return rc;
}

static int test_index_cache_reopen(void) {
    char path[] = "/tmp/aklv_cache_selftest_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    FILE *stream = fdopen(fd, "w");
    if (stream == NULL) {
        perror("fdopen");
        close(fd);
        unlink(path);
        return 1;
    }
    for (int i = 0; i < 20000; i++) {
        fprintf(stream, "[meta %d!line=%d common token payload target_%d\n", i, i, i % 7);
    }
    if (fclose(stream) != 0) {
        perror("fclose");
        unlink(path);
        return 1;
    }

    atomic_bool cancel;
    atomic_init(&cancel, false);
    AklvFile *first = NULL;
    char error[256] = {0};
    if (aklv_file_open_path(path, 1, &cancel, &first, error, sizeof(error)) != 0) {
        fprintf(stderr, "cache first open failed: %s\n", error);
        unlink(path);
        return 1;
    }
    if (!first->index.cache.available || first->index.cache.shard_count == 0 ||
        first->index.gram_index.count != 0) {
        fprintf(stderr, "cache was not stored after first open\n");
        aklv_file_release(first);
        unlink(path);
        return 1;
    }
    aklv_file_release(first);

    AklvFile *file = NULL;
    memset(error, 0, sizeof(error));
    if (aklv_file_open_path(path, 2, &cancel, &file, error, sizeof(error)) != 0) {
        fprintf(stderr, "cache reopen failed: %s\n", error);
        unlink(path);
        return 1;
    }
    if (!file->index.cache.available || file->index.cache.shard_count == 0 ||
        file->index.gram_index.count != 0 || file->index.count != 20000) {
        fprintf(stderr, "cache metadata mismatch on reopen\n");
        aklv_file_release(file);
        unlink(path);
        return 1;
    }
    AklvLineView view = aklv_file_line_fast(file, 1);
    const char expected_first_line[] = "[meta 0!line=0 common token payload target_0";
    if (view.len < strlen(expected_first_line) ||
        memcmp(view.start, expected_first_line, strlen(expected_first_line)) != 0) {
        fprintf(stderr, "cache-backed line view mismatch\n");
        aklv_file_release(file);
        unlink(path);
        return 1;
    }

    AklvSearchService *service = aklv_search_service_create();
    if (service == NULL) {
        aklv_file_release(file);
        unlink(path);
        return 1;
    }
    AklvFile *items[1] = {file};
    AklvFileSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.items = items;
    snapshot.count = 1;

    AklvSearchResults results;
    aklv_search_results_init(&results);
    aklv_search_service_submit(service, "target_3", &snapshot);
    int rc = 0;
    if (wait_for_search(service, &results) != 0 ||
        results.count != 2857 ||
        results.items[0].line_no != 4 ||
        results.items[results.count - 1].line_no != 19996 ||
        !file->index.cache.shards[0].gram_index_loaded) {
        fprintf(stderr, "cache-backed search failed: count=%" PRIu64 "\n", results.count);
        rc = 1;
    }
    if (rc == 0) {
        aklv_search_service_submit(service, "common", &snapshot);
        if (wait_for_search(service, &results) != 0 ||
            results.count != 20000 ||
            results.items[0].line_no != 1 ||
            results.items[results.count - 1].line_no != 20000) {
            fprintf(stderr, "cache-backed run chunk search failed: count=%" PRIu64 "\n", results.count);
            rc = 1;
        }
    }

    aklv_search_results_destroy(&results);
    aklv_search_service_destroy(service);
    aklv_file_release(file);
    unlink(path);
    return rc;
}

int main(void) {
    if (test_effective_line() != 0 ||
        test_index() != 0 ||
        test_search_service_stream() != 0 ||
        test_search_service_roaring_index() != 0 ||
        test_index_cache_reopen() != 0 ||
        aklv_search_selftest() != 0) {
        return 1;
    }
    puts("{\"type\":\"selftest\",\"status\":\"ok\"}");
    return 0;
}
