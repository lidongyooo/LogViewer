#include "aklv_loader.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct AklvLoadRequest {
    char *path;
    struct AklvLoadRequest *next;
} AklvLoadRequest;

typedef struct AklvLoadEventNode {
    AklvLoaderEvent event;
    struct AklvLoadEventNode *next;
} AklvLoadEventNode;

struct AklvLoader {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool stop;
    uint32_t next_file_id;
    uint32_t worker_count;
    pthread_t *workers;
    AklvLoadRequest *request_head;
    AklvLoadRequest *request_tail;
    AklvLoadEventNode *event_head;
    AklvLoadEventNode *event_tail;
    atomic_bool cancel;
};

static void aklv_loader_push_event(AklvLoader *loader, AklvLoaderEvent *event) {
    AklvLoadEventNode *node = calloc(1, sizeof(*node));
    if (node == NULL) {
        if (event->file != NULL) {
            aklv_file_release(event->file);
        }
        free(event->path);
        return;
    }
    node->event = *event;
    memset(event, 0, sizeof(*event));

    pthread_mutex_lock(&loader->mutex);
    if (loader->event_tail != NULL) {
        loader->event_tail->next = node;
    } else {
        loader->event_head = node;
    }
    loader->event_tail = node;
    pthread_mutex_unlock(&loader->mutex);
}

static void *aklv_loader_worker_main(void *arg) {
    AklvLoader *loader = (AklvLoader *)arg;

    while (true) {
        pthread_mutex_lock(&loader->mutex);
        while (!loader->stop && loader->request_head == NULL) {
            pthread_cond_wait(&loader->cond, &loader->mutex);
        }
        if (loader->stop) {
            pthread_mutex_unlock(&loader->mutex);
            return NULL;
        }

        AklvLoadRequest *request = loader->request_head;
        loader->request_head = request->next;
        if (loader->request_head == NULL) {
            loader->request_tail = NULL;
        }
        uint32_t file_id = loader->next_file_id++;
        pthread_mutex_unlock(&loader->mutex);

        AklvLoaderEvent event;
        memset(&event, 0, sizeof(event));
        event.path = aklv_strdup(request->path);
        char error[sizeof(event.error)] = {0};
        event.status = aklv_file_open_path(request->path, file_id, &loader->cancel, &event.file, error, sizeof(error));
        snprintf(event.error, sizeof(event.error), "%s", error);
        free(request->path);
        free(request);
        aklv_loader_push_event(loader, &event);
    }
}

AklvLoader *aklv_loader_create(uint32_t worker_count) {
    if (worker_count == 0) {
        long cores = sysconf(_SC_NPROCESSORS_ONLN);
        worker_count = cores > 8 ? 4 : 2;
    }
    if (worker_count > 8) {
        worker_count = 8;
    }

    AklvLoader *loader = calloc(1, sizeof(*loader));
    if (loader == NULL) {
        return NULL;
    }
    loader->worker_count = worker_count;
    loader->next_file_id = 1;
    atomic_init(&loader->cancel, false);
    if (pthread_mutex_init(&loader->mutex, NULL) != 0 ||
        pthread_cond_init(&loader->cond, NULL) != 0) {
        aklv_loader_destroy(loader);
        return NULL;
    }
    loader->workers = calloc(worker_count, sizeof(*loader->workers));
    if (loader->workers == NULL) {
        aklv_loader_destroy(loader);
        return NULL;
    }
    for (uint32_t i = 0; i < worker_count; i++) {
        int err = pthread_create(&loader->workers[i], NULL, aklv_loader_worker_main, loader);
        if (err != 0) {
            loader->worker_count = i;
            aklv_loader_destroy(loader);
            return NULL;
        }
    }
    return loader;
}

void aklv_loader_destroy(AklvLoader *loader) {
    if (loader == NULL) {
        return;
    }
    atomic_store(&loader->cancel, true);
    pthread_mutex_lock(&loader->mutex);
    loader->stop = true;
    pthread_cond_broadcast(&loader->cond);
    pthread_mutex_unlock(&loader->mutex);

    if (loader->workers != NULL) {
        for (uint32_t i = 0; i < loader->worker_count; i++) {
            if (loader->workers[i] != 0) {
                pthread_join(loader->workers[i], NULL);
            }
        }
        free(loader->workers);
    }

    AklvLoadRequest *request = loader->request_head;
    while (request != NULL) {
        AklvLoadRequest *next = request->next;
        free(request->path);
        free(request);
        request = next;
    }
    AklvLoadEventNode *event = loader->event_head;
    while (event != NULL) {
        AklvLoadEventNode *next = event->next;
        aklv_loader_event_destroy(&event->event);
        free(event);
        event = next;
    }
    pthread_cond_destroy(&loader->cond);
    pthread_mutex_destroy(&loader->mutex);
    free(loader);
}

bool aklv_loader_enqueue(AklvLoader *loader, const char *path) {
    if (loader == NULL || path == NULL || path[0] == '\0') {
        return false;
    }
    AklvLoadRequest *request = calloc(1, sizeof(*request));
    if (request == NULL) {
        return false;
    }
    request->path = aklv_strdup(path);
    if (request->path == NULL) {
        free(request);
        return false;
    }

    pthread_mutex_lock(&loader->mutex);
    if (loader->request_tail != NULL) {
        loader->request_tail->next = request;
    } else {
        loader->request_head = request;
    }
    loader->request_tail = request;
    pthread_cond_signal(&loader->cond);
    pthread_mutex_unlock(&loader->mutex);
    return true;
}

bool aklv_loader_poll(AklvLoader *loader, AklvLoaderEvent *event_out) {
    if (loader == NULL || event_out == NULL) {
        return false;
    }
    pthread_mutex_lock(&loader->mutex);
    AklvLoadEventNode *node = loader->event_head;
    if (node == NULL) {
        pthread_mutex_unlock(&loader->mutex);
        return false;
    }
    loader->event_head = node->next;
    if (loader->event_head == NULL) {
        loader->event_tail = NULL;
    }
    pthread_mutex_unlock(&loader->mutex);

    *event_out = node->event;
    free(node);
    return true;
}

void aklv_loader_event_destroy(AklvLoaderEvent *event) {
    if (event == NULL) {
        return;
    }
    if (event->file != NULL) {
        aklv_file_release(event->file);
    }
    free(event->path);
    memset(event, 0, sizeof(*event));
}
