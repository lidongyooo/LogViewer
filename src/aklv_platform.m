#include "aklv_platform.h"

#import <AppKit/AppKit.h>
#import <dispatch/dispatch.h>

#include <stdlib.h>
#include <string.h>

static char *aklv_copy_cstr(const char *text) {
    size_t len = strlen(text);
    char *out = malloc(len + 1);
    if (out != NULL) {
        memcpy(out, text, len + 1);
    }
    return out;
}

char **aklv_platform_open_files(size_t *count_out) {
    if (count_out != NULL) {
        *count_out = 0;
    }

    __block char **paths = NULL;
    __block size_t count = 0;
    void (^open_block)(void) = ^{
        NSOpenPanel *panel = [NSOpenPanel openPanel];
        panel.canChooseFiles = YES;
        panel.canChooseDirectories = NO;
        panel.allowsMultipleSelection = YES;
        panel.title = @"Open log files";
        NSInteger result = [panel runModal];
        if (result != NSModalResponseOK) {
            return;
        }
        NSArray<NSURL *> *urls = [panel URLs];
        count = (size_t)[urls count];
        paths = calloc(count, sizeof(*paths));
        if (paths == NULL) {
            count = 0;
            return;
        }
        for (size_t i = 0; i < count; i++) {
            const char *path = [[[urls objectAtIndex:i] path] fileSystemRepresentation];
            paths[i] = aklv_copy_cstr(path);
        }
    };

    if ([NSThread isMainThread]) {
        open_block();
    } else {
        dispatch_sync(dispatch_get_main_queue(), open_block);
    }

    if (count_out != NULL) {
        *count_out = count;
    }
    return paths;
}

void aklv_platform_free_open_files(char **paths, size_t count) {
    if (paths == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        free(paths[i]);
    }
    free(paths);
}
