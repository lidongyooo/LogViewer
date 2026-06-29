#include "aklv_index.h"
#include "aklv_loader.h"
#include "aklv_metal.h"
#include "aklv_platform.h"
#include "aklv_search.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_clipboard.h>

#include <inttypes.h>
#include <math.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define AKLV_NAV_H 34.0f
#define AKLV_NAV_TEXT_GAP 8.0f
#define AKLV_SEARCH_HEADER_H 34.0f
#define AKLV_SPLITTER_H 5.0f
#define AKLV_SCROLLBAR_W 12.0f
#define AKLV_LINE_NO_W 96.0f
#define AKLV_MAX_STATUS 512
#define AKLV_SEARCH_QUERY_CAP 256
#define AKLV_SEARCH_UNDO_DEPTH 32
#define AKLV_SEARCH_HISTORY_CAP 64
#define AKLV_TEXT_COPY_MAX (1024 * 1024)

typedef struct {
    float x;
    float y;
    float w;
    float h;
} AklvRect;

typedef struct {
    SDL_Window *window;
    AklvMetalRenderer *renderer;
    AklvLoader *loader;
    AklvSearchService *search_service;
    AklvSearchResults search_results;

    AklvFile **files;
    double *file_scroll_lines;
    double *file_scroll_cols;
    uint64_t *file_selected_lines;
    uint32_t file_count;
    uint32_t file_capacity;
    uint32_t active_file;

    double text_scroll_line;
    double text_scroll_col;
    double search_scroll;
    uint64_t selected_line;
    uint64_t selected_result;
    uint64_t search_anchor_line;
    uint32_t search_anchor_file_id;
    uint64_t text_cursor_line;
    size_t text_cursor_col;
    uint64_t text_select_anchor_line;
    size_t text_select_anchor_col;
    uint64_t text_select_focus_line;
    size_t text_select_focus_col;
    bool has_selected_result;
    bool has_search_anchor;
    bool has_text_cursor;
    bool has_text_selection;
    bool dragging_text_select;

    char search_query[AKLV_SEARCH_QUERY_CAP];
    size_t search_cursor;
    char search_undo[AKLV_SEARCH_UNDO_DEPTH][AKLV_SEARCH_QUERY_CAP];
    size_t search_undo_cursor[AKLV_SEARCH_UNDO_DEPTH];
    uint32_t search_undo_count;
    char search_history[AKLV_SEARCH_HISTORY_CAP][AKLV_SEARCH_QUERY_CAP];
    uint32_t search_history_count;
    int search_history_pos;
    char search_history_draft[AKLV_SEARCH_QUERY_CAP];
    size_t search_history_draft_cursor;
    char highlight_text[256];
    size_t highlight_len;
    bool search_input_focus;
    bool search_select_all;
    bool quit;
    bool resizing_search;
    bool dragging_text_scroll;
    bool dragging_search_scroll;
    bool mouse_down;
    float search_panel_h;
    float drag_grab_offset;
    char status[AKLV_MAX_STATUS];
} AklvApp;

static AklvColor C_BG = {0.105f, 0.118f, 0.145f, 1.0f};
static AklvColor C_PANEL = {0.135f, 0.150f, 0.185f, 1.0f};
static AklvColor C_PANEL_2 = {0.165f, 0.185f, 0.225f, 1.0f};
static AklvColor C_TAB = {0.150f, 0.172f, 0.210f, 1.0f};
static AklvColor C_TAB_ACTIVE = {0.090f, 0.250f, 0.360f, 1.0f};
static AklvColor C_TEXT = {0.850f, 0.885f, 0.920f, 1.0f};
static AklvColor C_MUTED = {0.520f, 0.590f, 0.650f, 1.0f};
static AklvColor C_ACCENT = {0.240f, 0.620f, 0.800f, 1.0f};
static AklvColor C_SELECT = {0.210f, 0.310f, 0.430f, 1.0f};
static AklvColor C_SEARCH_HIGHLIGHT = {0.660f, 0.520f, 0.160f, 0.72f};
static AklvColor C_WORD_HIGHLIGHT = {0.430f, 0.300f, 0.660f, 0.72f};
static AklvColor C_LINE_NO = {0.455f, 0.560f, 0.620f, 1.0f};
static AklvColor C_INPUT = {0.070f, 0.082f, 0.105f, 1.0f};

static bool aklv_rect_contains(AklvRect r, float x, float y) {
    return x >= r.x && y >= r.y && x < r.x + r.w && y < r.y + r.h;
}

static AklvFile *aklv_app_active_file(AklvApp *app) {
    if (app->file_count == 0 || app->active_file >= app->file_count) {
        return NULL;
    }
    return app->files[app->active_file];
}

static void aklv_app_save_active_view(AklvApp *app) {
    if (app->file_count == 0 || app->active_file >= app->file_count) {
        return;
    }
    app->file_scroll_lines[app->active_file] = app->text_scroll_line;
    app->file_scroll_cols[app->active_file] = app->text_scroll_col;
    app->file_selected_lines[app->active_file] = app->selected_line;
}

static void aklv_app_restore_active_view(AklvApp *app) {
    AklvFile *file = aklv_app_active_file(app);
    if (file == NULL) {
        app->text_scroll_line = 0.0;
        app->text_scroll_col = 0.0;
        app->selected_line = 0;
        return;
    }
    app->text_scroll_line = app->file_scroll_lines[app->active_file];
    app->text_scroll_col = app->file_scroll_cols[app->active_file];
    app->selected_line = app->file_selected_lines[app->active_file];
    if (app->selected_line == 0 && file->index.count > 0) {
        app->selected_line = 1;
    }
    if (app->selected_line > file->index.count) {
        app->selected_line = file->index.count;
    }
}

static void aklv_app_switch_file(AklvApp *app, uint32_t index) {
    if (index >= app->file_count) {
        return;
    }
    aklv_app_save_active_view(app);
    app->active_file = index;
    aklv_app_restore_active_view(app);
    app->highlight_len = 0;
    app->highlight_text[0] = '\0';
    app->has_text_cursor = false;
    app->has_text_selection = false;
    app->dragging_text_select = false;
}

static void aklv_app_set_status(AklvApp *app, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(app->status, sizeof(app->status), fmt, args);
    va_end(args);
}

static bool aklv_text_pos_before(uint64_t line_a, size_t col_a, uint64_t line_b, size_t col_b) {
    return line_a < line_b || (line_a == line_b && col_a < col_b);
}

static void aklv_app_normalized_text_selection(const AklvApp *app,
                                               uint64_t *start_line,
                                               size_t *start_col,
                                               uint64_t *end_line,
                                               size_t *end_col) {
    if (aklv_text_pos_before(app->text_select_focus_line,
                             app->text_select_focus_col,
                             app->text_select_anchor_line,
                             app->text_select_anchor_col)) {
        *start_line = app->text_select_focus_line;
        *start_col = app->text_select_focus_col;
        *end_line = app->text_select_anchor_line;
        *end_col = app->text_select_anchor_col;
    } else {
        *start_line = app->text_select_anchor_line;
        *start_col = app->text_select_anchor_col;
        *end_line = app->text_select_focus_line;
        *end_col = app->text_select_focus_col;
    }
}

static bool aklv_app_text_selection_non_empty(const AklvApp *app) {
    return app->has_text_selection &&
           (app->text_select_anchor_line != app->text_select_focus_line ||
            app->text_select_anchor_col != app->text_select_focus_col);
}

static void aklv_sanitize_copy_bytes(char *dst, const unsigned char *src, size_t len) {
    for (size_t i = 0; i < len; i++) {
        unsigned char c = src[i];
        if (c == '\t') {
            dst[i] = '\t';
        } else if (c < 0x20 || c == 0x7f) {
            dst[i] = ' ';
        } else if (c >= 0x80) {
            dst[i] = '?';
        } else {
            dst[i] = (char)c;
        }
    }
}

static bool aklv_app_copy_text_range(AklvApp *app) {
    if (!aklv_app_text_selection_non_empty(app)) {
        return false;
    }
    AklvFile *file = aklv_app_active_file(app);
    if (file == NULL) {
        return false;
    }

    uint64_t start_line = 0;
    uint64_t end_line = 0;
    size_t start_col = 0;
    size_t end_col = 0;
    aklv_app_normalized_text_selection(app, &start_line, &start_col, &end_line, &end_col);
    if (start_line == 0 || end_line == 0 || start_line > file->index.count) {
        return false;
    }
    if (end_line > file->index.count) {
        end_line = file->index.count;
    }

    size_t total = 0;
    bool truncated = false;
    for (uint64_t line_no = start_line; line_no <= end_line; line_no++) {
        AklvLineView line = aklv_file_line_fast(file, line_no);
        size_t begin = line_no == start_line ? start_col : 0;
        size_t end = line_no == end_line ? end_col : line.len;
        if (begin > line.len) {
            begin = line.len;
        }
        if (end > line.len) {
            end = line.len;
        }
        if (end < begin) {
            end = begin;
        }
        size_t add = end - begin;
        if (total + add > AKLV_TEXT_COPY_MAX) {
            add = AKLV_TEXT_COPY_MAX - total;
            truncated = true;
        }
        total += add;
        if (line_no != end_line && total < AKLV_TEXT_COPY_MAX) {
            total++;
        } else if (line_no != end_line) {
            truncated = true;
        }
        if (truncated || total >= AKLV_TEXT_COPY_MAX) {
            break;
        }
    }

    char *copy = malloc(total + 1);
    if (copy == NULL) {
        aklv_app_set_status(app, "copy failed: out of memory");
        return true;
    }
    size_t pos = 0;
    for (uint64_t line_no = start_line; line_no <= end_line && pos < total; line_no++) {
        AklvLineView line = aklv_file_line_fast(file, line_no);
        size_t begin = line_no == start_line ? start_col : 0;
        size_t end = line_no == end_line ? end_col : line.len;
        if (begin > line.len) {
            begin = line.len;
        }
        if (end > line.len) {
            end = line.len;
        }
        if (end < begin) {
            end = begin;
        }
        size_t add = end - begin;
        if (add > total - pos) {
            add = total - pos;
        }
        aklv_sanitize_copy_bytes(copy + pos, line.start + begin, add);
        pos += add;
        if (line_no != end_line && pos < total) {
            copy[pos++] = '\n';
        }
    }
    copy[total] = '\0';
    SDL_SetClipboardText(copy);
    free(copy);
    aklv_app_set_status(app, truncated ? "copied text selection (truncated to 1 MiB)" : "copied text selection");
    return true;
}

static void aklv_app_copy_text_selection(AklvApp *app) {
    if (aklv_app_copy_text_range(app)) {
        return;
    }
    if (app->highlight_len > 0) {
        SDL_SetClipboardText(app->highlight_text);
        aklv_app_set_status(app, "copied highlighted text");
        return;
    }

    AklvFile *file = aklv_app_active_file(app);
    if (file == NULL || app->selected_line == 0 || app->selected_line > file->index.count) {
        return;
    }

    AklvLineView line = aklv_file_line_fast(file, app->selected_line);
    size_t len = line.len;
    bool truncated = false;
    if (len > AKLV_TEXT_COPY_MAX) {
        len = AKLV_TEXT_COPY_MAX;
        truncated = true;
    }
    char *copy = malloc(len + 1);
    if (copy == NULL) {
        aklv_app_set_status(app, "copy failed: out of memory");
        return;
    }
    aklv_sanitize_copy_bytes(copy, line.start, len);
    copy[len] = '\0';
    SDL_SetClipboardText(copy);
    free(copy);
    aklv_app_set_status(app,
                        truncated ? "copied selected line (truncated to 1 MiB)" : "copied selected line");
}

static bool aklv_token_byte(unsigned char c) {
    return isalnum((int)c) || c == '_' || c == '-' || c == '.' ||
           c == '/' || c == ':' || c == '#' || c == '@';
}

static const unsigned char *aklv_memmem_simple(const unsigned char *haystack,
                                               size_t haystack_len,
                                               const char *needle,
                                               size_t needle_len) {
    if (needle_len == 0 || haystack_len < needle_len) {
        return NULL;
    }
    unsigned char first = (unsigned char)needle[0];
    const unsigned char *scan = haystack;
    size_t remaining = haystack_len;
    while (remaining >= needle_len) {
        const unsigned char *hit = aklv_find_byte_simd(scan, remaining - needle_len + 1, first);
        if (hit == NULL) {
            return NULL;
        }
        if (memcmp(hit, needle, needle_len) == 0) {
            return hit;
        }
        size_t consumed = (size_t)(hit - scan) + 1;
        scan += consumed;
        remaining -= consumed;
    }
    return NULL;
}

static const unsigned char *aklv_find_ascii_case_byte(const unsigned char *text,
                                                      size_t len,
                                                      unsigned char folded) {
    if (folded >= 'a' && folded <= 'z') {
        return aklv_find_either_byte_simd(text,
                                          len,
                                          folded,
                                          (unsigned char)(folded - ('a' - 'A')));
    }
    return aklv_find_byte_simd(text, len, folded);
}

static unsigned char aklv_ascii_lower_byte(unsigned char c) {
    if (c >= 'A' && c <= 'Z') {
        return (unsigned char)(c + ('a' - 'A'));
    }
    return c;
}

static void aklv_fold_ascii_pattern(unsigned char *out, const char *text, size_t len) {
    for (size_t i = 0; i < len; i++) {
        out[i] = aklv_ascii_lower_byte((unsigned char)text[i]);
    }
}

static bool aklv_ascii_case_equal_folded(const unsigned char *text,
                                         const unsigned char *folded_needle,
                                         size_t needle_len) {
    for (size_t i = 0; i < needle_len; i++) {
        if (aklv_ascii_lower_byte(text[i]) != folded_needle[i]) {
            return false;
        }
    }
    return true;
}

static const unsigned char *aklv_memmem_ascii_case_folded(const unsigned char *haystack,
                                                          size_t haystack_len,
                                                          const unsigned char *folded_needle,
                                                          size_t needle_len) {
    if (needle_len == 0 || haystack_len < needle_len) {
        return NULL;
    }
    unsigned char first = folded_needle[0];
    const unsigned char *scan = haystack;
    size_t remaining = haystack_len;
    while (remaining >= needle_len) {
        const unsigned char *hit =
            aklv_find_ascii_case_byte(scan, remaining - needle_len + 1, first);
        if (hit == NULL) {
            return NULL;
        }
        if (aklv_ascii_case_equal_folded(hit, folded_needle, needle_len)) {
            return hit;
        }
        size_t consumed = (size_t)(hit - scan) + 1;
        scan += consumed;
        remaining -= consumed;
    }
    return NULL;
}

static void aklv_draw_match_highlights(AklvMetalRenderer *r,
                                       float x,
                                       float y,
                                       float line_h,
                                       float char_w,
                                       const unsigned char *text,
                                       size_t len,
                                       const unsigned char *folded_needle,
                                       size_t needle_len,
                                       AklvColor color) {
    if (text == NULL || folded_needle == NULL || needle_len == 0 || len < needle_len) {
        return;
    }
    const unsigned char *scan = text;
    size_t remaining = len;
    while (remaining >= needle_len) {
        const unsigned char *hit = aklv_memmem_ascii_case_folded(scan,
                                                                 remaining,
                                                                 folded_needle,
                                                                 needle_len);
        if (hit == NULL) {
            break;
        }
        size_t hit_col = (size_t)(hit - text);
        size_t hit_offset = (size_t)(hit - scan);
        if (hit_offset > remaining) {
            break;
        }
        aklv_metal_push_rect(r,
                             x + (float)hit_col * char_w,
                             y,
                             (float)needle_len * char_w,
                             line_h,
                             color);
        size_t consumed = hit_offset + needle_len;
        if (consumed == 0 || consumed > remaining) {
            break;
        }
        scan += consumed;
        remaining -= consumed;
    }
}

static bool aklv_app_reserve_files(AklvApp *app, uint32_t capacity) {
    if (capacity <= app->file_capacity) {
        return true;
    }
    uint32_t new_capacity = app->file_capacity == 0 ? 8 : app->file_capacity * 2;
    while (new_capacity < capacity) {
        new_capacity *= 2;
    }
    AklvFile **new_files = calloc(new_capacity, sizeof(*new_files));
    double *new_scroll_lines = calloc(new_capacity, sizeof(*new_scroll_lines));
    double *new_scroll_cols = calloc(new_capacity, sizeof(*new_scroll_cols));
    uint64_t *new_selected_lines = calloc(new_capacity, sizeof(*new_selected_lines));
    if (new_files == NULL || new_scroll_lines == NULL ||
        new_scroll_cols == NULL || new_selected_lines == NULL) {
        free(new_files);
        free(new_scroll_lines);
        free(new_scroll_cols);
        free(new_selected_lines);
        return false;
    }
    if (app->file_count > 0) {
        memcpy(new_files, app->files, (size_t)app->file_count * sizeof(*new_files));
        memcpy(new_scroll_lines,
               app->file_scroll_lines,
               (size_t)app->file_count * sizeof(*new_scroll_lines));
        memcpy(new_scroll_cols,
               app->file_scroll_cols,
               (size_t)app->file_count * sizeof(*new_scroll_cols));
        memcpy(new_selected_lines,
               app->file_selected_lines,
               (size_t)app->file_count * sizeof(*new_selected_lines));
    }
    free(app->files);
    free(app->file_scroll_lines);
    free(app->file_scroll_cols);
    free(app->file_selected_lines);
    app->files = new_files;
    app->file_scroll_lines = new_scroll_lines;
    app->file_scroll_cols = new_scroll_cols;
    app->file_selected_lines = new_selected_lines;
    app->file_capacity = new_capacity;
    return true;
}

static void aklv_app_add_file(AklvApp *app, AklvFile *file) {
    if (!aklv_app_reserve_files(app, app->file_count + 1)) {
        aklv_app_set_status(app, "unable to add file: out of memory");
        return;
    }
    aklv_app_save_active_view(app);
    aklv_file_retain(file);
    app->files[app->file_count] = file;
    app->file_scroll_lines[app->file_count] = 0.0;
    app->file_scroll_cols[app->file_count] = 0.0;
    app->file_selected_lines[app->file_count] = file->index.count > 0 ? 1 : 0;
    app->active_file = app->file_count;
    app->file_count++;
    aklv_app_restore_active_view(app);
    aklv_app_set_status(app,
                        "opened %s: %" PRIu64 " lines, %.2f GiB",
                        file->name,
                        file->index.count,
                        (double)file->mapped.size / (1024.0 * 1024.0 * 1024.0));
}

static void aklv_app_close_file(AklvApp *app, uint32_t index) {
    if (index >= app->file_count) {
        return;
    }
    if (index != app->active_file) {
        aklv_app_save_active_view(app);
    }
    aklv_file_release(app->files[index]);
    memmove(app->files + index,
            app->files + index + 1,
            (size_t)(app->file_count - index - 1) * sizeof(*app->files));
    memmove(app->file_scroll_lines + index,
            app->file_scroll_lines + index + 1,
            (size_t)(app->file_count - index - 1) * sizeof(*app->file_scroll_lines));
    memmove(app->file_scroll_cols + index,
            app->file_scroll_cols + index + 1,
            (size_t)(app->file_count - index - 1) * sizeof(*app->file_scroll_cols));
    memmove(app->file_selected_lines + index,
            app->file_selected_lines + index + 1,
            (size_t)(app->file_count - index - 1) * sizeof(*app->file_selected_lines));
    app->file_count--;
    if (app->file_count == 0) {
        app->active_file = 0;
        app->selected_line = 0;
        app->text_scroll_line = 0.0;
        aklv_search_service_cancel(app->search_service);
        return;
    }
    if (app->active_file >= app->file_count) {
        app->active_file = app->file_count - 1;
    }
    aklv_app_restore_active_view(app);
    app->highlight_len = 0;
    app->highlight_text[0] = '\0';
}

static void aklv_app_destroy(AklvApp *app) {
    aklv_search_results_destroy(&app->search_results);
    if (app->search_service != NULL) {
        aklv_search_service_destroy(app->search_service);
    }
    if (app->loader != NULL) {
        aklv_loader_destroy(app->loader);
    }
    for (uint32_t i = 0; i < app->file_count; i++) {
        aklv_file_release(app->files[i]);
    }
    free(app->files);
    free(app->file_scroll_lines);
    free(app->file_scroll_cols);
    free(app->file_selected_lines);
    if (app->renderer != NULL) {
        aklv_metal_destroy(app->renderer);
    }
    if (app->window != NULL) {
        SDL_DestroyWindow(app->window);
    }
}

static void aklv_get_window_points(SDL_Window *window, int *w_out, int *h_out, float *scale_out) {
    int w = 1280;
    int h = 800;
    int pw = 1280;
    int ph = 800;
    SDL_GetWindowSize(window, &w, &h);
    SDL_GetWindowSizeInPixels(window, &pw, &ph);
    float scale = w > 0 ? (float)pw / (float)w : 1.0f;
    if (scale <= 0.0f) {
        scale = 1.0f;
    }
    *w_out = w;
    *h_out = h;
    *scale_out = scale;
}

static AklvRect aklv_text_rect(int w, int h, float search_h) {
    AklvRect r;
    r.x = 0.0f;
    r.y = AKLV_NAV_H + AKLV_NAV_TEXT_GAP;
    r.w = (float)w;
    r.h = (float)h - AKLV_NAV_H - AKLV_NAV_TEXT_GAP - search_h - AKLV_SPLITTER_H;
    if (r.h < 64.0f) {
        r.h = 64.0f;
    }
    return r;
}

static AklvRect aklv_splitter_rect(int w, int h, float search_h) {
    AklvRect text = aklv_text_rect(w, h, search_h);
    AklvRect r = {0.0f, text.y + text.h, (float)w, AKLV_SPLITTER_H};
    return r;
}

static AklvRect aklv_search_rect(int w, int h, float search_h) {
    AklvRect r = {0.0f, (float)h - search_h, (float)w, search_h};
    return r;
}

static double aklv_clamp_double(double value, double lo, double hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static void aklv_app_clamp_scroll(AklvApp *app, int win_w, int win_h) {
    AklvFile *file = aklv_app_active_file(app);
    AklvRect text = aklv_text_rect(win_w, win_h, app->search_panel_h);
    double line_h = aklv_metal_line_height(app->renderer);
    double visible = line_h > 0.0 ? floor(text.h / line_h) : 1.0;
    double max_scroll = 0.0;
    if (file != NULL && file->index.count > (uint64_t)visible) {
        max_scroll = (double)file->index.count - visible;
    }
    app->text_scroll_line = aklv_clamp_double(app->text_scroll_line, 0.0, max_scroll);
    if (app->text_scroll_col < 0.0) {
        app->text_scroll_col = 0.0;
    }

    AklvRect search = aklv_search_rect(win_w, win_h, app->search_panel_h);
    double result_rows = line_h > 0.0 ? floor((search.h - AKLV_SEARCH_HEADER_H) / line_h) : 1.0;
    double max_search_scroll = 0.0;
    if (app->search_results.count > (uint64_t)result_rows) {
        max_search_scroll = (double)app->search_results.count - result_rows;
    }
    app->search_scroll = aklv_clamp_double(app->search_scroll, 0.0, max_search_scroll);
}

static void aklv_app_center_line(AklvApp *app, uint64_t line_no, int win_w, int win_h) {
    AklvRect text = aklv_text_rect(win_w, win_h, app->search_panel_h);
    double line_h = aklv_metal_line_height(app->renderer);
    double visible = line_h > 0.0 ? floor(text.h / line_h) : 1.0;
    if (line_no == 0) {
        return;
    }
    app->selected_line = line_no;
    app->text_scroll_line = (double)(line_no - 1) - visible * 0.33;
    aklv_app_clamp_scroll(app, win_w, win_h);
}

static void aklv_app_select_highlight_at(AklvApp *app, uint64_t line_no, size_t byte_col) {
    AklvFile *file = aklv_app_active_file(app);
    app->highlight_len = 0;
    app->highlight_text[0] = '\0';
    if (file == NULL || line_no == 0 || line_no > file->index.count) {
        return;
    }
    AklvLineView line = aklv_file_line_fast(file, line_no);
    if (byte_col >= line.len) {
        return;
    }

    size_t start = byte_col;
    size_t end = byte_col + 1;
    unsigned char c = line.start[byte_col];
    if (aklv_token_byte(c)) {
        while (start > 0 && aklv_token_byte(line.start[start - 1])) {
            start--;
        }
        while (end < line.len && aklv_token_byte(line.start[end])) {
            end++;
        }
    } else if (c <= 0x20) {
        return;
    }

    size_t len = end - start;
    if (len >= sizeof(app->highlight_text)) {
        len = sizeof(app->highlight_text) - 1;
    }
    memcpy(app->highlight_text, line.start + start, len);
    app->highlight_text[len] = '\0';
    app->highlight_len = len;
    aklv_app_set_status(app, "highlight: %s", app->highlight_text);
}

static void aklv_app_text_pos_from_point(AklvApp *app,
                                         AklvRect rect,
                                         float x,
                                         float y,
                                         uint64_t *line_out,
                                         size_t *col_out) {
    AklvFile *file = aklv_app_active_file(app);
    float line_h = aklv_metal_line_height(app->renderer);
    float char_w = aklv_metal_char_width(app->renderer);
    uint64_t line = (uint64_t)floor(app->text_scroll_line + (double)((y - rect.y) / line_h)) + 1;
    if (file != NULL && file->index.count > 0) {
        if (line < 1) {
            line = 1;
        }
        if (line > file->index.count) {
            line = file->index.count;
        }
    }
    size_t col = (size_t)app->text_scroll_col;
    if (char_w > 0.0f && x > rect.x + AKLV_LINE_NO_W + 8.0f) {
        col += (size_t)floor((x - (rect.x + AKLV_LINE_NO_W + 8.0f)) / char_w + 0.5f);
    }
    if (file != NULL && line >= 1 && line <= file->index.count) {
        AklvLineView view = aklv_file_line_fast(file, line);
        if (col > view.len) {
            col = view.len;
        }
    }
    *line_out = line;
    *col_out = col;
}

static void aklv_app_open_dialog(AklvApp *app) {
    size_t count = 0;
    char **paths = aklv_platform_open_files(&count);
    for (size_t i = 0; i < count; i++) {
        if (paths[i] != NULL && !aklv_loader_enqueue(app->loader, paths[i])) {
            aklv_app_set_status(app, "failed to enqueue %s", paths[i]);
        }
    }
    aklv_platform_free_open_files(paths, count);
}

static void aklv_app_handle_loader(AklvApp *app) {
    AklvLoaderEvent event;
    while (aklv_loader_poll(app->loader, &event)) {
        if (event.status == 0 && event.file != NULL) {
            aklv_app_add_file(app, event.file);
        } else {
            aklv_app_set_status(app, "open failed: %s", event.error[0] ? event.error : event.path);
        }
        aklv_loader_event_destroy(&event);
    }
}

static size_t aklv_search_copy_text(char *dst, size_t dst_cap, const char *text) {
    if (dst_cap == 0) {
        return 0;
    }
    if (text == NULL) {
        text = "";
    }
    size_t len = strlen(text);
    if (len >= dst_cap) {
        len = dst_cap - 1;
    }
    memcpy(dst, text, len);
    dst[len] = '\0';
    return len;
}

static void aklv_app_search_reset_history_nav(AklvApp *app) {
    app->search_history_pos = -1;
    app->search_history_draft[0] = '\0';
    app->search_history_draft_cursor = 0;
}

static void aklv_app_search_push_undo(AklvApp *app) {
    size_t len = strlen(app->search_query);
    if (app->search_cursor > len) {
        app->search_cursor = len;
    }
    if (app->search_undo_count > 0) {
        uint32_t top = app->search_undo_count - 1;
        if (app->search_undo_cursor[top] == app->search_cursor &&
            strcmp(app->search_undo[top], app->search_query) == 0) {
            return;
        }
    }
    if (app->search_undo_count == AKLV_SEARCH_UNDO_DEPTH) {
        memmove(app->search_undo,
                app->search_undo + 1,
                sizeof(app->search_undo[0]) * (AKLV_SEARCH_UNDO_DEPTH - 1));
        memmove(app->search_undo_cursor,
                app->search_undo_cursor + 1,
                sizeof(app->search_undo_cursor[0]) * (AKLV_SEARCH_UNDO_DEPTH - 1));
        app->search_undo_count--;
    }
    uint32_t idx = app->search_undo_count++;
    memcpy(app->search_undo[idx], app->search_query, len + 1);
    app->search_undo_cursor[idx] = app->search_cursor;
}

static void aklv_app_set_search_text(AklvApp *app, const char *text, size_t cursor, bool select_all) {
    size_t len = aklv_search_copy_text(app->search_query, sizeof(app->search_query), text);
    if (cursor > len) {
        cursor = len;
    }
    app->search_cursor = cursor;
    app->search_select_all = select_all && len > 0;
}

static bool aklv_app_search_undo(AklvApp *app) {
    if (app->search_history_pos >= 0) {
        aklv_app_set_search_text(app,
                                 app->search_history_draft,
                                 app->search_history_draft_cursor,
                                 false);
        aklv_app_search_reset_history_nav(app);
        return true;
    }
    if (app->search_undo_count == 0) {
        return false;
    }
    uint32_t idx = --app->search_undo_count;
    aklv_app_set_search_text(app,
                             app->search_undo[idx],
                             app->search_undo_cursor[idx],
                             false);
    aklv_app_search_reset_history_nav(app);
    return true;
}

static void aklv_app_search_history_add(AklvApp *app, const char *text) {
    char entry[AKLV_SEARCH_QUERY_CAP];
    size_t len = aklv_search_copy_text(entry, sizeof(entry), text);
    if (len == 0) {
        return;
    }

    uint32_t out = 0;
    for (uint32_t i = 0; i < app->search_history_count; i++) {
        if (strcmp(app->search_history[i], entry) == 0) {
            continue;
        }
        if (out != i) {
            memcpy(app->search_history[out], app->search_history[i], sizeof(app->search_history[out]));
        }
        out++;
    }
    app->search_history_count = out;
    if (app->search_history_count == AKLV_SEARCH_HISTORY_CAP) {
        memmove(app->search_history,
                app->search_history + 1,
                sizeof(app->search_history[0]) * (AKLV_SEARCH_HISTORY_CAP - 1));
        app->search_history_count--;
    }
    memcpy(app->search_history[app->search_history_count], entry, len + 1);
    app->search_history_count++;
    aklv_app_search_reset_history_nav(app);
}

static bool aklv_app_search_history_move(AklvApp *app, int direction) {
    if (app->search_history_count == 0 || direction == 0) {
        return false;
    }
    if (direction < 0) {
        if (app->search_history_pos < 0) {
            size_t len = aklv_search_copy_text(app->search_history_draft,
                                               sizeof(app->search_history_draft),
                                               app->search_query);
            app->search_history_draft_cursor = app->search_cursor > len ? len : app->search_cursor;
            app->search_history_pos = (int)app->search_history_count - 1;
        } else if (app->search_history_pos > 0) {
            app->search_history_pos--;
        }
        aklv_app_set_search_text(app,
                                 app->search_history[app->search_history_pos],
                                 strlen(app->search_history[app->search_history_pos]),
                                 false);
        return true;
    }

    if (app->search_history_pos < 0) {
        return false;
    }
    if ((uint32_t)app->search_history_pos + 1 < app->search_history_count) {
        app->search_history_pos++;
        aklv_app_set_search_text(app,
                                 app->search_history[app->search_history_pos],
                                 strlen(app->search_history[app->search_history_pos]),
                                 false);
    } else {
        aklv_app_set_search_text(app,
                                 app->search_history_draft,
                                 app->search_history_draft_cursor,
                                 false);
        aklv_app_search_reset_history_nav(app);
    }
    return true;
}

static void aklv_app_submit_search(AklvApp *app) {
    if (app->search_query[0] == '\0' || app->file_count == 0) {
        return;
    }
    aklv_app_search_history_add(app, app->search_query);
    AklvFileSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.items = app->files;
    snapshot.count = app->file_count;
    snapshot.active_index = app->active_file;
    snapshot.active_selected_line = app->selected_line;
    aklv_search_service_submit(app->search_service, app->search_query, &snapshot);
    app->search_scroll = 0.0;
    app->has_selected_result = false;
    app->has_search_anchor = false;
    AklvFile *active = aklv_app_active_file(app);
    if (active != NULL) {
        app->search_anchor_file_id = active->id;
        app->search_anchor_line = app->selected_line;
        app->has_search_anchor = true;
    }
    aklv_app_set_status(app, "searching: %s", app->search_query);
}

static void aklv_app_focus_search(AklvApp *app) {
    app->search_input_focus = true;
    app->search_select_all = true;
    app->search_cursor = strlen(app->search_query);
    aklv_app_search_reset_history_nav(app);
}

static void aklv_app_replace_search_text(AklvApp *app, const char *text) {
    char next[AKLV_SEARCH_QUERY_CAP];
    size_t len = aklv_search_copy_text(next, sizeof(next), text);
    if (!app->search_select_all && app->search_cursor == len &&
        strcmp(app->search_query, next) == 0) {
        return;
    }
    aklv_app_search_push_undo(app);
    aklv_app_set_search_text(app, next, len, false);
    aklv_app_search_reset_history_nav(app);
}

static void aklv_app_append_search_text(AklvApp *app, const char *text) {
    if (app->search_select_all) {
        aklv_app_replace_search_text(app, text);
        return;
    }
    size_t len = strlen(app->search_query);
    size_t add = strlen(text);
    if (add == 0 || len >= sizeof(app->search_query) - 1) {
        return;
    }
    if (app->search_cursor > len) {
        app->search_cursor = len;
    }
    if (add > sizeof(app->search_query) - 1 - len) {
        add = sizeof(app->search_query) - 1 - len;
    }
    aklv_app_search_push_undo(app);
    memmove(app->search_query + app->search_cursor + add,
            app->search_query + app->search_cursor,
            len - app->search_cursor + 1);
    memcpy(app->search_query + app->search_cursor, text, add);
    app->search_cursor += add;
    app->search_select_all = false;
    aklv_app_search_reset_history_nav(app);
}

static bool aklv_app_select_next_result_after_active_line(AklvApp *app, int win_w, int win_h);

static void aklv_app_handle_search(AklvApp *app) {
    uint64_t old_gen = app->search_results.generation;
    uint64_t old_count = app->search_results.count;
    aklv_search_results_swap_from_service(app->search_service, &app->search_results);
    bool changed = app->search_results.generation != old_gen || app->search_results.count != old_count;
    if (changed && app->search_results.count > 0 && !app->has_selected_result) {
        int win_w = 0;
        int win_h = 0;
        float scale = 1.0f;
        aklv_get_window_points(app->window, &win_w, &win_h, &scale);
        if (aklv_app_select_next_result_after_active_line(app, win_w, win_h)) {
            app->search_input_focus = false;
            app->search_select_all = false;
        }
    }
    if (app->search_results.complete && changed) {
        aklv_app_set_status(app,
                            "search complete: %" PRIu64 " matches in %.3fs for '%s'",
                            app->search_results.count,
                            app->search_results.elapsed_sec,
                            app->search_results.query);
    }
}

static AklvFile *aklv_app_find_file_by_id(AklvApp *app, uint32_t file_id, uint32_t *index_out) {
    for (uint32_t i = 0; i < app->file_count; i++) {
        if (app->files[i]->id == file_id) {
            if (index_out != NULL) {
                *index_out = i;
            }
            return app->files[i];
        }
    }
    return NULL;
}

static void aklv_app_jump_to_result(AklvApp *app, uint64_t result_index, int win_w, int win_h) {
    if (result_index >= app->search_results.count) {
        return;
    }
    app->selected_result = result_index;
    app->has_selected_result = true;
    AklvSearchResult *result = &app->search_results.items[result_index];
    uint32_t file_index = 0;
    if (aklv_app_find_file_by_id(app, result->file_id, &file_index) == NULL) {
        aklv_app_set_status(app, "result file is closed");
        return;
    }
    aklv_app_switch_file(app, file_index);
    aklv_app_center_line(app, result->line_no, win_w, win_h);
}

static void aklv_app_ensure_result_visible(AklvApp *app, uint64_t result_index, int win_w, int win_h) {
    AklvRect search = aklv_search_rect(win_w, win_h, app->search_panel_h);
    double line_h = aklv_metal_line_height(app->renderer);
    double rows = line_h > 0.0 ? floor((search.h - AKLV_SEARCH_HEADER_H) / line_h) : 1.0;
    if (rows < 1.0) {
        rows = 1.0;
    }

    double top = floor(app->search_scroll);
    double bottom = top + rows - 1.0;
    if ((double)result_index < top) {
        app->search_scroll = (double)result_index;
    } else if ((double)result_index > bottom) {
        app->search_scroll = (double)result_index - rows + 1.0;
    }
    aklv_app_clamp_scroll(app, win_w, win_h);
}

static void aklv_app_select_result_without_jump(AklvApp *app, uint64_t result_index, int win_w, int win_h) {
    if (result_index >= app->search_results.count) {
        return;
    }
    app->selected_result = result_index;
    app->has_selected_result = true;
    aklv_app_ensure_result_visible(app, result_index, win_w, win_h);
}

static bool aklv_app_select_next_result_after_active_line(AklvApp *app, int win_w, int win_h) {
    if (app->search_results.count == 0) {
        return false;
    }

    AklvFile *file = aklv_app_active_file(app);
    uint32_t active_file_id = app->has_search_anchor
        ? app->search_anchor_file_id
        : (file != NULL ? file->id : 0);
    uint64_t anchor_line = app->has_search_anchor ? app->search_anchor_line : app->selected_line;
    uint64_t best = UINT64_MAX;
    for (uint64_t i = 0; i < app->search_results.count; i++) {
        AklvSearchResult *result = &app->search_results.items[i];
        if (result->file_id == active_file_id && result->line_no > anchor_line) {
            best = i;
            break;
        }
    }
    if (best == UINT64_MAX) {
        best = 0;
    }
    aklv_app_select_result_without_jump(app, best, win_w, win_h);
    return true;
}

static bool aklv_app_move_search_result(AklvApp *app, int delta, int win_w, int win_h) {
    if (app->search_results.count == 0) {
        return false;
    }

    uint64_t index = 0;
    if (app->has_selected_result && app->selected_result < app->search_results.count) {
        index = app->selected_result;
        if (delta > 0 && index + 1 < app->search_results.count) {
            index++;
        } else if (delta < 0 && index > 0) {
            index--;
        }
    }

    aklv_app_jump_to_result(app, index, win_w, win_h);
    aklv_app_ensure_result_visible(app, index, win_w, win_h);
    return true;
}

static float aklv_tab_width(AklvMetalRenderer *renderer, const char *name) {
    float cw = aklv_metal_char_width(renderer);
    size_t len = strlen(name);
    float width = 34.0f + (float)len * cw;
    if (width < 92.0f) {
        width = 92.0f;
    }
    if (width > 210.0f) {
        width = 210.0f;
    }
    return width;
}

static void aklv_draw_text_cstr(AklvMetalRenderer *renderer,
                                float x,
                                float y,
                                const char *text,
                                AklvColor color) {
    aklv_metal_push_text(renderer, x, y, text, strlen(text), color);
}

static void aklv_draw_nav(AklvApp *app, int win_w) {
    AklvMetalRenderer *r = app->renderer;
    aklv_metal_push_rect(r, 0.0f, 0.0f, (float)win_w, AKLV_NAV_H, C_PANEL);
    aklv_metal_push_rect(r, 0.0f, AKLV_NAV_H - 1.0f, (float)win_w, 1.0f, C_ACCENT);

    AklvRect open = {8.0f, 6.0f, 62.0f, 22.0f};
    aklv_metal_push_rect(r, open.x, open.y, open.w, open.h, C_TAB_ACTIVE);
    aklv_draw_text_cstr(r, open.x + 10.0f, open.y + 3.0f, "Open", C_TEXT);

    AklvRect close = {76.0f, 6.0f, 66.0f, 22.0f};
    aklv_metal_push_rect(r, close.x, close.y, close.w, close.h, app->file_count > 0 ? C_TAB : C_INPUT);
    aklv_draw_text_cstr(r, close.x + 10.0f, close.y + 3.0f, "Close", app->file_count > 0 ? C_TEXT : C_MUTED);

    float x = 150.0f;
    for (uint32_t i = 0; i < app->file_count && x < (float)win_w - 140.0f; i++) {
        AklvFile *file = app->files[i];
        float tw = aklv_tab_width(r, file->name);
        AklvColor tab = i == app->active_file ? C_TAB_ACTIVE : C_TAB;
        aklv_metal_push_rect(r, x, 5.0f, tw, 24.0f, tab);
        char label[128];
        snprintf(label, sizeof(label), "%s", file->name);
        aklv_draw_text_cstr(r, x + 10.0f, 8.0f, label, C_TEXT);
        aklv_draw_text_cstr(r, x + tw - 18.0f, 8.0f, "x", C_MUTED);
        x += tw + 4.0f;
    }

    if (app->status[0] != '\0') {
        float status_x = (float)win_w - 420.0f;
        if (status_x < x + 12.0f) {
            status_x = x + 12.0f;
        }
        if (status_x < (float)win_w - 20.0f) {
            aklv_metal_push_text(r, status_x, 9.0f, app->status, strlen(app->status), C_MUTED);
        }
    }
}

static void aklv_draw_scrollbar(AklvMetalRenderer *renderer,
                                float x,
                                float y,
                                float h,
                                double scroll,
                                double total,
                                double visible) {
    aklv_metal_push_rect(renderer, x, y, AKLV_SCROLLBAR_W, h, C_INPUT);
    if (total <= visible || total <= 0.0 || h <= 0.0) {
        return;
    }
    float thumb_h = (float)(h * (visible / total));
    if (thumb_h < 24.0f) {
        thumb_h = 24.0f;
    }
    if (thumb_h > h) {
        thumb_h = h;
    }
    double max_scroll = total - visible;
    float thumb_y = y;
    if (max_scroll > 0.0) {
        thumb_y = y + (float)((h - thumb_h) * (scroll / max_scroll));
    }
    aklv_metal_push_rect(renderer, x + 2.0f, thumb_y, AKLV_SCROLLBAR_W - 4.0f, thumb_h, C_ACCENT);
}

static void aklv_draw_text_view(AklvApp *app, AklvRect rect) {
    AklvMetalRenderer *r = app->renderer;
    AklvFile *file = aklv_app_active_file(app);
    float line_h = aklv_metal_line_height(r);
    float char_w = aklv_metal_char_width(r);
    aklv_metal_push_rect(r, rect.x, rect.y, rect.w, rect.h, C_BG);
    aklv_metal_push_rect(r, rect.x, rect.y, AKLV_LINE_NO_W, rect.h, C_PANEL);
    aklv_metal_push_rect(r, rect.x + AKLV_LINE_NO_W - 1.0f, rect.y, 1.0f, rect.h, C_PANEL_2);

    if (file == NULL) {
        aklv_draw_text_cstr(r, rect.x + 22.0f, rect.y + 30.0f, "Open or drop huge log files to start.", C_MUTED);
        return;
    }

    uint64_t first_line = (uint64_t)floor(app->text_scroll_line) + 1;
    double frac = app->text_scroll_line - floor(app->text_scroll_line);
    uint64_t visible = (uint64_t)(rect.h / line_h) + 3;
    size_t col = (size_t)app->text_scroll_col;
    size_t max_chars = (size_t)((rect.w - AKLV_LINE_NO_W - AKLV_SCROLLBAR_W - 18.0f) / char_w) + 2;
    const char *search_query = app->search_results.query;
    size_t search_query_len = strlen(search_query);
    unsigned char folded_search_query[AKLV_SEARCH_QUERY_CAP];
    if (search_query_len > sizeof(folded_search_query)) {
        search_query_len = sizeof(folded_search_query);
    }
    if (search_query_len > 0) {
        aklv_fold_ascii_pattern(folded_search_query, search_query, search_query_len);
    }
    bool has_text_selection = aklv_app_text_selection_non_empty(app);
    uint64_t sel_start_line = 0;
    uint64_t sel_end_line = 0;
    size_t sel_start_col = 0;
    size_t sel_end_col = 0;
    if (has_text_selection) {
        aklv_app_normalized_text_selection(app,
                                           &sel_start_line,
                                           &sel_start_col,
                                           &sel_end_line,
                                           &sel_end_col);
    }
    char number[32];

    for (uint64_t row = 0; row < visible; row++) {
        uint64_t line_no = first_line + row;
        if (line_no == 0 || line_no > file->index.count) {
            break;
        }
        float y = rect.y + (float)row * line_h - (float)(frac * line_h);
        if (y > rect.y + rect.h) {
            break;
        }
        if (line_no == app->selected_line) {
            aklv_metal_push_rect(r, rect.x, y, rect.w - AKLV_SCROLLBAR_W, line_h, C_SELECT);
        }
        int number_len = snprintf(number, sizeof(number), "%8" PRIu64, line_no);
        size_t number_size = number_len > 0 && (size_t)number_len < sizeof(number)
            ? (size_t)number_len
            : sizeof(number) - 1;
        aklv_metal_push_text(r, rect.x + 8.0f, y + 1.0f, number, number_size, C_LINE_NO);

        AklvLineView line = aklv_file_line_fast(file, line_no);
        if (has_text_selection) {
            if (line_no >= sel_start_line && line_no <= sel_end_line) {
                size_t begin = line_no == sel_start_line ? sel_start_col : 0;
                size_t end = line_no == sel_end_line ? sel_end_col : line.len;
                if (begin > line.len) {
                    begin = line.len;
                }
                if (end > line.len) {
                    end = line.len;
                }
                if (end > begin) {
                    size_t visible_begin = begin < col ? col : begin;
                    size_t visible_end = end;
                    size_t visible_limit = col + max_chars;
                    if (visible_end > visible_limit) {
                        visible_end = visible_limit;
                    }
                    if (visible_end > visible_begin) {
                        aklv_metal_push_rect(r,
                                             rect.x + AKLV_LINE_NO_W + 8.0f + (float)(visible_begin - col) * char_w,
                                             y,
                                             (float)(visible_end - visible_begin) * char_w,
                                             line_h,
                                             C_SELECT);
                    }
                }
            }
        }
        if (line.start != NULL && col < line.len) {
            size_t len = line.len - col;
            if (len > max_chars) {
                len = max_chars;
            }
            if (search_query_len > 0) {
                aklv_draw_match_highlights(r,
                                           rect.x + AKLV_LINE_NO_W + 8.0f,
                                           y,
                                           line_h,
                                           char_w,
                                           line.start + col,
                                           len,
                                           folded_search_query,
                                           search_query_len,
                                           C_SEARCH_HIGHLIGHT);
            }
            if (app->highlight_len > 0 && len >= app->highlight_len) {
                const unsigned char *scan = line.start + col;
                size_t remaining = len;
                while (remaining >= app->highlight_len) {
                    const unsigned char *hit = aklv_memmem_simple(scan,
                                                                  remaining,
                                                                  app->highlight_text,
                                                                  app->highlight_len);
                    if (hit == NULL) {
                        break;
                    }
                    size_t hit_col = (size_t)(hit - (line.start + col));
                    size_t hit_offset = (size_t)(hit - scan);
                    if (hit_offset > remaining) {
                        break;
                    }
                    aklv_metal_push_rect(r,
                                         rect.x + AKLV_LINE_NO_W + 8.0f + (float)hit_col * char_w,
                                         y,
                                         (float)app->highlight_len * char_w,
                                         line_h,
                                         C_WORD_HIGHLIGHT);
                    size_t consumed = hit_offset + app->highlight_len;
                    if (consumed == 0 || consumed > remaining) {
                        break;
                    }
                    scan += consumed;
                    remaining -= consumed;
                }
            }
            aklv_metal_push_text(r,
                                 rect.x + AKLV_LINE_NO_W + 8.0f,
                                 y + 1.0f,
                                 (const char *)line.start + col,
                                 len,
                                 C_TEXT);
        }
        if (app->has_text_cursor && line_no == app->text_cursor_line) {
            size_t cursor_col = app->text_cursor_col;
            if (cursor_col >= col && cursor_col <= col + max_chars) {
                aklv_metal_push_rect(r,
                                     rect.x + AKLV_LINE_NO_W + 8.0f + (float)(cursor_col - col) * char_w,
                                     y + 1.0f,
                                     1.0f,
                                     line_h - 2.0f,
                                     C_ACCENT);
            }
        }
    }

    double visible_lines = floor(rect.h / line_h);
    aklv_draw_scrollbar(r,
                        rect.x + rect.w - AKLV_SCROLLBAR_W,
                        rect.y,
                        rect.h,
                        app->text_scroll_line,
                        (double)file->index.count,
                        visible_lines);
}

static void aklv_draw_search_panel(AklvApp *app, AklvRect rect) {
    AklvMetalRenderer *r = app->renderer;
    float line_h = aklv_metal_line_height(r);
    float char_w = aklv_metal_char_width(r);
    aklv_metal_push_rect(r, rect.x, rect.y, rect.w, rect.h, C_PANEL);
    aklv_metal_push_rect(r, rect.x, rect.y, rect.w, 1.0f, C_ACCENT);

    AklvRect input = {rect.x + 76.0f, rect.y + 6.0f, rect.w - 340.0f, 22.0f};
    if (input.w < 160.0f) {
        input.w = 160.0f;
    }
    aklv_draw_text_cstr(r, rect.x + 12.0f, rect.y + 9.0f, "Search", C_TEXT);
    aklv_metal_push_rect(r, input.x, input.y, input.w, input.h, C_INPUT);
    aklv_metal_push_rect(r, input.x, input.y, input.w, 1.0f, app->search_input_focus ? C_ACCENT : C_PANEL_2);
    aklv_metal_push_rect(r, input.x, input.y + input.h - 1.0f, input.w, 1.0f, app->search_input_focus ? C_ACCENT : C_PANEL_2);
    size_t search_len = strlen(app->search_query);
    if (app->search_cursor > search_len) {
        app->search_cursor = search_len;
    }
    if (app->search_input_focus && app->search_select_all && search_len > 0) {
        float selected_w = (float)search_len * char_w;
        if (selected_w > input.w - 16.0f) {
            selected_w = input.w - 16.0f;
        }
        aklv_metal_push_rect(r, input.x + 8.0f, input.y + 3.0f, selected_w, input.h - 6.0f, C_SELECT);
    }
    aklv_metal_push_text(r, input.x + 8.0f, input.y + 3.0f, app->search_query, search_len, C_TEXT);
    if (app->search_input_focus) {
        float caret = input.x + 8.0f + (float)app->search_cursor * char_w;
        aklv_metal_push_rect(r, caret, input.y + 3.0f, 1.0f, input.h - 6.0f, C_ACCENT);
    }

    char summary[128];
    if (app->search_results.running) {
        snprintf(summary, sizeof(summary), "%" PRIu64 " matches, searching...", app->search_results.count);
    } else {
        snprintf(summary, sizeof(summary), "%" PRIu64 " matches", app->search_results.count);
    }
    aklv_draw_text_cstr(r, input.x + input.w + 14.0f, rect.y + 9.0f, summary, C_MUTED);
    if (app->search_results.complete && app->search_results.generation != 0) {
        char elapsed[64];
        int elapsed_len = snprintf(elapsed, sizeof(elapsed), "%.3fs", app->search_results.elapsed_sec);
        size_t elapsed_size = elapsed_len > 0 && (size_t)elapsed_len < sizeof(elapsed)
            ? (size_t)elapsed_len
            : sizeof(elapsed) - 1;
        float elapsed_w = (float)elapsed_size * char_w;
        aklv_metal_push_text(r,
                             rect.x + rect.w - AKLV_SCROLLBAR_W - elapsed_w - 12.0f,
                             rect.y + 9.0f,
                             elapsed,
                             elapsed_size,
                             C_ACCENT);
    }

    const char *query = app->search_results.query;
    size_t query_len = strlen(query);
    unsigned char folded_query[AKLV_SEARCH_QUERY_CAP];
    if (query_len > sizeof(folded_query)) {
        query_len = sizeof(folded_query);
    }
    if (query_len > 0) {
        aklv_fold_ascii_pattern(folded_query, query, query_len);
    }
    float list_y = rect.y + AKLV_SEARCH_HEADER_H;
    float list_h = rect.h - AKLV_SEARCH_HEADER_H;
    aklv_metal_push_rect(r, rect.x, list_y, rect.w, list_h, C_BG);

    uint64_t first = (uint64_t)floor(app->search_scroll);
    double frac = app->search_scroll - floor(app->search_scroll);
    uint64_t visible = (uint64_t)(list_h / line_h) + 2;
    uint32_t cached_file_id = 0;
    AklvFile *cached_result_file = NULL;
    bool cached_file_valid = false;
    for (uint64_t row = 0; row < visible; row++) {
        uint64_t index = first + row;
        if (index >= app->search_results.count) {
            break;
        }
        float y = list_y + (float)row * line_h - (float)(frac * line_h);
        if (y > rect.y + rect.h) {
            break;
        }
        if (app->has_selected_result && index == app->selected_result) {
            aklv_metal_push_rect(r, rect.x, y, rect.w - AKLV_SCROLLBAR_W, line_h, C_SELECT);
        } else if ((index & 1U) != 0) {
            aklv_metal_push_rect(r, rect.x, y, rect.w - AKLV_SCROLLBAR_W, line_h, (AklvColor){0.118f, 0.132f, 0.160f, 1.0f});
        }
        AklvSearchResult *item = &app->search_results.items[index];
        AklvFile *result_file = NULL;
        if (cached_file_valid && cached_file_id == item->file_id) {
            result_file = cached_result_file;
        } else {
            result_file = aklv_app_find_file_by_id(app, item->file_id, NULL);
            cached_file_id = item->file_id;
            cached_result_file = result_file;
            cached_file_valid = true;
        }
        const char *file_name = result_file != NULL ? result_file->name : "(closed)";
        char snippet[AKLV_SEARCH_SNIPPET_LIMIT + 4];
        size_t snippet_len = 0;
        bool snippet_truncated = false;
        if (result_file != NULL) {
            AklvLineView line = aklv_file_line_fast(result_file, item->line_no);
            snippet_len = aklv_copy_line_snippet(line, snippet, sizeof(snippet), &snippet_truncated);
            if (snippet_truncated && snippet_len + 3 < sizeof(snippet)) {
                snippet[snippet_len++] = '.';
                snippet[snippet_len++] = '.';
                snippet[snippet_len++] = '.';
                snippet[snippet_len] = '\0';
            }
        } else {
            snippet[0] = '\0';
        }
        char prefix[512];
        int prefix_result = snprintf(prefix,
                                     sizeof(prefix),
                                     "#%-6" PRIu64 " file:%s line:%" PRIu64 "  ",
                                     index + 1,
                                     file_name,
                                     item->line_no);
        size_t prefix_len = prefix_result > 0 && (size_t)prefix_result < sizeof(prefix)
            ? (size_t)prefix_result
            : sizeof(prefix) - 1;
        aklv_metal_push_text(r, rect.x + 10.0f, y + 1.0f, prefix, prefix_len, C_LINE_NO);
        float text_x = rect.x + 10.0f + (float)prefix_len * char_w;
        if (query_len > 0) {
            aklv_draw_match_highlights(r,
                                       text_x,
                                       y,
                                       line_h,
                                       char_w,
                                       (const unsigned char *)snippet,
                                       snippet_len,
                                       folded_query,
                                       query_len,
                                       C_SEARCH_HIGHLIGHT);
        }
        aklv_metal_push_text(r, text_x, y + 1.0f, snippet, snippet_len, C_TEXT);
    }

    double visible_rows = floor(list_h / line_h);
    aklv_draw_scrollbar(r,
                        rect.x + rect.w - AKLV_SCROLLBAR_W,
                        list_y,
                        list_h,
                        app->search_scroll,
                        (double)app->search_results.count,
                        visible_rows);
}

static void aklv_app_render(AklvApp *app) {
    int win_w = 0;
    int win_h = 0;
    float scale = 1.0f;
    aklv_get_window_points(app->window, &win_w, &win_h, &scale);
    int pixel_w = 0;
    int pixel_h = 0;
    SDL_GetWindowSizeInPixels(app->window, &pixel_w, &pixel_h);
    if (pixel_w <= 0 || pixel_h <= 0) {
        return;
    }

    float min_search = 90.0f;
    float max_search = (float)win_h - AKLV_NAV_H - 80.0f;
    if (app->search_panel_h < min_search) {
        app->search_panel_h = min_search;
    }
    if (app->search_panel_h > max_search) {
        app->search_panel_h = max_search;
    }
    aklv_app_clamp_scroll(app, win_w, win_h);

    if (!aklv_metal_begin(app->renderer, pixel_w, pixel_h, scale, C_BG)) {
        return;
    }
    aklv_draw_nav(app, win_w);
    AklvRect text = aklv_text_rect(win_w, win_h, app->search_panel_h);
    AklvRect split = aklv_splitter_rect(win_w, win_h, app->search_panel_h);
    AklvRect search = aklv_search_rect(win_w, win_h, app->search_panel_h);
    aklv_draw_text_view(app, text);
    aklv_metal_push_rect(app->renderer, split.x, split.y, split.w, split.h, app->resizing_search ? C_ACCENT : C_PANEL_2);
    aklv_draw_search_panel(app, search);
    aklv_metal_end(app->renderer);
}

static void aklv_app_nav_click(AklvApp *app, float x, float y) {
    if (y > AKLV_NAV_H) {
        return;
    }
    if (x >= 8.0f && x < 70.0f && y >= 6.0f && y < 28.0f) {
        aklv_app_open_dialog(app);
        return;
    }
    if (x >= 76.0f && x < 142.0f && y >= 6.0f && y < 28.0f) {
        if (app->file_count > 0) {
            aklv_app_close_file(app, app->active_file);
        }
        return;
    }
    float tab_x = 150.0f;
    for (uint32_t i = 0; i < app->file_count; i++) {
        float tw = aklv_tab_width(app->renderer, app->files[i]->name);
        if (x >= tab_x && x < tab_x + tw && y >= 5.0f && y < 29.0f) {
            if (x > tab_x + tw - 24.0f) {
                aklv_app_close_file(app, i);
            } else {
                aklv_app_switch_file(app, i);
            }
            return;
        }
        tab_x += tw + 4.0f;
    }
}

static void aklv_app_handle_mouse_down(AklvApp *app, SDL_Event *event) {
    int win_w = 0;
    int win_h = 0;
    float scale = 1.0f;
    aklv_get_window_points(app->window, &win_w, &win_h, &scale);
    float x = event->button.x;
    float y = event->button.y;
    app->mouse_down = true;

    if (y < AKLV_NAV_H) {
        aklv_app_nav_click(app, x, y);
        return;
    }

    AklvRect split = aklv_splitter_rect(win_w, win_h, app->search_panel_h);
    AklvRect text = aklv_text_rect(win_w, win_h, app->search_panel_h);
    AklvRect search = aklv_search_rect(win_w, win_h, app->search_panel_h);
    if (aklv_rect_contains(split, x, y)) {
        app->resizing_search = true;
        app->drag_grab_offset = y - split.y;
        return;
    }

    if (aklv_rect_contains(text, x, y)) {
        app->search_input_focus = false;
        if (x >= text.x + text.w - AKLV_SCROLLBAR_W) {
            app->dragging_text_scroll = true;
            return;
        }
        AklvFile *file = aklv_app_active_file(app);
        if (file != NULL) {
            float line_h = aklv_metal_line_height(app->renderer);
            uint64_t line = (uint64_t)floor(app->text_scroll_line + (double)((y - text.y) / line_h)) + 1;
            if (line >= 1 && line <= file->index.count) {
                app->selected_line = line;
                if (event->button.clicks >= 2 && x >= text.x + AKLV_LINE_NO_W + 8.0f) {
                    float char_w = aklv_metal_char_width(app->renderer);
                    size_t byte_col = (size_t)app->text_scroll_col +
                                      (size_t)floor((x - (text.x + AKLV_LINE_NO_W + 8.0f)) / char_w);
                    aklv_app_select_highlight_at(app, line, byte_col);
                    app->has_text_selection = false;
                    app->dragging_text_select = false;
                } else if (x >= text.x + AKLV_LINE_NO_W + 8.0f) {
                    uint64_t cursor_line = 0;
                    size_t cursor_col = 0;
                    aklv_app_text_pos_from_point(app, text, x, y, &cursor_line, &cursor_col);
                    app->text_cursor_line = cursor_line;
                    app->text_cursor_col = cursor_col;
                    app->text_select_anchor_line = cursor_line;
                    app->text_select_anchor_col = cursor_col;
                    app->text_select_focus_line = cursor_line;
                    app->text_select_focus_col = cursor_col;
                    app->has_text_cursor = true;
                    app->has_text_selection = true;
                    app->dragging_text_select = true;
                    app->highlight_len = 0;
                    app->highlight_text[0] = '\0';
                }
            }
        }
        return;
    }

    if (aklv_rect_contains(search, x, y)) {
        AklvRect input = {search.x + 76.0f, search.y + 6.0f, search.w - 340.0f, 22.0f};
        if (input.w < 160.0f) {
            input.w = 160.0f;
        }
        if (aklv_rect_contains(input, x, y)) {
            app->search_input_focus = true;
            app->search_select_all = false;
            float char_w = aklv_metal_char_width(app->renderer);
            size_t len = strlen(app->search_query);
            size_t cursor = 0;
            if (char_w > 0.0f && x > input.x + 8.0f) {
                cursor = (size_t)floor((x - (input.x + 8.0f)) / char_w + 0.5f);
                if (cursor > len) {
                    cursor = len;
                }
            }
            app->search_cursor = cursor;
            return;
        }
        app->search_input_focus = false;
        if (x >= search.x + search.w - AKLV_SCROLLBAR_W && y >= search.y + AKLV_SEARCH_HEADER_H) {
            app->dragging_search_scroll = true;
            return;
        }
        if (y >= search.y + AKLV_SEARCH_HEADER_H) {
            float line_h = aklv_metal_line_height(app->renderer);
            uint64_t result = (uint64_t)floor(app->search_scroll + (double)((y - search.y - AKLV_SEARCH_HEADER_H) / line_h));
            aklv_app_jump_to_result(app, result, win_w, win_h);
        }
    }
}

static void aklv_app_handle_mouse_motion(AklvApp *app, SDL_Event *event) {
    int win_w = 0;
    int win_h = 0;
    float scale = 1.0f;
    aklv_get_window_points(app->window, &win_w, &win_h, &scale);
    float x = event->motion.x;
    float y = event->motion.y;
    AklvRect text = aklv_text_rect(win_w, win_h, app->search_panel_h);
    AklvRect search = aklv_search_rect(win_w, win_h, app->search_panel_h);

    if (app->resizing_search) {
        app->search_panel_h = (float)win_h - y + app->drag_grab_offset;
        aklv_app_clamp_scroll(app, win_w, win_h);
        return;
    }
    if (app->dragging_text_scroll) {
        AklvFile *file = aklv_app_active_file(app);
        if (file != NULL && file->index.count > 0) {
            float line_h = aklv_metal_line_height(app->renderer);
            double visible = floor(text.h / line_h);
            double total = (double)file->index.count;
            double max_scroll = total > visible ? total - visible : 0.0;
            double t = (y - text.y) / text.h;
            app->text_scroll_line = aklv_clamp_double(t * max_scroll, 0.0, max_scroll);
        }
        return;
    }
    if (app->dragging_text_select) {
        uint64_t cursor_line = 0;
        size_t cursor_col = 0;
        aklv_app_text_pos_from_point(app, text, x, y, &cursor_line, &cursor_col);
        app->text_cursor_line = cursor_line;
        app->text_cursor_col = cursor_col;
        app->text_select_focus_line = cursor_line;
        app->text_select_focus_col = cursor_col;
        app->has_text_cursor = true;
        app->has_text_selection = true;

        if (y < text.y) {
            app->text_scroll_line -= 1.0;
        } else if (y >= text.y + text.h) {
            app->text_scroll_line += 1.0;
        }
        if (x < text.x + AKLV_LINE_NO_W + 8.0f) {
            app->text_scroll_col -= 1.0;
        } else if (x >= text.x + text.w - AKLV_SCROLLBAR_W) {
            app->text_scroll_col += 1.0;
        }
        aklv_app_clamp_scroll(app, win_w, win_h);
        return;
    }
    if (app->dragging_search_scroll) {
        float list_h = search.h - AKLV_SEARCH_HEADER_H;
        float line_h = aklv_metal_line_height(app->renderer);
        double visible = floor(list_h / line_h);
        double total = (double)app->search_results.count;
        double max_scroll = total > visible ? total - visible : 0.0;
        double t = (y - search.y - AKLV_SEARCH_HEADER_H) / list_h;
        app->search_scroll = aklv_clamp_double(t * max_scroll, 0.0, max_scroll);
    }
    (void)x;
}

static void aklv_app_handle_wheel(AklvApp *app, SDL_Event *event) {
    int win_w = 0;
    int win_h = 0;
    float scale = 1.0f;
    aklv_get_window_points(app->window, &win_w, &win_h, &scale);
    float mx = 0.0f;
    float my = 0.0f;
    SDL_GetMouseState(&mx, &my);
    AklvRect search = aklv_search_rect(win_w, win_h, app->search_panel_h);
    double dy = event->wheel.y;
    double dx = event->wheel.x;

    if (aklv_rect_contains(search, mx, my)) {
        app->search_scroll -= dy * 4.0;
    } else {
        if ((SDL_GetModState() & SDL_KMOD_SHIFT) != 0) {
            app->text_scroll_col -= dy * 8.0;
        } else {
            app->text_scroll_line -= dy * 4.0;
            app->text_scroll_col -= dx * 8.0;
        }
    }
    aklv_app_clamp_scroll(app, win_w, win_h);
}

static void aklv_app_handle_key(AklvApp *app, SDL_Event *event) {
    SDL_Keycode key = event->key.key;
    SDL_Keymod mod = event->key.mod;
    bool accel = (mod & (SDL_KMOD_CTRL | SDL_KMOD_GUI)) != 0;
    int win_w = 0;
    int win_h = 0;
    float scale = 1.0f;
    aklv_get_window_points(app->window, &win_w, &win_h, &scale);

    if (accel && key == SDLK_O) {
        aklv_app_open_dialog(app);
        return;
    }
    if (accel && key == SDLK_F) {
        aklv_app_focus_search(app);
        return;
    }

    if (app->search_input_focus) {
        size_t len = strlen(app->search_query);
        if (app->search_cursor > len) {
            app->search_cursor = len;
        }
        if (accel && key == SDLK_Z) {
            aklv_app_search_undo(app);
        } else if (accel && key == SDLK_A) {
            app->search_select_all = true;
            app->search_cursor = len;
        } else if (accel && key == SDLK_C) {
            SDL_SetClipboardText(app->search_query);
        } else if (accel && key == SDLK_X) {
            SDL_SetClipboardText(app->search_query);
            aklv_app_replace_search_text(app, "");
        } else if (accel && key == SDLK_V) {
            char *clip = SDL_GetClipboardText();
            if (clip != NULL) {
                if (app->search_select_all) {
                    aklv_app_replace_search_text(app, clip);
                } else {
                    aklv_app_append_search_text(app, clip);
                }
                SDL_free(clip);
            }
        } else if (key == SDLK_BACKSPACE && len > 0) {
            if (app->search_select_all) {
                aklv_app_search_push_undo(app);
                app->search_query[0] = '\0';
                app->search_cursor = 0;
                app->search_select_all = false;
                aklv_app_search_reset_history_nav(app);
            } else if (app->search_cursor > 0) {
                aklv_app_search_push_undo(app);
                memmove(app->search_query + app->search_cursor - 1,
                        app->search_query + app->search_cursor,
                        len - app->search_cursor + 1);
                app->search_cursor--;
                aklv_app_search_reset_history_nav(app);
            }
        } else if (key == SDLK_DELETE && len > 0) {
            if (app->search_select_all) {
                aklv_app_search_push_undo(app);
                app->search_query[0] = '\0';
                app->search_cursor = 0;
                app->search_select_all = false;
                aklv_app_search_reset_history_nav(app);
            } else if (app->search_cursor < len) {
                aklv_app_search_push_undo(app);
                memmove(app->search_query + app->search_cursor,
                        app->search_query + app->search_cursor + 1,
                        len - app->search_cursor);
                aklv_app_search_reset_history_nav(app);
            }
        } else if (key == SDLK_UP) {
            aklv_app_search_history_move(app, -1);
        } else if (key == SDLK_DOWN) {
            aklv_app_search_history_move(app, 1);
        } else if (key == SDLK_LEFT) {
            app->search_select_all = false;
            if (app->search_cursor > 0) {
                app->search_cursor--;
            }
        } else if (key == SDLK_RIGHT) {
            app->search_select_all = false;
            if (app->search_cursor < len) {
                app->search_cursor++;
            }
        } else if (key == SDLK_HOME) {
            app->search_select_all = false;
            app->search_cursor = 0;
        } else if (key == SDLK_END) {
            app->search_select_all = false;
            app->search_cursor = len;
        } else if (key == SDLK_RETURN || key == SDLK_KP_ENTER) {
            aklv_app_submit_search(app);
        } else if (key == SDLK_ESCAPE) {
            app->search_input_focus = false;
        }
        return;
    }

    if (key == SDLK_DOWN && aklv_app_move_search_result(app, 1, win_w, win_h)) {
        return;
    }
    if (key == SDLK_UP && aklv_app_move_search_result(app, -1, win_w, win_h)) {
        return;
    }

    if (accel && key == SDLK_C) {
        aklv_app_copy_text_selection(app);
        return;
    }

    AklvFile *file = aklv_app_active_file(app);
    if (file == NULL) {
        return;
    }
    AklvRect text = aklv_text_rect(win_w, win_h, app->search_panel_h);
    double visible = floor(text.h / aklv_metal_line_height(app->renderer));
    if (key == SDLK_DOWN && app->selected_line < file->index.count) {
        app->selected_line++;
        if ((double)app->selected_line > app->text_scroll_line + visible - 1.0) {
            app->text_scroll_line++;
        }
    } else if (key == SDLK_UP && app->selected_line > 1) {
        app->selected_line--;
        if ((double)app->selected_line <= app->text_scroll_line) {
            app->text_scroll_line--;
        }
    } else if (key == SDLK_PAGEDOWN) {
        uint64_t delta = (uint64_t)(visible > 1.0 ? visible - 1.0 : 1.0);
        app->selected_line = app->selected_line + delta > file->index.count ? file->index.count : app->selected_line + delta;
        app->text_scroll_line += (double)delta;
    } else if (key == SDLK_PAGEUP) {
        uint64_t delta = (uint64_t)(visible > 1.0 ? visible - 1.0 : 1.0);
        app->selected_line = app->selected_line > delta ? app->selected_line - delta : 1;
        app->text_scroll_line -= (double)delta;
    } else if (key == SDLK_HOME) {
        app->selected_line = 1;
        app->text_scroll_line = 0.0;
    } else if (key == SDLK_END) {
        app->selected_line = file->index.count;
        app->text_scroll_line = (double)file->index.count - visible;
    }
    aklv_app_clamp_scroll(app, win_w, win_h);
}

static void aklv_app_handle_text_input(AklvApp *app, SDL_Event *event) {
    if (!app->search_input_focus) {
        return;
    }
    const char *text = event->text.text;
    size_t add = strlen(text);
    if (add == 0) {
        return;
    }
    aklv_app_append_search_text(app, text);
}

static void aklv_app_handle_drop(AklvApp *app, SDL_Event *event) {
    if (event->drop.data != NULL) {
        aklv_loader_enqueue(app->loader, event->drop.data);
    }
}

static void aklv_app_process_events(AklvApp *app) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_EVENT_QUIT:
                app->quit = true;
                break;
            case SDL_EVENT_DROP_FILE:
                aklv_app_handle_drop(app, &event);
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    aklv_app_handle_mouse_down(app, &event);
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    if (app->dragging_text_select && !aklv_app_text_selection_non_empty(app)) {
                        app->has_text_selection = false;
                    }
                    app->mouse_down = false;
                    app->resizing_search = false;
                    app->dragging_text_scroll = false;
                    app->dragging_search_scroll = false;
                    app->dragging_text_select = false;
                }
                break;
            case SDL_EVENT_MOUSE_MOTION:
                aklv_app_handle_mouse_motion(app, &event);
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                aklv_app_handle_wheel(app, &event);
                break;
            case SDL_EVENT_KEY_DOWN:
                aklv_app_handle_key(app, &event);
                break;
            case SDL_EVENT_TEXT_INPUT:
                aklv_app_handle_text_input(app, &event);
                break;
            default:
                break;
        }
    }
}

static bool aklv_app_init(AklvApp *app, int argc, char **argv) {
    (void)argc;
    (void)argv;
    memset(app, 0, sizeof(*app));
    app->search_panel_h = 220.0f;
    app->active_file = 0;
    app->search_history_pos = -1;
    app->loader = aklv_loader_create(0);
    app->search_service = aklv_search_service_create();
    aklv_search_results_init(&app->search_results);
    if (app->loader == NULL || app->search_service == NULL) {
        fprintf(stderr, "failed to create background services\n");
        return false;
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }

    app->window = SDL_CreateWindow("AK Log Viewer",
                                   1280,
                                   820,
                                   SDL_WINDOW_METAL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (app->window == NULL) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }
    SDL_StartTextInput(app->window);

    char error[512] = {0};
    if (!aklv_metal_create(app->window, &app->renderer, error, sizeof(error))) {
        fprintf(stderr, "Metal renderer init failed: %s\n", error);
        return false;
    }

    aklv_app_set_status(app, "Click Open or press Ctrl+O to choose files");
    return true;
}

int main(int argc, char **argv) {
    AklvApp app;
    if (!aklv_app_init(&app, argc, argv)) {
        aklv_app_destroy(&app);
        SDL_Quit();
        return 1;
    }

    while (!app.quit) {
        aklv_app_process_events(&app);
        aklv_app_handle_loader(&app);
        aklv_app_handle_search(&app);
        aklv_app_render(&app);
        SDL_Delay(1);
    }

    aklv_app_destroy(&app);
    SDL_Quit();
    return 0;
}
