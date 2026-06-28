#ifndef AKLV_METAL_H
#define AKLV_METAL_H

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct AklvMetalRenderer AklvMetalRenderer;

typedef struct {
    float r;
    float g;
    float b;
    float a;
} AklvColor;

bool aklv_metal_create(SDL_Window *window, AklvMetalRenderer **out, char *error, size_t error_cap);
void aklv_metal_destroy(AklvMetalRenderer *renderer);

bool aklv_metal_begin(AklvMetalRenderer *renderer,
                      int drawable_width,
                      int drawable_height,
                      float ui_scale,
                      AklvColor clear_color);
void aklv_metal_end(AklvMetalRenderer *renderer);

void aklv_metal_push_rect(AklvMetalRenderer *renderer,
                          float x,
                          float y,
                          float w,
                          float h,
                          AklvColor color);
void aklv_metal_push_line(AklvMetalRenderer *renderer,
                          float x0,
                          float y0,
                          float x1,
                          float y1,
                          float thickness,
                          AklvColor color);
void aklv_metal_push_text(AklvMetalRenderer *renderer,
                          float x,
                          float y,
                          const char *text,
                          size_t len,
                          AklvColor color);

float aklv_metal_char_width(const AklvMetalRenderer *renderer);
float aklv_metal_line_height(const AklvMetalRenderer *renderer);

#endif
