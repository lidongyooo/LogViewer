#include "aklv_metal.h"

#include <SDL3/SDL_metal.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <simd/simd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    vector_float2 position;
    vector_float2 uv;
    vector_float4 color;
} AklvVertex;

typedef struct {
    float u0;
    float v0;
    float u1;
    float v1;
    float advance;
    float w;
    float h;
    float draw_x;
    float draw_y;
} AklvGlyph;

struct AklvMetalRenderer {
    SDL_Window *window;
    SDL_MetalView metal_view;
    CAMetalLayer *layer;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLRenderPipelineState> pipeline;
    id<MTLSamplerState> sampler;
    id<MTLTexture> atlas;
    id<CAMetalDrawable> drawable;
    id<MTLCommandBuffer> command_buffer;
    id<MTLRenderCommandEncoder> encoder;
    id<MTLBuffer> vertex_buffer;
    AklvGlyph glyphs[128];
    float atlas_scale;
    float char_width;
    float line_height;
    int atlas_width;
    int atlas_height;
    int drawable_width;
    int drawable_height;
    float ui_scale;
    AklvVertex *vertices;
    size_t vertex_count;
    size_t vertex_capacity;
};

static void aklv_set_error(char *error, size_t error_cap, const char *text) {
    if (error != NULL && error_cap > 0) {
        snprintf(error, error_cap, "%s", text == NULL ? "" : text);
    }
}

static bool aklv_metal_reserve(AklvMetalRenderer *renderer, size_t extra) {
    if (renderer->vertex_count + extra <= renderer->vertex_capacity) {
        return true;
    }
    size_t need = renderer->vertex_count + extra;
    size_t new_capacity = renderer->vertex_capacity == 0 ? 65536 : renderer->vertex_capacity;
    while (new_capacity < need) {
        if (new_capacity > (SIZE_MAX / 2)) {
            return false;
        }
        new_capacity *= 2;
    }
    AklvVertex *new_vertices = realloc(renderer->vertices, new_capacity * sizeof(*renderer->vertices));
    if (new_vertices == NULL) {
        return false;
    }
    renderer->vertices = new_vertices;
    renderer->vertex_capacity = new_capacity;
    return true;
}

static vector_float4 aklv_color_vec(AklvColor c) {
    return (vector_float4){c.r, c.g, c.b, c.a};
}

static void aklv_push_quad_uv(AklvMetalRenderer *renderer,
                              float x,
                              float y,
                              float w,
                              float h,
                              float u0,
                              float v0,
                              float u1,
                              float v1,
                              AklvColor color) {
    if (w <= 0.0f || h <= 0.0f || !aklv_metal_reserve(renderer, 6)) {
        return;
    }
    vector_float4 c = aklv_color_vec(color);
    AklvVertex *v = renderer->vertices + renderer->vertex_count;
    v[0] = (AklvVertex){(vector_float2){x, y}, (vector_float2){u0, v0}, c};
    v[1] = (AklvVertex){(vector_float2){x + w, y}, (vector_float2){u1, v0}, c};
    v[2] = (AklvVertex){(vector_float2){x, y + h}, (vector_float2){u0, v1}, c};
    v[3] = (AklvVertex){(vector_float2){x + w, y}, (vector_float2){u1, v0}, c};
    v[4] = (AklvVertex){(vector_float2){x + w, y + h}, (vector_float2){u1, v1}, c};
    v[5] = (AklvVertex){(vector_float2){x, y + h}, (vector_float2){u0, v1}, c};
    renderer->vertex_count += 6;
}

static bool aklv_metal_create_pipeline(AklvMetalRenderer *renderer, char *error, size_t error_cap) {
    static NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "struct Vertex { float2 position; float2 uv; float4 color; };\n"
         "struct Uniforms { float2 viewport; };\n"
         "struct Out { float4 position [[position]]; float2 uv; float4 color; };\n"
         "vertex Out vs_main(const device Vertex *vertices [[buffer(0)]], constant Uniforms &u [[buffer(1)]], uint vid [[vertex_id]]) {\n"
         "    Vertex v = vertices[vid];\n"
         "    float2 clip = float2((v.position.x / u.viewport.x) * 2.0 - 1.0, 1.0 - (v.position.y / u.viewport.y) * 2.0);\n"
         "    Out out;\n"
         "    out.position = float4(clip, 0.0, 1.0);\n"
         "    out.uv = v.uv;\n"
         "    out.color = v.color;\n"
         "    return out;\n"
         "}\n"
         "fragment half4 fs_main(Out in [[stage_in]], texture2d<float> atlas [[texture(0)]], sampler smp [[sampler(0)]]) {\n"
         "    float alpha = atlas.sample(smp, in.uv).a;\n"
         "    return half4(in.color.r, in.color.g, in.color.b, in.color.a * alpha);\n"
         "}\n";

    NSError *ns_error = nil;
    id<MTLLibrary> library = [renderer->device newLibraryWithSource:source options:nil error:&ns_error];
    if (library == nil) {
        aklv_set_error(error, error_cap, [[ns_error localizedDescription] UTF8String]);
        return false;
    }

    id<MTLFunction> vs = [library newFunctionWithName:@"vs_main"];
    id<MTLFunction> fs = [library newFunctionWithName:@"fs_main"];
    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vs;
    desc.fragmentFunction = fs;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    desc.colorAttachments[0].blendingEnabled = YES;
    desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

    renderer->pipeline = [renderer->device newRenderPipelineStateWithDescriptor:desc error:&ns_error];
    [desc release];
    [vs release];
    [fs release];
    [library release];

    if (renderer->pipeline == nil) {
        aklv_set_error(error, error_cap, [[ns_error localizedDescription] UTF8String]);
        return false;
    }

    MTLSamplerDescriptor *sampler_desc = [[MTLSamplerDescriptor alloc] init];
    sampler_desc.minFilter = MTLSamplerMinMagFilterLinear;
    sampler_desc.magFilter = MTLSamplerMinMagFilterLinear;
    sampler_desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sampler_desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    renderer->sampler = [renderer->device newSamplerStateWithDescriptor:sampler_desc];
    [sampler_desc release];
    return renderer->sampler != nil;
}

static bool aklv_metal_create_font_atlas(AklvMetalRenderer *renderer, char *error, size_t error_cap) {
    renderer->atlas_scale = 2.0f;
    const float font_size = 13.0f * renderer->atlas_scale;
    CTFontRef font = CTFontCreateWithName(CFSTR("Menlo-Regular"), font_size, NULL);
    if (font == NULL) {
        font = CTFontCreateWithName(CFSTR("Monaco"), font_size, NULL);
    }
    if (font == NULL) {
        aklv_set_error(error, error_cap, "unable to create CoreText monospace font");
        return false;
    }

    UniChar sample_char = 'M';
    CGGlyph sample_glyph = 0;
    CGSize advance = CGSizeZero;
    CTFontGetGlyphsForCharacters(font, &sample_char, &sample_glyph, 1);
    CTFontGetAdvancesForGlyphs(font, kCTFontOrientationDefault, &sample_glyph, &advance, 1);
    float ascent = CTFontGetAscent(font);
    float descent = CTFontGetDescent(font);
    float leading = CTFontGetLeading(font);
    int cell_w = (int)(advance.width + 8.0f);
    int cell_h = (int)(ascent + descent + leading + 8.0f);
    if (cell_w < 8) {
        cell_w = 8;
    }
    if (cell_h < 12) {
        cell_h = 12;
    }

    int cols = 16;
    int rows = 6;
    renderer->atlas_width = cols * cell_w + 2;
    renderer->atlas_height = rows * cell_h + 2;
    size_t atlas_bytes = (size_t)renderer->atlas_width * (size_t)renderer->atlas_height * 4;
    unsigned char *pixels = calloc(atlas_bytes, 1);
    if (pixels == NULL) {
        CFRelease(font);
        aklv_set_error(error, error_cap, "calloc failed while creating font atlas");
        return false;
    }
    pixels[3] = 255;

    CGColorSpaceRef rgb = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(pixels,
                                             (size_t)renderer->atlas_width,
                                             (size_t)renderer->atlas_height,
                                             8,
                                             (size_t)renderer->atlas_width * 4,
                                             rgb,
                                             (CGBitmapInfo)kCGImageAlphaPremultipliedLast);
    CGColorSpaceRelease(rgb);
    if (ctx == NULL) {
        free(pixels);
        CFRelease(font);
        aklv_set_error(error, error_cap, "CGBitmapContextCreate failed");
        return false;
    }

    CGContextSetShouldAntialias(ctx, true);
    CGContextSetAllowsAntialiasing(ctx, true);
    CGContextSetTextMatrix(ctx, CGAffineTransformIdentity);
    CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0);

    CGColorRef white = CGColorCreateGenericRGB(1.0, 1.0, 1.0, 1.0);
    NSDictionary *attrs = @{
        (id)kCTFontAttributeName: (id)font,
        (id)kCTForegroundColorAttributeName: (id)white
    };

    for (int ch = 32; ch < 127; ch++) {
        int index = ch - 32;
        int col = index % cols;
        int row = index / cols;
        int px = 1 + col * cell_w;
        int py = 1 + row * cell_h;
        unichar uch = (unichar)ch;
        NSString *s = [NSString stringWithCharacters:&uch length:1];
        NSAttributedString *attr = [[NSAttributedString alloc] initWithString:s attributes:attrs];
        CTLineRef line = CTLineCreateWithAttributedString((CFAttributedStringRef)attr);
        CGContextSetTextPosition(ctx, px + 4.0f, renderer->atlas_height - (py + 4.0f + ascent));
        CTLineDraw(line, ctx);
        CFRelease(line);
        [attr release];

        renderer->glyphs[ch].u0 = (float)px / (float)renderer->atlas_width;
        renderer->glyphs[ch].v0 = (float)py / (float)renderer->atlas_height;
        renderer->glyphs[ch].u1 = (float)(px + cell_w) / (float)renderer->atlas_width;
        renderer->glyphs[ch].v1 = (float)(py + cell_h) / (float)renderer->atlas_height;
        renderer->glyphs[ch].advance = (float)advance.width / renderer->atlas_scale;
        renderer->glyphs[ch].w = (float)cell_w / renderer->atlas_scale;
        renderer->glyphs[ch].h = (float)cell_h / renderer->atlas_scale;
        renderer->glyphs[ch].draw_x = 0.0f;
        renderer->glyphs[ch].draw_y = 0.0f;
    }

    CGContextRelease(ctx);
    CGColorRelease(white);
    renderer->char_width = (float)advance.width / renderer->atlas_scale;
    renderer->line_height = (float)cell_h / renderer->atlas_scale;

    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:(NSUInteger)renderer->atlas_width
                                                                                 height:(NSUInteger)renderer->atlas_height
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    renderer->atlas = [renderer->device newTextureWithDescriptor:td];
    if (renderer->atlas == nil) {
        free(pixels);
        CFRelease(font);
        aklv_set_error(error, error_cap, "Metal texture allocation failed for font atlas");
        return false;
    }
    MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)renderer->atlas_width, (NSUInteger)renderer->atlas_height);
    [renderer->atlas replaceRegion:region mipmapLevel:0 withBytes:pixels bytesPerRow:(NSUInteger)renderer->atlas_width * 4];
    free(pixels);
    CFRelease(font);
    return true;
}

bool aklv_metal_create(SDL_Window *window, AklvMetalRenderer **out, char *error, size_t error_cap) {
    *out = NULL;
    AklvMetalRenderer *renderer = calloc(1, sizeof(*renderer));
    if (renderer == NULL) {
        aklv_set_error(error, error_cap, "calloc failed while creating Metal renderer");
        return false;
    }

    renderer->window = window;
    renderer->metal_view = SDL_Metal_CreateView(window);
    if (renderer->metal_view == NULL) {
        aklv_set_error(error, error_cap, SDL_GetError());
        aklv_metal_destroy(renderer);
        return false;
    }

    renderer->device = MTLCreateSystemDefaultDevice();
    if (renderer->device == nil) {
        aklv_set_error(error, error_cap, "MTLCreateSystemDefaultDevice returned nil");
        aklv_metal_destroy(renderer);
        return false;
    }
    [renderer->device retain];

    renderer->layer = (CAMetalLayer *)SDL_Metal_GetLayer(renderer->metal_view);
    if (renderer->layer == nil) {
        aklv_set_error(error, error_cap, SDL_GetError());
        aklv_metal_destroy(renderer);
        return false;
    }
    [renderer->layer retain];
    renderer->layer.device = renderer->device;
    renderer->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    renderer->layer.framebufferOnly = YES;

    renderer->queue = [renderer->device newCommandQueue];
    if (renderer->queue == nil) {
        aklv_set_error(error, error_cap, "newCommandQueue failed");
        aklv_metal_destroy(renderer);
        return false;
    }
    if (!aklv_metal_create_pipeline(renderer, error, error_cap) ||
        !aklv_metal_create_font_atlas(renderer, error, error_cap)) {
        aklv_metal_destroy(renderer);
        return false;
    }

    *out = renderer;
    return true;
}

void aklv_metal_destroy(AklvMetalRenderer *renderer) {
    if (renderer == NULL) {
        return;
    }
    free(renderer->vertices);
    [renderer->vertex_buffer release];
    [renderer->atlas release];
    [renderer->sampler release];
    [renderer->pipeline release];
    [renderer->queue release];
    [renderer->layer release];
    [renderer->device release];
    if (renderer->metal_view != NULL) {
        SDL_Metal_DestroyView(renderer->metal_view);
    }
    free(renderer);
}

bool aklv_metal_begin(AklvMetalRenderer *renderer,
                      int drawable_width,
                      int drawable_height,
                      float ui_scale,
                      AklvColor clear_color) {
    renderer->vertex_count = 0;
    renderer->drawable_width = drawable_width;
    renderer->drawable_height = drawable_height;
    renderer->ui_scale = ui_scale <= 0.0f ? 1.0f : ui_scale;
    renderer->layer.drawableSize = CGSizeMake((CGFloat)drawable_width, (CGFloat)drawable_height);

    renderer->drawable = [renderer->layer nextDrawable];
    if (renderer->drawable == nil) {
        return false;
    }
    [renderer->drawable retain];
    renderer->command_buffer = [renderer->queue commandBuffer];
    [renderer->command_buffer retain];

    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = renderer->drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
    renderer->encoder = [renderer->command_buffer renderCommandEncoderWithDescriptor:pass];
    [renderer->encoder retain];
    return renderer->encoder != nil;
}

void aklv_metal_end(AklvMetalRenderer *renderer) {
    if (renderer->encoder == nil || renderer->command_buffer == nil || renderer->drawable == nil) {
        return;
    }

    if (renderer->vertex_count > 0) {
        NSUInteger bytes = (NSUInteger)(renderer->vertex_count * sizeof(*renderer->vertices));
        if (renderer->vertex_buffer == nil || renderer->vertex_buffer.length < bytes) {
            [renderer->vertex_buffer release];
            renderer->vertex_buffer = [renderer->device newBufferWithLength:bytes
                                                                     options:MTLResourceStorageModeShared];
        }
        if (renderer->vertex_buffer != nil) {
            memcpy(renderer->vertex_buffer.contents, renderer->vertices, bytes);
            [renderer->vertex_buffer didModifyRange:NSMakeRange(0, bytes)];
            vector_float2 viewport = {
                (float)renderer->drawable_width / renderer->ui_scale,
                (float)renderer->drawable_height / renderer->ui_scale
            };
            [renderer->encoder setRenderPipelineState:renderer->pipeline];
            [renderer->encoder setVertexBuffer:renderer->vertex_buffer offset:0 atIndex:0];
            [renderer->encoder setVertexBytes:&viewport length:sizeof(viewport) atIndex:1];
            [renderer->encoder setFragmentTexture:renderer->atlas atIndex:0];
            [renderer->encoder setFragmentSamplerState:renderer->sampler atIndex:0];
            [renderer->encoder drawPrimitives:MTLPrimitiveTypeTriangle
                                  vertexStart:0
                                  vertexCount:(NSUInteger)renderer->vertex_count];
        }
    }

    [renderer->encoder endEncoding];
    [renderer->command_buffer presentDrawable:renderer->drawable];
    [renderer->command_buffer commit];

    [renderer->encoder release];
    [renderer->command_buffer release];
    [renderer->drawable release];
    renderer->encoder = nil;
    renderer->command_buffer = nil;
    renderer->drawable = nil;
}

void aklv_metal_push_rect(AklvMetalRenderer *renderer,
                          float x,
                          float y,
                          float w,
                          float h,
                          AklvColor color) {
    float u = 0.5f / (float)renderer->atlas_width;
    float v = 0.5f / (float)renderer->atlas_height;
    aklv_push_quad_uv(renderer, x, y, w, h, u, v, u, v, color);
}

void aklv_metal_push_line(AklvMetalRenderer *renderer,
                          float x0,
                          float y0,
                          float x1,
                          float y1,
                          float thickness,
                          AklvColor color) {
    if (thickness <= 0.0f) {
        thickness = 1.0f;
    }
    if (x0 == x1) {
        float y = y0 < y1 ? y0 : y1;
        float h = y0 < y1 ? y1 - y0 : y0 - y1;
        aklv_metal_push_rect(renderer, x0 - thickness * 0.5f, y, thickness, h, color);
    } else if (y0 == y1) {
        float x = x0 < x1 ? x0 : x1;
        float w = x0 < x1 ? x1 - x0 : x0 - x1;
        aklv_metal_push_rect(renderer, x, y0 - thickness * 0.5f, w, thickness, color);
    }
}

void aklv_metal_push_text(AklvMetalRenderer *renderer,
                          float x,
                          float y,
                          const char *text,
                          size_t len,
                          AklvColor color) {
    if (text == NULL) {
        return;
    }
    float cursor = x;
    float max_x = (float)renderer->drawable_width / renderer->ui_scale;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\0') {
            break;
        }
        if (c == '\t') {
            cursor += renderer->char_width * 4.0f;
            continue;
        }
        if (c < 32 || c >= 127) {
            c = '?';
        }
        if (cursor > max_x) {
            break;
        }
        AklvGlyph *g = &renderer->glyphs[c];
        aklv_push_quad_uv(renderer, cursor, y, g->w, g->h, g->u0, g->v0, g->u1, g->v1, color);
        cursor += g->advance;
    }
}

float aklv_metal_char_width(const AklvMetalRenderer *renderer) {
    return renderer->char_width;
}

float aklv_metal_line_height(const AklvMetalRenderer *renderer) {
    return renderer->line_height;
}
