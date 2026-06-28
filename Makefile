CC ?= clang
OBJC ?= clang
PKG_CONFIG ?= pkg-config

BUILD_DIR := build
SRC_DIR := src

COMMON_CFLAGS := -std=c17 -O3 -Wall -Wextra -Wpedantic -D_DARWIN_C_SOURCE
OBJCFLAGS := -fblocks
PLATFORM_OBJCFLAGS := -fobjc-arc -fblocks
LDFLAGS :=
LDLIBS_CORE := -lpthread
LDLIBS_APP := -framework Metal -framework QuartzCore -framework Foundation -framework AppKit -framework CoreGraphics -framework CoreText

SDL_CFLAGS := $(shell $(PKG_CONFIG) --cflags sdl3 2>/dev/null || $(PKG_CONFIG) --cflags SDL3 2>/dev/null)
SDL_LIBS := $(shell $(PKG_CONFIG) --libs sdl3 2>/dev/null || $(PKG_CONFIG) --libs SDL3 2>/dev/null)

BREW_SDL_PREFIX := $(shell brew --prefix sdl3 2>/dev/null)
ifeq ($(SDL_CFLAGS),)
ifneq ($(wildcard $(BREW_SDL_PREFIX)/include/SDL3/SDL.h),)
SDL_CFLAGS := -I$(BREW_SDL_PREFIX)/include
SDL_LIBS := -L$(BREW_SDL_PREFIX)/lib -lSDL3
endif
endif

CORE_OBJS := \
	$(BUILD_DIR)/aklv_index.o \
	$(BUILD_DIR)/aklv_search.o \
	$(BUILD_DIR)/aklv_loader.o

APP_OBJS := \
	$(CORE_OBJS) \
	$(BUILD_DIR)/aklv_metal.o \
	$(BUILD_DIR)/aklv_platform.o \
	$(BUILD_DIR)/main.o

SELFTEST_OBJS := \
	$(CORE_OBJS) \
	$(BUILD_DIR)/selftest.o

.PHONY: all app selftest run-selftest clean check-sdl

all: selftest app

app: check-sdl $(BUILD_DIR)/aklv

selftest: $(BUILD_DIR)/aklv_selftest

run-selftest: selftest
	./$(BUILD_DIR)/aklv_selftest

check-sdl:
	@if [ -z "$(SDL_CFLAGS)$(SDL_LIBS)" ]; then \
		echo "SDL3 was not found. Install it with: brew install sdl3"; \
		exit 1; \
	fi

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(COMMON_CFLAGS) $(SDL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/aklv_index.o: $(SRC_DIR)/aklv_index.h
$(BUILD_DIR)/aklv_search.o: $(SRC_DIR)/aklv_index.h $(SRC_DIR)/aklv_search.h
$(BUILD_DIR)/aklv_loader.o: $(SRC_DIR)/aklv_index.h $(SRC_DIR)/aklv_loader.h
$(BUILD_DIR)/main.o: $(SRC_DIR)/aklv_index.h $(SRC_DIR)/aklv_search.h $(SRC_DIR)/aklv_loader.h $(SRC_DIR)/aklv_metal.h $(SRC_DIR)/aklv_platform.h
$(BUILD_DIR)/selftest.o: $(SRC_DIR)/aklv_index.h $(SRC_DIR)/aklv_search.h

$(BUILD_DIR)/aklv_metal.o: $(SRC_DIR)/aklv_metal.m | $(BUILD_DIR)
	$(OBJC) $(COMMON_CFLAGS) $(OBJCFLAGS) $(SDL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/aklv_platform.o: $(SRC_DIR)/aklv_platform.m | $(BUILD_DIR)
	$(OBJC) $(COMMON_CFLAGS) $(PLATFORM_OBJCFLAGS) $(SDL_CFLAGS) -c $< -o $@

$(BUILD_DIR)/aklv: $(APP_OBJS)
	$(OBJC) $(LDFLAGS) $^ $(SDL_LIBS) $(LDLIBS_APP) $(LDLIBS_CORE) -o $@

$(BUILD_DIR)/aklv_selftest: $(SELFTEST_OBJS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS_CORE) -o $@

clean:
	rm -rf $(BUILD_DIR)
