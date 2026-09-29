CC ?= clang
CFLAGS ?= -O3 -Wall -Wextra -Iinclude
LDFLAGS ?= -lm

ifeq ($(OS),Windows_NT)
    TARGET = vitna-anchor.exe
    ALIAS = vitna-engine.exe
    TESTS = vitna-anchor-tests.exe
    LDFLAGS += -lws2_32
    CP = cp
else
    TARGET = vitna-anchor
    ALIAS = vitna-engine
    TESTS = vitna-anchor-tests
    LDFLAGS += -lpthread
    CP = cp -f
endif

SRCS = $(sort $(wildcard src/*.c))
LIB_SRCS = $(filter-out src/main.c,$(SRCS))

all: $(TARGET) $(ALIAS)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

$(ALIAS): $(TARGET)
	$(CP) $(TARGET) $(ALIAS)

$(TESTS): $(LIB_SRCS) tests/unit_tests.c
	$(CC) $(CFLAGS) $(LIB_SRCS) tests/unit_tests.c -o $(TESTS) $(LDFLAGS)

test: $(TESTS)
	./$(TESTS)

clean:
	rm -f $(TARGET) $(ALIAS) $(TESTS) src/*.o

.PHONY: all clean test
