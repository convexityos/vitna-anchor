CC ?= clang
CFLAGS ?= -O3 -Wall -Wextra -Iinclude
LDFLAGS ?= -lm

ifeq ($(OS),Windows_NT)
    TARGET = vitna-anchor.exe
    ALIAS = vitna-engine.exe
    LDFLAGS += -lws2_32
    CP = cp
else
    TARGET = vitna-anchor
    ALIAS = vitna-engine
    LDFLAGS += -lpthread
    CP = cp -f
endif

SRCS = $(sort $(wildcard src/*.c))

all: $(TARGET) $(ALIAS)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

$(ALIAS): $(TARGET)
	$(CP) $(TARGET) $(ALIAS)

clean:
	rm -f $(TARGET) $(ALIAS) src/*.o

.PHONY: all clean

