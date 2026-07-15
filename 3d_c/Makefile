# First is meshsize then volfrac 
CC := gcc
PYTHON := python3

RAW_OUTPUT := raw
TARGET := $(RAW_OUTPUT)/topt

SOURCES := topt.c fem_solver.c mesh_struct.c
OBJECTS := $(SOURCES:%.c=$(RAW_OUTPUT)/%.o)
HEADERS := topt.h fem_solver.h mesh_struct.h

CFLAGS := -std=c11 -O2 -Wall -Wextra
LDLIBS := -lgmsh -lumfpack -lamd -lsuitesparseconfig -lblas -lm

PYTHON_RUNNER := main.py

OUTPUT_DIR := outputs
NPZ_OUTPUT := $(OUTPUT_DIR)/results.npz
MP4_OUTPUT := $(OUTPUT_DIR)/topt.mp4
VOLFRAC ?= 0.10
MESHSIZE ?= 0.30

.PHONY: all build optimise plot clean help

all: optimise
	@$(MAKE) --no-print-directory plot

build: $(TARGET)

$(TARGET): $(OBJECTS) | $(RAW_OUTPUT)
	@$(CC) $(OBJECTS) -o $@ $(LDLIBS)

$(RAW_OUTPUT)/%.o: %.c $(HEADERS) | $(RAW_OUTPUT)
	@$(CC) $(CFLAGS) -c $< -o $@

$(RAW_OUTPUT):
	@mkdir -p $(RAW_OUTPUT)

optimise: build
	@./$(TARGET) $(RAW_OUTPUT) $(VOLFRAC) $(MESHSIZE)

plot:
	@$(PYTHON) $(PYTHON_RUNNER) --skip-run --mp4 $(MP4_OUTPUT)

clean:
	@rm -rf $(RAW_OUTPUT)

help:
	@echo "Available targets:"
	@echo "  make             Compile, optimise, import results and create MP4"
	@echo "  make build       Compile the C executable only"
	@echo "  make optimise    Compile and run only the C optimisation"
	@echo "  make plot        Plot existing optimisation results"
	@echo "  make clean       Remove compiled files and raw results"