# CPU-only command-line programs built from the server's own sources UNCHANGED (A1 PREREG section 1 and the render-parity
# change): tool_parser_cli (server/src/server/tool_parser.cpp) and tokenize_cli (server/src/server/tokenizer.cpp). Needs g++ and
# nlohmann/json 3.11 headers; tokenize_cli also links libggml-base.so (the server's own ggml build, only its GGUF reader is used).
#   make                                    # in the repo: tool_parser_cli
#   make tokenize_cli GGML_LIBDIR=<dir with libggml-base.so>
#   make TP_SRC=src JSON_INC=-I/path        # on lucebox6: tool_parser.{cpp,h}, tokenizer.{cpp,h} copied byte-identical into ./src
SRV ?= ../../../../../server
TP_SRC ?= $(SRV)/src/server
GGML_INC ?= $(SRV)/deps/llama.cpp/ggml/include
GGML_LIBDIR ?= /mnt/kimi-k3/build/kvflash-am-cuda/deps/llama.cpp/ggml/src
BUILD ?= build
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall

all: $(BUILD)/tool_parser_cli

$(BUILD)/tool_parser_cli: tool_parser_cli.cpp $(TP_SRC)/tool_parser.cpp $(TP_SRC)/tool_parser.h
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(TP_SRC) $(JSON_INC) -o $@ tool_parser_cli.cpp $(TP_SRC)/tool_parser.cpp

tokenize_cli: $(BUILD)/tokenize_cli

$(BUILD)/tokenize_cli: tokenize_cli.cpp $(TP_SRC)/tokenizer.cpp $(TP_SRC)/tokenizer.h
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I$(TP_SRC) -I$(GGML_INC) $(JSON_INC) -o $@ tokenize_cli.cpp $(TP_SRC)/tokenizer.cpp \
	  -L$(GGML_LIBDIR) -lggml-base -Wl,-rpath,$(GGML_LIBDIR)

clean:
	rm -rf $(BUILD)

.PHONY: all tokenize_cli clean
