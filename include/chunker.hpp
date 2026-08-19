#pragma once

#include "hnsw.hpp"
#include <string>
#include <vector>

class Chunker {
public:
  Chunker(const std::string &filepath, int chunk_words = 200,
          int overlap_words = 50);
  ~Chunker();

  // Process the text file, compute embeddings, and insert into HNSW index
  // concurrently
  void process_and_index(HNSWIndex &index, int num_threads = 4);

  // Get a chunk from the file given offset and length
  std::string get_chunk(int offset, int length) const;

private:
  std::string filepath;
  int chunk_words;
  int overlap_words;

  int fd;
  size_t file_size;
  const char *mapped_text;

  struct ChunkInfo {
    int offset;
    int length;
    std::string text;
  };

  std::vector<ChunkInfo> build_chunks();
};
