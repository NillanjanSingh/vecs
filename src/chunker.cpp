#include "chunker.hpp"
#include "embedding.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

Chunker::Chunker(const std::string &filepath, int chunk_words,
                 int overlap_words)
    : filepath(filepath), chunk_words(chunk_words),
      overlap_words(overlap_words), fd(-1), mapped_text(nullptr) {

  fd = open(filepath.c_str(), O_RDONLY);
  if (fd == -1)
    throw std::runtime_error("Failed to open text file");

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    throw std::runtime_error("Failed to stat text file");
  }
  file_size = sb.st_size;

  mapped_text = static_cast<const char *>(
      mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (mapped_text == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Failed to mmap text file");
  }
}

Chunker::~Chunker() {
  if (mapped_text && mapped_text != MAP_FAILED) {
    munmap(const_cast<char *>(mapped_text), file_size);
  }
  if (fd != -1)
    close(fd);
}

std::string Chunker::get_chunk(int offset, int length) const {
  if (offset < 0 || offset + length > file_size)
    return "";
  return std::string(mapped_text + offset, length);
}

std::vector<Chunker::ChunkInfo> Chunker::build_chunks() {
  std::vector<ChunkInfo> chunks;
  int current_offset = 0;

  // We will find word boundaries
  std::vector<int> word_starts;
  std::vector<int> word_ends;

  bool in_word = false;
  for (size_t i = 0; i < file_size; i++) {
    if (std::isspace(mapped_text[i])) {
      if (in_word) {
        word_ends.push_back(i);
        in_word = false;
      }
    } else {
      if (!in_word) {
        word_starts.push_back(i);
        in_word = true;
      }
    }
  }
  if (in_word)
    word_ends.push_back(file_size);

  int total_words = word_starts.size();
  int i = 0;
  while (i < total_words) {
    int end_idx = std::min(i + chunk_words - 1, total_words - 1);
    int start_offset = word_starts[i];
    int end_offset = word_ends[end_idx];

    chunks.push_back(
        {start_offset, end_offset - start_offset,
         std::string(mapped_text + start_offset, end_offset - start_offset)});

    if (end_idx == total_words - 1)
      break;
    i += (chunk_words - overlap_words);
  }

  return chunks;
}

void Chunker::process_and_index(HNSWIndex &index, int num_threads) {
  auto chunks = build_chunks();
  std::cout << "Built " << chunks.size() << " chunks. Indexing concurrently..."
            << std::endl;

  ThreadPool pool(num_threads);
  std::vector<std::future<void>> futures;

  // Using thread_local Embedding to avoid loading the model for every single
  // chunk, but allowing concurrent evaluation across multiple threads. However,
  // thread_local Initialization within a task can be tricky, so we will pass an
  // Embedding instance per thread or just use thread_local inside the task.

  for (const auto &chunk : chunks) {
    futures.push_back(pool.enqueue([&index, chunk]() {
      static thread_local Embedding emb_model; // loaded once per thread
      std::vector<float> vec = emb_model.get_embedding(chunk.text);
      index.add_point(vec, chunk.offset, chunk.length);
    }));
  }

  // Wait for all to complete
  for (auto &fut : futures) {
    fut.get();
  }
  std::cout << "Indexing complete. Total elements in index: "
            << index.get_cur_elements() << std::endl;
}
