#pragma once

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

struct HNSWConfig {
  uint32_t dim = 384;
  uint32_t max_elements = 100000;
  uint32_t M = 16;
  uint32_t max_M0 = 32;
  uint32_t max_level = 16;
};

struct SearchResult {
  int id;
  float distance;

  bool operator<(const SearchResult &other) const {
    return distance < other.distance;
  }

  bool operator>(const SearchResult &other) const {
    return distance > other.distance;
  }
};

class HNSWIndex {
public:
  HNSWIndex(const std::string &index_path, const HNSWConfig &config,
            bool create = false);

  ~HNSWIndex();

  void add_point(const std::vector<float> &embedding, int text_offset,
                 int text_length);

  std::vector<SearchResult> search_knn(const std::vector<float> &query, int k,
                                       int ef_search = 50);

  std::vector<float> get_embedding(int id) const;

  std::pair<int, int> get_text_info(int id) const;

  int get_cur_elements() const;

private:
  struct Header {
    uint32_t magic;
    uint32_t max_elements;
    uint32_t cur_elements;
    uint32_t dim;
    uint32_t M;
    uint32_t max_M0;
    uint32_t max_level;
    int32_t ep_id;
    int32_t ep_level;
  };

  int fd = -1;
  size_t file_size = 0;
  size_t node_size = 0;

  void *mapped_data = nullptr;
  Header *header = nullptr;

  HNSWConfig config;

  std::vector<std::unique_ptr<std::shared_mutex>> node_locks;
  std::shared_mutex global_lock;

  int get_random_level();

  float calc_dist(const float *a, const float *b) const;

  std::vector<SearchResult> search_layer(const float *query, int ep, int ef,
                                         int layer);

  void add_connection(int src, int dst, int layer);

  void mutually_connect(int id1, int id2, int layer);
};
