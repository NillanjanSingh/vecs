#include "hnsw.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define HNSW_MAGIC 0x484E5357

HNSWIndex::HNSWIndex(const std::string &index_path, const HNSWConfig &cfg,
                     bool create)
    : config(cfg), fd(-1), mapped_data(nullptr) {

  if (config.dim == 0)
    throw std::runtime_error("Dimension cannot be zero");

  if (config.max_elements == 0)
    throw std::runtime_error("max_elements cannot be zero");

  if (config.M == 0)
    throw std::runtime_error("M cannot be zero");

  if (config.max_M0 == 0)
    throw std::runtime_error("max_M0 cannot be zero");

  if (config.max_level == 0)
    throw std::runtime_error("max_level cannot be zero");

  if (create) {
    fd = open(index_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);

    if (fd == -1)
      throw std::runtime_error("Failed to create index file");
  } else {
    fd = open(index_path.c_str(), O_RDWR);

    if (fd == -1)
      throw std::runtime_error("Failed to open index file");

    Header stored_header;

    if (pread(fd, &stored_header, sizeof(Header), 0) !=
        static_cast<ssize_t>(sizeof(Header))) {
      close(fd);
      throw std::runtime_error("Failed to read index header");
    }

    if (stored_header.magic != HNSW_MAGIC) {
      close(fd);
      throw std::runtime_error("Invalid index file");
    }

    config.dim = stored_header.dim;
    config.M = stored_header.M;
    config.max_M0 = stored_header.max_M0;
    config.max_elements = stored_header.max_elements;
  }

  const size_t connection_size =
      sizeof(int) * (1 + config.max_M0) +
      sizeof(int) * (config.max_level - 1) * (1 + config.M);

  node_size = sizeof(int) * 3 + sizeof(float) * config.dim + connection_size;

  file_size = sizeof(Header) + config.max_elements * node_size;

  if (create) {
    if (ftruncate(fd, file_size) == -1) {
      close(fd);
      throw std::runtime_error("Failed to allocate index file");
    }
  }

  mapped_data =
      mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  if (mapped_data == MAP_FAILED) {
    close(fd);
    mapped_data = nullptr;
    throw std::runtime_error("Failed to mmap index file");
  }

  header = static_cast<Header *>(mapped_data);

  if (create) {
    header->magic = HNSW_MAGIC;
    header->max_elements = config.max_elements;
    header->cur_elements = 0;
    header->dim = config.dim;
    header->M = config.M;
    header->max_M0 = config.max_M0;
    header->ep_id = -1;
    header->ep_level = -1;
  }

  node_locks.reserve(config.max_elements);

  for (size_t i = 0; i < config.max_elements; ++i)
    node_locks.push_back(std::make_unique<std::shared_mutex>());
}

HNSWIndex::~HNSWIndex() {
  if (mapped_data != nullptr && mapped_data != MAP_FAILED) {
    msync(mapped_data, file_size, MS_SYNC);
    munmap(mapped_data, file_size);
  }

  if (fd != -1)
    close(fd);
}

int HNSWIndex::get_cur_elements() const { return header->cur_elements; }

char *HNSWIndex::get_node_ptr(int id) const {
  if (id < 0 || id >= header->max_elements)
    throw std::out_of_range("Invalid node id");

  return static_cast<char *>(mapped_data) + sizeof(Header) +
         static_cast<size_t>(id) * node_size;
}

int *HNSWIndex::get_level_ptr(int id) const {
  return reinterpret_cast<int *>(get_node_ptr(id));
}

int *HNSWIndex::get_text_offset_ptr(int id) const {
  return reinterpret_cast<int *>(get_node_ptr(id) + sizeof(int));
}

int *HNSWIndex::get_text_length_ptr(int id) const {
  return reinterpret_cast<int *>(get_node_ptr(id) + 2 * sizeof(int));
}

float *HNSWIndex::get_embedding_ptr(int id) const {
  return reinterpret_cast<float *>(get_node_ptr(id) + 3 * sizeof(int));
}

int *HNSWIndex::get_connections_ptr(int id, int layer) const {
  int level = *get_level_ptr(id);

  if (layer < 0 || layer > level)
    return nullptr;

  char *base = get_node_ptr(id) + 3 * sizeof(int) + sizeof(float) * config.dim;

  if (layer == 0)
    return reinterpret_cast<int *>(base);

  size_t offset = sizeof(int) * (1 + config.max_M0) +
                  static_cast<size_t>(layer - 1) * sizeof(int) * (1 + config.M);

  return reinterpret_cast<int *>(base + offset);
}

int HNSWIndex::get_random_level() {
  static thread_local std::mt19937 generator(std::random_device{}());

  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  double r = -std::log(distribution(generator)) /
             std::log(static_cast<double>(config.M));

  return std::min(static_cast<int>(r), static_cast<int>(config.max_level) - 1);
}

float HNSWIndex::calc_dist(const float *a, const float *b) const {
  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;

  for (uint32_t i = 0; i < config.dim; ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  if (norm_a == 0.0f || norm_b == 0.0f)
    return 1.0f;

  return 1.0f - dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

std::vector<SearchResult> HNSWIndex::search_layer(const float *query, int ep,
                                                  int ef, int layer) {

  if (ep < 0 || ep >= header->cur_elements)
    return {};

  if (layer < 0 || layer >= config.max_level)
    return {};

  std::unordered_set<int> visited;

  std::priority_queue<SearchResult, std::vector<SearchResult>,
                      std::greater<SearchResult>>
      candidates;

  std::priority_queue<SearchResult, std::vector<SearchResult>,
                      std::less<SearchResult>>
      top_candidates;

  float distance = calc_dist(query, get_embedding_ptr(ep));

  candidates.push({ep, distance});
  top_candidates.push({ep, distance});
  visited.insert(ep);

  while (!candidates.empty()) {
    SearchResult current = candidates.top();
    candidates.pop();

    if (!top_candidates.empty() &&
        current.distance > top_candidates.top().distance) {
      break;
    }

    std::shared_lock<std::shared_mutex> lock(*node_locks[current.id]);

    int *connections = get_connections_ptr(current.id, layer);

    if (connections == nullptr)
      continue;

    int count = connections[0];

    for (int i = 0; i < count; ++i) {
      int neighbor = connections[i + 1];

      if (neighbor < 0 || neighbor >= header->cur_elements)
        continue;

      if (!visited.insert(neighbor).second)
        continue;

      float dist = calc_dist(query, get_embedding_ptr(neighbor));

      if (top_candidates.size() < static_cast<size_t>(ef) ||
          dist < top_candidates.top().distance) {

        candidates.push({neighbor, dist});
        top_candidates.push({neighbor, dist});

        if (top_candidates.size() > static_cast<size_t>(ef)) {
          top_candidates.pop();
        }
      }
    }
  }

  std::vector<SearchResult> result;

  while (!top_candidates.empty()) {
    result.push_back(top_candidates.top());
    top_candidates.pop();
  }

  std::reverse(result.begin(), result.end());

  return result;
}

void HNSWIndex::add_connection(int src, int dst, int layer) {

  if (src < 0 || src >= header->cur_elements)
    return;

  if (dst < 0 || dst >= header->cur_elements)
    return;

  if (src == dst)
    return;

  std::unique_lock<std::shared_mutex> lock(*node_locks[src]);

  int *connections = get_connections_ptr(src, layer);

  if (connections == nullptr)
    return;

  int max_connections = (layer == 0) ? config.max_M0 : config.M;

  int count = connections[0];

  for (int i = 0; i < count; ++i) {
    if (connections[i + 1] == dst)
      return;
  }

  if (count < max_connections) {
    connections[count + 1] = dst;
    connections[0] = count + 1;
  }
}

void HNSWIndex::mutually_connect(int id1, int id2, int layer) {
  add_connection(id1, id2, layer);
  add_connection(id2, id1, layer);
}

void HNSWIndex::add_point(const std::vector<float> &embedding, int text_offset,
                          int text_length) {

  if (embedding.size() != config.dim)
    throw std::runtime_error("Embedding dimension does not match index");

  int id;

  {
    std::unique_lock<std::shared_mutex> lock(global_lock);

    if (header->cur_elements >= config.max_elements) {
      throw std::runtime_error("Index full");
    }

    id = header->cur_elements++;
  }

  int level = get_random_level();

  *get_level_ptr(id) = level;
  *get_text_offset_ptr(id) = text_offset;
  *get_text_length_ptr(id) = text_length;

  float *embedding_ptr = get_embedding_ptr(id);

  std::copy(embedding.begin(), embedding.end(), embedding_ptr);

  for (int layer = 0; layer <= level; ++layer) {
    int *connections = get_connections_ptr(id, layer);

    connections[0] = 0;
  }

  int ep;
  int max_ep_level;

  {
    std::shared_lock<std::shared_mutex> lock(global_lock);

    ep = header->ep_id;
    max_ep_level = header->ep_level;
  }

  if (ep == -1) {
    std::unique_lock<std::shared_mutex> lock(global_lock);

    header->ep_id = id;
    header->ep_level = level;

    return;
  }

  int current_ep = ep;

  for (int layer = max_ep_level; layer > level; --layer) {

    auto result = search_layer(embedding.data(), current_ep, 1, layer);

    if (!result.empty())
      current_ep = result[0].id;
  }

  int highest_layer = std::min(level, max_ep_level);

  for (int layer = highest_layer; layer >= 0; --layer) {

    int ef = (layer == 0) ? config.max_M0 * 2 : config.M * 2;

    auto neighbors = search_layer(embedding.data(), current_ep, ef, layer);

    if (neighbors.empty())
      continue;

    int max_connections = (layer == 0) ? config.max_M0 : config.M;

    int count = std::min<int>(max_connections, neighbors.size());

    for (int i = 0; i < count; ++i) {
      mutually_connect(id, neighbors[i].id, layer);
    }

    current_ep = neighbors[0].id;
  }

  if (level > max_ep_level) {
    std::unique_lock<std::shared_mutex> lock(global_lock);

    header->ep_id = id;
    header->ep_level = level;
  }
}

std::vector<SearchResult> HNSWIndex::search_knn(const std::vector<float> &query,
                                                int k, int ef_search) {

  if (query.size() != config.dim)
    throw std::runtime_error("Query dimension does not match index");

  if (k <= 0)
    return {};

  if (ef_search < k)
    ef_search = k;

  int ep;
  int max_level;

  {
    std::shared_lock<std::shared_mutex> lock(global_lock);

    ep = header->ep_id;
    max_level = header->ep_level;
  }

  if (ep == -1)
    return {};

  int current_ep = ep;

  for (int layer = max_level; layer > 0; --layer) {

    auto result = search_layer(query.data(), current_ep, 1, layer);

    if (!result.empty())
      current_ep = result[0].id;
  }

  auto result = search_layer(query.data(), current_ep, ef_search, 0);

  if (result.size() > static_cast<size_t>(k))
    result.resize(k);

  return result;
}

std::vector<float> HNSWIndex::get_embedding(int id) const {

  if (id < 0 || id >= header->cur_elements)
    throw std::out_of_range("Invalid node id");

  float *ptr = get_embedding_ptr(id);

  return std::vector<float>(ptr, ptr + config.dim);
}

std::pair<int, int> HNSWIndex::get_text_info(int id) const {

  if (id < 0 || id >= header->cur_elements)
    throw std::out_of_range("Invalid node id");

  return {*get_text_offset_ptr(id), *get_text_length_ptr(id)};
}
