#include "hnsw.hpp"
#include "chunker.hpp"
#include "embedding.hpp"
#include <iostream>
#include <string>

void print_usage() {
    std::cout << "Usage:\n"
              << "  ./vecs index <text_file> <index_file>\n"
              << "  ./vecs search <text_file> <index_file> \"<query>\"\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage();
        return 1;
    }

    std::string mode = argv[1];
    std::string text_file = argv[2];
    std::string index_file = argv[3];

    HNSWConfig config;
    config.dim = 384; // BGE-small-en-v1.5 embedding size
    config.max_elements = 50000;

    if (mode == "index") {
        std::cout << "Creating new index..." << std::endl;
        HNSWIndex index(index_file, config, true);
        
        Chunker chunker(text_file, 100, 20); // 100 words per chunk, 20 words overlap
        chunker.process_and_index(index, 4); // 4 threads
        
    } else if (mode == "search") {
        if (argc < 5) {
            std::cout << "Missing query string for search.\n";
            print_usage();
            return 1;
        }
        std::string query = argv[4];
        
        std::cout << "Loading index for search..." << std::endl;
        HNSWIndex index(index_file, config, false);
        Chunker chunker(text_file); // Just for reading chunks
        
        std::cout << "Generating query embedding..." << std::endl;
        Embedding emb_model;
        std::vector<float> query_vec = emb_model.get_embedding(query);
        
        std::cout << "Searching..." << std::endl;
        auto results = index.search_knn(query_vec, 5); // Top 5
        
        std::cout << "\n--- Search Results ---\n";
        for (size_t i = 0; i < results.size(); i++) {
            auto text_info = index.get_text_info(results[i].id);
            std::string text = chunker.get_chunk(text_info.first, text_info.second);
            std::cout << "Result " << i + 1 << " (Distance: " << results[i].distance << "):\n";
            std::cout << text << "\n----------------------\n";
        }
    } else {
        print_usage();
        return 1;
    }

    return 0;
}
