#include "llama.h"
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

class Embedding {
private:
  llama_model *model;
  llama_context *ctx;
  const llama_vocab *vocab;

public:
  Embedding() {
    string model_path = "model/bge-small-en-v1.5.Q8_0.gguf";
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    this->model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!this->model) {
      throw runtime_error("Failed to load the embedding model.");
    }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    this->ctx = llama_init_from_model(this->model, ctx_params);
    if (!this->ctx) {
      throw runtime_error("Failed to create context.");
    }
    this->vocab = llama_model_get_vocab(this->model);
  }

  ~Embedding() {
    if (this->ctx)
      llama_free(this->ctx);
    if (this->model)
      llama_model_free(this->model);
    llama_backend_free();
  }

  vector<float> get_embedding(string chunk) {
    vector<llama_token> tokens(chunk.length() + 4);
    int n_tokens = llama_tokenize(this->vocab, chunk.c_str(), chunk.length(),
                                  tokens.data(), tokens.size(), true, false);

    if (n_tokens < 0) {
      tokens.resize(-n_tokens);
      n_tokens = llama_tokenize(this->vocab, chunk.c_str(), chunk.length(),
                                tokens.data(), tokens.size(), true, false);
    }
    tokens.resize(n_tokens);
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(this->ctx, batch) != 0) {
      throw runtime_error("Failed to process batch");
    }
    const int n_embd_out = llama_model_n_embd_out(this->model);
    const float *embd_ptr = nullptr;
    if (llama_pooling_type(this->ctx) != LLAMA_POOLING_TYPE_NONE) {
      embd_ptr = llama_get_embeddings_seq(this->ctx, 0);
    } else {
      embd_ptr = llama_get_embeddings_ith(this->ctx, n_tokens - 1);
    }
    if (!embd_ptr) {
      throw runtime_error("Failed to extract embeddings from context");
    }
    return vector<float>(embd_ptr, embd_ptr + n_embd_out);
  }
};
