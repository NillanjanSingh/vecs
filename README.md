# vecs: High-Performance C++ Vector Store

`vecs` is a lightning-fast, embedded vector search engine built entirely in C++ from scratch. It processes large text documents, generates dense vector embeddings using `llama.cpp` (specifically optimized for models like `bge-small-en-v1.5`), and builds a Hierarchical Navigable Small World (HNSW) graph to enable rapid approximate nearest neighbor (ANN) search. 

Designed for scalability and efficiency, `vecs` heavily utilizes memory mapping, thread pooling, and zero-copy text storage to provide a production-ready vector pipeline that operates efficiently out-of-core.

---

## The Pipeline: How It Works

Rather than explaining the codebase file-by-file, here is the logical flow of how data is ingested, processed, stored, and queried within the `vecs` system.

### 1. Ingestion & Chunking
When a user provides a large `.txt` file to index, `vecs` does not load the entire file into RAM. Instead, it relies on POSIX `mmap` to memory-map the file. The system scans the mapped file to identify word boundaries based on whitespace. 

To ensure that semantic meaning isn't lost at the edges of arbitrary text splits, `vecs` creates **overlapping chunks**. By default, it splits the text into windows of 100 words with a 20-word overlap. This guarantees that sentences straddling a boundary remain contextually intact for the embedding model.

### 2. Concurrent Vectorization
The generated text chunks are dispatched to a custom-built task queue managed by a standard C++ thread pool. 

To generate embeddings concurrently without stepping on the context limits of `llama.cpp`, the pipeline initializes a `thread_local` instance of the embedding model per worker thread. Because models like `bge-small` are highly quantized and lightweight, spinning up a few models in parallel consumes negligible memory while drastically improving throughput. Each chunk is tokenized, passed through a forward pass, and reduced to a 384-dimensional vector embedding.

### 3. Graph Construction (HNSW)
As worker threads generate embeddings, they immediately attempt to insert them into the Hierarchical Navigable Small World (HNSW) graph. 

The HNSW algorithm is an approximate nearest neighbor algorithm that builds multi-layered graphs. `vecs` implements this entirely from scratch:
* **Distance Metric:** Vector similarity is calculated using cosine distance.
* **Thread Safety:** Every node in the graph is given a unique `std::shared_mutex`. When a thread searches the graph to find the optimal insertion point, it takes a shared read-lock. When it modifies a node's connection array to link the new vector, it promotes it to a unique write-lock.

### 4. Zero-Copy Serialization
`vecs` avoids duplicating the original text data. When the HNSW index is serialized to a `.bin` file, it only stores:
1. The mathematical vectors (384 dimensions).
2. The multi-layer graph connections (Level 0, Level 1, etc.).
3. **Metadata offsets:** The start byte and length of the corresponding text chunk in the original text file.

By dynamically calculating pointer offsets during serialization, `vecs` achieves a perfect, dense binary layout.

### 5. Querying & Retrieval
During a search, the serialized `.bin` index is loaded back into memory using `mmap`. The operating system handles piecewise page swapping, allowing for extremely low overhead and the ability to search graphs larger than available RAM.

When a user submits a natural language query:
1. The query is vectorized using the same `llama.cpp` pipeline.
2. The HNSW graph is traversed from the top layer down to the bottom layer to find the nearest neighbors (the closest 384-dimensional vectors).
3. The result IDs are resolved to their text metadata (start offset and length).
4. `vecs` looks up those offsets directly in the memory-mapped original `.txt` file and prints the human-readable chunks back to the user.

---

## Features

- **Memory-Mapped Storage (`mmap`)**: Both the original text and the generated HNSW index are operated on via `mmap`, offloading memory management to the OS.
- **Concurrent Indexing**: A robust thread pool and fine-grained node-level locking allow multi-threaded graph insertion.
- **Zero-Copy Text Storage**: The database does not duplicate strings, dramatically reducing index size.
- **Native `llama.cpp` Integration**: Seamless tokenization and inference directly in C++ without Python overhead.

---

## Prerequisites

- **CMake** >= 3.10
- **C++ Compiler** with C++23 support (e.g., GCC or Clang)
- **Linux/POSIX environment** (Uses `sys/mman.h` and `fcntl.h`)
- Required `.gguf` model (e.g., `bge-small-en-v1.5.Q8_0.gguf`) placed in the `model/` directory.

## Build Instructions

First, download the required embedding model into a `model/` directory:

```bash
mkdir -p model
cd model
wget https://huggingface.co/compendiumlabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf -O bge-small-en-v1.5.Q8_0.gguf
cd ..
```

Then, compile the project:

```bash
mkdir build && cd build
cmake ..
make -j4
```

This will produce the `vecs` executable in the `build/` directory.

## Usage

The `vecs` CLI operates in two modes: `index` and `search`.

### 1. Indexing a File

To parse a large text file, chunk it, and generate the vector graph:

```bash
./vecs index <path_to_txt> <path_to_save_index.bin>
```
*Example:* `./vecs index my_book.txt book_index.bin`

### 2. Searching the Index

To query the memory-mapped graph using a natural language query:

```bash
./vecs search <path_to_txt> <path_to_saved_index.bin> "your query here"
```
*Example:* `./vecs search my_book.txt book_index.bin "Who was the main character?"`
