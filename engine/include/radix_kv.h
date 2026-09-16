/**
 * radix_kv.h - Radix Tree Attention for paged KV cache prefix sharing.
 *
 * Rules:
 * - Pure C11 zero external runtime dependencies
 * - No em-dashes anywhere in comments or code
 * - Zero-copy ancestor node sharing across branching agentic requests
 */

#ifndef VITNA_RADIX_KV_H
#define VITNA_RADIX_KV_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vitna_radix_node {
    uint32_t* token_ids;
    size_t num_tokens;
    uint32_t* physical_blocks;
    size_t num_blocks;
    size_t ref_count;
    struct vitna_radix_node** children;
    size_t num_children;
    size_t capacity_children;
    struct vitna_radix_node* parent;
} vitna_radix_node_t;

typedef struct {
    vitna_radix_node_t* root;
    size_t total_nodes;
    size_t total_tokens_cached;
} vitna_radix_tree_t;

/**
 * Initialize an empty Radix Tree.
 */
bool vitna_radix_tree_init(vitna_radix_tree_t* tree);

/**
 * Recursively free all nodes in the Radix Tree.
 */
void vitna_radix_tree_free(vitna_radix_tree_t* tree);

/**
 * Search the tree for the longest common prefix of incoming prompt tokens.
 * Returns the number of prefix tokens matched and copies their physical block IDs.
 */
size_t vitna_radix_tree_match(
    vitna_radix_tree_t* tree,
    const uint32_t* prompt_tokens,
    size_t num_tokens,
    uint32_t* out_block_ids,
    size_t max_blocks,
    vitna_radix_node_t** out_matched_node
);

/**
 * Insert a sequence branch into the Radix Tree under parent node.
 */
vitna_radix_node_t* vitna_radix_tree_insert(
    vitna_radix_tree_t* tree,
    vitna_radix_node_t* parent,
    const uint32_t* token_ids,
    size_t num_tokens,
    const uint32_t* block_ids,
    size_t num_blocks
);

/**
 * Increment reference count on a node and its ancestors.
 */
void vitna_radix_node_incref(vitna_radix_node_t* node);

/**
 * Decrement reference count. If leaf reaches zero, it may be pruned or reused.
 */
void vitna_radix_node_decref(vitna_radix_tree_t* tree, vitna_radix_node_t* node);

#ifdef __cplusplus
}
#endif

#endif /* VITNA_RADIX_KV_H */
