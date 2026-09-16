/**
 * radix_kv.c - Radix Tree Attention implementation for paged KV cache.
 *
 * Rules:
 * - Pure C11 zero external dependencies
 * - No em-dashes anywhere in comments or code
 * - Zero-copy ancestor sharing across branching sequences
 */

#include "radix_kv.h"
#include <stdlib.h>
#include <string.h>

static vitna_radix_node_t* create_node(
    const uint32_t* token_ids,
    size_t num_tokens,
    const uint32_t* block_ids,
    size_t num_blocks,
    vitna_radix_node_t* parent
) {
    vitna_radix_node_t* node = (vitna_radix_node_t*)calloc(1, sizeof(vitna_radix_node_t));
    if (!node) return NULL;

    node->num_tokens = num_tokens;
    if (num_tokens > 0 && token_ids) {
        node->token_ids = (uint32_t*)malloc(num_tokens * sizeof(uint32_t));
        if (node->token_ids) {
            memcpy(node->token_ids, token_ids, num_tokens * sizeof(uint32_t));
        }
    }

    node->num_blocks = num_blocks;
    if (num_blocks > 0 && block_ids) {
        node->physical_blocks = (uint32_t*)malloc(num_blocks * sizeof(uint32_t));
        if (node->physical_blocks) {
            memcpy(node->physical_blocks, block_ids, num_blocks * sizeof(uint32_t));
        }
    }

    node->ref_count = 1;
    node->parent = parent;
    node->capacity_children = 4;
    node->children = (vitna_radix_node_t**)calloc(node->capacity_children, sizeof(vitna_radix_node_t*));

    return node;
}

static void free_node_recursive(vitna_radix_node_t* node) {
    if (!node) return;

    for (size_t i = 0; i < node->num_children; i++) {
        free_node_recursive(node->children[i]);
    }

    free(node->children);
    free(node->token_ids);
    free(node->physical_blocks);
    free(node);
}

bool vitna_radix_tree_init(vitna_radix_tree_t* tree) {
    if (!tree) return false;

    tree->root = create_node(NULL, 0, NULL, 0, NULL);
    if (!tree->root) return false;

    tree->total_nodes = 1;
    tree->total_tokens_cached = 0;
    return true;
}

void vitna_radix_tree_free(vitna_radix_tree_t* tree) {
    if (!tree) return;
    if (tree->root) {
        free_node_recursive(tree->root);
        tree->root = NULL;
    }
    tree->total_nodes = 0;
    tree->total_tokens_cached = 0;
}

size_t vitna_radix_tree_match(
    vitna_radix_tree_t* tree,
    const uint32_t* prompt_tokens,
    size_t num_tokens,
    uint32_t* out_block_ids,
    size_t max_blocks,
    vitna_radix_node_t** out_matched_node
) {
    if (!tree || !tree->root || !prompt_tokens || num_tokens == 0) {
        return 0;
    }

    vitna_radix_node_t* current = tree->root;
    size_t matched_tokens = 0;
    size_t copied_blocks = 0;
    size_t offset = 0;

    while (offset < num_tokens) {
        vitna_radix_node_t* next_child = NULL;
        uint32_t target_tok = prompt_tokens[offset];

        for (size_t i = 0; i < current->num_children; i++) {
            vitna_radix_node_t* child = current->children[i];
            if (child && child->num_tokens > 0 && child->token_ids[0] == target_tok) {
                next_child = child;
                break;
            }
        }

        if (!next_child) {
            break;
        }

        /* Count matching tokens along this child edge */
        size_t edge_match = 0;
        while (edge_match < next_child->num_tokens &&
               (offset + edge_match) < num_tokens &&
               next_child->token_ids[edge_match] == prompt_tokens[offset + edge_match]) {
            edge_match++;
        }

        matched_tokens += edge_match;
        offset += edge_match;

        /* Copy physical block IDs */
        if (out_block_ids && next_child->physical_blocks) {
            size_t to_copy = next_child->num_blocks;
            if (copied_blocks + to_copy > max_blocks) {
                to_copy = max_blocks > copied_blocks ? (max_blocks - copied_blocks) : 0;
            }
            if (to_copy > 0) {
                memcpy(&out_block_ids[copied_blocks], next_child->physical_blocks, to_copy * sizeof(uint32_t));
                copied_blocks += to_copy;
            }
        }

        current = next_child;

        /* If partial edge match, cannot descend further */
        if (edge_match < next_child->num_tokens) {
            break;
        }
    }

    if (out_matched_node) {
        *out_matched_node = current;
    }

    return matched_tokens;
}

vitna_radix_node_t* vitna_radix_tree_insert(
    vitna_radix_tree_t* tree,
    vitna_radix_node_t* parent,
    const uint32_t* token_ids,
    size_t num_tokens,
    const uint32_t* block_ids,
    size_t num_blocks
) {
    if (!tree || !tree->root || !token_ids || num_tokens == 0) {
        return NULL;
    }

    if (!parent) {
        parent = tree->root;
    }

    /* Expand children array if needed */
    if (parent->num_children >= parent->capacity_children) {
        size_t new_cap = parent->capacity_children * 2;
        vitna_radix_node_t** new_children = (vitna_radix_node_t**)realloc(
            parent->children,
            new_cap * sizeof(vitna_radix_node_t*)
        );
        if (!new_children) return NULL;
        parent->children = new_children;
        parent->capacity_children = new_cap;
    }

    vitna_radix_node_t* child = create_node(token_ids, num_tokens, block_ids, num_blocks, parent);
    if (!child) return NULL;

    parent->children[parent->num_children++] = child;
    tree->total_nodes++;
    tree->total_tokens_cached += num_tokens;

    return child;
}

void vitna_radix_node_incref(vitna_radix_node_t* node) {
    vitna_radix_node_t* curr = node;
    while (curr) {
        curr->ref_count++;
        curr = curr->parent;
    }
}

void vitna_radix_node_decref(vitna_radix_tree_t* tree, vitna_radix_node_t* node) {
    if (!tree || !node || node == tree->root) return;

    if (node->ref_count > 0) {
        node->ref_count--;
    }

    if (node->parent) {
        vitna_radix_node_decref(tree, node->parent);
    }
}
