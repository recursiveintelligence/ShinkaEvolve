// EVOLVE-BLOCK-START
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <map> // For A* visited set
#include <iomanip>
#include <chrono>
#include <functional> // For std::hash
#include <cmath>      // For std::round
#include <random>     // For std::mt19937
#include <numeric>    // For std::iota
#include <queue>      // For A* search (priority_queue)

// Constants for tile connections
const int LEFT_MASK = 1;
const int UP_MASK = 2;
const int RIGHT_MASK = 4;
const int DOWN_MASK = 8;

// Max N value, actual N read from input
const int N_MAX_CONST = 10; 
int N_actual; // Actual N for the current test case
int T_param;  // Actual T for the current test case

const int DR_TILE_RELATIVE_TO_EMPTY[] = {-1, 1, 0, 0}; 
const int DC_TILE_RELATIVE_TO_EMPTY[] = {0, 0, -1, 1};
const char MOVE_CHARS[] = {'U', 'D', 'L', 'R'};


std::mt19937 zobrist_rng_engine(123456789); 
std::uniform_int_distribution<uint64_t> distrib_uint64;
uint64_t zobrist_tile_keys[N_MAX_CONST][N_MAX_CONST][16];


void init_zobrist_keys() {
    for (int i = 0; i < N_actual; ++i) {
        for (int j = 0; j < N_actual; ++j) {
            for (int k = 0; k < 16; ++k) { 
                zobrist_tile_keys[i][j][k] = distrib_uint64(zobrist_rng_engine);
            }
        }
    }
}

int hex_char_to_int(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    return c - 'a' + 10;
}


struct Board {
    std::array<std::array<char, N_MAX_CONST>, N_MAX_CONST> tiles;
    int empty_r, empty_c;
    uint64_t zobrist_hash_value;

    Board() : empty_r(0), empty_c(0), zobrist_hash_value(0) {}

    void calculate_initial_hash() {
        zobrist_hash_value = 0;
        for (int i = 0; i < N_actual; ++i) {
            for (int j = 0; j < N_actual; ++j) {
                zobrist_hash_value ^= zobrist_tile_keys[i][j][hex_char_to_int(tiles[i][j])];
            }
        }
    }
    
    void update_hash_after_move(int pos_tile_becomes_empty_r, int pos_tile_becomes_empty_c, 
                                int pos_empty_gets_tile_r, int pos_empty_gets_tile_c) {
        int moved_tile_val_int = hex_char_to_int(tiles[pos_empty_gets_tile_r][pos_empty_gets_tile_c]); 
        
        zobrist_hash_value ^= zobrist_tile_keys[pos_tile_becomes_empty_r][pos_tile_becomes_empty_c][moved_tile_val_int];
        zobrist_hash_value ^= zobrist_tile_keys[pos_empty_gets_tile_r][pos_empty_gets_tile_c][0];

        zobrist_hash_value ^= zobrist_tile_keys[pos_tile_becomes_empty_r][pos_tile_becomes_empty_c][0]; 
        zobrist_hash_value ^= zobrist_tile_keys[pos_empty_gets_tile_r][pos_empty_gets_tile_c][moved_tile_val_int];
    }

    bool apply_move_char(char move_char) {
        int move_dir_idx = -1;
        for(int i=0; i<4; ++i) if(MOVE_CHARS[i] == move_char) move_dir_idx = i;
        
        if(move_dir_idx == -1) return false;

        int tile_to_move_r = empty_r + DR_TILE_RELATIVE_TO_EMPTY[move_dir_idx];
        int tile_to_move_c = empty_c + DC_TILE_RELATIVE_TO_EMPTY[move_dir_idx];

        if (tile_to_move_r < 0 || tile_to_move_r >= N_actual || tile_to_move_c < 0 || tile_to_move_c >= N_actual) {
            return false; 
        }

        char moved_tile_hex_val = tiles[tile_to_move_r][tile_to_move_c];
        tiles[empty_r][empty_c] = moved_tile_hex_val; 
        tiles[tile_to_move_r][tile_to_move_c] = '0';  
        
        update_hash_after_move(tile_to_move_r, tile_to_move_c, empty_r, empty_c);
        
        empty_r = tile_to_move_r; 
        empty_c = tile_to_move_c;
        return true;
    }
};


struct ScoreComponents {
    int max_tree_size;
    int num_components; 
};
std::unordered_map<uint64_t, ScoreComponents> s_value_cache_by_hash;
const size_t MAX_SCORE_CACHE_SIZE_CONST = 2000000; 

struct DSU {
    std::vector<int> parent;
    std::vector<int> nodes_in_set; 
    std::vector<int> edges_in_set; 
    int N_sq_total_cells; 

    DSU(int current_N) : N_sq_total_cells(current_N * current_N) {
        parent.resize(N_sq_total_cells);
        std::iota(parent.begin(), parent.end(), 0);
        nodes_in_set.assign(N_sq_total_cells, 0); 
        edges_in_set.assign(N_sq_total_cells, 0);
    }

    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }

    void unite(int i_idx, int j_idx) { 
        int root_i = find(i_idx);
        int root_j = find(j_idx);
        
        if (nodes_in_set[root_i] < nodes_in_set[root_j]) std::swap(root_i, root_j);
        
        parent[root_j] = root_i;
        nodes_in_set[root_i] += nodes_in_set[root_j];
        edges_in_set[root_i] += edges_in_set[root_j];
    }
    
    void add_edge(int u_idx, int v_idx) {
        int root_u = find(u_idx);
        int root_v = find(v_idx);
        if (root_u != root_v) {
            unite(u_idx, v_idx); 
            edges_in_set[find(u_idx)]++; 
        } else {
            edges_in_set[root_u]++; 
        }
    }
};


ScoreComponents calculate_scores(const Board& board) {
    auto it_cache = s_value_cache_by_hash.find(board.zobrist_hash_value);
    if (it_cache != s_value_cache_by_hash.end()) {
        return it_cache->second;
    }

    DSU dsu(N_actual);

    for (int r = 0; r < N_actual; ++r) {
        for (int c = 0; c < N_actual; ++c) {
            int cell_idx = r * N_actual + c;
            if (board.tiles[r][c] != '0') {
                dsu.nodes_in_set[cell_idx] = 1; 
            } else {
                dsu.nodes_in_set[cell_idx] = 0; 
            }
        }
    }

    for (int r = 0; r < N_actual; ++r) {
        for (int c = 0; c < N_actual - 1; ++c) { 
            int tile1_val = hex_char_to_int(board.tiles[r][c]);
            int tile2_val = hex_char_to_int(board.tiles[r][c+1]);
            if (tile1_val != 0 && tile2_val != 0) { 
                if ((tile1_val & RIGHT_MASK) && (tile2_val & LEFT_MASK)) {
                    dsu.add_edge(r * N_actual + c, r * N_actual + (c + 1));
                }
            }
        }
    }
    for (int r = 0; r < N_actual - 1; ++r) { 
        for (int c = 0; c < N_actual; ++c) {
            int tile1_val = hex_char_to_int(board.tiles[r][c]);
            int tile2_val = hex_char_to_int(board.tiles[r+1][c]);
            if (tile1_val != 0 && tile2_val != 0) { 
                if ((tile1_val & DOWN_MASK) && (tile2_val & UP_MASK)) {
                    dsu.add_edge(r * N_actual + c, (r + 1) * N_actual + c);
                }
            }
        }
    }
    
    int max_tree_size = 0;
    int total_num_components = 0;

    for (int i = 0; i < dsu.N_sq_total_cells; ++i) {
        if (dsu.parent[i] == i && dsu.nodes_in_set[i] > 0) { 
            total_num_components++;
            if (dsu.edges_in_set[i] == dsu.nodes_in_set[i] - 1) { 
                if (dsu.nodes_in_set[i] > max_tree_size) {
                    max_tree_size = dsu.nodes_in_set[i];
                }
            }
        }
    }
    
    ScoreComponents result = {max_tree_size, total_num_components};
    if (s_value_cache_by_hash.size() < MAX_SCORE_CACHE_SIZE_CONST) { 
         s_value_cache_by_hash[board.zobrist_hash_value] = result;
    }
    return result;
}


int TARGET_EMPTY_R_GLOBAL_FOR_A_STAR, TARGET_EMPTY_C_GLOBAL_FOR_A_STAR; // Used by A* heuristic
bool A_STAR_PHASE_WAS_RUN = false; // Flag to adjust beam score empty penalty

double calculate_beam_score(const ScoreComponents& scores, int K_total, const Board& current_board_state) {
    int S = scores.max_tree_size;
    
    const double FULL_TREE_BASE_SCORE = 1e18; 
    if (S == N_actual * N_actual - 1) { 
        return FULL_TREE_BASE_SCORE + (double)(T_param * 2 - K_total); 
    }
    
    double W_S = 1e9; 
    double W_NC = W_S * 0.8; // Make W_NC very strong, almost as much as increasing S by 1.
    double W_K = 1.0; 
    double W_empty_dist_penalty_main;

    if (A_STAR_PHASE_WAS_RUN) { // A* moved empty to target initially
        W_empty_dist_penalty_main = W_K * 0.5; // Very low penalty, allow free movement
    } else { // Empty started at target, or A* failed (should not happen)
        W_empty_dist_penalty_main = W_K * 10.0; // Moderate penalty
    }
    
    double score_val = (double)S * W_S;
    if (scores.num_components > 1) { 
         score_val -= (double)(scores.num_components - 1) * W_NC; 
    } else if (scores.num_components == 0 && N_actual * N_actual - 1 > 0) {
         score_val -= (double)(N_actual * N_actual -1) * W_NC; 
    }

    // Bonus for being very close to a full tree and connected
    if (S >= (N_actual * N_actual - 1) - 2 && scores.num_components == 1 && S < N_actual * N_actual - 1) {
        score_val += W_S * 0.5; // Significant bonus to encourage the last step
    }

    score_val -= (double)K_total * W_K;

    // Penalty for empty square relative to (N-1,N-1)
    int dist_empty_to_corner = std::abs(current_board_state.empty_r - (N_actual - 1)) +
                               std::abs(current_board_state.empty_c - (N_actual - 1));
    score_val -= dist_empty_to_corner * W_empty_dist_penalty_main;
        
    return score_val;
}

double calculate_actual_score(int S, int K_total) {
    if (N_actual * N_actual - 1 == 0) return 0; 
    if (S == N_actual * N_actual - 1) {
        if (K_total > T_param) return 0; 
        return std::round(500000.0 * (2.0 - (double)K_total / T_param));
    } else {
        return std::round(500000.0 * (double)S / (N_actual * N_actual - 1.0));
    }
}

struct BeamHistoryEntry {
    int parent_history_idx; 
    char move_char_taken;   
};
std::vector<BeamHistoryEntry> beam_history_storage;
const size_t MAX_BEAM_HISTORY_STORAGE_SIZE_CONST = 3000000; 

struct BeamState {
    Board board; 
    double beam_score_val; 
    int k_beam_moves; 
    int history_idx; 
    int prev_move_direction_idx;

    bool operator<(const BeamState& other) const {
        return beam_score_val > other.beam_score_val;
    }
};

std::chrono::steady_clock::time_point T_START_CHRONO_MAIN;
const int TIME_LIMIT_MS_SLACK_CONST = 400; // Universal slack
long long TIME_LIMIT_MS_EFFECTIVE_MAIN;


std::mt19937 rng_stochastic_selection_main;
std::unordered_map<uint64_t, int> min_K_to_reach_by_hash_main; 
const size_t MAX_MIN_K_CACHE_SIZE_CONST = 2000000; 


struct AStarEmptyState {
    int r, c;
    int g_cost;
    std::string path;

    bool operator>(const AStarEmptyState& other) const {
        int h_cost_this = std::abs(r - TARGET_EMPTY_R_GLOBAL_FOR_A_STAR) + std::abs(c - TARGET_EMPTY_C_GLOBAL_FOR_A_STAR);
        int h_cost_other = std::abs(other.r - TARGET_EMPTY_R_GLOBAL_FOR_A_STAR) + std::abs(other.c - TARGET_EMPTY_C_GLOBAL_FOR_A_STAR);
        if (g_cost + h_cost_this != other.g_cost + h_cost_other) {
            return g_cost + h_cost_this > other.g_cost + h_cost_other;
        }
        return g_cost > other.g_cost; 
    }
};

std::string find_path_for_empty(const Board& initial_board_state_for_A_star, int target_r, int target_c) {
    TARGET_EMPTY_R_GLOBAL_FOR_A_STAR = target_r; 
    TARGET_EMPTY_C_GLOBAL_FOR_A_STAR = target_c;

    std::priority_queue<AStarEmptyState, std::vector<AStarEmptyState>, std::greater<AStarEmptyState>> pq;
    std::vector<std::vector<int>> min_g_cost_grid(N_actual, std::vector<int>(N_actual, T_param + 1));

    pq.push({initial_board_state_for_A_star.empty_r, initial_board_state_for_A_star.empty_c, 0, ""});
    min_g_cost_grid[initial_board_state_for_A_star.empty_r][initial_board_state_for_A_star.empty_c] = 0;

    int A_star_max_depth = N_actual * N_actual * 2; // Allow more depth just in case

    while(!pq.empty()){
        AStarEmptyState current = pq.top();
        pq.pop();

        if (current.g_cost > min_g_cost_grid[current.r][current.c]) {
             continue;
        }

        if (current.r == target_r && current.c == target_c) {
            return current.path;
        }

        if (current.g_cost >= A_star_max_depth) continue;

        for (int move_idx = 0; move_idx < 4; ++move_idx) { 
            int tile_that_moves_r = current.r + DR_TILE_RELATIVE_TO_EMPTY[move_idx];
            int tile_that_moves_c = current.c + DC_TILE_RELATIVE_TO_EMPTY[move_idx];
            
            if (tile_that_moves_r < 0 || tile_that_moves_r >= N_actual || tile_that_moves_c < 0 || tile_that_moves_c >= N_actual) {
                continue; 
            }
            
            int next_empty_r = tile_that_moves_r;
            int next_empty_c = tile_that_moves_c;
            
            int next_g_cost = current.g_cost + 1;

            if (min_g_cost_grid[next_empty_r][next_empty_c] <= next_g_cost) {
                continue;
            }
            min_g_cost_grid[next_empty_r][next_empty_c] = next_g_cost;
            pq.push({next_empty_r, next_empty_c, next_g_cost, current.path + MOVE_CHARS[move_idx]});
        }
    }
    return ""; 
}

std::string reconstruct_beam_path(int final_history_idx) {
    std::string path_str = "";
    int current_trace_hist_idx = final_history_idx;
    while(current_trace_hist_idx > 0 && 
          static_cast<size_t>(current_trace_hist_idx) < beam_history_storage.size() && 
          beam_history_storage[current_trace_hist_idx].parent_history_idx != -1) {
        path_str += beam_history_storage[current_trace_hist_idx].move_char_taken;
        current_trace_hist_idx = beam_history_storage[current_trace_hist_idx].parent_history_idx;
    }
    std::reverse(path_str.begin(), path_str.end());
    return path_str;
}


int main(int /*argc*/, char** /*argv*/) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    
    unsigned int random_seed_stochastic = std::chrono::steady_clock::now().time_since_epoch().count();
    rng_stochastic_selection_main.seed(random_seed_stochastic);

    T_START_CHRONO_MAIN = std::chrono::steady_clock::now();

    std::cin >> N_actual >> T_param;
    
    init_zobrist_keys();

    Board current_board_obj; 
    for (int i = 0; i < N_actual; ++i) {
        std::string row_str;
        std::cin >> row_str;
        for (int j = 0; j < N_actual; ++j) {
            current_board_obj.tiles[i][j] = row_str[j];
            if (current_board_obj.tiles[i][j] == '0') {
                current_board_obj.empty_r = i;
                current_board_obj.empty_c = j;
            }
        }
    }
    current_board_obj.calculate_initial_hash();
    
    std::string initial_empty_moves_path = "";
    int target_empty_final_r = N_actual - 1;
    int target_empty_final_c = N_actual - 1;

    if (current_board_obj.empty_r != target_empty_final_r || current_board_obj.empty_c != target_empty_final_c) {
        initial_empty_moves_path = find_path_for_empty(current_board_obj, target_empty_final_r, target_empty_final_c);
        A_STAR_PHASE_WAS_RUN = !initial_empty_moves_path.empty();
    }
    
    for (char move_char : initial_empty_moves_path) {
        current_board_obj.apply_move_char(move_char); 
    }
    int K_initial_empty_moves = initial_empty_moves_path.length();

    // Adaptive time limit after A*
    auto time_after_astar = std::chrono::steady_clock::now();
    long long elapsed_astar_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_after_astar - T_START_CHRONO_MAIN).count();
    TIME_LIMIT_MS_EFFECTIVE_MAIN = 2950 - elapsed_astar_ms - TIME_LIMIT_MS_SLACK_CONST;


    beam_history_storage.reserve(MAX_BEAM_HISTORY_STORAGE_SIZE_CONST); 
    s_value_cache_by_hash.reserve(MAX_SCORE_CACHE_SIZE_CONST); 
    min_K_to_reach_by_hash_main.reserve(MAX_MIN_K_CACHE_SIZE_CONST);

    std::vector<BeamState> current_beam;
    
    ScoreComponents initial_scores_for_beam = calculate_scores(current_board_obj);
    double initial_beam_eval_score = calculate_beam_score(initial_scores_for_beam, K_initial_empty_moves, current_board_obj);

    beam_history_storage.push_back({-1, ' '}); 
    current_beam.push_back({current_board_obj, initial_beam_eval_score, 0, 0, -1}); 

    double overall_best_actual_score = calculate_actual_score(initial_scores_for_beam.max_tree_size, K_initial_empty_moves);
    std::string overall_best_path_str = initial_empty_moves_path; 

    min_K_to_reach_by_hash_main[current_board_obj.zobrist_hash_value] = K_initial_empty_moves;

    int beam_width; 
    float elite_ratio = 0.2f; // Standard elite ratio
    int stochastic_sample_pool_factor = 3; 

    if (N_actual <= 6) { beam_width = 1200;} // N=6 is small, can afford wider
    else if (N_actual == 7) { beam_width = 1000;}
    else if (N_actual == 8) { beam_width = 700;} // Reduced from 800 to save time slightly
    else if (N_actual == 9) { beam_width = 400;} // Reduced from 500
    else { beam_width = 250;} // N=10, reduced from 300

    std::vector<BeamState> candidates_pool; 
    candidates_pool.reserve(beam_width * 4 + 10); 

    std::vector<BeamState> next_beam_states_temp; 
    next_beam_states_temp.reserve(beam_width + 10);

    std::vector<int> stochastic_selection_indices; 
    stochastic_selection_indices.reserve(stochastic_sample_pool_factor * beam_width + 10);
    
    int k_iter_count_beam = 0;

    for (int k_beam_iter = 0; K_initial_empty_moves + k_beam_iter < T_param; ++k_beam_iter) {
        k_iter_count_beam++;
        if (k_iter_count_beam % 10 == 0) { 
            if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - T_START_CHRONO_MAIN).count() > TIME_LIMIT_MS_EFFECTIVE_MAIN + elapsed_astar_ms) {
                 // Compare against total time budget, not just remaining for beam.
                 // Total time used > total budget minus slack
                if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - T_START_CHRONO_MAIN).count() > 2950 - TIME_LIMIT_MS_SLACK_CONST) {
                    break;
                }
            }
        }
        if (beam_history_storage.size() >= MAX_BEAM_HISTORY_STORAGE_SIZE_CONST - ( (size_t)beam_width * 4 + 100) ) { 
            break; 
        }
        
        candidates_pool.clear();

        for (const auto& current_state_in_beam : current_beam) {
            Board temp_board_for_moves = current_state_in_beam.board; 
            
            int parent_k_beam = current_state_in_beam.k_beam_moves;
            int parent_history_idx = current_state_in_beam.history_idx;
            int prev_m_dir_idx = current_state_in_beam.prev_move_direction_idx;

            for (int move_dir_idx = 0; move_dir_idx < 4; ++move_dir_idx) { 
                if (prev_m_dir_idx != -1) { 
                    if ((prev_m_dir_idx ^ 1) == move_dir_idx) { // Check for U/D or L/R reversal using XOR trick
                        continue; 
                    }
                }
                
                char current_move_char = MOVE_CHARS[move_dir_idx];
                int original_empty_r = temp_board_for_moves.empty_r; 
                int original_empty_c = temp_board_for_moves.empty_c;
                uint64_t original_hash = temp_board_for_moves.zobrist_hash_value;

                int tile_to_move_r = original_empty_r + DR_TILE_RELATIVE_TO_EMPTY[move_dir_idx];
                int tile_to_move_c = original_empty_c + DC_TILE_RELATIVE_TO_EMPTY[move_dir_idx];
                
                if (tile_to_move_r < 0 || tile_to_move_r >= N_actual || tile_to_move_c < 0 || tile_to_move_c >= N_actual) {
                    continue; 
                }

                char moved_tile_hex_val = temp_board_for_moves.tiles[tile_to_move_r][tile_to_move_c];
                temp_board_for_moves.tiles[original_empty_r][original_empty_c] = moved_tile_hex_val;
                temp_board_for_moves.tiles[tile_to_move_r][tile_to_move_c] = '0'; 
                temp_board_for_moves.empty_r = tile_to_move_r; 
                temp_board_for_moves.empty_c = tile_to_move_c;
                temp_board_for_moves.update_hash_after_move(tile_to_move_r, tile_to_move_c, original_empty_r, original_empty_c);
                
                int next_k_beam = parent_k_beam + 1;
                int next_K_total = K_initial_empty_moves + next_k_beam;

                bool already_reached_better = false;
                auto it_map = min_K_to_reach_by_hash_main.find(temp_board_for_moves.zobrist_hash_value);
                if (it_map != min_K_to_reach_by_hash_main.end()) {
                    if (it_map->second <= next_K_total) {
                        already_reached_better = true;
                    } else {
                         it_map->second = next_K_total; 
                    }
                } else { 
                    if (min_K_to_reach_by_hash_main.size() < MAX_MIN_K_CACHE_SIZE_CONST) {
                        min_K_to_reach_by_hash_main[temp_board_for_moves.zobrist_hash_value] = next_K_total;
                    }
                }

                if(already_reached_better) {
                    temp_board_for_moves.tiles[tile_to_move_r][tile_to_move_c] = moved_tile_hex_val; 
                    temp_board_for_moves.tiles[original_empty_r][original_empty_c] = '0'; 
                    temp_board_for_moves.empty_r = original_empty_r;
                    temp_board_for_moves.empty_c = original_empty_c;
                    temp_board_for_moves.zobrist_hash_value = original_hash; 
                    continue; 
                }

                ScoreComponents next_scores = calculate_scores(temp_board_for_moves);
                double next_beam_eval_score = calculate_beam_score(next_scores, next_K_total, temp_board_for_moves);
                
                beam_history_storage.push_back({parent_history_idx, current_move_char});
                int new_history_idx = beam_history_storage.size() - 1;
                
                candidates_pool.push_back({temp_board_for_moves, next_beam_eval_score, next_k_beam, new_history_idx, move_dir_idx});

                double current_actual_score_val = calculate_actual_score(next_scores.max_tree_size, next_K_total);
                if (current_actual_score_val > overall_best_actual_score) {
                    overall_best_actual_score = current_actual_score_val;
                    overall_best_path_str = initial_empty_moves_path + reconstruct_beam_path(new_history_idx);
                } else if (current_actual_score_val == overall_best_actual_score) {
                    // Prefer shorter paths for same score
                    if ((initial_empty_moves_path + reconstruct_beam_path(new_history_idx)).length() < overall_best_path_str.length()){
                         overall_best_path_str = initial_empty_moves_path + reconstruct_beam_path(new_history_idx);
                    }
                }
                
                temp_board_for_moves.tiles[tile_to_move_r][tile_to_move_c] = moved_tile_hex_val;
                temp_board_for_moves.tiles[original_empty_r][original_empty_c] = '0';
                temp_board_for_moves.empty_r = original_empty_r;
                temp_board_for_moves.empty_c = original_empty_c;
                temp_board_for_moves.zobrist_hash_value = original_hash; 
            }
        }

        if (candidates_pool.empty()) break; 

        std::sort(candidates_pool.begin(), candidates_pool.end()); 
        
        next_beam_states_temp.clear();
        int num_elites = std::min(static_cast<int>(candidates_pool.size()), static_cast<int>(beam_width * elite_ratio));
        num_elites = std::max(0, num_elites); 

        for(int i=0; i < num_elites && i < static_cast<int>(candidates_pool.size()); ++i) {
            next_beam_states_temp.push_back(candidates_pool[i]);
        }
        
        if (next_beam_states_temp.size() < static_cast<size_t>(beam_width) && candidates_pool.size() > static_cast<size_t>(num_elites)) {
            stochastic_selection_indices.clear();
            int pool_start_idx = num_elites;
            int pool_end_idx = std::min(static_cast<int>(candidates_pool.size()), num_elites + stochastic_sample_pool_factor * beam_width);
            
            for(int i = pool_start_idx; i < pool_end_idx; ++i) {
                stochastic_selection_indices.push_back(i);
            }
            if (!stochastic_selection_indices.empty()){
                std::shuffle(stochastic_selection_indices.begin(), stochastic_selection_indices.end(), rng_stochastic_selection_main);
            }

            for(size_t i=0; i < stochastic_selection_indices.size() && next_beam_states_temp.size() < static_cast<size_t>(beam_width); ++i) {
                next_beam_states_temp.push_back(candidates_pool[stochastic_selection_indices[i]]);
            }
        }
        
        current_beam = next_beam_states_temp; 
        if (current_beam.empty()) break; 
    }
    
    std::cout << overall_best_path_str << std::endl;
    
    return 0;
}

// EVOLVE-BLOCK-END