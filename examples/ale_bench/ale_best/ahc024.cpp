// EVOLVE-BLOCK-START
#pragma GCC optimize("O3,unroll-loops")

#include <iostream>
#include <vector>
#include <map>       // For temp_adj_deltas_map_global
#include <queue>
#include <algorithm> // For std::min, std::max, std::sort, std::unique, std::shuffle
#include <random>    // For XorShift and std::shuffle
#include <chrono>
#include <utility>   // For std::pair
#include <cmath>     // For std::exp, std::pow
#include <climits>   // For UINT_MAX

// --- Globals ---
const int N_FIXED = 50;
const int M_FIXED = 100; // Max ward ID, problem states M=100

std::vector<std::vector<int>> current_grid_state(N_FIXED, std::vector<int>(N_FIXED));
std::vector<std::vector<int>> best_grid_state(N_FIXED, std::vector<int>(N_FIXED));
int best_score_val = -1; // Stores count of 0-cells for the best state

struct XorShift {
    unsigned int x, y, z, w;
    XorShift() { 
        // Using std::random_device for better seed initialization
        std::random_device rd;
        x = rd();
        y = rd();
        z = rd();
        w = rd();
        // Ensure no zero initial state for w, which is common if rd() produces same values or all are 0
        if (x == 0 && y == 0 && z == 0 && w == 0) w = 1; // Or any non-zero value
    }
    unsigned int next_uint() {
        unsigned int t = x;
        t ^= t << 11;
        t ^= t >> 8;
        x = y; y = z; z = w;
        w ^= w >> 19;
        w ^= t;
        return w;
    }
    double next_double() { // In [0,1)
        return (double)next_uint() / ((double)UINT_MAX + 1.0);
    }
    int next_int(int exclusive_max_val) { // In [0, exclusive_max_val - 1]
        if (exclusive_max_val <= 0) return 0; 
        return next_uint() % exclusive_max_val;
    }
    // For std::shuffle
    using result_type = unsigned int;
    static constexpr unsigned int min() { return 0; }
    static constexpr unsigned int max() { return UINT_MAX; }
    unsigned int operator()() { return next_uint(); }
};
XorShift rnd_gen; // Global instance
auto G_START_TIME = std::chrono::high_resolution_clock::now();

double time_elapsed_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - G_START_TIME).count();
}

struct AdjacencyInfo {
    bool matrix[M_FIXED + 1][M_FIXED + 1];
    AdjacencyInfo() {
        for (int i = 0; i <= M_FIXED; ++i) for (int j = 0; j <= M_FIXED; ++j) matrix[i][j] = false;
    }
    void set_adj(int c1, int c2) {
        if (c1 == c2) return;
        matrix[std::min(c1, c2)][std::max(c1, c2)] = true;
    }
    bool is_adj(int c1, int c2) const {
        if (c1 == c2) return false; 
        return matrix[std::min(c1, c2)][std::max(c1, c2)];
    }
};
AdjacencyInfo required_adjacencies; 
bool ward_has_any_req_adj[M_FIXED + 1]; 

struct BorderEdgeTracker {
    int counts_arr[M_FIXED + 1][M_FIXED + 1];
    BorderEdgeTracker() { clear(); }
    void add_edge(int c1, int c2) {
        if (c1 == c2) return;
        counts_arr[std::min(c1, c2)][std::max(c1, c2)]++;
    }
    void remove_edge(int c1, int c2) {
        if (c1 == c2) return;
        counts_arr[std::min(c1, c2)][std::max(c1, c2)]--;
    }
    int get_count(int c1, int c2) const {
        if (c1 == c2) return 0;
        return counts_arr[std::min(c1, c2)][std::max(c1, c2)];
    }
    void clear() {
        for (int i = 0; i <= M_FIXED; ++i) for (int j = 0; j <= M_FIXED; ++j) counts_arr[i][j] = 0;
    }
};
BorderEdgeTracker current_border_edges_tracker; 

std::vector<std::vector<std::pair<int, int>>> cells_by_color(M_FIXED + 1);
std::vector<std::vector<int>> pos_in_color_list(N_FIXED, std::vector<int>(N_FIXED));

unsigned int visited_marker_grid[N_FIXED][N_FIXED]; 
unsigned int current_visit_marker = 0; 

std::queue<std::pair<int, int>> q_bfs_global; 

const int DR[] = {-1, 1, 0, 0}; 
const int DC[] = {0, 0, -1, 1};

inline bool is_cell_on_grid(int r, int c) { return r >= 0 && r < N_FIXED && c >= 0 && c < N_FIXED; }

void increment_bfs_marker() {
    current_visit_marker++;
    if (current_visit_marker == 0) { 
        for (int i = 0; i < N_FIXED; ++i) {
            for (int j = 0; j < N_FIXED; ++j) {
                visited_marker_grid[i][j] = 0;
            }
        }
        current_visit_marker = 1; 
    }
}

void clear_global_bfs_queue() {
    std::queue<std::pair<int, int>> empty_queue;
    std::swap(q_bfs_global, empty_queue);
}

void add_cell_to_color_ds(int r, int c, int color) {
    cells_by_color[color].push_back({r,c});
    pos_in_color_list[r][c] = cells_by_color[color].size() - 1;
}

void remove_cell_from_color_ds(int r, int c, int color) {
    int idx_to_remove = pos_in_color_list[r][c];
    std::pair<int,int> last_cell = cells_by_color[color].back();
    
    cells_by_color[color][idx_to_remove] = last_cell;
    pos_in_color_list[last_cell.first][last_cell.second] = idx_to_remove;
    
    cells_by_color[color].pop_back();
}

void initialize_all_data_structures(const std::vector<std::vector<int>>& initial_grid) {
    required_adjacencies = AdjacencyInfo(); 
    current_border_edges_tracker.clear();
    for(int i=0; i <= M_FIXED; ++i) cells_by_color[i].clear();

    for (int i = 0; i < N_FIXED; ++i) {
        for (int j = 0; j < N_FIXED; ++j) {
            current_grid_state[i][j] = initial_grid[i][j]; 
            add_cell_to_color_ds(i, j, initial_grid[i][j]);
        }
    }
    
    for (int i = 0; i < N_FIXED; ++i) {
        for (int j = 0; j < N_FIXED; ++j) {
            int initial_color_val = initial_grid[i][j];
            if (i == 0 || i == N_FIXED - 1 || j == 0 || j == N_FIXED - 1) {
                required_adjacencies.set_adj(0, initial_color_val);
            }
            if (j + 1 < N_FIXED && initial_color_val != initial_grid[i][j+1]) {
                required_adjacencies.set_adj(initial_color_val, initial_grid[i][j+1]);
            }
            if (i + 1 < N_FIXED && initial_color_val != initial_grid[i+1][j]) {
                required_adjacencies.set_adj(initial_color_val, initial_grid[i+1][j]);
            }

            int current_color_val = current_grid_state[i][j]; 
            if (i == 0) current_border_edges_tracker.add_edge(0, current_color_val);
            if (i == N_FIXED - 1) current_border_edges_tracker.add_edge(0, current_color_val);
            if (j == 0) current_border_edges_tracker.add_edge(0, current_color_val);
            if (j == N_FIXED - 1) current_border_edges_tracker.add_edge(0, current_color_val);
            
            if (j + 1 < N_FIXED && current_color_val != current_grid_state[i][j+1]) {
                current_border_edges_tracker.add_edge(current_color_val, current_grid_state[i][j+1]);
            }
            if (i + 1 < N_FIXED && current_color_val != current_grid_state[i+1][j]) {
                current_border_edges_tracker.add_edge(current_color_val, current_grid_state[i+1][j]);
            }
        }
    }

    for (int c1 = 0; c1 <= M_FIXED; ++c1) {
        ward_has_any_req_adj[c1] = false; 
        for (int c2 = 0; c2 <= M_FIXED; ++c2) {
            if (c1 == c2) continue;
            if (required_adjacencies.is_adj(c1, c2)) {
                ward_has_any_req_adj[c1] = true;
                break;
            }
        }
    }

    best_grid_state = current_grid_state;
    best_score_val = cells_by_color[0].size();
}

bool check_region_connectivity_bfs(int target_color) {
    const auto& cells_of_target_color = cells_by_color[target_color]; 
    if (cells_of_target_color.empty()) return true; 
    
    increment_bfs_marker();
    clear_global_bfs_queue();

    q_bfs_global.push(cells_of_target_color[0]); 
    visited_marker_grid[cells_of_target_color[0].first][cells_of_target_color[0].second] = current_visit_marker;
    
    int count_visited_cells = 0;
    while (!q_bfs_global.empty()) {
        std::pair<int, int> curr = q_bfs_global.front();
        q_bfs_global.pop();
        count_visited_cells++;

        for (int k = 0; k < 4; ++k) {
            int nr = curr.first + DR[k];
            int nc = curr.second + DC[k];
            if (is_cell_on_grid(nr, nc) && 
                current_grid_state[nr][nc] == target_color && 
                visited_marker_grid[nr][nc] != current_visit_marker) {
                visited_marker_grid[nr][nc] = current_visit_marker;
                q_bfs_global.push({nr, nc});
            }
        }
    }
    return count_visited_cells == cells_of_target_color.size();
}

bool check_region_0_connectivity_full() {
    const auto& cells_c0 = cells_by_color[0];
    if (cells_c0.empty()) {
        return true; 
    }

    increment_bfs_marker();
    clear_global_bfs_queue();

    bool any_boundary_zero_cell_found = false;
    for (const auto& cell_coord : cells_c0) {
        int r = cell_coord.first;
        int c = cell_coord.second;
        if (r == 0 || r == N_FIXED - 1 || c == 0 || c == N_FIXED - 1) {
            if (visited_marker_grid[r][c] != current_visit_marker) { 
                 q_bfs_global.push(cell_coord);
                 visited_marker_grid[r][c] = current_visit_marker;
            }
            any_boundary_zero_cell_found = true;
        }
    }

    if (!any_boundary_zero_cell_found) {
        return false;
    }

    while (!q_bfs_global.empty()) {
        std::pair<int, int> curr = q_bfs_global.front();
        q_bfs_global.pop();

        for (int k_dir = 0; k_dir < 4; ++k_dir) {
            int nr = curr.first + DR[k_dir];
            int nc = curr.second + DC[k_dir];
            if (is_cell_on_grid(nr, nc) &&
                current_grid_state[nr][nc] == 0 && 
                visited_marker_grid[nr][nc] != current_visit_marker) { 
                visited_marker_grid[nr][nc] = current_visit_marker;
                q_bfs_global.push({nr, nc});
            }
        }
    }

    for (const auto& cell_coord : cells_c0) {
        if (visited_marker_grid[cell_coord.first][cell_coord.second] != current_visit_marker) {
            return false; 
        }
    }
    return true;
}

std::map<std::pair<int, int>, int> temp_adj_deltas_map_global;

bool attempt_change_cell_color_and_validate(int r, int c, int old_color, int new_color) {
    current_grid_state[r][c] = new_color;
    remove_cell_from_color_ds(r, c, old_color); 
    add_cell_to_color_ds(r, c, new_color);

    temp_adj_deltas_map_global.clear();
    for (int k_adj=0; k_adj<4; ++k_adj) {
        int nr = r + DR[k_adj];
        int nc = c + DC[k_adj];
        int neighbor_actual_color = is_cell_on_grid(nr,nc) ? current_grid_state[nr][nc] : 0; 
        
        if (old_color != neighbor_actual_color) { 
             temp_adj_deltas_map_global[{std::min(old_color, neighbor_actual_color), std::max(old_color, neighbor_actual_color)}]--;
        }
        if (new_color != neighbor_actual_color) { 
             temp_adj_deltas_map_global[{std::min(new_color, neighbor_actual_color), std::max(new_color, neighbor_actual_color)}]++;
        }
    }
    for(const auto& entry : temp_adj_deltas_map_global) {
        int c1 = entry.first.first; int c2 = entry.first.second; int delta = entry.second;
        if (delta > 0) for(int i=0; i<delta; ++i) current_border_edges_tracker.add_edge(c1,c2);
        else for(int i=0; i<-delta; ++i) current_border_edges_tracker.remove_edge(c1,c2);
    }

    bool is_change_valid = true;

    for(const auto& entry : temp_adj_deltas_map_global) {
        int c1 = entry.first.first; int c2 = entry.first.second;
        bool has_edge_now = current_border_edges_tracker.get_count(c1, c2) > 0;
        bool needs_edge = required_adjacencies.is_adj(c1, c2);
        if (has_edge_now != needs_edge) {
            is_change_valid = false; break;
        }
    }
    
    if (is_change_valid && old_color != 0 && cells_by_color[old_color].empty() && ward_has_any_req_adj[old_color]) {
        is_change_valid = false;
    }
    
    if (is_change_valid && old_color != 0 && !cells_by_color[old_color].empty()) {
        if (!check_region_connectivity_bfs(old_color)) is_change_valid = false;
    }

    if (is_change_valid && new_color != 0) { 
         if (!check_region_connectivity_bfs(new_color)) is_change_valid = false;
    }
    
    if (is_change_valid && (old_color == 0 || new_color == 0)) { 
        if (!cells_by_color[0].empty()) { 
            if (!check_region_0_connectivity_full()) is_change_valid = false;
        } else { 
            if (ward_has_any_req_adj[0]) { 
                 is_change_valid = false;
            }
        }
    }

    if (!is_change_valid) { 
        current_grid_state[r][c] = old_color; 
        remove_cell_from_color_ds(r, c, new_color); 
        add_cell_to_color_ds(r, c, old_color);      
        
        for(const auto& entry : temp_adj_deltas_map_global) {
            int c1_ = entry.first.first; int c2_ = entry.first.second; int delta = entry.second;
            if (delta > 0) for(int i=0; i<delta; ++i) current_border_edges_tracker.remove_edge(c1_,c2_);
            else for(int i=0; i<-delta; ++i) current_border_edges_tracker.add_edge(c1_,c2_);
        }
        return false; 
    }
    return true; 
}

void solve_main_logic() {
    std::vector<std::vector<int>> initial_grid_from_input(N_FIXED, std::vector<int>(N_FIXED));
    for (int i = 0; i < N_FIXED; ++i) for (int j = 0; j < N_FIXED; ++j) std::cin >> initial_grid_from_input[i][j];
    
    initialize_all_data_structures(initial_grid_from_input);

    const double GREEDY_PASS_BUDGET_MS = 300.0; 
    double greedy_pass_start_abs_time = time_elapsed_ms();
    
    std::vector<std::pair<int,int>> all_cells_shuffled;
    all_cells_shuffled.reserve(N_FIXED * N_FIXED);
    for(int r_idx=0; r_idx<N_FIXED; ++r_idx) for(int c_idx=0; c_idx<N_FIXED; ++c_idx) all_cells_shuffled.push_back({r_idx,c_idx});
    
    std::shuffle(all_cells_shuffled.begin(), all_cells_shuffled.end(), rnd_gen);
    for (const auto& cell_coords : all_cells_shuffled) {
        if (time_elapsed_ms() - greedy_pass_start_abs_time > GREEDY_PASS_BUDGET_MS) break;
        
        int r = cell_coords.first; int c = cell_coords.second;
        int original_color = current_grid_state[r][c];
        if (original_color == 0) continue; 

        if (attempt_change_cell_color_and_validate(r, c, original_color, 0)) {
            int current_zeros_count = cells_by_color[0].size();
            if (current_zeros_count > best_score_val) {
                best_score_val = current_zeros_count;
                best_grid_state = current_grid_state; 
            }
        }
    }
    
    double sa_start_temp = 2.0; 
    double sa_end_temp = 0.01; 
    const double TOTAL_COMPUTATION_TIME_MS = 1950.0; 
    
    double sa_start_abs_time = time_elapsed_ms();
    double sa_total_duration_ms = TOTAL_COMPUTATION_TIME_MS - sa_start_abs_time;
    if (sa_total_duration_ms <= 0) sa_total_duration_ms = 1.0; 
    
    int iter_count = 0;
    while(true) {
        iter_count++;
        if(iter_count % 256 == 0) { 
             if (time_elapsed_ms() >= TOTAL_COMPUTATION_TIME_MS) break;
        }

        double time_spent_in_sa = time_elapsed_ms() - sa_start_abs_time;
        double progress_ratio = (sa_total_duration_ms > 1e-9) ? (time_spent_in_sa / sa_total_duration_ms) : 1.0;
        progress_ratio = std::min(progress_ratio, 1.0); 
        
        double current_temperature = sa_start_temp * std::pow(sa_end_temp / sa_start_temp, progress_ratio);
        current_temperature = std::max(current_temperature, sa_end_temp); 
        
        int r_coord = rnd_gen.next_int(N_FIXED);
        int c_coord = rnd_gen.next_int(N_FIXED);
        int original_color_at_cell = current_grid_state[r_coord][c_coord];
        
        int candidate_new_colors[5]; 
        int num_candidate_options = 0;
        candidate_new_colors[num_candidate_options++] = 0; 
        for(int k_neighbor_idx=0; k_neighbor_idx<4; ++k_neighbor_idx) {
            int nr = r_coord + DR[k_neighbor_idx];
            int nc = c_coord + DC[k_neighbor_idx];
            if (is_cell_on_grid(nr,nc)) {
                candidate_new_colors[num_candidate_options++] = current_grid_state[nr][nc];
            } else { 
                candidate_new_colors[num_candidate_options++] = 0;
            }
        }
        int new_proposed_color = candidate_new_colors[rnd_gen.next_int(num_candidate_options)];

        if (original_color_at_cell == new_proposed_color) continue; 
        
        int delta_in_score_metric = 0; 
        if (new_proposed_color == 0 && original_color_at_cell != 0) delta_in_score_metric = 1;
        else if (new_proposed_color != 0 && original_color_at_cell == 0) delta_in_score_metric = -1;
        
        if (attempt_change_cell_color_and_validate(r_coord, c_coord, original_color_at_cell, new_proposed_color)) {
            bool accept_this_move = false;
            if (delta_in_score_metric >= 0) { 
                accept_this_move = true;
                if (cells_by_color[0].size() > best_score_val) { 
                    best_score_val = cells_by_color[0].size();
                    best_grid_state = current_grid_state; 
                }
            } else { 
                if (current_temperature > 1e-9 && rnd_gen.next_double() < std::exp((double)delta_in_score_metric / current_temperature)) {
                    accept_this_move = true;
                } else {
                    accept_this_move = false;
                }
            }

            if (!accept_this_move) { 
                current_grid_state[r_coord][c_coord] = original_color_at_cell; 
                remove_cell_from_color_ds(r_coord, c_coord, new_proposed_color); 
                add_cell_to_color_ds(r_coord, c_coord, original_color_at_cell);      
                
                for(const auto& entry : temp_adj_deltas_map_global) { 
                    int c1_ = entry.first.first; int c2_ = entry.first.second; int delta = entry.second;
                    if (delta > 0) for(int i=0; i<delta; ++i) current_border_edges_tracker.remove_edge(c1_,c2_);
                    else for(int i=0; i<-delta; ++i) current_border_edges_tracker.add_edge(c1_,c2_);
                }
            }
        } 
    }

    for (int i = 0; i < N_FIXED; ++i) {
        for (int j = 0; j < N_FIXED; ++j) {
            std::cout << best_grid_state[i][j] << (j == N_FIXED - 1 ? "" : " ");
        }
        std::cout << std::endl;
    }
}

int main() {
    std::ios_base::sync_with_stdio(false); std::cin.tie(NULL);
    G_START_TIME = std::chrono::high_resolution_clock::now();
    
    int n_in_dummy, m_in_dummy; 
    std::cin >> n_in_dummy >> m_in_dummy; 
    
    solve_main_logic();
    return 0;
}

// EVOLVE-BLOCK-END